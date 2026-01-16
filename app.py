import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import beta

# --- 설정 ---
st.set_page_config(page_title="Ad Analytics System v2", layout="wide")
st.title("🎯 광고 매체 통계분석 시스템")
st.markdown("**Empirical Bayes & CUSUM 기반 소재 성과 분석**")
st.markdown("---")

# --- 데이터 로드 ---
def load_and_clean_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file, sep='\t' if uploaded_file.name.endswith('.tsv') else ',')
        
        df.columns = [c.strip() for c in df.columns]
        mapping = {
            '날짜': ['날짜', '일자', 'date'], 
            '상품': ['상품명', '상품', 'product'], 
            '소재': ['소재명', '소재', 'material'],
            '노출': ['노출수', '노출', 'impressions'], 
            '클릭': ['클릭수', '클릭', 'clicks'], 
            '비용': ['비용', '지출', 'cost']
        }
        
        final_df = pd.DataFrame()
        for k, v in mapping.items():
            for col in v:
                if col in df.columns: 
                    final_df[k] = df[col]
                    break
        
        final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
        for col in ['노출', '클릭', '비용']:
            final_df[col] = pd.to_numeric(
                final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), 
                errors='coerce'
            ).fillna(0)
        
        final_df['CTR(%)'] = (final_df['클릭'] / (final_df['노출'] + 1e-9) * 100)
        final_df['CPC'] = final_df['비용'] / (final_df['클릭'] + 1e-9)
        final_df['ID'] = "[" + final_df['상품'].astype(str) + "] " + final_df['소재'].astype(str)
        
        return final_df.dropna(subset=['날짜']).sort_values(['ID', '날짜'])
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

# --- Empirical Bayes 분석 ---
def analyze_empirical_bayes(df):
    global_ctr = df['클릭'].sum() / (df['노출'].sum() + 1e-9)
    id_stats = df.groupby('ID').agg({'클릭': 'sum', '노출': 'sum', '비용': 'sum'})
    id_ctrs = id_stats['클릭'] / (id_stats['노출'] + 1e-9)
    var_ctr = max(id_ctrs.var(), 1e-7)
    
    kappa = (global_ctr * (1 - global_ctr) / var_ctr) - 1
    kappa = np.clip(kappa, 10, 1000)
    
    alpha_0, beta_0 = global_ctr * kappa, (1 - global_ctr) * kappa
    
    agg = id_stats.reset_index()
    agg['post_alpha'] = alpha_0 + agg['클릭']
    agg['post_beta'] = beta_0 + (agg['노출'] - agg['클릭'])
    agg['exp_ctr'] = agg['post_alpha'] / (agg['post_alpha'] + agg['post_beta'])
    agg['raw_ctr'] = agg['클릭'] / (agg['노출'] + 1e-9)
    
    # 몬테카를로 시뮬레이션
    samples = np.random.beta(
        agg['post_alpha'].values[:, None], 
        agg['post_beta'].values[:, None], 
        size=(len(agg), 5000)
    )
    agg['prob_is_best'] = np.bincount(
        np.argmax(samples, axis=0), 
        minlength=len(agg)
    ) / 5000
    
    # 최근 7일 평균 비용 (전체 기간 기준)
    max_date = df['날짜'].max()
    date_7d_ago = max_date - timedelta(days=6)  # 7일 = 오늘 포함 6일 전
    last_7d = df[df['날짜'] >= date_7d_ago]
    
    # 소재별 최근 7일 총비용 / 7 = 일평균
    last_costs = last_7d.groupby('ID')['비용'].sum() / 7
    agg = agg.merge(last_costs.rename('avg_cost_7d'), on='ID', how='left').fillna(0)
    
    return agg, (alpha_0, beta_0, kappa, global_ctr)

# --- CUSUM 분석 ---
def get_binomial_cusum(clicks, imps, p0):
    p1 = np.clip(p0 * 0.85, 1e-6, 1-1e-6)
    p0 = np.clip(p0, 1e-6, 1-1e-6)
    llr = clicks * np.log(p1/p0) + (imps - clicks) * np.log((1-p1)/(1-p0))
    s = 0
    cusum = []
    for val in llr:
        s = min(0, s + val)
        cusum.append(s)
    return np.array(cusum)

@st.cache_data
def estimate_h_via_arl(p0, imps_series, target_arl=30, sims=1000):
    """
    몬테카를로 시뮬레이션으로 목표 ARL을 달성하는 h 추정
    
    Parameters:
    - p0: 기준 CTR (정상 상태)
    - imps_series: 노출수 샘플 (실제 데이터에서 추출)
    - target_arl: 목표 평균 런 길이 (일)
    - sims: 시뮬레이션 반복 횟수
    
    Returns:
    - h: 임계값
    - actual_arl: 실제 달성된 ARL
    """
    p1 = np.clip(p0 * 0.85, 1e-6, 1-1e-6)
    p0_clip = np.clip(p0, 1e-6, 1-1e-6)
    llr_success = np.log(p1 / p0_clip)
    llr_failure = np.log((1 - p1) / (1 - p0_clip))
    
    # h 후보 범위를 동적으로 설정
    h_candidates = np.arange(1.0, 30.0, 0.5)
    
    for h in h_candidates:
        run_lengths = []
        
        for _ in range(sims):
            s = 0
            t = 0
            max_iter = 500  # 충분히 긴 시뮬레이션
            
            while t < max_iter:
                t += 1
                # 실제 노출 분포에서 샘플링
                n = np.random.choice(imps_series) if len(imps_series) > 0 else 100000
                c = np.random.binomial(int(n), p0_clip)
                
                # CUSUM 업데이트
                s = min(0, s + (c * llr_success + (int(n) - c) * llr_failure))
                
                if s < -h:
                    break
            
            run_lengths.append(t)
        
        actual_arl = np.mean(run_lengths)
        
        # 목표 ARL에 도달하면 반환
        if actual_arl >= target_arl:
            return h, actual_arl
    
    # 못 찾으면 최대값 반환 (경고와 함께)
    return h_candidates[-1], np.mean(run_lengths)

# --- UI ---
uploaded_file = st.file_uploader(
    "📂 캠페인 데이터 업로드 (CSV/XLSX/TSV)", 
    type=['csv', 'xlsx', 'tsv']
)

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    
    if not df.empty:
        res_agg, (a0, b0, k_est, global_ctr) = analyze_empirical_bayes(df)
        ids = sorted(df['ID'].unique())
        
        # 모드 선택
        st.markdown("---")
        analysis_mode = st.radio(
            "📊 분석 모드 선택",
            ["🎯 실무 모드 (권장)", "🔬 전문가 모드"],
            horizontal=True,
            help="실무 모드: 일상 의사결정에 필요한 핵심 기능만 | 전문가 모드: 통계 분석 및 진단 도구 포함"
        )
        
        # 탭 구성
        if analysis_mode == "🎯 실무 모드 (권장)":
            tabs = st.tabs([
                "📊 성과 요약",
                "🎯 오늘의 액션", 
                "⏰ 소재 수명 예측",
                "📄 주간 리포트"
            ])
            tab_mapping = {0: 0, 1: 1, 2: 2, 3: 6}  # 실무 모드 탭 → 실제 탭 번호
        else:
            tabs = st.tabs([
                "📊 Executive Summary", 
                "🎯 오늘의 액션",
                "⏰ 소재 수명 예측",
                "🧬 Bayesian Analysis", 
                "📉 CUSUM & Anomaly Detection",
                "🎮 예산 시뮬레이터",
                "📄 주간 리포트"
            ])
            tab_mapping = {i: i for i in range(7)}  # 전문가 모드: 모든 탭
        
        # ====================
        # TAB 0: Executive Summary
        # ====================
        tab_idx = 0 if analysis_mode == "🎯 실무 모드 (권장)" else 0
        with tabs[tab_idx]:
            st.markdown("### 📊 핵심 지표 요약")
            
            if analysis_mode == "🎯 실무 모드 (권장)":
                st.info("💡 **실무 모드**: 일상 의사결정에 필요한 핵심 정보만 표시합니다. 통계 분석이 필요하면 '전문가 모드'를 선택하세요.")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("전체 평균 CTR", f"{global_ctr*100:.2f}%")
            col2.metric("분석 기간", f"{(df['날짜'].max() - df['날짜'].min()).days}일")
            col3.metric("총 소재 수", len(ids))
            col4.metric("총 집행 비용", f"{df['비용'].sum():,.0f}원")
            
            with st.expander("ℹ️ CTR(Click-Through Rate)이란?"):
                st.markdown("""
                **CTR = (클릭수 / 노출수) × 100%**
                
                광고가 1000번 노출되어 10번 클릭되었다면 CTR = 1.0%
                - 높을수록: 광고가 사용자에게 매력적
                - 낮을수록: 소재 개선 필요 또는 타겟팅 문제
                - 업계 평균: 디스플레이 0.5~1%, 검색광고 2~5%
                """)
            
            st.markdown("---")
            st.markdown("### 🏆 최고 성과 소재 확률")
            st.markdown("*Bayesian 사후확률 기반 - 5000회 Monte Carlo 시뮬레이션*")
            
            with st.expander("ℹ️ 이 확률은 무엇을 의미하나요?"):
                st.markdown("""
                **"각 소재가 실제로 최고 CTR을 가질 확률"**
                
                예시:
                - 소재 A: 65% → 100번 중 65번은 A가 최고
                - 소재 B: 25% → 100번 중 25번은 B가 최고
                - 소재 C: 10% → 100번 중 10번은 C가 최고
                
                ⚠️ **주의사항:**
                - 이는 **현재 데이터 기준** 확률입니다
                - 향후 성과를 보장하지 않습니다
                - 확률이 비슷하면 → 더 많은 데이터 필요
                - 과도한 집중 투자는 위험할 수 있습니다
                """)
            
            # 확률 바 차트
            fig_prob = px.bar(
                res_agg.sort_values('prob_is_best', ascending=True),
                x='prob_is_best',
                y='ID',
                orientation='h',
                text=res_agg.sort_values('prob_is_best', ascending=True)['prob_is_best'].apply(lambda x: f"{x*100:.1f}%"),
                title="각 소재가 최고 CTR일 확률"
            )
            fig_prob.update_traces(textposition='outside')
            fig_prob.update_xaxes(title="확률", tickformat='.0%')
            st.plotly_chart(fig_prob, use_container_width=True)
            
            st.info(f"""
            **📖 해석:**
            - 가장 높은 확률의 소재가 실제로 최고 성과일 가능성이 가장 높습니다
            - 확률이 비슷하면 → 소재 간 차이가 미미하거나 더 많은 데이터 필요
            - 확률이 명확하게 차이나면 → 통계적으로 유의미한 성과 차이 존재
            """)
            
            st.markdown("---")
            st.markdown("### 📈 소재별 상세 성과")
            
            display_df = res_agg[['ID', 'raw_ctr', 'exp_ctr', '노출', '클릭', '비용', 'prob_is_best']].copy()
            # 먼저 값 변환
            display_df['raw_ctr'] = display_df['raw_ctr'] * 100
            display_df['exp_ctr'] = display_df['exp_ctr'] * 100
            display_df['prob_is_best'] = display_df['prob_is_best'] * 100
            # 그 다음 컬럼명 변경
            display_df.columns = ['소재', '원본CTR(%)', '보정CTR(%)', '노출수', '클릭수', '비용', '최고확률']
            
            st.dataframe(
                display_df.style.format({
                    '원본CTR(%)': '{:.2f}',
                    '보정CTR(%)': '{:.2f}',
                    '노출수': '{:,.0f}',
                    '클릭수': '{:,.0f}',
                    '비용': '{:,.0f}원',
                    '최고확률': '{:.1f}%'
                }).background_gradient(subset=['보정CTR(%)'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            with st.expander("ℹ️ 원본CTR vs 보정CTR 차이"):
                st.markdown("""
                **원본CTR:** 각 소재의 실제 클릭률 (클릭/노출)
                **보정CTR:** Empirical Bayes로 극단값을 완화한 CTR
                
                예시:
                - 소재 A: 노출 100만, 클릭 5000 → 원본 0.5%
                - 소재 B: 노출 100, 클릭 10 → 원본 10% (!)
                
                소재 B는 데이터가 적어서 우연히 높을 가능성 큼
                → 보정CTR은 전체 평균 쪽으로 조정 (예: 2.1%)
                
                **언제 차이가 크나?**
                - 노출수가 적을수록
                - 전체 평균과 차이가 클수록
                """)
        
        # ====================
        # TAB 1: 오늘의 액션
        # ====================
        tab_idx = 1 if analysis_mode == "🎯 실무 모드 (권장)" else 1
        with tabs[tab_idx]:
            st.markdown("### 🎯 오늘의 액션 리스트")
            st.markdown(f"**분석 기준일: {df['날짜'].max().strftime('%Y년 %m월 %d일')}**")
            
            with st.expander("ℹ️ 이 탭은 무엇을 하나요?"):
                st.markdown("""
                **매일 아침 확인할 의사결정 체크리스트**
                
                각 소재를:
                - 🔴 즉시 조치 필요
                - 🟡 주의 관찰
                - 🟢 증액 검토
                - ⚪ 현상 유지
                
                4가지로 분류하여 우선순위를 제시합니다.
                
                **판단 기준:**
                1. CUSUM 이상 감지 (통계적 하락)
                2. 최근 3일 추세 (상승/하락)
                3. Bayesian 최고 확률 (성과 신뢰도)
                """)
            
            st.markdown("---")
            
            # 소재별 액션 판정
            actions = []
            for _, material in res_agg.iterrows():
                mat_id = material['ID']
                mat_data = df[df['ID'] == mat_id].sort_values('날짜')
                
                # 최근 3일 추세
                if len(mat_data) >= 3:
                    recent_3 = mat_data.tail(3)['CTR(%)']
                    trend_change = (recent_3.iloc[-1] - recent_3.iloc[0]) / recent_3.iloc[0] if recent_3.iloc[0] > 0 else 0
                else:
                    trend_change = 0
                
                # CUSUM 상태 (간단히 최근 성과 vs EB CTR)
                recent_ctr = mat_data.tail(3)['CTR(%)'].mean()
                baseline_ctr = material['exp_ctr'] * 100
                cusum_alert = recent_ctr < baseline_ctr * 0.85  # 15% 이상 하락
                
                # 액션 판정
                if cusum_alert and trend_change < -0.1:
                    status = "🔴 즉시 중단 권장"
                    priority = 1
                    reason = f"기준선 대비 15% 이상 하락, 3일 추세 {trend_change*100:.1f}%"
                    action = "예산 재배분 또는 소재 교체"
                elif trend_change < -0.05:
                    status = "🟡 모니터링 강화"
                    priority = 2
                    reason = f"3일 추세 {trend_change*100:.1f}% 하락"
                    action = "1~2일 추가 관찰 후 조치"
                elif material['prob_is_best'] > 0.4 and trend_change > 0.05:
                    status = "🟢 예산 증액 검토"
                    priority = 3
                    reason = f"최고 확률 {material['prob_is_best']*100:.0f}%, 3일 추세 +{trend_change*100:.1f}%"
                    action = "점진적 증액 테스트 (+20~30%)"
                else:
                    status = "⚪ 현상 유지"
                    priority = 4
                    reason = "안정적 성과 유지 중"
                    action = "정기 모니터링"
                
                actions.append({
                    'ID': mat_id,
                    'status': status,
                    'priority': priority,
                    'reason': reason,
                    'action': action,
                    'current_cost': material['avg_cost_7d']
                })
            
            # 우선순위 정렬
            actions_df = pd.DataFrame(actions).sort_values('priority')
            
            # 출력
            for _, action in actions_df.iterrows():
                if action['priority'] == 1:
                    color = "red"
                elif action['priority'] == 2:
                    color = "orange"
                elif action['priority'] == 3:
                    color = "green"
                else:
                    color = "gray"
                
                with st.container():
                    st.markdown(f"### {action['status']}")
                    st.markdown(f"**소재:** {action['ID']}")
                    st.markdown(f"**현재 일평균 비용:** {action['current_cost']:,.0f}원")
                    st.markdown(f"**판단 근거:** {action['reason']}")
                    st.markdown(f"**권장 조치:** {action['action']}")
                    st.markdown("---")
        
        # ====================
        # TAB 2: 소재 수명 예측
        # ====================
        tab_idx = 2 if analysis_mode == "🎯 실무 모드 (권장)" else 2
        with tabs[tab_idx]:
            st.markdown("### ⏰ 소재 수명 예측")
            
            with st.expander("ℹ️ 소재 수명이란?"):
                st.markdown("""
                **소재 피로도 (Creative Fatigue)**
                
                같은 광고를 반복 노출하면:
                - 사용자가 무시하기 시작
                - CTR 점진적 하락
                - 비용 효율 악화
                
                **수명 예측 방법:**
                - 최근 추세를 선형 회귀로 분석
                - 현재 CTR의 50%까지 떨어지는 시점 추정
                - "D-day" 형태로 교체 권장일 제시
                
                ⚠️ **주의:**
                - 단순 선형 가정 (실제는 비선형일 수 있음)
                - 외부 요인(시즌, 경쟁사) 미반영
                - 참고용으로만 활용
                """)
            
            st.markdown("---")
            
            for mat_id in ids:
                mat_data = df[df['ID'] == mat_id].sort_values('날짜')
                
                if len(mat_data) < 5:
                    st.warning(f"{mat_id}: 데이터 부족 (5일 이상 필요)")
                    continue
                
                # 선형 회귀
                from sklearn.linear_model import LinearRegression
                X = np.arange(len(mat_data)).reshape(-1, 1)
                y = mat_data['CTR(%)'].values
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                
                current_ctr = y[-1]
                threshold_ctr = current_ctr * 0.5
                
                # 수명 계산
                if slope < -0.001:  # 하락 추세
                    days_left = (current_ctr - threshold_ctr) / abs(slope)
                    days_left = max(0, int(days_left))
                    
                    if days_left == 0:
                        lifespan_status = "⚠️ 교체 권장 (한계 도달)"
                        color = "red"
                    elif days_left <= 3:
                        lifespan_status = f"🔴 D-{days_left} (긴급)"
                        color = "red"
                    elif days_left <= 7:
                        lifespan_status = f"🟡 D-{days_left} (주의)"
                        color = "orange"
                    else:
                        lifespan_status = f"🟢 D-{days_left} (안정)"
                        color = "green"
                else:
                    lifespan_status = "✅ 안정적 (하락 추세 없음)"
                    color = "green"
                    days_left = None
                
                # 출력
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"### {mat_id}")
                    st.markdown(f"**수명 상태:** {lifespan_status}")
                    st.markdown(f"**현재 CTR:** {current_ctr:.2f}%")
                    st.markdown(f"**일평균 하락률:** {slope:.4f}%p")
                    if days_left is not None and days_left > 0:
                        rec_date = df['날짜'].max() + timedelta(days=days_left)
                        st.markdown(f"**교체 권장일:** {rec_date.strftime('%Y-%m-%d')}")
                
                with col2:
                    # 미니 차트
                    fig_mini = go.Figure()
                    fig_mini.add_trace(go.Scatter(
                        y=y,
                        mode='lines+markers',
                        name='실제 CTR'
                    ))
                    # 추세선
                    trend_line = model.predict(X)
                    fig_mini.add_trace(go.Scatter(
                        y=trend_line,
                        mode='lines',
                        name='추세',
                        line=dict(dash='dash')
                    ))
                    fig_mini.update_layout(
                        height=200,
                        showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    st.plotly_chart(fig_mini, use_container_width=True)
                
                st.markdown("---")
        
        # ====================
        # TAB 3: Bayesian Analysis (전문가 모드만)
        # ====================
        if analysis_mode == "🔬 전문가 모드":
            with tabs[3]:
                st.markdown("### 🧬 Empirical Bayes 방법론")
            
            with st.expander("ℹ️ Empirical Bayes란? (초보자용 설명)"):
                st.markdown("""
                **문제 상황:**
                - 소재 A: 노출 100만, 클릭 5000 → CTR 0.50%
                - 소재 B: 노출 100, 클릭 5 → CTR 5.00% (10배!)
                
                소재 B가 정말 10배 좋을까요? **아닙니다.** 데이터가 적어서 우연일 가능성 높습니다.
                
                **Empirical Bayes 해결책:**
                1. 전체 평균 CTR을 "사전 믿음"으로 설정
                2. 각 소재의 실제 데이터와 결합
                3. 극단적인 값을 전체 평균 쪽으로 "당김"
                
                **결과:**
                - 데이터 많은 소재 → 거의 그대로
                - 데이터 적은 소재 → 전체 평균에 가깝게 보정
                
                이렇게 하면 "운 좋게 높은" 소재에 속지 않습니다!
                """)
            
            st.markdown(f"""
            **핵심 개념:**
            - 소표본에서 CTR은 변동성이 큽니다 (클릭 몇 개로 100% or 0% 가능)
            - 전체 평균을 사전 정보로 활용해 극단값을 보정합니다
            - "전체적으로 CTR이 {global_ctr*100:.2f}%인데, 이 소재만 {global_ctr*100*3:.1f}%는 의심스럽다"
            """)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Prior α₀", f"{a0:.1f}")
            col2.metric("Prior β₀", f"{b0:.1f}")
            col3.metric("신뢰도 κ (Kappa)", f"{k_est:.1f}")
            
            with st.expander("ℹ️ κ(Kappa) 값이란?"):
                st.markdown(f"""
                **κ(Kappa) = 사전 정보의 강도**
                
                κ가 크면 → 전체 평균을 더 신뢰 (보수적 평가)
                κ가 작으면 → 개별 소재 데이터를 더 신뢰
                
                **현재 값: {k_est:.1f}**
                - κ = 10 정도: 개별 데이터 중시 (10번 시행 정도의 신뢰도)
                - κ = 100 정도: 균형 (100번 시행 정도의 신뢰도)
                - κ = 1000 정도: 전체 평균 중시 (1000번 시행 정도의 신뢰도)
                
                적정 범위: 10~1000 (현재 {'✅ 적절' if 10 < k_est < 1000 else '⚠️ 조정 필요'})
                
                ⚠️ **기술적 한계:**
                현재 κ는 관측 CTR 분산 기반 (Method of Moments)
                노출수 차이가 큰 경우 과소 추정 가능
                → 향후 Beta-Binomial MLE 또는 Hierarchical Bayes 권장
                """)
            
            st.markdown("---")
            st.markdown("### 📊 사후확률 분포 (Posterior Distribution)")
            
            with st.expander("ℹ️ 사후확률 분포란?"):
                st.markdown("""
                **이 그래프는 "각 소재의 진짜 CTR 범위"를 보여줍니다**
                
                예시:
                - 소재 A: 0.4%~0.6% 사이에 95% 확률로 존재
                - 소재 B: 0.3%~0.9% 사이에 95% 확률로 존재
                
                **분포가 좁으면:**
                - 데이터가 많거나 일관성이 높음
                - "이 소재의 진짜 성과를 확신함"
                
                **분포가 넓으면:**
                - 데이터가 적거나 변동성이 큼
                - "더 많은 테스트 필요"
                
                **분포가 겹치면:**
                - 소재 간 차이가 통계적으로 명확하지 않음
                - "둘 다 비슷할 가능성 높음"
                
                **💡 Tip:** 그래프 위에 마우스를 올리면 각 곡선의 소재명을 확인할 수 있습니다!
                """)
            
            fig_post = go.Figure()
            colors = px.colors.qualitative.Set2  # 구분 가능한 색상 팔레트
            
            for idx, (_, row) in enumerate(res_agg.iterrows()):
                x = np.linspace(0, 0.03, 500)
                y = beta.pdf(x, row['post_alpha'], row['post_beta'])
                
                fig_post.add_trace(go.Scatter(
                    x=x*100, 
                    y=y, 
                    name=row['ID'],
                    mode='lines',
                    fill='tozeroy',
                    opacity=0.6,
                    line=dict(color=colors[idx % len(colors)], width=3),
                    hovertemplate='<b>%{fullData.name}</b><br>CTR: %{x:.2f}%<br>확률밀도: %{y:.1f}<extra></extra>'
                ))
            
            fig_post.update_layout(
                title="각 소재의 실제 CTR 분포 추정",
                xaxis_title="CTR (%)",
                yaxis_title="확률 밀도",
                hovermode='closest',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            )
            st.plotly_chart(fig_post, use_container_width=True)
        
        # ====================
        # TAB 4: CUSUM (전문가 모드만)
        # ====================
        if analysis_mode == "🔬 전문가 모드":
            with tabs[4]:
            st.markdown("### 📉 CUSUM 기반 이상 감지")
            st.markdown("**Cumulative Sum Control Chart - 성과 하락 조기 경보 시스템**")
            
            t_id = st.selectbox("분석할 소재 선택", ids, key='cusum_material')
            sub = df[df['ID'] == t_id].sort_values('날짜')
            
            # 기준 CTR 설정
            if len(sub) >= 7:
                baseline = sub.head(7)
                p0_val = baseline['클릭'].sum() / (baseline['노출'].sum() + 1e-9)
                st.info(f"**기준선 설정:** 초기 7일 평균 CTR = {p0_val*100:.2f}%")
            else:
                p0_val = sub['클릭'].sum() / (sub['노출'].sum() + 1e-9)
                st.warning(f"데이터 부족: 전체 평균 CTR = {p0_val*100:.2f}% 사용")
            
            # CUSUM 계산
            cusum_vals = get_binomial_cusum(sub['클릭'].values, sub['노출'].values, p0_val)
            
            # 몬테카를로로 임계값 추정
            with st.spinner('임계값 계산 중... (몬테카를로 시뮬레이션)'):
                h_threshold, achieved_arl = estimate_h_via_arl(
                    p0_val, 
                    sub['노출'].values,
                    target_arl=30,
                    sims=500
                )
            h_threshold = -h_threshold  # CUSUM은 음수 방향이므로
            
            # CUSUM 차트
            fig_cusum = go.Figure()
            fig_cusum.add_trace(go.Scatter(
                x=sub['날짜'],
                y=cusum_vals,
                mode='lines+markers',
                name='CUSUM',
                line=dict(color='blue', width=2)
            ))
            fig_cusum.add_hline(
                y=h_threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"임계값 h={-h_threshold:.2f} (ARL≈{achieved_arl:.0f}일)"
            )
            fig_cusum.update_layout(
                title=f"{t_id} - CUSUM 추세",
                xaxis_title="날짜",
                yaxis_title="CUSUM 값",
                hovermode='x unified'
            )
            st.plotly_chart(fig_cusum, use_container_width=True)
            
            # 이상 감지 결과
            if cusum_vals[-1] < h_threshold:
                st.error(f"""
                ⚠️ **성과 하락 감지!**
                - 현재 CUSUM: {cusum_vals[-1]:.2f}
                - 최근 성과가 기준선 대비 유의미하게 하락했습니다
                - **권장 조치:** 소재 교체 또는 예산 축소 검토
                """)
            elif cusum_vals[-1] < h_threshold * 0.5:
                st.warning(f"""
                ⚡ **주의 필요**
                - 현재 CUSUM: {cusum_vals[-1]:.2f}
                - 하락 추세가 감지되고 있습니다
                - 모니터링 강화 필요
                """)
            else:
                st.success(f"""
                ✅ **정상 범위**
                - 현재 CUSUM: {cusum_vals[-1]:.2f}
                - 성과 안정적으로 유지 중
                """)
            
            st.markdown("---")
            st.markdown("### 📈 일별 CTR 추이")
            
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(
                x=sub['날짜'],
                y=sub['CTR(%)'],
                mode='lines+markers',
                name='일별 CTR',
                line=dict(color='green')
            ))
            fig_daily.add_hline(
                y=p0_val*100,
                line_dash="dot",
                line_color="orange",
                annotation_text="기준 CTR"
            )
            fig_daily.update_layout(
                title="일별 CTR 변화",
                xaxis_title="날짜",
                yaxis_title="CTR (%)"
            )
            st.plotly_chart(fig_daily, use_container_width=True)
            
            st.info(f"""
            **📖 CUSUM 방법론:**
            - **임계값 h = {-h_threshold:.2f}** (Monte Carlo {500}회 시뮬레이션)
            - **목표 ARL = 30일** (정상 상태에서 평균 30일마다 1회 오경보)
            - **달성 ARL = {achieved_arl:.0f}일** (실제 시뮬레이션 결과)
            
            **작동 원리:**
            - 기준선({p0_val*100:.2f}%) 대비 "누적 편차" 계산
            - 값이 음수로 떨어질수록 → 성과 하락 신호
            - 임계값 돌파 시 → 통계적으로 유의미한 변화
            - 장점: 작은 변화도 빠르게 감지 (일별 비교보다 민감)
            
            **왜 이 임계값인가?**
            - 너무 낮으면 → 오경보 많음 (정상인데 경고)
            - 너무 높으면 → 감지 늦음 (문제 놓침)
            - ARL 30일 = "정상 상태에서 한 달에 한 번 정도만 오경보"
            """)
            
            with st.expander("ℹ️ ARL(Average Run Length)이란?"):
                st.markdown("""
                **ARL = 오경보가 나기까지 평균 시간**
                
                예시:
                - ARL = 10일: 정상 상태에서 평균 10일마다 1회 오경보
                - ARL = 30일: 정상 상태에서 평균 30일마다 1회 오경보
                - ARL = 100일: 정상 상태에서 평균 100일마다 1회 오경보
                
                **왜 ARL이 중요한가?**
                오경보가 너무 많으면:
                - 실무자가 경고를 무시하게 됨
                - "늑대가 나타났다" 효과
                - 의사결정 피로
                
                **적정 ARL 선택:**
                - 공격적 감지: ARL 10일 (신규 소재 테스트)
                - 균형: ARL 30일 (일반 모니터링)
                - 보수적: ARL 100일 (안정적 장기 캠페인)
                """)
            
            # ========== 추가 분석 섹션 ==========
            st.markdown("---")
            st.markdown("### 🔬 고급 통계 분석")
            
            with st.expander("ℹ️ 고급 분석이란?"):
                st.markdown("""
                **이 섹션은 CUSUM의 성능을 심층 분석합니다**
                
                1. **ARL 곡선**: h값에 따라 오경보 빈도가 어떻게 변하나?
                2. **목표 ARL 비교**: 공격적 vs 보수적 설정의 차이
                3. **Power 분석**: 실제 하락 시 며칠 만에 감지하나?
                
                **누구에게 유용한가?**
                - 통계 담당자: 파라미터 조정 근거
                - 의사결정자: "15% 하락하면 며칠 만에 알 수 있나?"
                - 감사/검증: 시스템 신뢰성 증명
                """)
            
            analysis_tab = st.radio(
                "분석 선택:",
                ["ARL 곡선 (h값의 영향)", "목표 ARL 비교", "Power 분석 (감지 속도)"],
                horizontal=True
            )
            
            if analysis_tab == "ARL 곡선 (h값의 영향)":
                st.markdown("**h값에 따른 ARL(오경보 간격) 변화**")
                
                st.warning("""
                ⚠️ **전문가용 진단 도구**
                
                이 분석은 CUSUM 파라미터 조정 및 시스템 검증을 위한 것입니다.
                실무 의사결정에는 "오늘의 액션" 탭을 권장합니다.
                
                **활용 사례:**
                - 여러 캠페인에 동일 시스템 적용 시
                - 시스템 신뢰성 검증 (감사용)
                - 파라미터 표준화 필요 시
                """)
                
                with st.expander("ℹ️ 이 분석이 왜 필요한가요?"):
                    st.markdown("""
                    **질문: "h를 7로 할까 10으로 할까?"**
                    
                    이 그래프는 그 질문에 답합니다.
                    
                    **X축:** 임계값 h (낮을수록 민감, 높을수록 보수적)
                    **Y축:** 정상 상태에서 오경보까지 평균 일수
                    
                    **활용:**
                    - 빨간 선(목표 ARL 30일)과 교차하는 점 = 최적 h
                    - 곡선이 가파르면 → h에 민감 (신중히 선택)
                    - 곡선이 완만하면 → h 선택이 덜 중요
                    """)
                
                with st.spinner('ARL 곡선 계산 중... (약 10초 소요)'):
                    h_range = np.arange(1.0, 20.0, 1.0)
                    arl_values = []
                    
                    for h_test in h_range:
                        run_lengths = []
                        p1 = np.clip(p0_val * 0.85, 1e-6, 1-1e-6)
                        p0_clip = np.clip(p0_val, 1e-6, 1-1e-6)
                        llr_s = np.log(p1 / p0_clip)
                        llr_f = np.log((1 - p1) / (1 - p0_clip))
                        
                        for _ in range(200):  # 빠른 계산을 위해 200회
                            s = 0
                            t = 0
                            while t < 200:
                                t += 1
                                n = np.random.choice(sub['노출'].values)
                                c = np.random.binomial(int(n), p0_clip)
                                s = min(0, s + (c * llr_s + (int(n) - c) * llr_f))
                                if s < -h_test:
                                    break
                            run_lengths.append(t)
                        
                        arl_values.append(np.mean(run_lengths))
                
                fig_arl = go.Figure()
                fig_arl.add_trace(go.Scatter(
                    x=h_range,
                    y=arl_values,
                    mode='lines+markers',
                    name='ARL',
                    line=dict(color='purple', width=3)
                ))
                fig_arl.add_hline(
                    y=30, 
                    line_dash="dot", 
                    line_color="red",
                    annotation_text="목표 ARL = 30일"
                )
                fig_arl.add_vline(
                    x=-h_threshold,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"현재 h = {-h_threshold:.1f}"
                )
                fig_arl.update_layout(
                    title="임계값 h에 따른 ARL 변화",
                    xaxis_title="임계값 h",
                    yaxis_title="ARL (일)",
                    hovermode='x'
                )
                st.plotly_chart(fig_arl, use_container_width=True)
                
                st.info("""
                **📖 해석:**
                - X축: 임계값 h (클수록 보수적)
                - Y축: 정상 상태에서 오경보까지 평균 일수
                - h가 클수록 → ARL 증가 → 오경보 감소 BUT 감지 늦어짐
                - 녹색 선: 현재 사용 중인 h값 (목표 ARL 30일 달성)
                - 빨간 선: 목표 ARL (교차점이 최적 h)
                """)
            
            elif analysis_tab == "목표 ARL 비교":
                st.markdown("**서로 다른 ARL 목표에 따른 임계값 비교**")
                
                with st.expander("ℹ️ 실무 시나리오별 ARL 선택 가이드"):
                    st.markdown("""
                    **시나리오 1: 신규 소재 테스트 초기**
                    - 목표 ARL: 10일 (공격적)
                    - 이유: 빠르게 문제 파악, 오경보 감수
                    - 예산: 높음 (손실 위험 큼)
                    
                    **시나리오 2: 일반 상시 모니터링**
                    - 목표 ARL: 30일 (균형) ← **권장**
                    - 이유: 오경보와 감지 속도의 균형
                    - 예산: 중간
                    
                    **시나리오 3: 안정적 장기 캠페인**
                    - 목표 ARL: 100일 (보수적)
                    - 이유: 확실한 변화만 반응, 변동성 높은 상품
                    - 예산: 낮음 or 계절성 높음
                    """)
                
                with st.spinner('다양한 ARL 시나리오 계산 중...'):
                    target_arls = [10, 20, 30, 50, 100]
                    comparison_results = []
                    
                    for target in target_arls:
                        h_val, actual = estimate_h_via_arl(
                            p0_val,
                            sub['노출'].values,
                            target_arl=target,
                            sims=300  # 빠른 계산
                        )
                        comparison_results.append({
                            '목표ARL': target,
                            '임계값h': h_val,
                            '실제ARL': actual
                        })
                
                comp_df = pd.DataFrame(comparison_results)
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=comp_df['목표ARL'],
                    y=comp_df['임계값h'],
                    name='임계값 h',
                    marker_color='lightblue',
                    text=comp_df['임계값h'].apply(lambda x: f"{x:.1f}"),
                    textposition='outside'
                ))
                fig_comp.update_layout(
                    title="목표 ARL에 따른 필요 임계값",
                    xaxis_title="목표 ARL (일)",
                    yaxis_title="임계값 h",
                    showlegend=False
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                st.dataframe(
                    comp_df.style.format({
                        '목표ARL': '{:.0f}일',
                        '임계값h': '{:.2f}',
                        '실제ARL': '{:.1f}일'
                    }).background_gradient(subset=['임계값h'], cmap='Blues'),
                    use_container_width=True
                )
                
                st.info("""
                **📖 실무 적용 가이드:**
                - **ARL 10일**: 공격적 감지 (오경보 많지만 빠름)
                  → 신규 소재 테스트 초기, 고비용 캠페인
                - **ARL 30일**: 균형잡힌 설정 (권장)
                  → 일반적인 상시 모니터링
                - **ARL 100일**: 보수적 감지 (확실한 변화만)
                  → 안정적인 장기 캠페인, 계절성 높은 상품
                """)
            
            else:  # Power 분석
                st.markdown("### 🎯 Power 분석 (감지 속도)")
                st.markdown("*'CTR이 실제로 하락하면 며칠 만에 감지할 수 있나?'*")
                
                st.warning("""
                ⚠️ **전문가용 성능 평가**
                
                이 분석은 CUSUM의 감지 성능을 정량적으로 평가합니다.
                
                **한계:**
                - 매체 데이터만으로는 "하락의 중요도" 판단 불가
                - 비즈니스 임팩트(매출 영향) 미반영
                - SLA 제공 목적이 아니면 참고용
                
                **실무에서는:**
                "오늘의 액션" 탭에서 실시간 신호 확인 권장
                """)
                
                with st.expander("ℹ️ Power 분석이란?"):
                    st.markdown("""
                    **ARL vs Power 분석의 차이**
                    
                    **ARL (이전 분석):**
                    - 질문: "정상일 때 오경보가 얼마나 자주?"
                    - 목적: 오경보 관리
                    
                    **Power 분석 (지금):**
                    - 질문: "진짜 문제가 생기면 얼마나 빨리 알 수 있나?"
                    - 목적: 감지 능력 평가
                    
                    **실무 예시:**
                    소재 피로도로 CTR이 15% 하락했다면
                    → CUSUM이 평균 며칠 만에 경고를 줄까?
                    
                    **활용:**
                    - "5일 안에 알 수 있다" → 빠른 대응 가능
                    - "20일 걸린다" → 손실 큼, 임계값 조정 필요
                    
                    ⚠️ **가정:**
                    현재 분석은 CTR의 "상대적 하락"(예: 15% 감소) 기준
                    절대적 하락(예: 0.1%p 감소)이 중요하면 다른 접근 필요
                    """)
                
                with st.spinner('Power 분석 실행 중...'):
                    decline_scenarios = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
                    detection_times = []
                    
                    for decline_ratio in decline_scenarios:
                        p1_scenario = p0_val * decline_ratio
                        p1_clip = np.clip(p1_scenario, 1e-6, 1-1e-6)
                        p0_clip = np.clip(p0_val, 1e-6, 1-1e-6)
                        llr_s = np.log(p1_clip / p0_clip)
                        llr_f = np.log((1 - p1_clip) / (1 - p0_clip))
                        
                        detection_days = []
                        for _ in range(300):
                            s = 0
                            t = 0
                            while t < 100:
                                t += 1
                                n = np.random.choice(sub['노출'].values)
                                c = np.random.binomial(int(n), p1_clip)
                                s = min(0, s + (c * llr_s + (int(n) - c) * llr_f))
                                if s < h_threshold:
                                    break
                            detection_days.append(t)
                        
                        detection_times.append({
                            '하락률': f"{(1-decline_ratio)*100:.0f}%",
                            '하락후CTR': p1_scenario * 100,
                            '평균감지일': np.mean(detection_days),
                            '중앙값': np.median(detection_days),
                            '90%감지일': np.percentile(detection_days, 90)
                        })
                
                power_df = pd.DataFrame(detection_times)
                
                fig_power = go.Figure()
                fig_power.add_trace(go.Scatter(
                    x=power_df['하락률'],
                    y=power_df['평균감지일'],
                    mode='lines+markers',
                    name='평균',
                    line=dict(color='red', width=3),
                    marker=dict(size=10)
                ))
                fig_power.add_trace(go.Scatter(
                    x=power_df['하락률'],
                    y=power_df['90%감지일'],
                    mode='lines+markers',
                    name='90% 분위',
                    line=dict(color='orange', width=2, dash='dash'),
                    marker=dict(size=8)
                ))
                fig_power.update_layout(
                    title=f"하락 정도별 감지 소요 시간 (h={-h_threshold:.2f})",
                    xaxis_title="CTR 하락률",
                    yaxis_title="감지까지 소요 일수",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_power, use_container_width=True)
                
                st.dataframe(
                    power_df.style.format({
                        '하락후CTR': '{:.3f}%',
                        '평균감지일': '{:.1f}일',
                        '중앙값': '{:.1f}일',
                        '90%감지일': '{:.1f}일'
                    }).background_gradient(subset=['평균감지일'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
                
                st.info(f"""
                **📖 해석:**
                - **30% 하락** (CTR {p0_val*100:.2f}% → {p0_val*0.7*100:.2f}%): 
                  평균 {power_df.iloc[0]['평균감지일']:.1f}일 만에 감지
                - **15% 하락** (CTR {p0_val*100:.2f}% → {p0_val*0.85*100:.2f}%): 
                  평균 {power_df.iloc[3]['평균감지일']:.1f}일 만에 감지
                - **5% 하락**: 감지가 매우 느림 → 노이즈와 구분 어려움
                
                **실무 시사점:**
                - 큰 하락(20%+)은 빠르게 감지 가능
                - 작은 하락(5~10%)은 2주 이상 관찰 필요
                - 90% 분위 = "최악의 경우 이 정도 걸림"
                """)
            
        
        # ====================
        # TAB 6: 예산 시뮬레이터
        # ====================
        with tabs[5]:
            st.markdown("### 🎮 인터랙티브 예산 시뮬레이터")
            
            with st.expander("ℹ️ 이 도구의 목적과 한계"):
                st.markdown("""
                **목적:**
                "총 예산을 어떻게 배분할까?" 시뮬레이션
                
                **방법:**
                - 슬라이더로 소재별 배분 비율 조정
                - 실시간으로 예상 성과 계산
                
                ⚠️ **중요한 한계:**
                - **선형 가정**: 예산 2배 = 노출 2배 (실제는 X)
                - **CTR 불변 가정**: 노출 늘려도 CTR 동일 (실제는 하락 가능)
                - **인과 무시**: 예산 증액이 성과 보장 안 함
                
                **올바른 사용법:**
                - "대략적 방향성" 탐색
                - 실제 적용 시 점진적 테스트 필수
                - A/B 테스트로 검증
                """)
            
            st.markdown("---")
            
            total_budget = st.number_input(
                "총 일예산 (원)",
                min_value=0,
                value=int(res_agg['avg_cost_7d'].sum()),
                step=100000
            )
            
            st.markdown("### 소재별 예산 배분")
            
            allocations = {}
            for _, material in res_agg.iterrows():
                mat_id = material['ID']
                current_pct = material['avg_cost_7d'] / res_agg['avg_cost_7d'].sum() * 100
                
                allocations[mat_id] = st.slider(
                    f"{mat_id}",
                    0, 100,
                    int(current_pct),
                    key=f"slider_{mat_id}"
                )
            
            # 합계 체크
            total_pct = sum(allocations.values())
            
            if abs(total_pct - 100) > 1:
                st.error(f"⚠️ 총 배분: {total_pct}% (100%가 되어야 합니다)")
            else:
                st.success(f"✅ 총 배분: {total_pct}%")
                
                st.markdown("---")
                st.markdown("### 📊 시뮬레이션 결과")
                
                sim_results = []
                for mat_id, pct in allocations.items():
                    material = res_agg[res_agg['ID'] == mat_id].iloc[0]
                    
                    # 배분 금액
                    allocated_budget = total_budget * (pct / 100)
                    
                    # 단순 추정 (선형 가정)
                    current_avg_cost = material['avg_cost_7d']
                    if current_avg_cost > 0:
                        scale_factor = allocated_budget / current_avg_cost
                        expected_clicks = material['클릭'] / 7 * scale_factor
                        expected_impressions = material['노출'] / 7 * scale_factor
                    else:
                        expected_clicks = 0
                        expected_impressions = 0
                    
                    sim_results.append({
                        '소재': mat_id,
                        '배분(%)': pct,
                        '배분금액': allocated_budget,
                        '예상클릭': int(expected_clicks),
                        '예상노출': int(expected_impressions),
                        'CTR': material['exp_ctr'] * 100
                    })
                
                sim_df = pd.DataFrame(sim_results)
                
                st.dataframe(
                    sim_df.style.format({
                        '배분(%)': '{:.1f}%',
                        '배분금액': '{:,.0f}원',
                        '예상클릭': '{:,.0f}회',
                        '예상노출': '{:,.0f}회',
                        'CTR': '{:.2f}%'
                    }),
                    use_container_width=True
                )
                
                # 총계
                col1, col2, col3 = st.columns(3)
                col1.metric("총 예산", f"{total_budget:,.0f}원")
                col2.metric("예상 총 클릭", f"{sim_df['예상클릭'].sum():,.0f}회")
                col3.metric("예상 평균 CPC", f"{total_budget / sim_df['예상클릭'].sum():,.0f}원" if sim_df['예상클릭'].sum() > 0 else "N/A")
        
        # ====================
        # TAB 6: 주간 리포트
        # ====================
        tab_idx = 3 if analysis_mode == "🎯 실무 모드 (권장)" else 6
        with tabs[tab_idx]:
            st.markdown("### 📄 주간 성과 리포트")
            
            # 기간 선택
            date_range = st.date_input(
                "분석 기간 선택",
                value=(df['날짜'].min().date(), df['날짜'].max().date()),
                max_value=df['날짜'].max().date()
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                period_df = df[(df['날짜'].dt.date >= start_date) & (df['날짜'].dt.date <= end_date)]
                
                if len(period_df) == 0:
                    st.warning("선택한 기간에 데이터가 없습니다.")
                else:
                    st.markdown(f"**분석 기간: {start_date} ~ {end_date} ({(end_date - start_date).days + 1}일)**")
                    st.markdown("---")
                    
                    # 전체 요약
                    st.markdown("### ✨ 핵심 요약")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_cost = period_df['비용'].sum()
                    total_clicks = period_df['클릭'].sum()
                    total_impressions = period_df['노출'].sum()
                    avg_ctr = total_clicks / total_impressions * 100 if total_impressions > 0 else 0
                    avg_cpc = total_cost / total_clicks if total_clicks > 0 else 0
                    
                    col1.metric("총 집행비", f"{total_cost:,.0f}원")
                    col2.metric("총 클릭수", f"{total_clicks:,}회")
                    col3.metric("평균 CTR", f"{avg_ctr:.2f}%")
                    col4.metric("평균 CPC", f"{avg_cpc:,.0f}원")
                    
                    st.markdown("---")
                    
                    # 매체별/상품별/소재별 집행
                    st.markdown("### 💰 예산 집행 현황")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📱 매체별**")
                        media_summary = period_df.groupby('매체')['비용'].sum().sort_values(ascending=False)
                        for media, cost in media_summary.items():
                            pct = cost / total_cost * 100
                            st.write(f"├─ {media}: {cost:,.0f}원 ({pct:.1f}%)")
                    
                    with col2:
                        st.markdown("**📦 상품별**")
                        product_summary = period_df.groupby('상품')['비용'].sum().sort_values(ascending=False)
                        for product, cost in product_summary.items():
                            pct = cost / total_cost * 100
                            st.write(f"├─ {product}: {cost:,.0f}원 ({pct:.1f}%)")
                    
                    st.markdown("**🎨 소재별**")
                    material_summary = period_df.groupby('ID')['비용'].sum().sort_values(ascending=False)
                    for mat_id, cost in material_summary.items():
                        pct = cost / total_cost * 100
                        st.write(f"├─ {mat_id}: {cost:,.0f}원 ({pct:.1f}%)")
                    
                    st.markdown("---")
                    
                    # 베스트/워스트
                    st.markdown("### 🏆 성과 분석")
                    
                    col1, col2 = st.columns(2)
                    
                    material_perf = period_df.groupby('ID').agg({
                        '클릭': 'sum',
                        '노출': 'sum',
                        '비용': 'sum'
                    })
                    material_perf['CTR'] = material_perf['클릭'] / material_perf['노출'] * 100
                    material_perf['CPC'] = material_perf['비용'] / material_perf['클릭']
                    
                    with col1:
                        st.markdown("**🥇 베스트 소재 (CTR 기준)**")
                        best = material_perf.nlargest(1, 'CTR').iloc[0]
                        st.success(f"""
                        **{material_perf.nlargest(1, 'CTR').index[0]}**
                        - CTR: {best['CTR']:.2f}%
                        - 총 클릭: {int(best['클릭']):,}회
                        - 총 비용: {int(best['비용']):,}원
                        """)
                    
                    with col2:
                        st.markdown("**⚠️ 개선 필요 소재 (CTR 기준)**")
                        worst = material_perf.nsmallest(1, 'CTR').iloc[0]
                        st.warning(f"""
                        **{material_perf.nsmallest(1, 'CTR').index[0]}**
                        - CTR: {worst['CTR']:.2f}%
                        - 총 클릭: {int(worst['클릭']):,}회
                        - 총 비용: {int(worst['비용']):,}원
                        """)
        
        # ====================
        # TAB 8: Budget Optimization (기존, 이제 사용 안 함)
        # ====================
        # 삭제됨 - 기능이 시뮬레이터로 통합됨
            st.markdown("### 💰 예산 효율 분석")
            
            with st.expander("ℹ️ 이 탭의 목적과 한계"):
                st.markdown("""
                **목적:**
                "현재 예산을 어떻게 재분배하면 좋을까?" 시뮬레이션
                
                **방법:**
                - CTR과 비용을 기반으로 효율 점수 계산
                - 상위 소재에 집중 vs 효율 비례 배분 비교
                
                ⚠️ **중요: 이 탭의 한계**
                - **전환/매출 데이터 없음** → 진짜 ROI를 모름
                - **인과 추론 아님** → 예산 늘린다고 성과 보장 안 됨
                - **한계 효율 무시** → 예산 2배 ≠ 클릭 2배
                - **목적:** "만약 이렇게 하면?" 시나리오 탐색용
                
                **실제 의사결정 시 필요한 것:**
                1. A/B 테스트로 인과 효과 확인
                2. 전환율/LTV 데이터와 결합
                3. 점진적 증액으로 한계 효율 측정
                4. 경쟁 입찰, 노출 한계 등 외부 요인 고려
                """)
            
            res_agg['효율점수'] = res_agg['exp_ctr'] / (res_agg['avg_cost_7d'] / 100000 + 1e-9)
            
            with st.expander("ℹ️ 효율점수 계산 방식"):
                st.markdown("""
                **효율점수 = CTR / (비용 / 10만)**
                
                예시:
                - 소재 A: CTR 0.5%, 일평균 10만원 → 점수 = 0.5 / 1 = 0.5
                - 소재 B: CTR 1.0%, 일평균 30만원 → 점수 = 1.0 / 3 = 0.33
                
                **해석:**
                점수가 높을수록 "적은 비용으로 높은 CTR"
                
                ⚠️ 단, CTR ≠ 수익이므로 참고용으로만 사용
                """)
            
            fig_scatter = px.scatter(
                res_agg,
                x='avg_cost_7d',
                y='exp_ctr',
                size='노출',
                color='ID',
                text='ID',
                title="비용 대비 성과 분포",
                labels={'avg_cost_7d': '일평균 비용 (최근 7일, 원)', 'exp_ctr': '보정 CTR (%)'}
            )
            fig_scatter.update_traces(textposition='top center')
            # Y축을 퍼센트로 표시
            fig_scatter.update_yaxes(tickformat='.1%')
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🎯 예산 재분배 시뮬레이션")
            
            with st.expander("ℹ️ 각 전략 설명"):
                st.markdown("""
                **1. 현재 유지**
                - 아무것도 바꾸지 않음
                - 기준선으로 사용
                
                **2. 상위 집중 (70%)**
                - 상위 2개 소재에 70% 집중 (각 35%)
                - 나머지 소재에 30% 분배
                - 공격적 전략: 승자에 베팅
                
                **3. 효율 비례 배분**
                - 효율 점수에 비례해서 배분
                - 균형 전략: 모든 소재 고려
                
                ⚠️ 모든 전략은 **총 예산 불변** (제로섬)
                """)
            
            strategy = st.radio(
                "전략 선택:",
                ["현재 유지", "상위 집중 (70%)", "효율 비례 배분"]
            )
            
            if st.button("💡 시뮬레이션 실행"):
                total_budget = res_agg['avg_cost_7d'].sum()
                
                if strategy == "현재 유지":
                    res_agg['제안예산'] = res_agg['avg_cost_7d']
                    
                elif strategy == "상위 집중 (70%)":
                    # 상위 2개에 70%, 나머지에 30% 분배
                    top2 = res_agg.nlargest(2, 'exp_ctr')['ID'].values
                    n_others = len(res_agg) - 2  # 나머지 소재 수
                    
                    res_agg['제안예산'] = res_agg.apply(
                        lambda x: total_budget * 0.35 if x['ID'] in top2 
                                  else total_budget * 0.30 / n_others if n_others > 0 
                                  else 0,
                        axis=1
                    )
                    
                else:  # 효율 비례
                    total_efficiency = res_agg['효율점수'].sum()
                    res_agg['제안예산'] = res_agg['효율점수'] / total_efficiency * total_budget
                
                result_df = res_agg[['ID', 'avg_cost_7d', '제안예산', 'exp_ctr']].copy()
                result_df['변화율'] = (
                    (result_df['제안예산'] - result_df['avg_cost_7d']) / result_df['avg_cost_7d'] * 100
                )
                result_df.columns = ['소재', '현재 일평균', '제안 일평균', '보정CTR(%)', '변화율(%)']
                
                st.dataframe(
                    result_df.style.format({
                        '현재 일평균': '{:,.0f}원',
                        '제안 일평균': '{:,.0f}원',
                        '보정CTR(%)': '{:.2%}',
                        '변화율(%)': '{:+.1f}%'
                    }).background_gradient(subset=['변화율(%)'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # 예산 총합 검증
                current_sum = result_df['현재 일평균'].sum()
                proposed_sum = result_df['제안 일평균'].sum()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("현재 총예산", f"{current_sum:,.0f}원")
                col2.metric("제안 총예산", f"{proposed_sum:,.0f}원")
                col3.metric("차이", f"{proposed_sum - current_sum:,.0f}원", 
                           delta=f"{(proposed_sum/current_sum - 1)*100:.2f}%" if current_sum > 0 else "0%")
                
                if abs(proposed_sum - current_sum) < 1:  # 반올림 오차 허용
                    st.success("✅ 총 예산이 유지됩니다 (제로섬)")
                else:
                    st.warning(f"⚠️ 예산 차이 발생: {abs(proposed_sum - current_sum):,.0f}원")
    else:
        st.warning("데이터를 로드할 수 없습니다. 파일 형식을 확인해주세요.")
else:
    st.info("👆 상단에서 데이터 파일을 업로드하세요")