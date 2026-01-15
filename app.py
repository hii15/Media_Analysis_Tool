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
    
    # 최근 7일 평균 비용
    max_date = df['날짜'].max()
    last_costs = df[df['날짜'] >= max_date - timedelta(days=7)].groupby('ID')['비용'].mean()
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
        
        # 탭 구성
        tabs = st.tabs([
            "📊 Executive Summary", 
            "🧬 Bayesian Analysis", 
            "📉 Trend & Anomaly Detection",
            "💰 Budget Optimization"
        ])
        
        # ====================
        # TAB 1: Executive Summary
        # ====================
        with tabs[0]:
            st.markdown("### 📊 핵심 지표 요약")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("전체 평균 CTR", f"{global_ctr*100:.2f}%")
            col2.metric("분석 기간", f"{(df['날짜'].max() - df['날짜'].min()).days}일")
            col3.metric("총 소재 수", len(ids))
            col4.metric("총 집행 비용", f"₩{df['비용'].sum()/10000:.0f}만")
            
            st.markdown("---")
            st.markdown("### 🏆 최고 성과 소재 확률")
            st.markdown("*Bayesian 사후확률 기반 - 5000회 시뮬레이션*")
            
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
                    '비용': '₩{:,.0f}',
                    '최고확률': '{:.1f}%'
                }).background_gradient(subset=['보정CTR(%)'], cmap='RdYlGn'),
                use_container_width=True
            )
        
        # ====================
        # TAB 2: Bayesian Analysis
        # ====================
        with tabs[1]:
            st.markdown("### 🧬 Empirical Bayes 방법론")
            
            st.markdown(f"""
            **핵심 개념:**
            - 소표본에서 CTR은 변동성이 큽니다 (클릭 몇 개로 100% or 0% 가능)
            - 전체 평균을 사전 정보로 활용해 극단값을 보정합니다
            - "전체적으로 CTR이 {global_ctr*100:.2f}%인데, 이 소재만 {global_ctr*100*3:.1f}%는 의심스럽다"
            """)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Prior α₀", f"{a0:.1f}")
            col2.metric("Prior β₀", f"{b0:.1f}")
            col3.metric("신뢰도 κ", f"{k_est:.1f}")
            
            st.markdown(f"""
            **κ (Kappa) 해석:**
            - 현재 값: **{k_est:.1f}**
            - κ가 클수록 → 전체 평균을 더 신뢰 (보수적 평가)
            - κ가 작을수록 → 개별 소재 데이터를 더 신뢰
            - 적정 범위: 10~1000 (현재 {'✅ 적절' if 10 < k_est < 1000 else '⚠️ 조정 필요'})
            """)
            
            st.markdown("---")
            st.markdown("### 📊 사후확률 분포 (Posterior Distribution)")
            
            fig_post = go.Figure()
            for _, row in res_agg.iterrows():
                x = np.linspace(0, 0.03, 500)
                y = beta.pdf(x, row['post_alpha'], row['post_beta'])
                fig_post.add_trace(go.Scatter(
                    x=x*100, y=y, 
                    name=row['ID'],
                    mode='lines',
                    fill='tozeroy',
                    opacity=0.6
                ))
            
            fig_post.update_layout(
                title="각 소재의 실제 CTR 분포 추정",
                xaxis_title="CTR (%)",
                yaxis_title="확률 밀도",
                hovermode='x unified'
            )
            st.plotly_chart(fig_post, use_container_width=True)
            
            st.info("""
            **📖 그래프 해석:**
            - X축: 해당 소재의 "진짜" CTR 범위
            - Y축: 각 CTR일 확률 (높을수록 그 값일 가능성 높음)
            - 분포가 좁을수록 → 확신도 높음 (데이터 많거나 일관성 높음)
            - 분포가 넓을수록 → 불확실성 높음 (더 많은 테스트 필요)
            - 분포가 겹치면 → 소재 간 차이가 통계적으로 명확하지 않음
            """)
        
        # ====================
        # TAB 3: Trend & CUSUM
        # ====================
        with tabs[2]:
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
            - **임계값 h = {-h_threshold:.2f}** (몬테카를로 {500}회 시뮬레이션)
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
            
            # ========== 추가 분석 섹션 ==========
            st.markdown("---")
            st.markdown("### 🔬 고급 통계 분석")
            
            analysis_tab = st.radio(
                "분석 선택:",
                ["ARL 곡선 (h값의 영향)", "목표 ARL 비교", "Power 분석 (감지 속도)"],
                horizontal=True
            )
            
            if analysis_tab == "ARL 곡선 (h값의 영향)":
                st.markdown("**h값에 따른 ARL(오경보 간격) 변화**")
                
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
                st.markdown("**성과 하락 시 감지 속도 분석**")
                st.markdown("*'15% CTR 하락이 발생하면 며칠 만에 감지할 수 있나?'*")
                
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
        # TAB 4: Budget Optimization
        # ====================
        with tabs[3]:
            st.markdown("### 💰 예산 효율 분석")
            
            res_agg['효율점수'] = res_agg['exp_ctr'] / (res_agg['avg_cost_7d'] / 100000 + 1e-9)
            
            fig_scatter = px.scatter(
                res_agg,
                x='avg_cost_7d',
                y='exp_ctr',
                size='노출',
                color='ID',
                text='ID',
                title="비용 대비 성과 분포",
                labels={'avg_cost_7d': '일평균 비용 (최근 7일)', 'exp_ctr': '보정 CTR'}
            )
            fig_scatter.update_traces(textposition='top center')
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🎯 예산 재분배 시뮬레이션")
            
            strategy = st.radio(
                "전략 선택:",
                ["현재 유지", "상위 집중 (70%)", "효율 비례 배분"]
            )
            
            if st.button("💡 시뮬레이션 실행"):
                total_budget = res_agg['avg_cost_7d'].sum()
                
                if strategy == "현재 유지":
                    res_agg['제안예산'] = res_agg['avg_cost_7d']
                    
                elif strategy == "상위 집중 (70%)":
                    top2 = res_agg.nlargest(2, 'exp_ctr')['ID'].values
                    res_agg['제안예산'] = res_agg.apply(
                        lambda x: total_budget * 0.35 if x['ID'] in top2 else total_budget * 0.15,
                        axis=1
                    )
                    
                else:  # 효율 비례
                    res_agg['제안예산'] = (
                        res_agg['효율점수'] / res_agg['효율점수'].sum() * total_budget
                    )
                
                result_df = res_agg[['ID', 'avg_cost_7d', '제안예산', 'exp_ctr']].copy()
                result_df['변화율'] = (
                    (result_df['제안예산'] - result_df['avg_cost_7d']) / result_df['avg_cost_7d'] * 100
                )
                result_df.columns = ['소재', '현재 일평균', '제안 일평균', '보정CTR(%)', '변화율(%)']
                
                st.dataframe(
                    result_df.style.format({
                        '현재 일평균': '₩{:,.0f}',
                        '제안 일평균': '₩{:,.0f}',
                        '보정CTR(%)': '{:.2%}',
                        '변화율(%)': '{:+.1f}%'
                    }).background_gradient(subset=['변화율(%)'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                st.success(f"✅ 총 예산: ₩{total_budget:,.0f} (변동 없음)")
    else:
        st.warning("데이터를 로드할 수 없습니다. 파일 형식을 확인해주세요.")
else:
    st.info("👆 상단에서 데이터 파일을 업로드하세요")