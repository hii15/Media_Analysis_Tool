import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import beta
from statsmodels.tsa.seasonal import seasonal_decompose

# --- [0. 시스템 기본 설정] ---
st.set_page_config(page_title="Ad Intelligence System v36.1", layout="wide")

st.title("🛡️ 매체 라이브 관련 의사결정 보조 도구")
st.markdown("---")

# --- [1. 데이터 엔지니어링 레이어] ---
def load_and_clean_data(uploaded_file):
    """
    데이터 로드 및 정제: CTR은 분모/분자를 보존한 상태에서 계산하여 통계적 왜곡을 방지합니다.
    """
    try:
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file)
        
        df.columns = [c.strip() for c in df.columns]
        mapping = {
            '날짜': ['날짜', '일자'], '상품': ['상품명', '상품'], '소재': ['소재명', '소재'],
            '노출': ['노출수', '노출'], '클릭': ['클릭수', '클릭'], '조회': ['조회수', '조회'], '비용': ['비용', '지출']
        }
        
        final_df = pd.DataFrame()
        for k, v in mapping.items():
            for col in v:
                if col in df.columns: final_df[k] = df[col]; break
        
        final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
        for col in ['노출', '클릭', '조회', '비용']:
            final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
        
        final_df['CTR(%)'] = (final_df['클릭'] / (final_df['노출'] + 1e-9) * 100)
        final_df['ID'] = "[" + final_df['상품'].astype(str).str.upper() + "] " + final_df['소재'].astype(str)
        return final_df.dropna(subset=['날짜']).sort_values(['ID', '날짜'])
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}"); return pd.DataFrame()

# --- [2. 통계 엔진 함수 정의] ---
def analyze_empirical_bayes(df):
    """
    Empirical Bayes 보정 알고리즘: 소표본 데이터의 CTR 변동성을 완화합니다.
    """
    global_ctr = df['클릭'].sum() / (df['노출'].sum() + 1e-9)
    # 비용은 전체 기여도 파악을 위해 sum으로 집계
    id_stats = df.groupby('ID').agg({'클릭': 'sum', '노출': 'sum', '비용': 'sum'})
    id_ctrs = id_stats['클릭'] / (id_stats['노출'] + 1e-9)
    var_ctr = max(id_ctrs.var(), 1e-7)
    
    # Kappa(신뢰도) 상한선을 노출수 중앙값으로 제한하여 방어적 설계 적용
    kappa = (global_ctr * (1 - global_ctr) / var_ctr) - 1
    kappa = np.clip(kappa, 10, min(1000, df.groupby('ID')['노출'].median()))
    
    alpha_0, beta_0 = global_ctr * kappa, (1 - global_ctr) * kappa
    agg = id_stats.reset_index()
    agg['post_alpha'] = alpha_0 + agg['클릭']
    agg['post_beta'] = beta_0 + (agg['노출'] - agg['클릭'])
    agg['exp_ctr'] = agg['post_alpha'] / (agg['post_alpha'] + agg['post_beta'])
    
    # Thompson Sampling: 우수 소재 확률
    samples = np.random.beta(agg['post_alpha'].values[:, None], 
                             agg['post_beta'].values[:, None], size=(len(agg), 5000))
    agg['prob_is_best'] = np.bincount(np.argmax(samples, axis=0), minlength=len(agg)) / 5000
    
    # 최근 7일 평균 비용 (운영 기준값)
    max_date = df['날짜'].max()
    last_costs = df[df['날짜'] >= max_date - timedelta(days=7)].groupby('ID')['비용'].mean()
    agg = agg.merge(last_costs.rename('avg_cost_7d'), on='ID', how='left').fillna(0)
    return agg, (alpha_0, beta_0, kappa)

def get_binomial_cusum(clicks, imps, p0, p1_ratio=0.85):
    p1 = np.clip(p0 * p1_ratio, 1e-6, 1-1e-6)
    p0 = np.clip(p0, 1e-6, 1-1e-6)
    llr = clicks * np.log(p1/p0) + (imps - clicks) * np.log((1-p1)/(1-p0))
    s = 0
    cusum = []
    for val in llr:
        s = min(0, s + val)
        cusum.append(s)
    return np.array(cusum)

@st.cache_data
def estimate_h_arl(p0, imps_series, target_arl=30, sims=500):
    p1 = np.clip(p0 * 0.85, 1e-6, 1-1e-6)
    p0 = np.clip(p0, 1e-6, 1-1e-6)
    llr_s, llr_f = np.log(p1/p0), np.log((1-p1)/(1-p0))
    for h in np.arange(2.0, 15.0, 1.0):
        rls = []
        for _ in range(sims):
            s, t = 0, 0
            while t < 100:
                t += 1
                n = np.random.choice(imps_series)
                c = np.random.binomial(int(n), p0)
                s = min(0, s + (c * llr_s + (int(n) - c) * llr_f))
                if s < -h: break
            rls.append(t)
        if np.mean(rls) >= target_arl: return h
    return 5.0

# --- [3. 메인 UI 및 탭별 로직] ---
uploaded_file = st.file_uploader("캠페인 데이터를 업로드하세요", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    if not df.empty:
        res_agg, (a0, b0, k_est) = analyze_empirical_bayes(df)
        ids = sorted(df['ID'].unique())
        
        tabs = st.tabs(["📊 통합 대시보드", "🧬 통계적 신뢰도 분석", "📉 추세 및 하락 감지", "🎯 예산 효율 곡선"])

        with tabs[0]:
            st.markdown("### 📊 통합 대시보드")
            st.caption("비중은 누적 기여도(Sum)로, 기대 CTR은 보정된 실력치(EB)로 파악합니다.")
            col1, col2 = st.columns(2)
            metric = col1.selectbox("비중 분석 지표", ["비용", "노출", "클릭"])
            col1.plotly_chart(px.pie(df.groupby('ID')[metric].sum().reset_index(), values=metric, names='ID', hole=0.4), use_container_width=True)
            col2.plotly_chart(px.bar(res_agg, x='ID', y='exp_ctr', title="통계 보정된 기대 CTR (%)"), use_container_width=True)

        with tabs[1]:
            st.markdown("### 🧬 분석 방법론: Empirical Bayes (수치 보정 알고리즘)")
            st.write("""
            **왜 이 분석이 필요한가요?** 노출수가 적은 초기 소재는 단 몇 번의 클릭만으로도 CTR이 0%가 되거나 매우 높게 나오는 등 수치가 불안정합니다.
            
            **어떻게 해결하나요?** **Empirical Bayes** 기법은 데이터 전체의 평균을 사전 정보로 활용합니다. 노출이 적은 소재는 전체 평균 쪽으로 보정(Shrinkage)하고, 
            노출이 충분히 쌓인 소재는 실제 수치를 반영합니다. 신뢰도 지수($\kappa$)는 데이터 규모에 의해 상한이 제한되도록 설계되어 과도한 확신을 방지합니다.
            """)
            st.info(f"데이터 기반 추정 사전 신뢰도(κ): {k_est:.2f}")
            fig_post = go.Figure()
            for _, row in res_agg.iterrows():
                samples = np.random.beta(row['post_alpha'], row['post_beta'], 3000)
                fig_post.add_trace(go.Box(x=samples, name=row['ID'], boxpoints=False))
            fig_post.update_layout(title="소재별 성과 신뢰 구간", xaxis_title="기대 CTR 범위")
            st.plotly_chart(fig_post, use_container_width=True)

        with tabs[2]:
            st.markdown("### 📉 분석 방법론: 시계열 분해 및 CUSUM 하락 감지")
            st.write("""
            **동적 기준점(p0) 설정 로직:** 데이터 기간에 따라 기준점을 유연하게 결정합니다. 
            - **14일 이하**: 일별 변동성이 크므로, EB 기법으로 보정된 **'통계적 기대 실력치'**를 기준점으로 삼아 리스크를 관리합니다.
            - **28일 이상**: 최근 14일을 제외한 **'과거 안정 구간의 평균'**을 기준점으로 삼아 현재의 하락 여부를 판단합니다.
            """)
            
            t_id = st.selectbox("분석 대상 소재 선택", ids)
            sub = df[df['ID'] == t_id].sort_values('날짜')
            exp_ctr = res_agg[res_agg['ID'] == t_id]['exp_ctr'].values[0]
            
            # 데이터 길이에 따른 p0 설정 전략
            if len(sub) >= 28:
                stable_period = sub[sub['날짜'] < (sub['날짜'].max() - timedelta(days=14))]
                p0_val = stable_period['클릭'].sum() / (stable_period['노출'].sum() + 1e-9)
                strategy_txt = "과거 안정 구간 기준"
            else:
                p0_val = exp_ctr # 데이터가 적을 땐 보정된 실력치 사용
                strategy_txt = "통계적 보정 평균 기준"
            
            st.caption(f"현재 선택된 기준점 설정 방식: **{strategy_txt}** (p0 = {p0_val:.4f})")
            
            # CUSUM 시각화
            h_opt = estimate_h_arl(p0_val, sub['노출'].values)
            cusum_v = get_binomial_cusum(sub['클릭'].values, sub['노출'].values, p0_val)
            fig_cusum = go.Figure()
            fig_cusum.add_trace(go.Scatter(x=sub['날짜'], y=cusum_v, name="하락 신호 누적치", fill='tozeroy', line_color='red'))
            fig_cusum.add_hline(y=-h_opt, line_dash="dash", line_color="black", annotation_text="하락 경계선")
            fig_cusum.update_layout(title=f"소재 성과 드리프트 탐지 (기준: {strategy_txt})")
            st.plotly_chart(fig_cusum, use_container_width=True)

        with tabs[3]:
            st.markdown("### 🎯 분석 방법론: 예산 효율 및 실험적 최적화")
            st.write("""
            **비용 탄력성 분석:** 집행 규모(최근 7일 평균) 대비 유입 효율(CTR)의 분포를 파악하여 한계 효율 지점을 탐색합니다.
            
            **주의사항:** 본 예산 제안은 **비용 대비 유입 효율**을 기반으로 산출된 실험적 지표입니다. 실제 운영 시에는 소재의 정성적 가치를 반드시 병행 검토해야 하며, 본 시스템은 자동 집행을 전제로 하지 않습니다.
            """)
            fig_scatter = px.scatter(res_agg, x='avg_cost_7d', y='exp_ctr', size='노출', color='ID',
                                     labels={'avg_cost_7d': '최근 7일 평균 비용', 'exp_ctr': '기대 CTR'},
                                     title="비용 효율 프론티어 (우상단 소재가 고효율)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            if st.button("실험적 예산 배분 정책 실행"):
                # 예산 대비 효율(Efficiency) 스코어 계산
                res_agg['score'] = res_agg['exp_ctr'] / (res_agg['avg_cost_7d'] + 1e-9)
                avg_s = res_agg['score'].mean() + 1e-9
                res_agg['proposed'] = res_agg['avg_cost_7d'] * (res_agg['score'] / avg_s)
                res_agg['최종제안액'] = res_agg.apply(lambda r: np.clip(r['proposed'], r['avg_cost_7d']*0.7, r['avg_cost_7d']*1.3), axis=1)
                st.table(res_agg[['ID', 'exp_ctr', '최종제안액']].style.format({'exp_ctr': '{:.4f}', '최종제안액': '{:,.0f}'}))