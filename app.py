import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import beta
from statsmodels.tsa.seasonal import seasonal_decompose

# --- [0. 시스템 기본 설정] ---
st.set_page_config(page_title="Ad Intelligence System v35.1", layout="wide")

st.title("🛡️ 매체 라이브 관련 의사결정 보조 도구")
st.markdown("---")

# --- [1. 데이터 엔지니어링 레이어] ---
def load_and_clean_data(uploaded_file):
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
    global_ctr = df['클릭'].sum() / (df['노출'].sum() + 1e-9)
    id_stats = df.groupby('ID').agg({'클릭': 'sum', '노출': 'sum', '비용': 'last'})
    id_ctrs = id_stats['클릭'] / (id_stats['노출'] + 1e-9)
    var_ctr = max(id_ctrs.var(), 1e-7)
    kappa = (global_ctr * (1 - global_ctr) / var_ctr) - 1
    kappa = np.clip(kappa, 10, 1000)
    alpha_0, beta_0 = global_ctr * kappa, (1 - global_ctr) * kappa
    
    agg = id_stats.reset_index()
    agg['post_alpha'] = alpha_0 + agg['클릭']
    agg['post_beta'] = beta_0 + (agg['노출'] - agg['클릭'])
    agg['exp_ctr'] = agg['post_alpha'] / (agg['post_alpha'] + agg['post_beta'])
    
    samples = np.random.beta(agg['post_alpha'].values[:, None], 
                             agg['post_beta'].values[:, None], size=(len(agg), 5000))
    agg['prob_is_best'] = np.bincount(np.argmax(samples, axis=0), minlength=len(agg)) / 5000
    
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

def get_time_decomposition(df, target_col='CTR(%)'):
    if len(df) < 14: return None
    df_ts = df.set_index('날짜')[target_col].resample('D').mean().interpolate()
    try:
        return seasonal_decompose(df_ts, model='additive', period=7)
    except: return None

# --- [3. 메인 UI 및 탭별 분석 로직] ---
uploaded_file = st.file_uploader("캠페인 성과 데이터를 업로드하세요", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    if not df.empty:
        res_agg, (a0, b0, k_est) = analyze_empirical_bayes(df)
        ids = sorted(df['ID'].unique())
        
        tabs = st.tabs(["📊 통합 대시보드", "🧬 통계적 신뢰도 분석", "📉 추세 및 하락 감지", "🎯 예산 효율 곡선"])

        with tabs[0]:
            st.markdown("### 📊 통합 대시보드")
            st.caption("전체 캠페인의 현황을 한눈에 파악합니다. 우측 차트의 CTR은 통계적으로 보정되어 신뢰도가 높습니다.")
            col1, col2 = st.columns(2)
            metric = col1.selectbox("비중 분석 지표", ["비용", "노출", "클릭"])
            col1.plotly_chart(px.pie(df.groupby('ID')[metric].sum().reset_index(), values=metric, names='ID', hole=0.4), use_container_width=True)
            col2.plotly_chart(px.bar(res_agg, x='ID', y='exp_ctr', title="통계 보정된 기대 CTR (%)"), use_container_width=True)

        with tabs[1]:
            st.markdown("### 🧬 분석 방법론: Empirical Bayes (수치 보정 알고리즘)")
            st.write("""
            **왜 이 분석이 필요한가요?** 노출수가 적은 소재는 단 몇 번의 클릭만으로도 CTR이 0%가 되거나 50%가 되는 등 수치가 매우 불안정합니다. 이를 '소표본 왜곡'이라고 합니다.
            
            **어떻게 해결하나요?** **Empirical Bayes** 기법은 데이터 전체의 평균을 '사전 정보'로 활용합니다. 노출이 적은 소재는 전체 평균 쪽으로 수치를 보정(Shrinkage)하고, 노출이 충분히 쌓인 소재는 실제 수치를 그대로 반영합니다. 
            이를 통해 **"운 좋게 높게 나온 수치"와 "진짜 실력"을 구분**해낼 수 있습니다.
            """)
            
            st.divider()
            st.info(f"전체 데이터 기반 추정된 사전 신뢰도(κ): {k_est:.2f}")
            fig_post = go.Figure()
            for _, row in res_agg.iterrows():
                samples = np.random.beta(row['post_alpha'], row['post_beta'], 3000)
                fig_post.add_trace(go.Box(x=samples, name=row['ID'], boxpoints=False))
            fig_post.update_layout(title="소재별 성과 신뢰 구간 (박스가 좁을수록 수치가 확실함을 의미)", xaxis_title="기대 CTR 범위")
            st.plotly_chart(fig_post, use_container_width=True)

        with tabs[2]:
            st.markdown("### 📉 분석 방법론: 시계열 분해 및 CUSUM 하락 감지")
            st.write("""
            **1. 시계열 분해 (Trend Extraction)** 광고 성과는 요일(주말/평일)에 따라 춤을 춥니다. 단순히 어제보다 CTR이 떨어졌다고 해서 성과 하락으로 판단하면 오류가 생깁니다.  
            본 시스템은 **가법적 시계열 분해**를 통해 요일 반복성을 제거하고, 소재가 가진 **순수 성과 추세(Trend)**만 추출하여 보여줍니다.
            
            **2. CUSUM 하락 감지 (Structural Drift Detection)** 소재 피로도는 서서히 일어납니다. **CUSUM(누적합)** 방식은 매일 발생하는 미세한 하락 신호를 누적으로 합산하여, 통계적 임계치를 넘어서는 순간 알람을 울립니다.  
            단순한 변동(Noise)인지, 구조적인 성과 하락(Signal)인지를 과학적으로 판별합니다.
            """)
            
            st.divider()
            t_id = st.selectbox("분석 대상 소재 선택", ids)
            sub = df[df['ID'] == t_id].sort_values('날짜')
            p0_val = res_agg[res_agg['ID'] == t_id]['exp_ctr'].values[0]
            
            decomp = get_time_decomposition(sub)
            if decomp:
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name="요일 효과가 제거된 순수 추세", line=dict(width=4)))
                fig_trend.add_trace(go.Scatter(x=sub['날짜'], y=sub['CTR(%)'], name="원본 CTR 데이터", opacity=0.2))
                fig_trend.update_layout(title="소재 성과 추세 분석")
                st.plotly_chart(fig_trend, use_container_width=True)
            
            h_opt = estimate_h_arl(p0_val, sub['노출'].values)
            cusum_v = get_binomial_cusum(sub['클릭'].values, sub['노출'].values, p0_val)
            fig_cusum = go.Figure()
            fig_cusum.add_trace(go.Scatter(x=sub['날짜'], y=cusum_v, name="하락 신호 누적치", fill='tozeroy', line_color='red'))
            fig_cusum.add_hline(y=-h_opt, line_dash="dash", line_color="black", annotation_text="통계적 위험 경계선")
            fig_cusum.update_layout(title="소재 피로도 및 하락 신호 탐지 (그래프가 경계선 밑으로 내려가면 교체 권장)")
            st.plotly_chart(fig_cusum, use_container_width=True)

        with tabs[3]:
            st.markdown("### 🎯 분석 방법론: 예산 효율 곡선 및 최적화")
            st.write("""
            **비용 탄력성 분석 (Spend Elasticity)** 돈을 많이 쓴다고 해서 클릭률이 계속 유지되지는 않습니다. 특정 금액 이상에서는 효율이 급격히 떨어지는 구간이 존재합니다.  
            본 탭에서는 **집행 규모 대비 기대 CTR의 분포**를 시각화하여, 현재 예산이 효율적으로 배분되고 있는지 확인합니다.
            
            **Thompson Sampling 기반 정책 제안** 단순히 CTR이 높은 곳에 돈을 몰아주는 것이 아니라, **"이 소재가 실제로 가장 우수할 확률"**과 **"기대되는 개선량"**을 계산하여 예산 증액/감액 비율을 제안합니다.
            """)
            st.divider()
            fig_scatter = px.scatter(res_agg, x='avg_cost_7d', y='exp_ctr', size='노출', color='ID',
                                     labels={'avg_cost_7d': '최근 7일 평균 집행 비용', 'exp_ctr': '통계적 기대 CTR'},
                                     title="집행 비용 대비 성과 프론티어 (우상단 소재가 고효율)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            if st.button("최적 예산 배분 정책 제안 실행"):
                res_agg['score'] = res_agg['exp_ctr'] * res_agg['prob_is_best']
                avg_s = res_agg['score'].mean() + 1e-9
                res_agg['proposed'] = res_agg['avg_cost_7d'] * (res_agg['score'] / avg_s)
                res_agg['최종제안액'] = res_agg.apply(lambda r: np.clip(r['proposed'], r['avg_cost_7d']*0.7, r['avg_cost_7d']*1.3), axis=1)
                st.table(res_agg[['ID', 'exp_ctr', 'prob_is_best', '최종제안액']].style.format({'exp_ctr': '{:.4f}', 'prob_is_best': '{:.2f}', '최종제안액': '{:,.0f}'}))