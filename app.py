import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import beta

# --- [0. 기본 설정] ---
st.set_page_config(page_title="Ad Intelligence Pro v34.1", layout="wide")

# --- [1. 데이터 엔진: 로딩 및 전처리] ---
def load_and_clean_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file)
        
        df.columns = [c.strip() for c in df.columns]
        mapping = {'날짜':['날짜','일자'], '상품':['상품명','상품'], '소재':['소재명','소재'],
                   '노출':['노출수','노출'], '클릭':['클릭수','클릭'], '조회':['조회수','조회'], '비용':['비용','지출']}
        
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
        st.error(f"데이터 로드 에러: {e}"); return pd.DataFrame()

# --- [2. 핵심 엔진: Empirical Bayes (Moment Matching Kappa)] ---
def analyze_empirical_bayes(df):
    global_ctr = df['클릭'].sum() / (df['노출'].sum() + 1e-9)
    # ID별 CTR 분산 계산 (Prior Strength 추정용)
    id_stats = df.groupby('ID').agg({'클릭':'sum', '노출':'sum', '비용':'last'})
    id_ctrs = id_stats['클릭'] / (id_stats['노출'] + 1e-9)
    var_ctr = max(id_ctrs.var(), 1e-7)
    
    # Moment Matching: kappa = [p(1-p)/var] - 1
    kappa = (global_ctr * (1 - global_ctr) / var_ctr) - 1
    kappa = np.clip(kappa, 10, 1000) # 수치적 안정성 가이드
    
    alpha_0, beta_0 = global_ctr * kappa, (1 - global_ctr) * kappa
    
    agg = id_stats.reset_index()
    agg['post_alpha'] = alpha_0 + agg['클릭']
    agg['post_beta'] = beta_0 + (agg['노출'] - agg['클릭'])
    agg['exp_ctr'] = agg['post_alpha'] / (agg['post_alpha'] + agg['post_beta'])
    
    # Thompson Sampling
    samples = np.random.beta(agg['post_alpha'].values[:, None], 
                             agg['post_beta'].values[:, None], size=(len(agg), 5000))
    agg['prob_is_best'] = np.bincount(np.argmax(samples, axis=0), minlength=len(agg)) / 5000
    
    # 최근 비용(최근 3일 평균) 가져오기
    max_date = df['날짜'].max()
    last_costs = df[df['날짜'] >= max_date - timedelta(days=3)].groupby('ID')['비용'].mean()
    agg = agg.merge(last_costs.rename('last_3d_avg_cost'), on='ID', how='left').fillna(0)
    return agg, (alpha_0, beta_0, kappa)

# --- [3. 탐지 엔진: Binomial CUSUM & Bootstrap ARL] ---
def get_binomial_cusum(clicks, imps, p0, p1_ratio=0.85):
    """ Binomial Log-Likelihood Ratio CUSUM (하락 감지 전용) """
    p1 = p0 * p1_ratio
    p0 = np.clip(p0, 1e-6, 1-1e-6)
    p1 = np.clip(p1, 1e-6, 1-1e-6)
    
    # 우도비 계산
    llr = clicks * np.log(p1/p0) + (imps - clicks) * np.log((1-p1)/(1-p0))
    s = 0
    cusum = []
    for val in llr:
        s = min(0, s + val) # One-sided (하락만 누적)
        cusum.append(s)
    return np.array(cusum)

@st.cache_data
def estimate_h_arl(p0, imps_series, target_arl=30, sims=500):
    """ Monte Carlo 기반 ARL 임계치 추정 """
    p1 = p0 * 0.85
    p0 = np.clip(p0, 1e-6, 1-1e-6)
    p1 = np.clip(p1, 1e-6, 1-1e-6)
    llr_s, llr_f = np.log(p1/p0), np.log((1-p1)/(1-p0))
    
    for h in np.arange(2.0, 15.0, 1.0):
        rls = []
        for _ in range(sims):
            s, t = 0, 0
            while t < 100: # Capped ARL (최대 100일)
                t += 1
                n = np.random.choice(imps_series) # 노출수 변동성 반영
                c = np.random.binomial(int(n), p0)
                s = min(0, s + (c * llr_s + (int(n) - c) * llr_f))
                if s < -h: break
            rls.append(t)
        if np.mean(rls) >= target_arl: return h
    return 5.0

# --- [4. 메인 UI 흐름] ---
uploaded_file = st.file_uploader("캠페인 데이터를 업로드하세요 (CSV/XLSX)", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    if not df.empty:
        # 엔진 실행
        res_agg, (a0, b0, kappa_est) = analyze_empirical_bayes(df)
        ids = sorted(df['ID'].unique())
        
        tabs = st.tabs(["📊 성과 대시보드", "🧬 EB Shrinkage 진단", "📉 하락 감지(CUSUM)", "🎯 예산 정책 제안", "🧪 시스템 리포트"])

        with tabs[0]: # 대시보드
            st.info("**[가이드]** 상품별 물량 비중과 기대 CTR을 비교합니다.")
            c1, c2 = st.columns(2)
            pie_m = c1.selectbox("비중 지표", ["비용", "노출", "클릭"])
            c1.plotly_chart(px.pie(df.groupby('ID')[pie_m].sum().reset_index(), values=pie_m, names='ID', hole=0.4), use_container_width=True)
            c2.plotly_chart(px.bar(res_agg, x='ID', y='exp_ctr', title="Empirical Bayes 추정 기대 CTR(%)"), use_container_width=True)

        with tabs[1]: # EB Shrinkage
            st.info(f"**Prior Strength (κ) 자동 추정치: {kappa_est:.2f}**")
            st.write("데이터의 분산을 고려하여 개별 소재의 CTR을 보정합니다. (Shrinkage 효과)")
            
            fig_post = go.Figure()
            for _, row in res_agg.iterrows():
                samples = np.random.beta(row['post_alpha'], row['post_beta'], 3000)
                fig_post.add_trace(go.Box(x=samples, name=row['ID'], boxpoints=False))
            fig_post.update_layout(title="ID별 사후 분포 (Posteriors)", xaxis_title="Expected CTR")
            st.plotly_chart(fig_post, use_container_width=True)

        with tabs[2]: # CUSUM (에러 발생했던 지점)
            st.info("**[가이드]** 하락 전용 우도비 감지기 (One-sided Fatigue Detector)")
            target_id = st.selectbox("분석 대상 선택", ids)
            t_df = df[df['ID']==target_id].sort_values('날짜')
            
            # 함수 호출 전 p0 정의
            p0_val = res_agg[res_agg['ID']==target_id]['exp_ctr'].values[0]
            
            # 1. h 산출 (Bootstrap ARL)
            h_opt = estimate_h_arl(p0_val, t_df['노출'].values)
            
            # 2. CUSUM 누적 (함수 호출)
            cusum_v = get_binomial_cusum(t_df['클릭'].values, t_df['노출'].values, p0_val)
            
            is_alarm = cusum_v[-1] < -h_opt
            
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=t_df['날짜'], y=cusum_v, name="Log-Likelihood Ratio Sum", fill='tozeroy'))
            fig_c.add_hline(y=-h_opt, line_dash="dash", line_color="red", annotation_text=f"ARL-30 Threshold (h={h_opt})")
            st.plotly_chart(fig_c, use_container_width=True)
            if is_alarm: st.error("🚨 **구조적 하락 감지**: 성과가 통계적 신뢰 한계를 벗어나 하락 중입니다.")

        with tabs[3]: # 예산 최적화
            st.info("**[가이드]** 기대 성과 및 승리 확률 기반 자원 배분 정책")
            if st.button("예산 정책 실행"):
                res_agg['score'] = res_agg['exp_ctr'] * res_agg['prob_is_best']
                avg_score = res_agg['score'].mean() + 1e-9
                res_agg['proposed'] = res_agg['비용'] * (res_agg['score'] / avg_score)
                
                # Safety Rail (Budget Inertia)
                res_agg['final_proposed'] = res_agg.apply(lambda r: np.clip(r['proposed'], r['비용']*0.7, r['비용']*1.3), axis=1)
                
                st.table(res_agg[['ID', 'exp_ctr', 'prob_is_best', 'final_proposed']].style.format(
                    {'exp_ctr':'{:.4f}', 'prob_is_best':'{:.2f}', 'final_proposed':'{:,.0f}'}))

        with tabs[4]: # 시스템 리포트
            st.subheader("📊 Methodological Transparency")
            st.markdown(f"""
            - **Estimation**: Empirical Bayes (Moment Matching). $\kappa$ = {kappa_est:.2f}
            - **Detection**: Binomial Log-Likelihood Ratio CUSUM.
            - **Thresholding**: Monte Carlo-estimated Capped ARL.
            - **Exposure Variance**: Bootstrap sampling applied.
            """)
            st.success("이 시스템은 통계적 공정 관리(SPC) 원칙을 준수합니다.")