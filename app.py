import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import beta

# --- [0. 설정 및 디자인] ---
st.set_page_config(page_title="Ad Intelligence System v36.4", layout="wide")
st.title("🎯 광고 매체 통계분석 시스템")
st.markdown("### **Empirical Bayes & CUSUM 기반 소재 성과 분석**")
st.info("💡 이 도구는 단순 평균 비교의 오류를 방지하고, 통계적 확신을 바탕으로 예산 결정을 돕습니다.")

# --- [1. 데이터 로드 및 정제] ---
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
            '노출': ['노출수', '노출'], '클릭': ['클릭수', '클릭'], '비용': ['비용', '지출']
        }
        
        final_df = pd.DataFrame()
        for k, v in mapping.items():
            for col in v:
                if col in df.columns: final_df[k] = df[col]; break
        
        final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
        for col in ['노출', '클릭', '비용']:
            final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
        
        final_df['ID'] = "[" + final_df['상품'].astype(str) + "] " + final_df['소재'].astype(str)
        return final_df.dropna(subset=['날짜']).sort_values(['ID', '날짜'])
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}"); return pd.DataFrame()

# --- [2. 통계 엔진] ---
def analyze_empirical_bayes(df):
    global_ctr = df['클릭'].sum() / (df['노출'].sum() + 1e-9)
    id_stats = df.groupby('ID').agg({'클릭': 'sum', '노출': 'sum', '비용': 'sum'})
    id_ctrs = id_stats['클릭'] / (id_stats['노출'] + 1e-9)
    var_ctr = max(id_ctrs.var(), 1e-7)
    
    # Kappa (신뢰도 파라미터)
    kappa = (global_ctr * (1 - global_ctr) / var_ctr) - 1
    kappa = np.clip(kappa, 10, 1000)
    
    alpha_0, beta_0 = global_ctr * kappa, (1 - global_ctr) * kappa
    agg = id_stats.reset_index()
    agg['post_alpha'] = alpha_0 + agg['클릭']
    agg['post_beta'] = beta_0 + (agg['노출'] - agg['클릭'])
    agg['exp_ctr'] = agg['post_alpha'] / (agg['post_alpha'] + agg['post_beta'])
    agg['raw_ctr'] = agg['클릭'] / (agg['노출'] + 1e-9)
    
    # Thompson Sampling (최고 소재 확률)
    samples = np.random.beta(agg['post_alpha'].values[:, None], 
                             agg['post_beta'].values[:, None], size=(len(agg), 5000))
    agg['prob_is_best'] = np.bincount(np.argmax(samples, axis=0), minlength=len(agg)) / 5000
    
    # 최근 7일 평균 비용
    max_date = df['날짜'].max()
    last_costs = df[df['날짜'] >= max_date - timedelta(days=7)].groupby('ID')['비용'].mean()
    agg = agg.merge(last_costs.rename('avg_cost_7d'), on='ID', how='left').fillna(0)
    return agg, (alpha_0, beta_0, kappa, global_ctr)

def get_binomial_cusum(clicks, imps, p0):
    p1 = np.clip(p0 * 0.85, 1e-6, 1-1e-6) # 기준 대비 15% 하락 감지 타겟
    p0 = np.clip(p0, 1e-6, 1-1e-6)
    llr = clicks * np.log(p1/p0) + (imps - clicks) * np.log((1-p1)/(1-p0))
    s, cusum = 0, []
    for val in llr:
        s = min(0, s + val)
        cusum.append(s)
    return np.array(cusum)

# --- [3. UI 레이아웃] ---
uploaded_file = st.file_uploader("📂 캠페인 데이터 업로드 (CSV/XLSX)", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    if not df.empty:
        res_agg, (a0, b0, k_est, global_ctr) = analyze_empirical_bayes(df)
        
        tabs = st.tabs(["📊 요약 보고서", "🧬 성과 신뢰도 분석", "📉 피로도 탐지", "💰 예산 최적화"])

        with tabs[0]:
            st.markdown("### 📊 핵심 지표 Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("전체 평균 CTR", f"{global_ctr*100:.2f}%")
            c2.metric("분석 기간", f"{(df['날짜'].max() - df['날짜'].min()).days}일")
            c3.metric("소재 수", len(res_agg))
            c4.metric("신뢰도 지수(κ)", f"{k_est:.0f}")

            st.markdown("---")
            st.markdown("### 🏆 최고 성과(Winner) 소재 확률")
            st.caption("5,000회 시뮬레이션을 통해 산출된 '이 소재가 진짜 1등일 확률'입니다.")
            
            fig_prob = px.bar(res_agg.sort_values('prob_is_best'), x='prob_is_best', y='ID', orientation='h',
                              text=res_agg['prob_is_best'].apply(lambda x: f"{x*100:.1f}%"),
                              color='prob_is_best', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_prob, use_container_width=True)
            
            st.info("""
            **📖 인사이트 가이드:**
            - **확률 70% 이상:** 승리 소재가 명확합니다. 해당 소재에 예산을 집중하세요.
            - **확률이 고르게 분포:** 소재 간 유의미한 차이가 없습니다. 더 많은 데이터가 쌓일 때까지 지켜보세요.
            """)

        with tabs[1]:
            st.markdown("### 🧬 Bayesian 성과 보정 (Empirical Bayes)")
            st.write("노출이 적은 신규 소재의 CTR은 우연에 의해 0%나 100%가 되기 쉽습니다. 본 시스템은 이를 전체 평균 방향으로 보정하여 '진짜 실력'을 추정합니다.")
            
            # 사후 분포 시각화
            fig_dist = go.Figure()
            for _, row in res_agg.iterrows():
                x = np.linspace(0, global_ctr * 3, 200)
                y = beta.pdf(x, row['post_alpha'], row['post_beta'])
                fig_dist.add_trace(go.Scatter(x=x*100, y=y, name=row['ID'], fill='tozeroy', opacity=0.4))
            fig_dist.update_layout(title="소재별 실제 성과 분포 추정", xaxis_title="CTR (%)")
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.markdown(f"""
            **현재 Kappa(κ) 값: {k_est:.1f}**
            - 이 값이 클수록 시스템은 **전체 평균**을 더 신뢰합니다. 
            - 현재 상태: **{'보수적(안정 중시)' if k_est > 100 else '공격적(개별 데이터 중시)'}**
            """)

        with tabs[2]:
            st.markdown("### 📉 소재 피로도 및 성과 하락 감지 (CUSUM)")
            target_id = st.selectbox("분석 대상 소재 선택", res_agg['ID'].unique())
            sub_df = df[df['ID'] == target_id].sort_values('날짜')
            
            p0_val = sub_df.head(7)['클릭'].sum() / (sub_df.head(7)['노출'].sum() + 1e-9)
            cusum_vals = get_binomial_cusum(sub_df['클릭'].values, sub_df['노출'].values, p0_val)
            
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=sub_df['날짜'], y=cusum_vals, fill='tozeroy', name="누적 편차", line_color='orange'))
            fig_c.add_hline(y=-5.0, line_dash="dash", line_color="red", annotation_text="경고선(h=-5.0)")
            fig_c.update_layout(title=f"[{target_id}] 성과 하락 추적")
            st.plotly_chart(fig_c, use_container_width=True)
            
            if cusum_vals[-1] < -5.0:
                st.error("⚠️ **성과 하락(Creative Fatigue) 감지!** 소재 교체 또는 캠페인 재검토가 필요합니다.")
            else:
                st.success("✅ 성과가 기준점 대비 안정적으로 유지되고 있습니다.")

        with tabs[3]:
            st.markdown("### 💰 예산 효율 최적화 가이드")
            res_agg['eff_score'] = res_agg['exp_ctr'] / (res_agg['avg_cost_7d'] + 1e-9)
            
            fig_scatter = px.scatter(res_agg, x='avg_cost_7d', y='exp_ctr', size='노출', color='ID',
                                     title="비용 대비 성과 잠재력 (우상단 소재가 가장 효율적)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            if st.button("🚀 AI 예산 배분 제안"):
                total = res_agg['avg_cost_7d'].sum()
                res_agg['weight'] = res_agg['eff_score'] / res_agg['eff_score'].sum()
                res_agg['Proposed_Budget'] = res_agg['weight'] * total
                
                st.table(res_agg[['ID', 'avg_cost_7d', 'Proposed_Budget', 'exp_ctr']]
                         .style.format({'avg_cost_7d': '₩{:,.0f}', 'Proposed_Budget': '₩{:,.0f}', 'exp_ctr': '{:.2%}'}))