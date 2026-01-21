import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import beta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="게임 마케팅 통합 분석", layout="wide")
st.title("🎮 게임 마케팅 통합 분석 시스템")
st.markdown("**Bayesian 통계 기반 성과 분석 & 의사결정 지원**")
st.markdown("---")

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
            '매체': ['매체', 'media'],
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

def analyze_empirical_bayes(df, benchmark_df=None, use_manual_prior=False):
    global_ctr = df['클릭'].sum() / (df['노출'].sum() + 1e-9)
    id_stats = df.groupby('ID').agg({'클릭': 'sum', '노출': 'sum', '비용': 'sum', '매체': 'first'})
    id_ctrs = id_stats['클릭'] / (id_stats['노출'] + 1e-9)
    
    agg = id_stats.reset_index()
    agg['raw_ctr'] = id_ctrs.values
    
    if use_manual_prior and benchmark_df is not None:
        benchmark_dict = benchmark_df.set_index('매체')['업계평균CTR(%)'].to_dict()
        strength_dict = benchmark_df.set_index('매체')['Prior강도'].to_dict()
        
        for idx, row in agg.iterrows():
            media = row['매체']
            if media in benchmark_dict:
                prior_ctr = benchmark_dict[media] / 100
                prior_strength = strength_dict[media]
                
                alpha_0 = prior_ctr * prior_strength
                beta_0 = (1 - prior_ctr) * prior_strength
            else:
                alpha_0, beta_0 = 1, 99
            
            agg.loc[idx, 'post_alpha'] = alpha_0 + row['클릭']
            agg.loc[idx, 'post_beta'] = beta_0 + (row['노출'] - row['클릭'])
            agg.loc[idx, 'alpha_0'] = alpha_0
            agg.loc[idx, 'beta_0'] = beta_0
    else:
        var_ctr = max(id_ctrs.var(), 1e-7)
        kappa = (global_ctr * (1 - global_ctr) / var_ctr) - 1
        kappa = np.clip(kappa, 10, 1000)
        
        alpha_0, beta_0 = global_ctr * kappa, (1 - global_ctr) * kappa
        
        agg['post_alpha'] = alpha_0 + agg['클릭']
        agg['post_beta'] = beta_0 + (agg['노출'] - agg['클릭'])
        agg['alpha_0'] = alpha_0
        agg['beta_0'] = beta_0
    
    agg['exp_ctr'] = agg['post_alpha'] / (agg['post_alpha'] + agg['post_beta'])
    
    samples = np.random.beta(
        agg['post_alpha'].values[:, None], 
        agg['post_beta'].values[:, None], 
        size=(len(agg), 5000)
    )
    agg['prob_is_best'] = np.bincount(
        np.argmax(samples, axis=0), 
        minlength=len(agg)
    ) / 5000
    
    max_date = df['날짜'].max()
    date_7d_ago = max_date - timedelta(days=6)
    last_7d = df[df['날짜'] >= date_7d_ago]
    last_costs = last_7d.groupby('ID')['비용'].sum() / 7
    agg = agg.merge(last_costs.rename('avg_cost_7d'), on='ID', how='left').fillna(0)
    
    return agg

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
def estimate_h_via_arl(p0, imps_series, target_arl=30, sims=500):
    p1 = np.clip(p0 * 0.85, 1e-6, 1-1e-6)
    p0_clip = np.clip(p0, 1e-6, 1-1e-6)
    llr_success = np.log(p1 / p0_clip)
    llr_failure = np.log((1 - p1) / (1 - p0_clip))
    
    h_candidates = np.arange(1.0, 30.0, 0.5)
    
    for h in h_candidates:
        run_lengths = []
        for _ in range(sims):
            s, t = 0, 0
            while t < 500:
                t += 1
                n = np.random.choice(imps_series) if len(imps_series) > 0 else 100000
                c = np.random.binomial(int(n), p0_clip)
                s = min(0, s + (c * llr_success + (int(n) - c) * llr_failure))
                if s < -h:
                    break
            run_lengths.append(t)
        
        actual_arl = np.mean(run_lengths)
        if actual_arl >= target_arl:
            return h, actual_arl
    
    return h_candidates[-1], np.mean(run_lengths)

def get_confidence_level(material, df):
    mat_id = material['ID']
    mat_data = df[df['ID'] == mat_id]
    
    data_score = 1 if material['노출'] > 1000000 else (0.5 if material['노출'] > 100000 else 0)
    
    if len(mat_data) >= 7:
        daily_ctr_std = mat_data['CTR(%)'].std()
        stability_score = 1 if daily_ctr_std < material['exp_ctr'] * 50 else (0.5 if daily_ctr_std < material['exp_ctr'] * 100 else 0)
    else:
        stability_score = 0
    
    total_score = (data_score + stability_score) / 2
    
    if total_score >= 0.7:
        return "🟢 높음", "충분한 데이터와 안정적 패턴"
    elif total_score >= 0.4:
        return "🟡 보통", "추가 관찰 권장"
    else:
        return "🔴 낮음", "데이터 부족 또는 변동성 높음"

with st.sidebar:
    st.markdown("## ⚙️ 분석 설정")
    
    st.markdown("### 📊 Prior 설정 방식")
    prior_mode = st.radio(
        "Prior 설정",
        ["자동 (데이터 기반)", "수동 (벤치마크 기반)"],
        help="자동: 현재 데이터로 Prior 추정 / 수동: 업계 벤치마크 입력"
    )
    
    benchmark_df = None
    if prior_mode == "수동 (벤치마크 기반)":
        st.markdown("### 📋 상품별 벤치마크 입력")
        
        if 'benchmark_data' not in st.session_state:
            st.session_state.benchmark_data = pd.DataFrame({
                '매체': ['네이버 GFA', '유튜브', 'GDN', '페이스북'],
                '업계평균CTR(%)': [0.8, 2.5, 0.3, 1.2],
                'Prior강도': [100, 100, 100, 100]
            })
        
        edited_benchmark = st.data_editor(
            st.session_state.benchmark_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                '매체': st.column_config.TextColumn("매체명", help="데이터의 '매체' 컬럼과 정확히 일치해야 함"),
                '업계평균CTR(%)': st.column_config.NumberColumn("업계 평균 CTR (%)", min_value=0.0, max_value=10.0, format="%.2f"),
                'Prior강도': st.column_config.NumberColumn("Prior 강도", min_value=10, max_value=1000, help="높을수록 벤치마크 의존도 증가")
            }
        )
        st.session_state.benchmark_data = edited_benchmark
        benchmark_df = edited_benchmark
        
        with st.expander("ℹ️ Prior 강도란?"):
            st.markdown("""
            **Prior 강도 = 가상의 "과거 데이터" 양**
            
            - **100**: 벤치마크 CTR로 노출 10만회 본 것처럼 취급
            - **500**: 노출 50만회 (벤치마크 강하게 신뢰)
            - **10**: 노출 1만회 (실제 데이터에 빠르게 적응)
            
            **권장:**
            - 신뢰할 수 있는 업계 데이터: 200-500
            - 대략적 추정치: 50-100
            - 데이터 2주치만 있으면: 100 권장
            """)

uploaded_file = st.file_uploader("📂 캠페인 데이터 업로드 (CSV/XLSX/TSV)", type=['csv', 'xlsx', 'tsv'])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    
    if not df.empty:
        use_manual_prior = (prior_mode == "수동 (벤치마크 기반)")
        res_agg = analyze_empirical_bayes(df, benchmark_df, use_manual_prior)
        ids = sorted(df['ID'].unique())
        
        st.markdown("---")
        
        tabs = st.tabs([
            "📋 주간 체크리스트", 
            "📊 성과 대시보드", 
            "🧬 Bayesian 분석",
            "⏰ 조기 경고",
            "📉 CUSUM 모니터링"
        ])
        
        with tabs[0]:
            st.markdown("## 📋 주간 의사결정 체크리스트")
            st.markdown(f"**분석 기준일: {df['날짜'].max().strftime('%Y년 %m월 %d일')}**")
            st.markdown("---")
            
            today = df['날짜'].max()
            this_week_start = today - timedelta(days=6)
            last_week_start = this_week_start - timedelta(days=7)
            last_week_end = this_week_start - timedelta(days=1)
            
            this_week = df[df['날짜'] >= this_week_start]
            last_week = df[(df['날짜'] >= last_week_start) & (df['날짜'] <= last_week_end)]
            
            st.markdown("### 🚨 즉시 조치 필요")
            
            critical_items = []
            
            for material in res_agg.iterrows():
                _, mat = material
                mat_id = mat['ID']
                
                mat_this_week = this_week[this_week['ID'] == mat_id]['CTR(%)'].mean()
                mat_last_week = last_week[last_week['ID'] == mat_id]['CTR(%)'].mean()
                
                if mat_last_week > 0:
                    change_pct = (mat_this_week - mat_last_week) / mat_last_week
                    if change_pct < -0.3:
                        critical_items.append({
                            '소재': mat_id,
                            '문제': f'CTR {abs(change_pct)*100:.0f}% 급락',
                            '이번주': f'{mat_this_week:.2f}%',
                            '지난주': f'{mat_last_week:.2f}%',
                            '액션': '소재 교체 또는 타겟 재설정',
                            '우선순위': 1
                        })
                
                mat_cost = this_week[this_week['ID'] == mat_id]['비용'].sum()
                total_cost = this_week['비용'].sum()
                cost_share = mat_cost / total_cost if total_cost > 0 else 0
                
                mat_clicks = this_week[this_week['ID'] == mat_id]['클릭'].sum()
                total_clicks = this_week['클릭'].sum()
                click_share = mat_clicks / total_clicks if total_clicks > 0 else 0
                
                if cost_share > 0.4 and click_share < 0.3:
                    critical_items.append({
                        '소재': mat_id,
                        '문제': f'비용 {cost_share*100:.0f}%, 클릭 {click_share*100:.0f}%',
                        '이번주': f'{mat_cost:,.0f}원',
                        '지난주': '-',
                        '액션': '예산 재분배 또는 입찰가 조정',
                        '우선순위': 1
                    })
            
            if len(critical_items) > 0:
                st.error(f"⚠️ {len(critical_items)}건의 긴급 이슈")
                for idx, item in enumerate(critical_items, 1):
                    with st.expander(f"🔴 [{idx}] {item['소재']}: {item['문제']}", expanded=True):
                        col1, col2 = st.columns(2)
                        col1.metric("이번주", item['이번주'])
                        col2.metric("지난주", item['지난주'])
                        st.warning(f"**권장 액션:** {item['액션']}")
            else:
                st.success("✅ 긴급 조치 필요한 항목 없음")
            
            st.markdown("---")
            st.markdown("### 💡 개선 기회")
            
            opportunities = []
            
            material_perf = this_week.groupby('ID').agg({
                'CTR(%)': 'mean',
                '비용': 'sum',
                '클릭': 'sum'
            }).reset_index()
            
            if len(material_perf) > 0:
                best_ctr = material_perf.loc[material_perf['CTR(%)'].idxmax()]
                if best_ctr['비용'] / this_week['비용'].sum() < 0.4:
                    opportunities.append({
                        '기회': f"🟢 고성과 소재 '{best_ctr['ID']}' 증액 기회",
                        '근거': f"CTR {best_ctr['CTR(%)']:.2f}%로 1위, 예산 점유율 {best_ctr['비용']/this_week['비용'].sum()*100:.0f}%",
                        '제안': "10-20% 점진 증액 후 3일 모니터링"
                    })
            
            media_diversity = this_week.groupby('매체')['비용'].sum()
            if len(media_diversity) > 0 and (media_diversity / media_diversity.sum()).max() > 0.6:
                opportunities.append({
                    '기회': f"📱 매체 다각화 필요 ({media_diversity.idxmax()} {media_diversity.max()/media_diversity.sum()*100:.0f}%)",
                    '근거': "단일 매체 의존도 높음",
                    '제안': "타 매체 소규모 테스트 시작"
                })
            
            if len(opportunities) > 0:
                for idx, opp in enumerate(opportunities, 1):
                    with st.expander(f"💡 [{idx}] {opp['기회']}", expanded=False):
                        st.info(f"**근거:** {opp['근거']}")
                        st.success(f"**제안:** {opp['제안']}")
            else:
                st.info("추가 개선 기회 없음 (현상 유지)")
            
            st.markdown("---")
            st.markdown("### 📊 이번주 성과 요약")
            
            col1, col2, col3, col4 = st.columns(4)
            
            this_week_cost = this_week['비용'].sum()
            last_week_cost = last_week['비용'].sum()
            cost_change = (this_week_cost - last_week_cost) / last_week_cost if last_week_cost > 0 else 0
            
            this_week_clicks = this_week['클릭'].sum()
            last_week_clicks = last_week['클릭'].sum()
            clicks_change = (this_week_clicks - last_week_clicks) / last_week_clicks if last_week_clicks > 0 else 0
            
            this_week_ctr = (this_week['클릭'].sum() / this_week['노출'].sum()) * 100
            last_week_ctr = (last_week['클릭'].sum() / last_week['노출'].sum()) * 100
            ctr_change = this_week_ctr - last_week_ctr
            
            this_week_cpc = this_week_cost / this_week_clicks if this_week_clicks > 0 else 0
            last_week_cpc = last_week_cost / last_week_clicks if last_week_clicks > 0 else 0
            cpc_change = this_week_cpc - last_week_cpc
            
            col1.metric("총 지출", f"{this_week_cost:,.0f}원", f"{cost_change*100:+.1f}%")
            col2.metric("총 클릭", f"{this_week_clicks:,}회", f"{clicks_change*100:+.1f}%")
            col3.metric("평균 CTR", f"{this_week_ctr:.2f}%", f"{ctr_change:+.2f}%p")
            col4.metric("평균 CPC", f"{this_week_cpc:,.0f}원", f"{cpc_change:+.0f}원")
        
        with tabs[1]:
            st.markdown("### 📊 성과 대시보드")
            
            col1, col2, col3, col4 = st.columns(4)
            global_ctr = df['클릭'].sum() / (df['노출'].sum() + 1e-9)
            col1.metric("전체 평균 CTR", f"{global_ctr*100:.2f}%")
            col2.metric("분석 기간", f"{(df['날짜'].max() - df['날짜'].min()).days}일")
            col3.metric("총 소재 수", len(ids))
            col4.metric("총 집행 비용", f"{df['비용'].sum():,.0f}원")
            
            st.markdown("---")
            st.markdown("### 🏆 소재별 최고 성과 확률")
            
            fig_prob = px.bar(
                res_agg.sort_values('prob_is_best', ascending=True),
                x='prob_is_best', y='ID', orientation='h',
                text=res_agg.sort_values('prob_is_best', ascending=True)['prob_is_best'].apply(lambda x: f"{x*100:.1f}%")
            )
            fig_prob.update_xaxes(tickformat='.0%', title='최고 성과 확률')
            fig_prob.update_yaxes(title='')
            fig_prob.update_traces(textposition='outside')
            st.plotly_chart(fig_prob, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📈 소재별 상세 성과")
            
            display_df = res_agg[['ID', 'raw_ctr', 'exp_ctr', '노출', '클릭', '비용', 'prob_is_best', 'avg_cost_7d']].copy()
            display_df['raw_ctr'] = display_df['raw_ctr'] * 100
            display_df['exp_ctr'] = display_df['exp_ctr'] * 100
            display_df['prob_is_best'] = display_df['prob_is_best'] * 100
            display_df.columns = ['소재', '원본CTR(%)', '보정CTR(%)', '노출수', '클릭수', '비용', '최고확률(%)', '일평균비용']
            
            st.dataframe(
                display_df.style.format({
                    '원본CTR(%)': '{:.2f}',
                    '보정CTR(%)': '{:.2f}',
                    '노출수': '{:,.0f}',
                    '클릭수': '{:,.0f}',
                    '비용': '{:,.0f}',
                    '최고확률(%)': '{:.1f}',
                    '일평균비용': '{:,.0f}'
                }).background_gradient(subset=['보정CTR(%)'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("### 📊 CTR 추이")
            
            daily_ctr = df.groupby(['날짜', 'ID']).agg({
                '클릭': 'sum',
                '노출': 'sum'
            }).reset_index()
            daily_ctr['CTR'] = (daily_ctr['클릭'] / daily_ctr['노출']) * 100
            
            fig = px.line(daily_ctr, x='날짜', y='CTR', color='ID', markers=True)
            fig.update_layout(yaxis_title='CTR (%)', xaxis_title='')
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.markdown("### 🧬 Bayesian 분석 상세")
            
            st.markdown("#### Prior 설정 현황")
            
            if use_manual_prior:
                st.success("✅ 수동 설정 모드 (벤치마크 기반)")
                
                prior_summary = res_agg[['ID', '매체', 'alpha_0', 'beta_0']].copy()
                prior_summary['Prior_CTR(%)'] = (prior_summary['alpha_0'] / (prior_summary['alpha_0'] + prior_summary['beta_0'])) * 100
                prior_summary['Prior_강도'] = prior_summary['alpha_0'] + prior_summary['beta_0']
                
                st.dataframe(
                    prior_summary[['ID', '매체', 'Prior_CTR(%)', 'Prior_강도']].style.format({
                        'Prior_CTR(%)': '{:.2f}',
                        'Prior_강도': '{:.0f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("ℹ️ 자동 설정 모드 (데이터 기반)")
                
                alpha_0 = res_agg['alpha_0'].iloc[0]
                beta_0 = res_agg['beta_0'].iloc[0]
                kappa = alpha_0 + beta_0
                prior_ctr = alpha_0 / (alpha_0 + beta_0)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Prior α₀", f"{alpha_0:.1f}")
                col2.metric("Prior β₀", f"{beta_0:.1f}")
                col3.metric("κ (Kappa)", f"{kappa:.1f}")
                
                st.markdown(f"""
                **Prior CTR:** {prior_ctr*100:.2f}%  
                **Prior 강도(κ):** {kappa:.1f} (가상 노출 {kappa*10000:,.0f}회 상당)
                """)
            
            st.markdown("---")
            st.markdown("#### Posterior 분포 (실제 CTR 추정)")
            
            fig_post = go.Figure()
            colors = px.colors.qualitative.Set2
            
            for idx, (_, row) in enumerate(res_agg.iterrows()):
                x = np.linspace(0, 0.05, 500)
                y = beta.pdf(x, row['post_alpha'], row['post_beta'])
                fig_post.add_trace(go.Scatter(
                    x=x*100, y=y, name=row['ID'],
                    mode='lines', fill='tozeroy', opacity=0.6,
                    line=dict(color=colors[idx % len(colors)], width=3)
                ))
            
            fig_post.update_layout(
                title="각 소재의 실제 CTR 분포 추정 (Posterior Distribution)",
                xaxis_title="CTR (%)",
                yaxis_title="확률 밀도",
                height=500
            )
            st.plotly_chart(fig_post, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 신뢰도 평가")
            
            conf_data = []
            for _, material in res_agg.iterrows():
                conf_level, conf_reason = get_confidence_level(material, df)
                conf_data.append({
                    '소재': material['ID'],
                    '신뢰도': conf_level,
                    '이유': conf_reason,
                    '노출수': material['노출'],
                    '데이터일수': len(df[df['ID'] == material['ID']])
                })
            
            conf_df = pd.DataFrame(conf_data)
            st.dataframe(
                conf_df.style.format({'노출수': '{:,.0f}'}),
                use_container_width=True
            )
        
        with tabs[3]:
            st.markdown("### ⏰ 소재 피로도 조기 경고")
            
            st.markdown("""
            **소재 피로도(Creative Fatigue):** 동일 소재 반복 노출 시 CTR 하락 현상  
            선형 회귀로 추세를 분석하여 교체 시점을 조기 예측합니다.
            """)
            
            st.markdown("---")
            
            for mat_id in ids:
                mat_data = df[df['ID'] == mat_id].sort_values('날짜')
                
                if len(mat_data) < 5:
                    st.warning(f"**{mat_id}**: 데이터 부족 (최소 5일 필요)")
                    continue
                
                X = np.arange(len(mat_data)).reshape(-1, 1)
                y = mat_data['CTR(%)'].values
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                current_ctr = y[-1]
                
                if slope < -0.001:
                    days_left = max(0, int((current_ctr - current_ctr * 0.5) / abs(slope)))
                    
                    if days_left == 0:
                        lifespan_status = "⚠️ 즉시 교체 검토"
                    elif days_left <= 3:
                        lifespan_status = f"🔴 긴급 (추정 D-{days_left})"
                    elif days_left <= 7:
                        lifespan_status = f"🟡 주의 (추정 D-{days_left})"
                    else:
                        lifespan_status = f"🟢 안정 (추정 D-{days_left})"
                else:
                    lifespan_status = "✅ 하락 추세 없음"
                    days_left = None
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{mat_id}**")
                    st.markdown(f"**상태:** {lifespan_status}")
                    st.markdown(f"현재 CTR: {current_ctr:.2f}% | 일평균 변화: {slope:.4f}%p")
                    if days_left is not None and days_left > 0:
                        st.caption("※ 선형 추세 기준 참고 추정치")
                
                with col2:
                    fig_mini = go.Figure()
                    fig_mini.add_trace(go.Scatter(
                        x=mat_data['날짜'], y=y, 
                        mode='lines+markers', name='실제'
                    ))
                    trend_line = model.predict(X)
                    fig_mini.add_trace(go.Scatter(
                        x=mat_data['날짜'], y=trend_line,
                        mode='lines', name='추세', 
                        line=dict(dash='dash', color='red')
                    ))
                    fig_mini.update_layout(
                        height=200, showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0),
                        yaxis_title='CTR(%)'
                    )
                    st.plotly_chart(fig_mini, use_container_width=True)
                
                st.markdown("---")
        
        with tabs[4]:
            st.markdown("### 📉 CUSUM 이상 감지")
            
            st.markdown("""
            **CUSUM(Cumulative Sum):** 통계적 공정 관리 기법  
            기준 성과 대비 누적 이탈도를 추적하여 성과 하락을 조기 감지합니다.
            """)
            
            st.markdown("---")
            
            selected_material = st.selectbox("소재 선택", ids)
            sub = df[df['ID'] == selected_material].sort_values('날짜')
            
            if len(sub) >= 7:
                p0_val = sub.head(7)['클릭'].sum() / (sub.head(7)['노출'].sum() + 1e-9)
            else:
                p0_val = sub['클릭'].sum() / (sub['노출'].sum() + 1e-9)
            
            cusum_vals = get_binomial_cusum(sub['클릭'].values, sub['노출'].values, p0_val)
            h_threshold, achieved_arl = estimate_h_via_arl(p0_val, sub['노출'].values, sims=200)
            h_threshold = -h_threshold
            
            col1, col2, col3 = st.columns(3)
            col1.metric("기준 CTR (p0)", f"{p0_val*100:.2f}%")
            col2.metric("감지 임계값 (h)", f"{h_threshold:.2f}")
            col3.metric("현재 CUSUM", f"{cusum_vals[-1]:.2f}")
            
            fig_cusum = go.Figure()
            fig_cusum.add_trace(go.Scatter(
                x=sub['날짜'], y=cusum_vals,
                mode='lines+markers', name='CUSUM',
                line=dict(color='blue', width=2)
            ))
            fig_cusum.add_hline(
                y=h_threshold, line_dash="dash",
                line_color="red", annotation_text="임계값"
            )
            fig_cusum.update_layout(
                title=f"{selected_material} - CUSUM 모니터링",
                xaxis_title="날짜",
                yaxis_title="CUSUM 값",
                height=400
            )
            st.plotly_chart(fig_cusum, use_container_width=True)
            
            if cusum_vals[-1] < h_threshold:
                st.error(f"⚠️ **성과 하락 감지** (CUSUM: {cusum_vals[-1]:.2f} < 임계값: {h_threshold:.2f})")
                st.markdown("""
                **권장 조치:**
                - 소재 즉시 교체 검토
                - 타겟팅 설정 재확인
                - 경쟁사 동향 분석
                """)
            else:
                st.success(f"✅ **정상 범위** (CUSUM: {cusum_vals[-1]:.2f})")
            
            with st.expander("ℹ️ CUSUM 해석 가이드"):
                st.markdown("""
                **CUSUM 값의 의미:**
                - **0 부근:** 기준 성과 대비 정상 범위
                - **음수 증가:** 성과가 지속적으로 하락 중
                - **임계값 돌파:** 통계적으로 유의미한 하락 감지
                
                **장점:**
                - 작은 변화도 누적하여 조기 감지
                - 일시적 변동과 구조적 하락 구분
                
                **한계:**
                - 외부 요인(시즌, 경쟁사) 미반영
                - 상승 전환 감지는 별도 설정 필요
                """)
        
        st.markdown("---")
        
        with st.expander("🔍 현재 데이터로 답할 수 없는 질문", expanded=False):
            st.markdown("""
            ### ❌ 현재 데이터의 한계
            
            **1. 전환 성과 분석 불가**
            - 질문: "CTR 높은 소재가 실제 Install/매출 기여하는가?"
            - 필요 데이터: Install, 회원가입, 인앱 결제 전환 데이터
            - 영향: CTR만으로 판단 시 CPI가 높은 비효율적 소재 선택 위험
            
            **2. 인과 관계 추정 불가**
            - 질문: "예산 2배 증액 시 Install 몇 개 증가?"
            - 필요 데이터: 과거 예산 변경 실험 데이터 (A/B 테스트)
            - 영향: 선형 가정만 가능, 실제론 비선형 반응
            
            **3. 타겟 최적화 제한**
            - 질문: "어떤 유저 세그먼트가 전환율 높은가?"
            - 필요 데이터: 연령/성별/관심사별 성과 분해
            - 영향: 광범위 타겟팅만 가능, 정밀 최적화 불가
            
            **4. 장기 예측 불가**
            - 질문: "이 소재가 3개월 후에도 성과 유지?"
            - 필요 데이터: 최소 3-6개월 이상의 장기 추적 데이터
            - 영향: 2주 데이터로는 추세만 파악, 예측 신뢰도 낮음
            
            ---
            
            ### ✅ 현재 데이터로 답할 수 있는 질문
            
            **1. 조기 경고**
            - 어떤 소재가 성과 하락 중인가? (CUSUM, 선형 회귀)
            - 언제 소재를 교체해야 하는가? (피로도 추정)
            
            **2. 효율성 비교**
            - 예산이 효율적으로 분배되고 있나? (비용/클릭 비율)
            - 매체별/상품별 성과 차이는? (CTR, CPC 비교)
            
            **3. 통계적 우열 판단**
            - 소재 A와 B 중 어느 쪽이 통계적으로 우수한가? (Bayesian)
            - 우연인가 실력인가? (신뢰 구간)
            
            **4. 단기 의사결정**
            - 내일/이번주 어떤 액션을 취해야 하나? (체크리스트)
            - 어떤 소재에 우선 예산을 배분할까? (최고 확률)
            
            ---
            
            **→ 이 시스템의 포지셔닝:**  
            "완벽한 예측 시스템"이 아닌 **"지금 당장 조치 필요한 것을 찾는 조기 경보 시스템"**
            """)
    else:
        st.warning("데이터를 로드할 수 없습니다. 파일 형식을 확인해주세요.")
else:
    st.info("👆 데이터 파일을 업로드하세요")
    
    st.markdown("---")
    st.markdown("### 📋 시스템 기능 소개")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### ✨ 핵심 기능
        
        **1. 벤치마크 기반 Prior 설정**
        - 매체별 업계 평균 CTR 입력
        - Prior 강도 조정 (10~1000)
        - 소량 데이터에서도 안정적 추정
        
        **2. 주간 체크리스트**
        - 즉시 조치 필요 항목 자동 분류
        - 개선 기회 포착
        - WoW 성과 비교
        
        **3. Bayesian 분석**
        - 소재별 실제 CTR 분포 추정
        - 최고 성과 확률 계산
        - 신뢰도 평가
        """)
    
    with col2:
        st.markdown("""
        #### 🎯 활용 시나리오
        
        **신규 캠페인 런칭 (D+1~14)**
        - 벤치마크 CTR 입력 (예: 네이버 GFA 0.8%, 유튜브 2.5%)
        - Prior로 매체 특성 반영
        - 2-3일 데이터로 초기 판단
        - CUSUM으로 빠른 이상 감지
        
        **정기 운영 (D+15~)**
        - 주간 체크리스트로 월요일 의사결정
        - 소재 피로도 모니터링
        - 예산 재분배 기회 포착
        
        **데이터 축적 후**
        - 자동 Prior로 전환
        - 전환 데이터 연동하여 CPI/ROAS 분석
        """)
    
    st.markdown("---")
    st.markdown("### 💡 시작 가이드")
    
    st.markdown("""
    **1단계: 데이터 준비**
    - 필수 컬럼: 날짜, 매체, 상품, 소재, 노출, 클릭, 비용
    - 형식: CSV, XLSX, TSV 지원
    - 최소 기간: 5일 이상 권장
    
    **2단계: Prior 설정 선택**
    - **자동**: 현재 데이터로 Prior 추정 (14일 이상 데이터 있을 때)
    - **수동**: 매체별 업계 벤치마크 입력 (2주 미만 데이터일 때 권장)
      - 예: 네이버 GFA 0.8%, 유튜브 2.5%, GDN 0.3%
    
    **3단계: 분석 실행**
    - 주간 체크리스트에서 액션 아이템 확인
    - Bayesian 분석에서 통계적 우열 판단
    - CUSUM에서 이상 징후 모니터링
    """)
    
    st.markdown("---")
    st.caption("💡 Tip: 사이드바에서 Prior 설정 방식을 변경할 수 있습니다")