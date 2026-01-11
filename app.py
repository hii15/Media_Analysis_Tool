import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from scipy.optimize import minimize
from datetime import timedelta
import logging

# 1. 설정 및 로그 제어
logging.getLogger('prophet').setLevel(logging.WARNING)
st.set_page_config(page_title="Marketing Analytics & Budget Optimizer", layout="wide")

# --- [엔진 1: 데이터 정제 및 베이지안 보정] ---
def clean_and_process_pro(df):
    mapping = {
        '날짜': ['날짜', '일자', 'Date'],
        '매체': ['매체', '채널', 'Media'],
        '상품명': ['상품명', '상품', 'Product'],
        '소재명': ['소재명', '소재', 'Creative'],
        '노출수': ['노출수', '노출', 'Impression'],
        '클릭수': ['클릭수', '클릭', 'Click'],
        '비용': ['비용', '지출', 'Cost']
    }
    final_df = pd.DataFrame()
    for std_key, patterns in mapping.items():
        found = [c for c in df.columns if str(c).strip() in patterns]
        if found: final_df[std_key] = df[found[0]]
        else:
            found_sub = [c for c in df.columns if any(p in str(c) for p in patterns)]
            if found_sub: final_df[std_key] = df[found_sub[0]]
    
    if len(final_df.columns) < len(mapping): return pd.DataFrame()

    final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
    for col in ['노출수', '클릭수', '비용']:
        final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    
    # 베이지안 보정 CTR (Empirical Bayes Shrinkage)
    global_mean = final_df['클릭수'].sum() / final_df['노출수'].sum()
    K = 100 # 신뢰 가중치
    final_df['Adj_CTR'] = (final_df['클릭수'] + K * global_mean) / (final_df['노출수'] + K) * 100
    final_df['ID'] = "[" + final_df['상품명'].astype(str) + "] " + final_df['소재명'].astype(str)
    
    return final_df.dropna(subset=['날짜'])

# --- [엔진 2: 수명 예측 및 추세 추출] ---
def get_forecast_and_slope(data):
    valid_df = data[data['노출수'] >= 10].sort_values('날짜').copy()
    if len(valid_df) < 7: return None, 0
    
    try:
        p = np.clip(valid_df['Adj_CTR'].values / 100, 0.0001, 0.9999)
        valid_df['y_logit'] = np.log(p / (1 - p))
        m = Prophet(interval_width=0.8, uncertainty_samples=500, daily_seasonality=False, weekly_seasonality=True)
        m.fit(valid_df[['날짜', 'y_logit']].rename(columns={'날짜': 'ds', 'y_logit': 'y'}))
        
        future = m.make_future_dataframe(periods=7)
        forecast = m.predict(future)
        
        # 최근 7일간의 기울기(Slope) 계산
        yhat = forecast['yhat'].values
        slope = (yhat[-1] - yhat[-7]) / 7
        
        def inv_logit(x): return (np.exp(x) / (1 + np.exp(x))) * 100
        res = pd.DataFrame({'ds': forecast['ds'], 'yhat': inv_logit(forecast['yhat']), 
                            'yhat_lower': inv_logit(forecast['yhat_lower']), 'yhat_upper': inv_logit(forecast['yhat_upper'])})
        return res, slope
    except: return None, 0

# --- [엔진 3: 한계 효용 및 최적화] ---
def hill_performance_model(budget, current_spend, avg_cpc, slope):
    if budget <= 0 or avg_cpc <= 0: return 0
    base_clicks = budget / avg_cpc
    spend_ratio = budget / (current_spend + 1e-6)
    slope_penalty = 1.0 + abs(min(0, slope)) * 3.0 # 하락세일수록 페널티 강화
    efficiency = 1.0 / (1.0 + (0.15 * slope_penalty * (max(0, spend_ratio - 1.0))**1.2))
    return base_clicks * efficiency

# --- [UI 메인] ---
st.title("🔬 마케팅 사이언스 통합 의사결정 시스템")
st.caption("Statistical Diagnosis & Multi-Scenario Optimization Engine")

uploaded_file = st.file_uploader("CSV 또는 XLSX 데이터 업로드", type=['csv', 'xlsx'])

if uploaded_file:
    # 데이터 로드 로직
    if uploaded_file.name.endswith('xlsx'):
        df_raw = pd.read_excel(uploaded_file)
    else:
        df_raw = pd.read_csv(uploaded_file)
        
    full_df = clean_and_process_pro(df_raw)
    
    if not full_df.empty:
        ids = sorted(full_df['ID'].unique())
        tabs = st.tabs(["💎 성과 리포트", "📈 수명 예측", "🕹️ 시뮬레이터", "🎯 예산 최적화"])

        # 사전 계산: 모든 소재의 추세 데이터 확보
        forecast_cache = {}
        for i in ids:
            f_res, f_slope = get_forecast_and_slope(full_df[full_df['ID'] == i])
            forecast_cache[i] = {'res': f_res, 'slope': f_slope}

        with tabs[0]: # 성과 리포트
            st.header("📊 전주 대비 성과(WoW) 및 베이지안 보정")
            max_date = full_df['날짜'].max()
            this_week = full_df[full_df['날짜'] > max_date - timedelta(days=7)]
            last_week = full_df[(full_df['날짜'] <= max_date - timedelta(days=7)) & (full_df['날짜'] > max_date - timedelta(days=14))]
            
            c1, c2, c3 = st.columns(3)
            def calc_ctr(d): return (d['클릭수'].sum() / d['노출수'].sum() * 100) if d['노출수'].sum() > 0 else 0
            tw_ctr, lw_ctr = calc_ctr(this_week), calc_ctr(last_week)
            c1.metric("이번 주 CTR", f"{tw_ctr:.2f}%", f"{(tw_ctr-lw_ctr):.2f}%")
            c2.metric("총 집행 비용", f"{this_week['비용'].sum():,.0f}원")
            c3.metric("베이지안 보정 리프트", "신뢰도 기반")

            st.plotly_chart(px.bar(full_df.groupby('ID')['Adj_CTR'].mean().reset_index(), x='ID', y='Adj_CTR', title="보정된 소재별 평균 성과 (Shrinkage applied)"), use_container_width=True)

        with tabs[1]: # 수명 예측
            st.header("📈 확률적 수명 추세 분석")
            sel_id = st.selectbox("소재 선택", ids, key="life_sel")
            f_data = forecast_cache[sel_id]['res']
            if f_data is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=full_df[full_df['ID']==sel_id]['날짜'], y=full_df[full_df['ID']==sel_id]['CTR(%)'], name="원시 실적", mode='markers'))
                fig.add_trace(go.Scatter(x=f_data['ds'], y=f_data['yhat_upper'], line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=f_data['ds'], y=f_data['yhat_lower'], fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line=dict(width=0), name="80% 예측 구간"))
                fig.add_trace(go.Scatter(x=f_data['ds'], y=f_data['yhat'], name="추세선", line=dict(color='red', dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"기울기 지수(Slope Index): {forecast_cache[sel_id]['slope']:.4f} (음수일수록 피로도 높음)")

        with tabs[2]: # 시뮬레이션
            st.header("🕹️ 한계 효용 시뮬레이터 (What-If)")
            st.markdown("특정 소재의 예산 변경 시, **한계 효용 체감**과 **수명 추세**를 반영하여 예상 성과를 도출합니다.")
            sim_id = st.selectbox("시뮬레이션 대상", ids, key="sim_sel")
            target_data = full_df[full_df['ID'] == sim_id]
            curr_spend = target_data['비용'].sum()
            avg_cpc = curr_spend / target_data['클릭수'].sum() if target_data['클릭수'].sum() > 0 else 0
            
            new_spend = st.slider("예산 변경 시뮬레이션 (원)", 0.0, curr_spend * 3, float(curr_spend))
            pred_clicks = hill_performance_model(new_spend, curr_spend, avg_cpc, forecast_cache[sim_id]['slope'])
            
            sc1, sc2 = st.columns(2)
            sc1.metric("현재 클릭수", f"{target_data['클릭수'].sum():,.0f}")
            sc2.metric("예상 클릭수", f"{pred_clicks:,.0f}", f"{pred_clicks - target_data['클릭수'].sum():,.0f}")

        with tabs[3]: # 예산 최적화
            st.header("🎯 통계적 예산 최적 배분 제안")
            if st.button("🚀 최적화 알고리즘(SLSQP) 가동"):
                total_b = full_df['비용'].sum()
                summary = full_df.groupby('ID').agg({'비용':'sum', '클릭수':'sum'}).reset_index()
                
                def objective(budgets):
                    t_clicks = 0
                    for i, b in enumerate(budgets):
                        ad_id = ids[i]
                        c_spend = summary[summary['ID']==ad_id]['비용'].iloc[0]
                        c_cpc = c_spend / summary[summary['ID']==ad_id]['클릭수'].iloc[0] if summary[summary['ID']==ad_id]['클릭수'].iloc[0] > 0 else 999999
                        t_clicks += hill_performance_model(b, c_spend, c_cpc, forecast_cache[ad_id]['slope'])
                    return -t_clicks

                cons = ({'type': 'eq', 'fun': lambda b: sum(b) - total_b})
                bnds = [(0, summary[summary['ID']==id_]['비용'].iloc[0]*3) for id_ in ids]
                init_guess = [total_b / len(ids)] * len(ids)
                
                opt_res = minimize(objective, init_guess, method='SLSQP', bounds=bnds, constraints=cons)
                
                res_df = pd.DataFrame({'ID': ids, '현재 예산': [summary[summary['ID']==id_]['비용'].iloc[0] for id_ in ids], '최적화 제안': opt_res.x})
                res_df['차이'] = res_df['최적화 제안'] - res_df['현재 예산']
                
                st.dataframe(res_df.style.format({'현재 예산':'{:,.0f}', '최적화 제안':'{:,.0f}', '차이':'+{:,.0f}'}))
                st.plotly_chart(px.bar(res_df, x='ID', y=['현재 예산', '최적화 제안'], barmode='group', title="예산 재배분 권고안"), use_container_width=True)
                st.success(f"최적화 완료: 현재 예산 범위 내에서 예상 클릭수가 약 {(-opt_res.fun / summary['클릭수'].sum() - 1)*100:.1f}% 개선될 것으로 예측됩니다.")