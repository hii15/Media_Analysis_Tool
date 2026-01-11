import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from scipy.optimize import minimize
from datetime import datetime, timedelta  # timedelta 추가
import logging

# 1. 설정 및 로그 제어
logging.getLogger('prophet').setLevel(logging.WARNING)
st.set_page_config(page_title="Marketing Analytics & Optimizer", layout="wide")

# --- [엔진 1: 데이터 처리 및 베이지안 보정] ---
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
    
    # 기본 지표 계산
    final_df['CTR(%)'] = np.where(final_df['노출수'] > 0, (final_df['클릭수'] / final_df['노출수'] * 100), 0.0)
    
    # 베이지안 보정 CTR
    global_mean = final_df['클릭수'].sum() / (final_df['노출수'].sum() + 1e-6)
    K = 100 
    final_df['Adj_CTR'] = (final_df['클릭수'] + K * global_mean) / (final_df['노출수'] + K) * 100
    final_df['ID'] = "[" + final_df['상품명'].astype(str) + "] " + final_df['소재명'].astype(str)
    
    return final_df.dropna(subset=['날짜'])

# --- [엔진 2: 예측 및 최적화 로직] ---
def get_forecast_and_slope(data):
    valid_df = data[data['노출수'] >= 10].sort_values('날짜').copy()
    if len(valid_df) < 7: return None, 0
    
    try:
        # Logit 변환 예측
        p = np.clip(valid_df['Adj_CTR'].values / 100, 0.0001, 0.9999)
        valid_df['y_logit'] = np.log(p / (1 - p))
        m = Prophet(interval_width=0.8, daily_seasonality=False, weekly_seasonality=True)
        m.fit(valid_df[['날짜', 'y_logit']].rename(columns={'날짜': 'ds', 'y_logit': 'y'}))
        
        future = m.make_future_dataframe(periods=7)
        forecast = m.predict(future)
        slope = (forecast['yhat'].values[-1] - forecast['yhat'].values[-7]) / 7
        
        def inv_logit(x): return (np.exp(x) / (1 + np.exp(x))) * 100
        res = pd.DataFrame({'ds': forecast['ds'], 'yhat': inv_logit(forecast['yhat']), 
                            'yhat_lower': inv_logit(forecast['yhat_lower']), 'yhat_upper': inv_logit(forecast['yhat_upper'])})
        return res, slope
    except: return None, 0

def hill_model(budget, current_spend, avg_cpc, slope):
    if budget <= 0 or avg_cpc <= 0: return 0
    base_clicks = budget / avg_cpc
    penalty = 1.0 + abs(min(0, slope)) * 3.0
    efficiency = 1.0 / (1.0 + (0.15 * penalty * (max(0, budget/(current_spend+1e-6) - 1.0))**1.2))
    return base_clicks * efficiency

# --- [UI 메인] ---
st.title("🔬 마케팅 사이언스 통합 의사결정 시스템")

# 1. 초기화 (NameError 방지 핵심)
full_df = pd.DataFrame() 

uploaded_file = st.file_uploader("데이터 업로드", type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('xlsx'): df_raw = pd.read_excel(uploaded_file)
    else: df_raw = pd.read_csv(uploaded_file)
    full_df = clean_and_process_pro(df_raw)

# 2. 데이터 유무 체크 로직 수정
if not full_df.empty:
    ids = sorted(full_df['ID'].unique())
    forecast_cache = {}
    for i in ids:
        f_res, f_slope = get_forecast_and_slope(full_df[full_df['ID'] == i])
        forecast_cache[i] = {'res': f_res, 'slope': f_slope}

    tabs = st.tabs(["📊 성과", "📈 수명", "🕹️ 시뮬레이션", "🎯 최적화"])

    with tabs[0]: # 성과
        st.header("📊 전주 대비 성과(WoW)")
        max_date = full_df['날짜'].max()
        this_week = full_df[full_df['날짜'] > max_date - timedelta(days=7)]
        last_week = full_df[(full_df['날짜'] <= max_date - timedelta(days=7)) & (full_df['날짜'] > max_date - timedelta(days=14))]
        
        c1, c2 = st.columns(2)
        c1.metric("이번 주 지출", f"{this_week['비용'].sum():,.0f}원")
        st.plotly_chart(px.bar(full_df.groupby('ID')['Adj_CTR'].mean().reset_index(), x='ID', y='Adj_CTR', title="보정된 소재별 평균 성과"), use_container_width=True)

    with tabs[1]: # 수명 (KeyError 방지 수정 완료)
        st.header("📈 확률적 수명 분석")
        sel_id = st.selectbox("소재 선택", ids)
        target_df = full_df[full_df['ID'] == sel_id]
        f_data = forecast_cache[sel_id]['res']
        
        if f_data is not None:
            fig = go.Figure()
            # KeyError 발생 지점 수정: 'CTR(%)' 컬럼 존재 확인
            fig.add_trace(go.Scatter(x=target_df['날짜'], y=target_df['CTR(%)'], name="원시 실적", mode='markers'))
            fig.add_trace(go.Scatter(x=f_data['ds'], y=f_data['yhat'], name="추세선", line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=f_data['ds'], y=f_data['yhat_upper'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=f_data['ds'], y=f_data['yhat_lower'], fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line=dict(width=0), name="예측 범위"))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]: # 시뮬레이션
        st.header("🕹️ What-If 시뮬레이터")
        sim_id = st.selectbox("시뮬레이션 대상", ids, key="sim")
        t_data = full_df[full_df['ID'] == sim_id]
        c_spend = t_data['비용'].sum()
        c_cpc = c_spend / (t_data['클릭수'].sum() + 1e-6)
        
        n_spend = st.slider("예산 변경 (원)", 0.0, c_spend * 3.0, float(c_spend))
        p_clicks = hill_model(n_spend, c_spend, c_cpc, forecast_cache[sim_id]['slope'])
        st.metric("예상 클릭수", f"{p_clicks:,.0f}", f"{p_clicks - t_data['클릭수'].sum():,.0f}")

    with tabs[3]: # 최적화
        st.header("🎯 예산 최적화 제안")
        if st.button("🚀 최적 배분 계산"):
            total_b = full_df['비용'].sum()
            summary = full_df.groupby('ID').agg({'비용':'sum', '클릭수':'sum'}).reset_index()
            def objective(b_list):
                total_clicks = 0
                for idx, b in enumerate(b_list):
                    target_id = ids[idx]
                    cur_s = summary[summary['ID']==target_id]['비용'].iloc[0]
                    cur_cpc = cur_s / (summary[summary['ID']==target_id]['클릭수'].iloc[0] + 1e-6)
                    total_clicks += hill_model(b, cur_s, cur_cpc, forecast_cache[target_id]['slope'])
                return -total_clicks
            
            res = minimize(objective, [total_b/len(ids)]*len(ids), method='SLSQP', 
                           bounds=[(0, s*3) for s in summary['비용']], constraints={'type':'eq', 'fun': lambda b: sum(b)-total_b})
            
            res_df = pd.DataFrame({'ID': ids, '현재 예산': summary['비용'], '최적화 제안': res.x})
            st.dataframe(res_df.style.format({'현재 예산':'{:,.0f}', '최적화 제안':'{:,.0f}'}))
else:
    st.info("데이터를 업로드하면 분석이 시작됩니다.")