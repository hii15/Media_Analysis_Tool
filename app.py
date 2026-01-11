import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import logging

# 1. 설정 및 로그 제어
logging.getLogger('prophet').setLevel(logging.WARNING)
st.set_page_config(page_title="Marketing Analytics Pro", layout="wide")

# --- [엔진: 데이터 처리 및 정교화된 매핑] ---
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
        if found:
            final_df[std_key] = df[found[0]]
        else:
            found_sub = [c for c in df.columns if any(p in str(c) for p in patterns)]
            if found_sub: final_df[std_key] = df[found_sub[0]]
    
    if len(final_df.columns) < len(mapping):
        return pd.DataFrame()

    final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
    for col in ['노출수', '클릭수', '비용']:
        final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    
    final_df['CTR(%)'] = np.where(final_df['노출수'] > 0, (final_df['클릭수'] / final_df['노출수'] * 100), 0.0)
    final_df['CPC'] = np.where(final_df['클릭수'] > 0, (final_df['비용'] / final_df['클릭수']), 0.0)
    final_df['ID'] = "[" + final_df['상품명'].astype(str) + "] " + final_df['소재명'].astype(str)
    
    return final_df.dropna(subset=['날짜'])

# --- [엔진: 확률적 예측 로직] ---
def robust_forecast(data):
    valid_df = data[data['노출수'] >= 100].sort_values('날짜').copy()
    if len(valid_df) < 7: return None, None
    
    try:
        # Logit 변환
        p = np.clip(valid_df['CTR(%)'].values / 100, 0.0001, 0.9999)
        valid_df['y_logit'] = np.log(p / (1 - p))
        
        m = Prophet(interval_width=0.8, daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=True)
        m.fit(valid_df[['날짜', 'y_logit']].rename(columns={'날짜': 'ds', 'y_logit': 'y'}))
        
        future = m.make_future_dataframe(periods=7)
        forecast = m.predict(future)
        
        def inv_logit(x): return (np.exp(x) / (1 + np.exp(x))) * 100
        
        res = pd.DataFrame({
            'ds': forecast['ds'],
            'yhat': inv_logit(forecast['yhat']),
            'yhat_lower': inv_logit(forecast['yhat_lower']),
            'yhat_upper': inv_logit(forecast['yhat_upper'])
        })
        
        # Fit Quality (R^2)
        y_true = valid_df['y_logit'].values
        y_pred = forecast.iloc[:len(y_true)]['yhat'].values
        fit_quality = max(0, 1 - (np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-6)))
        
        return res, fit_quality
    except:
        return None, None

# --- [UI 메인] ---
st.title("🔬 고신뢰도 마케팅 분석 시스템 (Ver. Pro)")
st.warning("⚠️ 본 도구는 의사결정 '참고용'입니다.")

uploaded_file = st.file_uploader("분석 데이터 업로드", type=['xlsx', 'csv'])

# 변수 초기화 (NameError 방지)
full_df = pd.DataFrame()

if uploaded_file:
    all_dfs = []
    if uploaded_file.name.endswith('xlsx'):
        xl = pd.ExcelFile(uploaded_file)
        for sheet in xl.sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name=sheet)
            processed = clean_and_process_pro(df)
            if not processed.empty: all_dfs.append(processed)
    else:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        processed = clean_and_process_pro(df)
        if not processed.empty: all_dfs.append(processed)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)

# 데이터가 로드된 경우에만 탭 표시 (NameError 해결)
if not full_df.empty:
    tab1, tab2, tab3, tab4 = st.tabs(["💎 성과 요약", "🔍 전체 리포트", "⚖️ 베이지안 진단", "📈 확률적 수명 예측"])
    
    ids = sorted(full_df['ID'].unique())

    with tab1:
        st.header("🏢 상품별 성과 효율")
        p_sum = full_df.groupby('상품명').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum', 'CTR(%)':'mean'}).reset_index()
        p_sum['CPC'] = (p_sum['비용'] / p_sum['클릭수'].replace(0, 1))
        p_sum['효율성점수'] = (p_sum['CTR(%)'] / p_sum['CPC'].replace(0, 0.001))
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(p_sum, values='비용', names='상품명', title="상품별 예산 비중"), use_container_width=True)
        c2.plotly_chart(px.bar(p_sum, x='상품명', y='효율성점수', title="예산 효율성 가이드"), use_container_width=True)

    with tab2:
        st.header("🔍 모든 상품/소재 성과 일람")
        total_summary = full_df.groupby(['ID', '매체']).agg({'노출수':'sum', '클릭수':'sum', '비용':'sum'}).reset_index()
        total_summary['CTR(%)'] = (total_summary['클릭수'] / total_summary['노출수'] * 100).fillna(0)
        total_summary['CPC'] = (total_summary['비용'] / total_summary['클릭수']).replace([np.inf, -np.inf], 0).fillna(0)
        
        # ImportError 방지: 스타일링을 단순화하거나 필수 라이브러리 체크
        try:
            st.dataframe(
                total_summary.style.background_gradient(cmap='Blues', subset=['CTR(%)'])
                .format({'비용': '{:,.0f}', 'CPC': '{:,.1f}', 'CTR(%)': '{:.2f}%'}),
                use_container_width=True
            )
        except:
            st.dataframe(total_summary, use_container_width=True)

    with tab3:
        st.header("⚖️ 소재간 베이지안 진단")
        st.markdown("**📊 가이드:** 두 산의 거리가 멀수록 성과 차이가 '실력'일 확률이 높습니다.")
        c_sel1, c_sel2 = st.columns(2)
        sel_a = c_sel1.selectbox("기준 소재 (A)", ids, index=0)
        sel_b = c_sel2.selectbox("비교 소재 (B)", ids, index=min(1, len(ids)-1))
        df_a, df_b = full_df[full_df['ID']==sel_a], full_df[full_df['ID']==sel_b]
        s_a, s_b = df_a[['노출수','클릭수']].sum(), df_b[['노출수','클릭수']].sum()
        if s_a['노출수'] > 100 and s_b['노출수'] > 100:
            dist_a = np.random.beta(s_a['클릭수']+1, s_a['노출수']-s_a['클릭수']+1, 10000)
            dist_b = np.random.beta(s_b['클릭수']+1, s_b['노출수']-s_b['클릭수']+1, 10000)
            prob_b_win = (dist_b > dist_a).mean()
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=dist_a, name=f"A: {sel_a}", opacity=0.5, marker_color='blue'))
            fig.add_trace(go.Histogram(x=dist_b, name=f"B: {sel_b}", opacity=0.5, marker_color='red'))
            st.plotly_chart(fig, use_container_width=True)
            winner = sel_b if prob_b_win > 0.5 else sel_a
            st.success(f"🏆 진단 결과: **[{winner}]**가 더 우수할 확률이 **{(prob_b_win if prob_b_win > 0.5 else 1-prob_b_win)*100:.1f}%**입니다.")

    with tab4:
        st.header("📈 확률적 수명 예측")
        sel_target = st.selectbox("분석 대상 소재 선택", ids)
        target_df = full_df[full_df['ID'] == sel_target]
        
        forecast_res, fit_score = robust_forecast(target_df)
        
        if forecast_res is not None:
            fig = go.Figure()
            # 실측치
            fig.add_trace(go.Scatter(x=target_df['날짜'], y=target_df['CTR(%)'], name="실측 CTR", mode='lines+markers', line=dict(color='black')))
            # 예측 범위 (Uncertainty)
            fig.add_trace(go.Scatter(x=forecast_res['ds'], y=forecast_res['yhat_upper'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_res['ds'], y=forecast_res['yhat_lower'], fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', line=dict(width=0), name="80% 예측 구간"))
            # 예측선
            fig.add_trace(go.Scatter(x=forecast_res['ds'], y=forecast_res['yhat'], name="기대 추세", line=dict(color='red', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns(2)
            c1.metric("모델 적합도 (Fit Quality)", f"{fit_score*100:.1f}%")
            
            curr_ctr, pred_ctr = target_df['CTR(%)'].iloc[-1], forecast_res['yhat'].iloc[-1]
            if pred_ctr < curr_ctr * 0.8:
                st.error(f"📉 **추세 주의:** 하향 신호가 감지되었습니다. (7일 후 기대값: {pred_ctr:.2f}%)")
            else:
                st.success(f"📈 **추세 유지:** 현재 성과 범위 내에서 안정적입니다.")
        else:
            st.warning("예측을 위한 충분한 데이터(노출 100회 이상, 7일 이상)가 부족합니다.")