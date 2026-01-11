import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.linear_model import HuberRegressor
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="Advanced Marketing Analyzer", layout="wide")

# --- [엔진: 실무 데이터 예외 처리 및 정제] ---
def clean_and_process(df):
    col_map_patterns = {
        '날짜': ['날짜', '일자', 'Date', 'Day', '일시'],
        '매체': ['매체', '채널', 'Media', 'Channel', 'Platform'],
        '상품명': ['상품명', '상품', 'Product', 'Campaign'],
        '소재명': ['소재명', '소재', 'Creative', 'AdName', 'Content'],
        '노출수': ['노출수', '노출', 'Imp', 'Impression'],
        '클릭수': ['클릭수', '클릭', 'Click'],
        '비용': ['비용', '지출', 'Cost', 'Spend']
    }
    
    final_df = pd.DataFrame()
    for std_key, patterns in col_map_patterns.items():
        found_col = None
        for actual_col in df.columns:
            clean_actual = str(actual_col).strip().replace(" ", "")
            if any(p in clean_actual for p in patterns):
                found_col = actual_col
                break
        if found_col is not None:
            final_df[std_key] = df[found_col]
        else:
            return pd.DataFrame(), std_key # 실패 시 어떤 컬럼 때문인지 반환

    # 데이터 정제
    final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
    for col in ['노출수', '클릭수', '비용']:
        final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    
    final_df['CTR(%)'] = np.where(final_df['노출수'] > 0, (final_df['클릭수'] / final_df['노출수'] * 100), 0.0)
    final_df['ID'] = "[" + final_df['매체'].astype(str) + "] " + final_df['상품명'].astype(str) + "_" + final_df['소재명'].astype(str)
    
    return final_df.dropna(subset=['날짜']), None

def ml_forecast(data):
    y = data['CTR(%)'].values
    x = np.arange(len(y)).reshape(-1, 1)
    model = HuberRegressor()
    model.fit(x, y)
    forecast = model.predict(np.arange(len(y), len(y) + 7).reshape(-1, 1))
    future_dates = [data['날짜'].max() + timedelta(days=i) for i in range(1, 8)]
    return future_dates, forecast

# --- [UI 메인] ---
st.title("🚀 통합 마케팅 성과 분석 시스템")

uploaded_file = st.file_uploader("파일을 업로드하세요 (xlsx, csv)", type=['xlsx', 'csv'])

if uploaded_file:
    # 파일 로드
    if uploaded_file.name.endswith('xlsx'):
        xl = pd.ExcelFile(uploaded_file)
        selected_sheet = st.selectbox("📄 분석할 시트 선택", xl.sheet_names)
        raw_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    else:
        raw_df = pd.read_csv(uploaded_file)

    # 분석 실행 버튼 추가
    if st.button("📊 분석 시작"):
        df, missing_col = clean_and_process(raw_df)
        
        if df.empty:
            st.error(f"❌ '{missing_col}' 컬럼을 찾을 수 없거나 데이터 형식이 잘못되었습니다. 양식을 확인해주세요.")
        else:
            st.success("✅ 데이터 분석 준비 완료")

            # --- [Part 1: 매체별 합산 성과 (Top-View)] ---
            st.header("🌐 1. 매체별 통합 성과 요약")
            media_summary = df.groupby('매체').agg({
                '노출수': 'sum', '클릭수': 'sum', '비용': 'sum'
            }).reset_index()
            media_summary['CTR(%)'] = (media_summary['클릭수'] / media_summary['노출수'] * 100).fillna(0)
            
            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(media_summary, values='비용', names='매체', title="매체별 지출 비중")
                st.plotly_chart(fig_pie)
            with c2:
                fig_bar = px.bar(media_summary, x='매체', y='CTR(%)', title="매체별 평균 CTR", color='매체')
                st.plotly_chart(fig_bar)

            # --- [Part 2: 소재별 상세 분석 (Drill-down)] ---
            st.divider()
            st.header("🎯 2. 소재별 머신러닝 예측 및 비교")
            
            ids = sorted(df['ID'].unique())
            sel_id = st.selectbox("상세 분석할 소재 선택", ids)
            target = df[df['ID'] == sel_id].sort_values('날짜')

            if len(target) >= 7:
                f_dates, f_vals = ml_forecast(target)
                
                col_m1, col_m2 = st.columns([1, 2])
                with col_m1:
                    curr_ctr = target['CTR(%)'].iloc[-1]
                    pred_ctr = f_vals[-1]
                    st.metric("현재 CTR", f"{curr_ctr:.2f}%")
                    st.metric("7일 뒤 예측 CTR", f"{pred_ctr:.2f}%", f"{pred_ctr - curr_ctr:.2f}%")
                
                with col_m2:
                    fig_ml = go.Figure()
                    fig_ml.add_trace(go.Scatter(x=target['날짜'], y=target['CTR(%)'], name="과거 성과"))
                    fig_ml.add_trace(go.Scatter(x=f_dates, y=f_vals, name="머신러닝 예측", line=dict(dash='dash', color='red')))
                    fig_ml.update_layout(title=f"{sel_id} 성과 추이 및 미래 예측")
                    st.plotly_chart(fig_ml)
            else:
                st.warning("머신러닝 분석을 위해선 해당 소재의 데이터가 7일 이상 필요합니다.")
                st.line_chart(target.set_index('날짜')['CTR(%)'])