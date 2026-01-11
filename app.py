import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import HuberRegressor
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="Marketing Intelligence Pro", layout="wide")

# --- [엔진: 데이터 통합 및 정제] ---
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
            clean_actual = str(actual_col).strip().replace(" ", "").replace("_", "")
            if any(p in clean_actual for p in patterns):
                found_col = actual_col
                break
        if found_col is not None:
            final_df[std_key] = df[found_col]
        else:
            return pd.DataFrame(), std_key

    final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
    for col in ['노출수', '클릭수', '비용']:
        final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    
    final_df['CTR(%)'] = np.where(final_df['노출수'] > 0, (final_df['클릭수'] / final_df['노출수'] * 100), 0.0)
    final_df['ID'] = "[" + final_df['매체'].astype(str) + "] " + final_df['상품명'].astype(str) + "_" + final_df['소재명'].astype(str)
    
    return final_df.dropna(subset=['날짜']), None

def ml_forecast(data):
    if len(data) < 5: return None, None # 최소 데이터 기준
    y = data['CTR(%)'].values
    x = np.arange(len(y)).reshape(-1, 1)
    model = HuberRegressor()
    model.fit(x, y)
    forecast = model.predict(np.arange(len(y), len(y) + 7).reshape(-1, 1))
    future_dates = [data['날짜'].max() + timedelta(days=i) for i in range(1, 8)]
    return future_dates, forecast

# --- [UI 메인] ---
st.title("📊 통합 마케팅 성과 분석 시스템")

uploaded_file = st.file_uploader("파일을 업로드하세요 (xlsx, csv)", type=['xlsx', 'csv'])

if uploaded_file:
    # 1. 모든 시트 데이터 통합 로직
    all_dfs = []
    if uploaded_file.name.endswith('xlsx'):
        xl = pd.ExcelFile(uploaded_file)
        for sheet in xl.sheet_names:
            temp_df = pd.read_excel(uploaded_file, sheet_name=sheet)
            processed, _ = clean_and_process(temp_df)
            if not processed.empty:
                all_dfs.append(processed)
    else:
        try:
            raw_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        except:
            raw_df = pd.read_csv(uploaded_file, encoding='cp949')
        processed, _ = clean_and_process(raw_df)
        if not processed.empty:
            all_dfs.append(processed)

    if not all_dfs:
        st.error("❌ 분석 가능한 데이터를 찾을 수 없습니다. 컬럼명을 확인해주세요.")
    else:
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # --- PART 1: 매체별 통합 지표 (자동 실행) ---
        st.header("🌐 1. 매체별 통합 성과 요약 (모든 시트 합산)")
        m_sum = full_df.groupby('매체').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum'}).reset_index()
        m_sum['CTR(%)'] = (m_sum['클릭수'] / m_sum['노출수'] * 100).fillna(0)
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.plotly_chart(px.pie(m_sum, values='비용', names='매체', title="전체 매체별 광고비 비중"), use_container_width=True)
        with c_right:
            st.plotly_chart(px.bar(m_sum, x='매체', y='CTR(%)', color='매체', title="매체별 평균 CTR (%)"), use_container_width=True)

        # --- PART 2 & 3: 소재 비교 및 머신러닝 (인터랙티브) ---
        st.divider()
        st.header("⚖️ 2. 소재간 베이지안 승률 & 머신러닝 예측")
        
        ids = sorted(full_df['ID'].unique())
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1: sel_a = st.selectbox("기준 소재 (A)", ids, index=0, key="sb_a")
        with col_sel2: sel_b = st.selectbox("비교 소재 (B)", ids, index=min(1, len(ids)-1), key="sb_b")
        
        df_a = full_df[full_df['ID']==sel_a].sort_values('날짜')
        df_b = full_df[full_df['ID']==sel_b].sort_values('날짜')
        
        # 베이지안 계산
        s_a, s_b = df_a[['노출수','클릭수']].sum(), df_b[['노출수','클릭수']].sum()
        dist_a = np.random.beta(s_a['클릭수']+1, s_a['노출수']-s_a['클릭수']+1, 10000)
        dist_b = np.random.beta(s_b['클릭수']+1, s_b['노출수']-s_b['클릭수']+1, 10000)
        prob_b_win = (dist_b > dist_a).mean()
        
        st.subheader(f"🔍 {sel_b}의 상대적 승률: {prob_b_win*100:.1f}%")

        # 머신러닝 이중 그래프
        st.write("#### 📈 소재별 성과 추이 및 미래 예측 비교")
        fig_ml = go.Figure()

        # 소재 A 시각화
        f_dates_a, f_vals_a = ml_forecast(df_a)
        fig_ml.add_trace(go.Scatter(x=df_a['날짜'], y=df_a['CTR(%)'], name=f"{sel_a} (실적)", line=dict(color='blue', width=1)))
        if f_dates_a:
            fig_ml.add_trace(go.Scatter(x=f_dates_a, y=f_vals_a, name=f"{sel_a} (예측)", line=dict(dash='dash', color='blue', width=2)))

        # 소재 B 시각화
        f_dates_b, f_vals_b = ml_forecast(df_b)
        fig_ml.add_trace(go.Scatter(x=df_b['날짜'], y=df_b['CTR(%)'], name=f"{sel_b} (실적)", line=dict(color='red', width=1)))
        if f_dates_b:
            fig_ml.add_trace(go.Scatter(x=f_dates_b, y=f_vals_b, name=f"{sel_b} (예측)", line=dict(dash='dash', color='red', width=2)))

        fig_ml.update_layout(height=500, xaxis_title="날짜", yaxis_title="CTR (%)", hovermode="x unified")
        st.plotly_chart(fig_ml, use_container_width=True)

        if not f_dates_a or not f_dates_b:
            st.warning("일부 소재의 데이터가 부족하여 머신러닝 예측 점선이 표시되지 않을 수 있습니다. (최소 5일치 이상 필요)")