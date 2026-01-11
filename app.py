import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.linear_model import HuberRegressor
from datetime import datetime, timedelta
import io

# 1. 페이지 설정
st.set_page_config(page_title="Marketing Intelligence Pro", layout="wide")

# --- [엔진: 데이터 정제 및 예외 처리] ---
def clean_and_process(df):
    # 컬럼 매핑 사전 (더 광범위하게 보완)
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
            # 공백 및 특수문자 제거 후 비교
            clean_actual = str(actual_col).strip().replace(" ", "").replace("_", "")
            if any(p in clean_actual for p in patterns):
                found_col = actual_col
                break
        if found_col is not None:
            final_df[std_key] = df[found_col]
        else:
            return pd.DataFrame(), std_key

    # 데이터 타입 강제 변환 및 빈칸 처리
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
st.title("📊 통합 마케팅 성과 분석 시스템")

uploaded_file = st.file_uploader("파일을 업로드하세요 (xlsx, csv)", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('xlsx'):
            xl = pd.ExcelFile(uploaded_file)
            selected_sheet = st.selectbox("📄 분석할 시트 선택", xl.sheet_names)
            raw_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        else:
            # CSV 인코딩 문제 해결을 위한 시도
            try:
                raw_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            except:
                raw_df = pd.read_csv(uploaded_file, encoding='cp949')

        if st.button("🚀 분석 시작"):
            df, missing_col = clean_and_process(raw_df)
            
            if df.empty:
                st.error(f"❌ '{missing_col}' 컬럼을 찾을 수 없습니다. 시트의 첫 줄(헤더) 이름을 확인해주세요.")
            else:
                # 1. 매체별 통합 성과
                st.header("🌐 1. 매체별 지출 및 성과 요약")
                m_sum = df.groupby('매체').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum'}).reset_index()
                m_sum['CTR(%)'] = (m_sum['클릭수'] / m_sum['노출수'] * 100).fillna(0)
                
                col_left, col_right = st.columns(2)
                with col_left:
                    st.plotly_chart(px.pie(m_sum, values='비용', names='매체', title="매체별 광고비 비중"), use_container_width=True)
                with col_right:
                    st.plotly_chart(px.bar(m_sum, x='매체', y='CTR(%)', color='매체', title="매체별 평균 CTR (%)"), use_container_width=True)

                # 2. 베이지안 승률 비교
                st.divider()
                st.header("⚖️ 2. 소재간 베이지안 승률 비교")
                ids = sorted(df['ID'].unique())
                c1, c2 = st.columns(2)
                with c1: sel_a = st.selectbox("기준 소재 (A)", ids, index=0)
                with c2: sel_b = st.selectbox("비교 소재 (B)", ids, index=min(1, len(ids)-1))
                
                df_a, df_b = df[df['ID']==sel_a], df[df['ID']==sel_b]
                s_a, s_b = df_a[['노출수','클릭수']].sum(), df_b[['노출수','클릭수']].sum()
                
                # 몬테카를로 시뮬레이션
                dist_a = np.random.beta(s_a['클릭수']+1, s_a['노출수']-s_a['클릭수']+1, 10000)
                dist_b = np.random.beta(s_b['클릭수']+1, s_b['노출수']-s_b['클릭수']+1, 10000)
                prob_b_win = (dist_b > dist_a).mean()
                
                st.metric(f"{sel_b}가 더 우수할 확률", f"{prob_b_win*100:.1f}%")

                # 3. 머신러닝 예측
                st.divider()
                st.header("📈 3. 머신러닝 기반 성과 예측")
                target = df_b.sort_values('날짜')
                if len(target) >= 7:
                    f_dates, f_vals = ml_forecast(target)
                    fig_ml = go.Figure()
                    fig_ml.add_trace(go.Scatter(x=target['날짜'], y=target['CTR(%)'], name="과거 실적"))
                    fig_ml.add_trace(go.Scatter(x=f_dates, y=f_vals, name="머신러닝 예측", line=dict(dash='dash', color='red')))
                    st.plotly_chart(fig_ml, use_container_width=True)
                    
                    diff = f_vals[-1] - target['CTR(%)'].iloc[-1]
                    st.info(f"💡 머신러닝 예측: 7일 뒤 CTR은 현재보다 약 {abs(diff):.2f}%p {'상승' if diff>0 else '하락'}할 것으로 보입니다.")
                else:
                    st.warning("머신러닝 예측을 위해선 최소 7일 이상의 데이터가 필요합니다.")

    except Exception as e:
        st.error(f"⚠️ 파일 처리 오류: {e}")