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
    """
    1. 컬럼명 유연 감지 (매핑)
    2. 날짜 형식 강제 변환
    3. 빈칸(Null)을 0으로 대체
    4. 숫자 데이터 기호 제거 및 수치화
    """
    # 유연한 컬럼 매핑 사전
    col_map_patterns = {
        '날짜': ['날짜', '일자', 'Date', 'Day', '일시'],
        '매체': ['매체', '채널', 'Media', 'Channel', 'Platform'],
        '상품명': ['상품명', '상품', 'Product', 'Campaign', '목표'],
        '소재명': ['소재명', '소재', 'Creative', 'AdName', 'Content', '제목'],
        '노출수': ['노출수', '노출', 'Imp', 'Impression', 'View'],
        '클릭수': ['클릭수', '클릭', 'Click'],
        '비용': ['비용', '지출', 'Cost', 'Spend', 'Amount', '금액']
    }
    
    final_df = pd.DataFrame()
    
    # [기능 1] 컬럼명 자동 매칭
    for std_key, patterns in col_map_patterns.items():
        found_col = None
        for actual_col in df.columns:
            # 공백 제거 및 소문자화하여 비교
            clean_actual = str(actual_col).strip().replace(" ", "")
            if any(p in clean_actual for p in patterns):
                found_col = actual_col
                break
        
        if found_col is not None:
            final_df[std_key] = df[found_col]
        else:
            st.error(f"❌ 필수 항목인 **'{std_key}'**를 찾을 수 없습니다. (후보: {patterns})")
            return pd.DataFrame()

    # [기능 2] 빈칸 처리 - 수치 데이터는 0으로, 텍스트는 '미분류'로
    final_df['매체'] = final_df['매체'].fillna('Unknown')
    final_df['상품명'] = final_df['상품명'].fillna('Default')
    final_df['소재명'] = final_df['소재명'].fillna('Unnamed')

    # [기능 3] 날짜 형식 보정
    final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
    
    # [기능 4] 숫자 데이터 정제 (콤마, 통화기호 제거 및 0 채우기)
    for col in ['노출수', '클릭수', '비용']:
        final_df[col] = (
            final_df[col].astype(str)
            .str.replace(r'[^\d.]', '', regex=True) # 숫자와 소수점 제외 제거
            .replace('', '0')
        )
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
    
    # 지표 계산 및 고유 ID 생성
    final_df['CTR(%)'] = np.where(final_df['노출수'] > 0, (final_df['클릭수'] / final_df['노출수'] * 100), 0.0)
    final_df['ID'] = "[" + final_df['매체'].astype(str) + "] " + final_df['상품명'].astype(str) + "_" + final_df['소재명'].astype(str)
    
    return final_df.dropna(subset=['날짜'])

def ml_forecast(data, days_to_predict=7):
    """Huber Regression 기반 머신러닝 예측 모델"""
    y = data['CTR(%)'].values
    x = np.arange(len(y)).reshape(-1, 1)
    
    model = HuberRegressor()
    model.fit(x, y)
    
    future_x = np.arange(len(y), len(y) + days_to_predict).reshape(-1, 1)
    forecast = model.predict(future_x)
    
    last_date = data['날짜'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    return future_dates, forecast

# --- [UI 섹션 시작] ---
st.title("📊 통합 마케팅 성과 분석 (머신러닝 & 통계)")
st.info("💡 엑셀의 컬럼명이 '날짜', '매체', '소재명', '노출수', '클릭수' 등을 포함하면 자동으로 인식합니다.")

uploaded_file = st.file_uploader("엑셀 또는 CSV 파일을 업로드하세요", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        # 파일 타입에 따른 로드
        if uploaded_file.name.endswith('xlsx'):
            xl = pd.ExcelFile(uploaded_file)
            sheet_names = xl.sheet_names
            selected_sheet = st.selectbox("📄 분석할 시트를 선택하세요", sheet_names)
            raw_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        else:
            raw_df = pd.read_csv(uploaded_file)

        # 데이터 정제 실행 (컬럼 매핑, 날짜 보정, 빈칸 처리 포함)
        df = clean_and_process(raw_df)
        
        if not df.empty:
            st.success("✅ 데이터 전처리가 완료되었습니다.")
            
            # 분석 대상 선택
            ids = sorted(df['ID'].unique())
            st.divider()
            c1, c2 = st.columns(2)
            with c1: sel_a = st.selectbox("기준 소재 (A)", ids, index=0)
            with c2: sel_b = st.selectbox("비교 소재 (B)", ids, index=min(1, len(ids)-1))

            df_a = df[df['ID'] == sel_a].sort_values('날짜')
            df_b = df[df['ID'] == sel_b].sort_values('날짜')

            # --- [SECTION 1: 베이지안 통계 분석] ---
            st.header("1️⃣ 베이지안 성과 비교 (누적 실적)")
            sum_a = df_a[['노출수', '클릭수']].sum()
            sum_b = df_b[['노출수', '클릭수']].sum()
            
            s_a = np.random.beta(sum_a['클릭수']+1, sum_a['노출수']-sum_a['클릭수']+1, 10000)
            s_b = np.random.beta(sum_b['클릭수']+1, sum_b['노출수']-sum_b['클릭수']+1, 10000)
            prob_b_win = (s_b > s_a).mean()

            m1, m2, m3 = st.columns(3)
            m1.metric(f"{sel_b} 승리 확률", f"{prob_b_win*100:.1f}%")
            m2.metric(f"{sel_a} 누적 CTR", f"{(sum_a['클릭수']/sum_a['노출수']*100 if sum_a['노출수']>0 else 0):.2f}%")
            m3.metric(f"{sel_b} 누적 CTR", f"{(sum_b['클릭수']/sum_b['노출수']*100 if sum_b['노출수']>0 else 0):.2f}%")

            # --- [SECTION 2: 머신러닝 추세 분석] ---
            st.divider()
            st.header("2️⃣ 머신러닝 추세 분석 및 예측")
            
            data_count = len(df_b)
            if data_count < 7:
                st.warning(f"⚠️ 데이터가 {data_count}일치로 부족합니다. 머신러닝 예측을 위해선 7일 이상의 데이터가 권장됩니다.")
                fig = px.line(df_b, x='날짜', y='CTR(%)', markers=True, title="기초 성과 추이")
                st.plotly_chart(fig, use_container_width=True)
            else:
                f_dates, f_values = ml_forecast(df_b)
                col_res, col_chart = st.columns([1, 2])
                
                with col_res:
                    curr_val = df_b['CTR(%)'].iloc[-1]
                    next_val = f_values[-1]
                    diff = next_val - curr_val
                    st.write("#### 소재 건강도 진단")
                    st.metric("7일 뒤 예상 CTR", f"{next_val:.2f}%", f"{diff:.2f}%")
                    if diff < -0.05: st.error("🚨 피로도 높음: 교체 권장")
                    elif diff < 0: st.warning("🟡 하락세 감지: 주의")
                    else: st.success("🟢 안정적: 유지 권장")
                
                with col_chart:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_b['날짜'], y=df_b['CTR(%)'], name="실적", line=dict(color='#1f77b4', width=3)))
                    fig.add_trace(go.Scatter(x=f_dates, y=f_values, name="ML예측", line=dict(color='#d62728', dash='dash')))
                    fig.update_layout(xaxis_title="날짜", yaxis_title="CTR (%)")
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ 분석 중 오류 발생: {e}")
        st.info("팁: 엑셀 파일의 첫 번째 행이 컬럼명인지 확인하고, 데이터가 비어있지 않은지 점검해 보세요.")