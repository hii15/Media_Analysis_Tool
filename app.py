import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.linear_model import HuberRegressor
from datetime import datetime, timedelta
import re

# 1. 페이지 설정
st.set_page_config(page_title="Marketing Data Science Tool", layout="wide")

# --- [엔진: 데이터 정제 및 머신러닝 분석 로직] ---
def clean_and_process(df):
    """컬럼 표준화 및 숫자 데이터 정제"""
    col_map = {
        '날짜': ['날짜', 'Date', '일자'], '매체': ['매체', 'Media', '채널'],
        '상품명': ['상품명', 'Product'], '소재명': ['소재명', 'Creative'],
        '노출수': ['노출수', 'Impression'], '클릭수': ['클릭수', 'Click'], '비용': ['비용', 'Cost']
    }
    for std, vars in col_map.items():
        for v in vars:
            if v in df.columns:
                df = df.rename(columns={v: std}); break

    # 날짜 처리
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    
    # 숫자 데이터 정제 (콤마 제거 및 수치화)
    for col in ['노출수', '클릭수', '비용']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    # 지표 계산 및 ID 생성
    df['CTR(%)'] = np.where(df['노출수'] > 0, (df['클릭수'] / df['노출수'] * 100), 0.0)
    df['ID'] = "[" + df['매체'].astype(str) + "] " + df['상품명'].astype(str) + "_" + df['소재명'].astype(str)
    
    return df.dropna(subset=['날짜', 'ID'])

def ml_forecast(data, days_to_predict=7):
    """Huber Regression 기반 머신러닝 예측 모델"""
    y = data['CTR(%)'].values
    x = np.arange(len(y)).reshape(-1, 1)
    
    # 이상치에 강한 머신러닝 모델 학습
    model = HuberRegressor()
    model.fit(x, y)
    
    # 미래 날짜 및 예측값 생성
    future_x = np.arange(len(y), len(y) + days_to_predict).reshape(-1, 1)
    forecast = model.predict(future_x)
    
    last_date = data['날짜'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    return future_dates, forecast

# --- [UI 섹션 시작] ---
st.title("📊 마케팅 통계 및 머신러닝 통합 분석 시스템")
st.markdown("엑셀 파일을 업로드하고 분석할 시트를 선택하면 **베이지안 승률**과 **머신러닝 예측** 리포트를 생성합니다.")

# 파일 업로드 (xlsx, csv)
uploaded_file = st.file_uploader("파일을 업로드하세요 (xlsx, csv)", type=['xlsx', 'csv'])

if uploaded_file:
    # 1. 파일 타입에 따른 로드 및 시트 선택 (기능 1번 적용)
    try:
        if uploaded_file.name.endswith('xlsx'):
            xl = pd.ExcelFile(uploaded_file)
            sheet_names = xl.sheet_names
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox("📄 분석할 시트를 선택하세요", sheet_names)
            else:
                selected_sheet = sheet_names[0]
            raw_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        else:
            raw_df = pd.read_csv(uploaded_file)

        # 데이터 정제 실행
        df = clean_and_process(raw_df)
        
        if not df.empty:
            st.success(f"✅ '{selected_sheet if uploaded_file.name.endswith('xlsx') else uploaded_file.name}' 데이터 로드 완료")
            
            # 소재 선택 인터페이스
            ids = sorted(df['ID'].unique())
            st.divider()
            c1, c2 = st.columns(2)
            with c1: sel_a = st.selectbox("기준 소재 (A)", ids, index=0)
            with c2: sel_b = st.selectbox("비교 소재 (B)", ids, index=1 if len(ids)>1 else 0)

            df_a = df[df['ID'] == sel_a].sort_values('날짜')
            df_b = df[df['ID'] == sel_b].sort_values('날짜')

            # --- [SECTION 1: 베이지안 통계 분석 (과거 누적)] ---
            st.header("1️⃣ 베이지안 기반 성과 비교 (과거 누적)")
            
            sum_a = df_a[['노출수', '클릭수']].sum()
            sum_b = df_b[['노출수', '클릭수']].sum()
            
            # 몬테카를로 시뮬레이션 (1만 번 실행)
            s_a = np.random.beta(sum_a['클릭수']+1, sum_a['노출수']-sum_a['클릭수']+1, 10000)
            s_b = np.random.beta(sum_b['클릭수']+1, sum_b['노출수']-sum_b['클릭수']+1, 10000)
            prob_b_win = (s_b > s_a).mean()

            m1, m2, m3 = st.columns(3)
            m1.metric(f"{sel_b}의 승리 확률", f"{prob_b_win*100:.1f}%")
            m2.metric("A의 누적 CTR", f"{(sum_a['클릭수']/sum_a['노출수']*100 if sum_a['노출수']>0 else 0):.2f}%")
            m3.metric("B의 누적 CTR", f"{(sum_b['클릭수']/sum_b['노출수']*100 if sum_b['노출수']>0 else 0):.2f}%")

            # --- [SECTION 2: 머신러닝 추세 분석 및 예측 (미래 예측)] ---
            st.divider()
            st.header("2️⃣ 머신러닝 추세 분석 (최근 흐름 및 예측)")
            
            # 신뢰도 체크 로직
            data_count = len(df_b)
            if data_count < 7:
                st.warning(f"⚠️ 현재 {sel_b}의 데이터가 {data_count}일치에 불과합니다. 머신러닝 예측은 7일 이상의 데이터가 있어야 신뢰도가 확보됩니다.")
                # 데이터가 적을 때는 단순 그래프만 표시
                fig = px.line(df_b, x='날짜', y='CTR(%)', markers=True, title=f"{sel_b} 최근 성과 추이 (기초 통계)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 7일 이상일 때 머신러닝 가동
                f_dates, f_values = ml_forecast(df_b)
                
                col_res, col_chart = st.columns([1, 2])
                with col_res:
                    curr_val = df_b['CTR(%)'].iloc[-1]
                    next_val = f_values[-1]
                    change = next_val - curr_val
                    
                    st.write(f"#### {sel_b} 머신러닝 진단 결과")
                    st.metric("7일 뒤 예상 CTR", f"{next_val:.2f}%", f"{change:.2f}%")
                    
                    if change < -0.05:
                        st.error("🚨 머신러닝 분석 결과: 소재 피로도가 심각합니다. 교체를 강력 권장합니다.")
                    elif change < 0:
                        st.warning("⚠️ 머신러닝 분석 결과: 성과 하락세가 감지되었습니다. 주의가 필요합니다.")
                    else:
                        st.success("✨ 머신러닝 분석 결과: 성과가 안정적입니다. 운영 유지를 권장합니다.")
                
                with col_chart:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_b['날짜'], y=df_b['CTR(%)'], name="과거 성과 실적", line=dict(color='#1f77b4', width=3)))
                    fig.add_trace(go.Scatter(x=f_dates, y=f_values, name="머신러닝 추세 예측", line=dict(color='#d62728', dash='dash', width=2)))
                    fig.update_layout(title=f"{sel_b} 머신러닝 추세선 리포트", xaxis_title="날짜", yaxis_title="CTR (%)")
                    st.plotly_chart(fig, use_container_width=True)

            # --- [SECTION 3: 전체 요약 데이터] ---
            st.divider()
            st.subheader("📋 전체 소재별 성과 요약 (집계)")
            raw_summary = df.groupby('ID').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum', 'CTR(%)':'mean'}).reset_index()
            st.dataframe(raw_summary.sort_values('CTR(%)', ascending=False), use_container_width=True)

        else:
            st.error("데이터 정제 후 분석할 유효 행이 없습니다. 컬럼명과 날짜 형식을 확인하세요.")
            
    except Exception as e:
        st.error(f"⚠️ 파일 처리 중 오류 발생: {e}")

else:
    st.info("💡 엑셀 파일을 업로드하면 분석이 시작됩니다. 여러 시트가 있는 경우 원하는 시트를 선택할 수 있습니다.")