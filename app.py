import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="AE 매체/유형별 대시보드", layout="wide")

# 스타일 설정
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 22px; color: #1f77b4; }
    section[data-testid="stSidebar"] { background-color: #ffffff; }
    .stButton>button { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# 데이터 초기화
if 'daily_data' not in st.session_state:
    st.session_state.daily_data = []

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 데이터 입력")
    t_date = st.date_input("날짜", datetime.now())
    
    # [추가] 소재 유형 구분 (배너 vs 영상)
    c_type = st.radio("소재 유형", ["배너(DA)", "영상(Video)"], horizontal=True)
    
    m_name = st.selectbox("매체", ["네이버", "카카오", "구글", "메타", "유튜브", "기타"])
    p_name = st.text_input("상품명", "웹툰빅배너")
    
    c1, c2 = st.columns(2)
    with c1: imps = st.number_input("노출수", min_value=0, value=1000)
    with c2: clicks = st.number_input("클릭수", min_value=0, value=10)
    cost = st.number_input("비용", min_value=0, value=100000)
    
    if st.button("➕ 데이터 기록", use_container_width=True):
        st.session_state.daily_data.append({
            "날짜": t_date, "유형": c_type, "매체": m_name, "상품": p_name,
            "Imps": imps, "Clicks": clicks, "Cost": cost,
            "ID": f"{t_date}_{c_type}_{m_name}_{len(st.session_state.daily_data)}"
        })
        st.rerun()

    if st.session_state.daily_data:
        st.divider()
        st.subheader("🗑️ 데이터 관리")
        df_tmp = pd.DataFrame(st.session_state.daily_data)
        to_del = st.multiselect("삭제 항목", options=df_tmp['ID'].tolist())
        if st.button("선택 삭제"):
            st.session_state.daily_data = [d for d in st.session_state.daily_data if d['ID'] not in to_del]
            st.rerun()

# --- 메인 화면 ---
st.title("🎯 매체/유형별 성과 대시보드")

if st.session_state.daily_data:
    df = pd.DataFrame(st.session_state.daily_data)
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values(by='날짜')
    
    # 지표 계산
    df['CTR'] = (df['Clicks'] / df['Imps'] * 100).fillna(0)
    df['CPC'] = (df['Cost'] / df['Clicks']).replace([float('inf')], 0).fillna(0)
    df['CPM'] = (df['Cost'] / df['Imps'] * 1000).replace([float('inf')], 0).fillna(0)
    
    # [추가] 차트 필터링: 통합 / 배너별 / 영상별
    st.divider()
    view_option = st.segmented_control("📊 보기 설정", ["통합", "배너(DA)", "영상(Video)"], default="통합")
    
    if view_option == "통합":
        plot_df = df