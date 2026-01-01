import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="AE 퍼포먼스 대시보드", layout="wide")

# 스타일 설정
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 22px; color: #1f77b4; }
    section[data-testid="stSidebar"] { background-color: #ffffff; }
    .stButton>button { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# 데이터 저장소 초기화
if 'daily_data' not in st.session_state:
    st.session_state.daily_data = []

# --- 사이드바: 데이터 입력 및 소재 관리 ---
with st.sidebar:
    st.header("⚙️ 데이터 입력")
    t_date = st.date_input("날짜", datetime.now())
    
    # 유형 및 매체 선택
    c_type = st.radio("소재 유형", ["배너(DA)", "영상(Video)"], horizontal=True)
    m_name = st.selectbox("매체", ["네이버", "카카오", "구글", "메타", "유튜브", "기타"])
    
    # 소재명 입력 (기본값 제공 및 직접 입력 가능)
    st.divider()
    creative_options = ["소재 A", "소재 B", "소재 C", "직접 입력"]
    selected_opt = st.selectbox("소재 선택/입력", creative_options)
    
    if selected_opt == "직접 입력":
        creative_name = st.text_input("소재명 직접 입력", "신규 소재_01")
    else:
        creative_name = selected_opt

    # 수치 입력
    c1, c2 = st.columns(2)
    with c1: imps = st.number_input("노출수(Imp)", min_value=0, value=1000)
    with c2: clicks = st.number_input("클릭수(Click)", min_value=0, value=10)
    cost = st.number_input("비용(Cost)", min_value=0, value=100000)
    
    if st.button("➕ 데이터 기록", use_container_width=True):
        st.session_state.daily_data.append({
            "날짜": t_date, "유형": c_type, "매체": m_name, "소재명": creative_name,
            "Imps": imps, "Clicks": clicks, "Cost": cost,
            "ID": f"{t_date}_{m_name}_{creative_name}_{len(st.session_state.daily_data)}"
        })
        st.rerun()

    # 데이터 삭제 관리
    if st.session_state.daily_data:
        st.divider()
        st.subheader("🗑️ 데이터 관리")
        df_tmp = pd.DataFrame(st.session_state.daily_data)
        to_del = st.multiselect("삭제 항목 선택", options=df_tmp['ID'].tolist())
        if st.button("선택 삭제"):
            st.session_state.daily