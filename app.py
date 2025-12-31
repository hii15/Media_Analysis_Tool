import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="AE 퍼포먼스 대시보드 (소재별 분석)", layout="wide")

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
    
    c_type = st.radio("소재 유형", ["배너(DA)", "영상(Video)"], horizontal=True)
    m_name = st.selectbox("매체", ["네이버", "카카오", "구글", "메타", "유튜브", "기타"])
    
    # [개선] 소재명 입력 (디폴트 값 제공 및 사용자 수정 가능)
    st.divider()
    creative_default = ["소재 A", "소재 B", "소재 C", "직접 입력"]
    selected_creative = st.selectbox("소재 선택/입력", creative_default)
    
    if selected_creative == "직접 입력":
        creative_name = st.text_input("소재명을 입력하세요", "신규 소재_01")
    else:
        creative_name = selected_creative

    c1, c2 = st.columns(2)
    with c1: imps = st.number_input("노출수", min_value=0, value=1000)
    with c2: clicks = st.number_input("클릭수", min_value=0, value=10)
    cost = st.number_input("비용", min_value=0, value=100000)
    
    if st.button("➕ 데이터 기록", use_container_width=True):
        st.session_state.daily_data.append({
            "날짜": t_date, "유형": c_type, "매체": m_name, "소재명": creative_name,
            "Imps": imps, "Clicks": clicks, "Cost": cost,
            "ID": f"{t_date}_{m_name}_{creative_name}_{len(st.session_state.daily_data)}"
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
st.title("🎯 소재별 성과 대시보드")

if st.session_state.daily_data:
    df = pd.DataFrame(st.session_state.daily_data)
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values(by='날짜')
    
    # 지표 계산
    df['CTR'] = (df['Clicks'] / df['Imps'] * 100).fillna(0)
    df['CPC'] = (df['Cost'] / df['Clicks']).replace([float('inf')], 0).fillna(0)
    df['CPM'] = (df['Cost'] / df['Imps'] * 1000).replace([float('inf')], 0).fillna(0)
    
    # 보기 설정 필터
    st.divider()
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        view_type = st.segmented_control("📊 유형 필터", ["통합", "배너(DA)", "영상(Video)"], default="통합")
    with col_f2