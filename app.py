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
            st.session_state.daily_data = [d for d in st.session_state.daily_data if d['ID'] not in to_del]
            st.rerun()

# --- 메인 대시보드 화면 ---
st.title("🎯 소재별 통합 성과 대시보드")

if st.session_state.daily_data:
    # 데이터프레임 생성 및 정렬
    df = pd.DataFrame(st.session_state.daily_data)
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values(by='날짜', ascending=True) # [요청] 날짜순 자동 나열
    
    # 지표 계산 (CTR, CPC, CPM)
    df['CTR'] = (df['Clicks'] / df['Imps'] * 100).fillna(0)
    df['CPC'] = (df['Cost'] / df['Clicks']).replace([float('inf')], 0).fillna(0)
    df['CPM'] = (df['Cost'] / df['Imps'] * 1000).replace([float('inf')], 0).fillna(0)
    
    # --- 상단 필터 레이아웃 ---
    st.divider()
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        v_type = st.radio("📊 유형 필터", ["통합", "배너(DA)", "영상(Video)"], horizontal=True)
    with f_col2:
        m_list = ["전체 매체"] + sorted(df['매체'].unique().tolist())
        v_media = st.selectbox("🎯 매체 필터", m_list)

    # 필터링 적용
    plot_df = df.copy()
    if v_type != "통합":
        plot_df = plot_df[plot_df['유형'] == v_type]
    if v_media != "전체 매체":
        plot_df = plot_df[plot_df['매체'] == v_media]

    # --- 핵심 성과 요약 (KPI) ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 비용", f"₩{plot_df['Cost'].sum():,}")
    k2.metric("평균 CTR", f"{plot_df['CTR'].mean():.2f}%")
    k3.metric("평균 CPC", f"₩{int(plot_df['CPC'].mean()):,}")
    k4.metric("평균 CPM", f"₩{int(plot_df['CPM'].mean()):,}")

    # --- 차트 영역 ---
    st.markdown("<br>", unsafe_allow_html=True)
    c_col_l, c_col_r = st.columns([2, 1])
    
    with c_col_l:
        st.markdown(f"#### 📈 {v_type} 성과 추이")
        m_choice = st.radio("지표 선택:", ["CTR", "Cost", "Clicks", "CPM"], horizontal=True)
        # 소재별 성과를 선으로 구분
        fig_line = px.line(plot_df, x="날짜", y=m_choice, color="소재명", symbol="매체",
                           markers=True, template="plotly_white", height=450)
        st.plotly_chart(fig_line, use_container_width=True)

    with c_col_r:
        st.markdown("#### 📊 소재별 비용 비중")
        fig_pie = px.pie(plot_df, values='Cost', names='소재명', hole=0.5, 
                         template="plotly_white", height=450)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- 전체 데이터 표 (요청하신 열 순서) ---
    st.divider()
    st.subheader("📝 상세 데이터 내역")
    # 순서: 날짜, 매체, 소재명, imp, click, ctr, cpc, cpm, cost (유형 추가)
    display_df = df[['날짜', '매체', '소재명', '유형', 'Imps', 'Clicks', 'CTR', 'CPC', 'CPM', 'Cost']]
    display_df['날짜'] = display_df['날짜'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(display_df, use_container_width=True)

else:
    st.info("사이드바에서 데이터를 입력하고 '데이터 기록'을 누르면 대시보드가 구성됩니다.")