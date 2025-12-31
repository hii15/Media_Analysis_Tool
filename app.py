import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="Performance Dashboard", layout="wide")

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
    m_name = st.selectbox("매체", ["네이버", "카카오", "구글", "메타", "기타"])
    p_name = st.text_input("상품명", "웹툰빅배너")
    
    c1, c2 = st.columns(2)
    with c1: imps = st.number_input("노출수", min_value=0, value=1000)
    with c2: clicks = st.number_input("클릭수", min_value=0, value=10)
    cost = st.number_input("비용", min_value=0, value=100000)
    
    # [기능 유지] 데이터 '기록' 버튼을 누를 때마다 리스트에 추가됨
    if st.button("➕ 데이터 기록", use_container_width=True):
        st.session_state.daily_data.append({
            "날짜": t_date, "매체": m_name, "상품": p_name,
            "Imps": imps, "Clicks": clicks, "Cost": cost,
            "ID": f"{t_date}_{m_name}_{p_name}_{len(st.session_state.daily_data)}"
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
st.title("🎯 매체 성과 대시보드")

if st.session_state.daily_data:
    df = pd.DataFrame(st.session_state.daily_data)
    
    # [업데이트 1] 날짜 자동 정렬
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values(by='날짜', ascending=True)
    
    # 지표 계산
    df['CTR'] = (df['Clicks'] / df['Imps'] * 100).fillna(0)
    df['CPC'] = (df['Cost'] / df['Clicks']).replace([float('inf')], 0).fillna(0)
    df['CPM'] = (df['Cost'] / df['Imps'] * 1000).replace([float('inf')], 0).fillna(0)
    
    # 상단 요약 카드
    st.subheader("📍 성과 요약")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 비용", f"₩{df['Cost'].sum():,}")
    k2.metric("총 노출", f"{df['Imps'].sum():,}")
    k3.metric("평균 CTR", f"{df['CTR'].mean():.2f}%")
    k4.metric("평균 CPC", f"₩{int(df['CPC'].mean()):,}")

    st.divider()

    # 중단 차트 영역
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("### 📈 성과 추이")
        m_choice = st.radio("표시 지표:", ["CTR", "Cost", "Clicks"], horizontal=True)
        # 괄호 닫힘 확인 완료 (에러 지점)
        fig_line = px.line(df, x="날짜", y=m_choice, color="매체", markers=True,
                           template="plotly_white", title=f"일별 {m_choice} 변화")
        st.plotly_chart(fig_line, use_container_width=True)

    with col_r:
        st.markdown("### 📊 비용 비중")
        fig_pie = px.pie(df, values='Cost', names='매체', hole=0.5, template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)

    # [업데이트 2] 표 열 순서 조절 및 데이터 확인
    st.divider()
    st.subheader("📝 전체 데이터 확인")
    
    # 요청하신 순서: 날짜, 매체, 상품, imp, click, ctr, cpc, cpm, cost
    display_df = df[['날짜', '매체', '상품', 'Imps', 'Clicks', 'CTR', 'CPC', 'CPM', 'Cost']]
    
    # 날짜 포맷 깔끔하게 변경
    display_df['날짜'] = display_df['날짜'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(display_df, use_container_width=True)

else:
    st.info("사이드바에서 데이터를 입력하고 '데이터 기록' 버튼을 누르세요.")