import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# 1. 페이지 설정 (넓은 화면 모드)
st.set_page_config(page_title="Performance Dashboard", layout="wide")

# 2. 배경색 및 카드 스타일링 (보내주신 이미지 느낌의 CSS)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #1f77b4; }
    div.stButton > button { width: 100%; border-radius: 5px; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# 데이터 저장소 초기화
if 'daily_data' not in st.session_state:
    st.session_state.daily_data = []

# --- 사이드바: 입력 및 관리 ---
with st.sidebar:
    st.header("⚙️ 데이터 입력")
    target_date = st.date_input("날짜 선택", datetime.now())
    m_name = st.selectbox("매체 선택", ["네이버", "카카오", "구글", "메타", "기타"])
    p_name = st.text_input("상품명", "GFA")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        imps = st.number_input("노출수", value=0)
        cost = st.number_input("비용", value=0)
    with col_in2:
        clicks = st.number_input("클릭수", value=0)
    
    if st.button("➕ 데이터 기록"):
        st.session_state.daily_data.append({
            "날짜": target_date, "매체": m_name, "상품": p_name,
            "Imps": imps, "Clicks": clicks, "Cost": cost,
            "ID": f"{target_date}_{m_name}_{p_name}"
        })
        st.rerun()

    if st.session_state.daily_data:
        st.divider()
        st.subheader("🗑️ 데이터 관리")
        df_tmp = pd.DataFrame(st.session_state.daily_data)
        to_delete = st.multiselect("삭제 항목", options=df_tmp['ID'].tolist())
        if st.button("선택 삭제"):
            st.session_state.daily_data = [d for d in st.session_state.daily_data if d['ID'] not in to_delete]
            st.rerun()

# --- 메인 대시보드 영역 ---
st.title("📊 매체 퍼포먼스 대시보드")

if st.session_state.daily_data:
    df = pd.DataFrame(st.session_state.daily_data)
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values('날짜')
    
    # 지표 계산
    total_imps = df['Imps'].sum()
    total_clicks = df['Clicks'].sum()
    total_cost = df['Cost'].sum()
    avg_ctr = (total_clicks / total_imps * 100) if total_imps > 0 else 0
    avg_cpc = (total_cost / total_clicks) if total_clicks > 0 else 0

    # [LAYOUT 1] 상단 요약 지표 (이미지의 Gauge 차트 느낌)
    st.subheader("📍 핵심 성과 요약")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("총 노출수", f"{total_imps:,}")
    m2.metric("총 클릭수", f"{total_clicks:,}")
    m3.metric("총 집행비용", f"₩{total_cost:,}")
    m4.metric("평균 CTR", f"{avg_ctr:.2f}%")
    m5.metric("평균 CPC", f"₩{int(avg_cpc):,}")

    st.divider()

    # [LAYOUT 2] 중간 차트 영역 (2분할 카드 레이아웃)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📈 일별 성과 추이")
        metric_choice = st.segmented_control("지표 선택", ["CTR", "Cost", "Clicks"], default="CTR")
        fig_line = px.line(df, x="날짜", y=metric_choice, color="매체", markers=True, 
                           template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_line, use_container_width=True)

    with col_right:
        st.markdown("### 🥧 매체별 비용 비중")
        fig_pie = px.pie(df, values='Cost', names='매체', hole=0.4,
                         template="plotly_white", color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig_pie, use_container_width=True)

    # [LAYOUT 3] 하단 상세 데이터
    st.divider()
    with st.expander("📝 상세 데이터 테이블 보기"):
        st.dataframe(df.drop(columns=['ID']), use_container_width=True)

else:
    st.info("왼쪽 사이드바에서 데이터를 입력하면 대시보드가 활성화됩니다.")