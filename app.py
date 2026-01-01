import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="매체 성과 대시보드", layout="wide")

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
    
    c_type = st.radio("소재 유형", ["배너(DA)", "영상(Video)"], horizontal=True)
    m_name = st.selectbox("매체", ["네이버", "카카오", "구글", "메타", "유튜브", "네트워크매체", "인벤", "루리웹", "디시인사이드"])
    
    st.divider()
    creative_options = ["소재 A", "소재 B", "소재 C", "직접 입력"]
    selected_opt = st.selectbox("소재 선택/입력", creative_options)
    
    if selected_opt == "직접 입력":
        creative_name = st.text_input("소재명 직접 입력", "신규 소재_01")
    else:
        creative_name = selected_opt

    c1, c2 = st.columns(2)
    with c1: imps = st.number_input("노출수(Imp)", min_value=1, value=1000) # [수정] 0 나누기 방지를 위해 min 1 설정
    with c2: clicks = st.number_input("클릭수(Click)", min_value=0, value=10)
    cost = st.number_input("비용(Cost)", min_value=0, value=100000)
    
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
        to_del = st.multiselect("삭제 항목 선택", options=df_tmp['ID'].tolist())
        if st.button("선택 삭제"):
            st.session_state.daily_data = [d for d in st.session_state.daily_data if d['ID'] not in to_del]
            st.rerun()

# --- 메인 대시보드 화면 ---
st.title("🎯 소재별 통합 성과 대시보드")

if st.session_state.daily_data:
    df = pd.DataFrame(st.session_state.daily_data)
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values(by='날짜', ascending=True)
    
    # 지표 계산
    df['CTR'] = (df['Clicks'] / df['Imps'] * 100).fillna(0)
    df['CPC'] = (df['Cost'] / df['Clicks']).replace([float('inf')], 0).fillna(0)
    df['CPM'] = (df['Cost'] / df['Imps'] * 1000).replace([float('inf')], 0).fillna(0)
    
    st.divider()
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        # [수정] 최신 버전 Streamlit 권장 함수 사용 (segmented_control 사용 가능 시 변경 권장)
        v_type = st.pills("📊 유형 필터", ["통합", "배너(DA)", "영상(Video)"], default="통합") 
    with f_col2:
        m_list = ["전체 매체"] + sorted(df['매체'].unique().tolist())
        v_media = st.selectbox("🎯 매체 필터", m_list)

    # 필터링 적용
    plot_df = df.copy()
    if v_type != "통합":
        plot_df = plot_df[plot_df['유형'] == v_type]
    if v_media != "전체 매체":
        plot_df = plot_df[plot_df['매체'] == v_media]

    # --- KPI 요약 ---
    if not plot_df.empty: # [수정] 필터링 결과가 없을 때 에러 방지
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("총 비용", f"₩{plot_df['Cost'].sum():,}")
        k2.metric("평균 CTR", f"{plot_df['CTR'].mean():.2f}%")
        k3.metric("평균 CPC", f"₩{int(plot_df['CPC'].mean()):,}")
        k4.metric("평균 CPM", f"₩{int(plot_df['CPM'].mean()):,}")

        st.markdown("<br>", unsafe_allow_html=True)
        c_col_l, c_col_r = st.columns([2, 1])
        
        with c_col_l:
            st.markdown(f"#### 📈 {v_type} 성과 추이")
            m_choice = st.radio("지표 선택:", ["CTR", "Cost", "Clicks", "CPM"], horizontal=True, key="metric_radio")
            
            # [수정] 데이터가 1개일 때 라인 차트 오류 방지를 위해 markers=True 유지 및 예외 처리
            fig_line = px.line(plot_df, x="날짜", y=m_choice, color="소재명", symbol="매체",
                               markers=True, template="plotly_white", height=450)
            st.plotly_chart(fig_line, use_container_width=True)

        with c_col_r:
            st.markdown("#### 📊 소재별 비용 비중")
            fig_pie = px.pie(plot_df, values='Cost', names='소재명', hole=0.5, 
                             template="plotly_white", height=450)
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("선택한 필터에 해당하는 데이터가 없습니다.")

    # --- 상세 데이터 표 ---
    st.divider()
    st.subheader("📝 상세 데이터 내역")
    # [수정] 데이터 표 가독성을 위해 날짜 포맷 적용 및 열 재정렬
    display_df = df[['날짜', '매체', '소재명', '유형', 'Imps', 'Clicks', 'CTR', 'CPC', 'CPM', 'Cost']].copy()
    display_df['날짜'] = display_df['날짜'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(display_df, use_container_width=True)

else:
    st.info("사이드바에서 데이터를 입력하고 '데이터 기록'을 누르면 대시보드가 구성됩니다.")