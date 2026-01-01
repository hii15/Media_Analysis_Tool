import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="AE 퍼포먼스 대시보드 PRO", layout="wide")

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

# --- 사이드바: 데이터 입력 모드 ---
with st.sidebar:
    st.header("⚙️ 데이터 입력 모드")
    input_mode = st.radio("입력 방식 선택", ["개별 입력", "엑셀 붙여넣기(대량)"])

    if input_mode == "개별 입력":
        t_date = st.date_input("날짜", datetime.now())
        c_type = st.radio("소재 유형", ["배너(DA)", "영상(Video)"], horizontal=True)
        m_name = st.selectbox("매체", ["네이버", "카카오", "구글", "메타", "유튜브", "네트워크매체", "인벤", "루리웹", "디시인사이드"])
        product_name = st.text_input(f"[{m_name}] 상품명", placeholder="예: 웹툰빅배너")
        
        st.divider()
        creative_options = ["소재 A", "소재 B", "소재 C", "직접 입력"]
        selected_opt = st.selectbox("소재 선택", creative_options)
        creative_name = st.text_input("소재명 직접 입력", "신규 소재_01") if selected_opt == "직접 입력" else selected_opt

        c1, c2 = st.columns(2)
        with c1: imps = st.number_input("노출수", min_value=1, value=1000)
        with c2: clicks = st.number_input("클릭수", min_value=0, value=10)
        cost = st.number_input("비용", min_value=0, value=100000)
        
        if st.button("➕ 데이터 기록", use_container_width=True):
            st.session_state.daily_data.append({
                "날짜": t_date, "유형": c_type, "매체": m_name, "상품명": product_name, "소재명": creative_name,
                "Imps": imps, "Clicks": clicks, "Cost": cost,
                "ID": f"{t_date}_{m_name}_{creative_name}_{len(st.session_state.daily_data)}"
            })
            st.rerun()

    else:
        st.info("💡 아래 표에 엑셀 내용을 복사(Ctrl+C)해서 붙여넣기(Ctrl+V) 하세요.")
        # 붙여넣기용 임시 데이터프레임 구조
        df_template = pd.DataFrame(columns=["날짜", "유형", "매체", "상품명", "소재명", "Imps", "Clicks", "Cost"])
        edited_df = st.data_editor(df_template, num_rows="dynamic", use_container_width=True)
        
        if st.button("🚀 붙여넣은 데이터 일괄 저장"):
            if not edited_df.empty:
                new_data = edited_df.to_dict('records')
                # ID 생성 및 추가
                for item in new_data:
                    item['ID'] = f"batch_{datetime.now().strftime('%M%S')}_{item['소재명']}"
                st.session_state.daily_data.extend(new_data)
                st.success(f"{len(new_data)}건 저장 완료!")
                st.rerun()

    # 삭제 관리
    if st.session_state.daily_data:
        st.divider()
        st.subheader("🗑️ 관리")
        df_tmp = pd.DataFrame(st.session_state.daily_data)
        to_del = st.multiselect("삭제 항목", options=df_tmp['ID'].tolist())
        if st.button("선택 삭제"):
            st.session_state.daily_data = [d for d in st.session_state.daily_data if d['ID'] not in to_del]
            st.rerun()

# --- 메인 대시보드 ---
st.title("🎯 소재별 통합 성과 대시보드")

if st.session_state.daily_data:
    df = pd.DataFrame(st.session_state.daily_data)
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values(by='날짜', ascending=True)
    
    # 기본 지표 계산 (에러 방지용)
    df['Imps'] = pd.to_numeric(df['Imps'], errors='coerce').fillna(0)
    df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce').fillna(0)
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce').fillna(0)
    
    df['CTR'] = (df['Clicks'] / df['Imps'] * 100).fillna(0)
    df['CPC'] = (df['Cost'] / df['Clicks']).replace([float('inf')], 0).fillna(0)
    df['CPM'] = (df['Cost'] / df['Imps'] * 1000).replace([float('inf')], 0).fillna(0)
    
    # 상단 필터 및 추이 분석 (기존 코드 유지)
    st.divider()
    f1, f2, f3 = st.columns([1, 1, 1])
    with f1: v_type = st.pills("📊 유형", ["통합", "배너(DA)", "영상(Video)"], default="통합") 
    with f2: v_media = st.selectbox("🎯 매체 필터", ["전체 매체"] + sorted(df['매체'].unique().tolist()))
    with f3: time_unit = st.segmented_control("📅 시간 단위", ["일", "주", "월"], default="일")

    # 필터링 및 시간 단위 그룹화
    plot_df = df.copy()
    if v_type != "통합": plot_df = plot_df[plot_df['유형'] == v_type]
    if v_media != "전체 매체": plot_df = plot_df[plot_df['매체'] == v_media]

    if time_unit == "주": plot_df['날짜'] = plot_df['날짜'].dt.to_period('W').apply(lambda r: r.start_time)
    elif time_unit == "월": plot_df['날짜'] = plot_df['날짜'].dt.to_period('M').apply(lambda r: r.start_time)
    
    plot_df = plot_df.groupby(['날짜', '매체', '상품명', '소재명']).agg({'Imps':'sum','Clicks':'sum','Cost':'sum'}).reset_index()
    plot_df['CTR'] = (plot_df['Clicks']/plot_df['Imps']*100).fillna(0)
    plot_df['CPC'] = (plot_df['Cost']/plot_df['Clicks']).replace([float('inf')], 0).fillna(0)
    plot_df['CPM'] = (plot_df['Cost']/plot_df['Imps']*1000).replace([float('inf')], 0).fillna(0)

    # 지표 시각화 (기존 코드 유지)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 비용", f"₩{int(plot_df['Cost'].sum()):,}")
    k2.metric("평균 CTR", f"{plot_df['CTR'].mean():.2f}%")
    k3.metric("평균 CPC", f"₩{int(plot_df['CPC'].mean()):,}")
    k4.metric("평균 CPM", f"₩{int(plot_df['CPM'].mean()):,}")

    c_col_l, c_col_r = st.columns([2, 1])
    with c_col_l:
        m_choice = st.radio("지표 선택:", ["CTR", "Cost", "Clicks", "CPM"], horizontal=True)
        fig_line = px.line(plot_df, x="날짜", y=m_choice, color="소재명", markers=True, template="plotly_white", height=400)
        fig_line.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_line, use_container_width=True)
    with c_col_r:
        fig_pie = px.pie(plot_df, values='Cost', names='소재명', hole=0.5, template="plotly_white", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- 하단 심화 성과 분석 ---
    st.divider()
    st.subheader("🧐 심화 성과 분석 (Advanced Analytics)")
    tab1, tab2 = st.tabs(["📉 매체별 성과 안정성", "🎯 소재별 효율 분포 Map"])

    with tab1:
        st.markdown("#### 매체별 CTR 변동 범위 분석")
        fig_box = px.box(df, x="매체", y="CTR", color="매체", points="all", template="plotly_white", height=450)
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        st.markdown("#### 소재별 가성비(CPM) 대비 반응(CTR) 효율")
        fig_scatter = px.scatter(plot_df, x="CPM", y="CTR", size="Cost", color="소재명", 
                                 hover_data=["매체", "상품명"], text="소재명", template="plotly_white", height=500)
        fig_scatter.add_hline(y=plot_df['CTR'].mean(), line_dash="dot", annotation_text="평균 CTR")
        fig_scatter.add_vline(x=plot_df['CPM'].mean(), line_dash="dot", annotation_text="평균 CPM")
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.info("데이터를 입력하거나 엑셀 내용을 붙여넣으세요.")