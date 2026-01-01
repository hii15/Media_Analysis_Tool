import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="AE 통합 성과 대시보드", layout="wide")

# 스타일 설정
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 22px; color: #1f77b4; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 소재별 통합 성과 대시보드")

# --- 1. 데이터 입력 섹션 (메인 화면 상단) ---
st.subheader("📝 데이터 입력 및 편집")
st.info("💡 아래 표에 직접 입력하거나, 엑셀 데이터를 복사(Ctrl+C)해서 붙여넣기(Ctrl+V) 하세요.")

# 초기 샘플 데이터 구조
if 'df_master' not in st.session_state:
    st.session_state.df_master = pd.DataFrame([
        {"날짜": "2025-12-30", "매체": "네이버", "상품명": "웹툰빅배너", "소재명": "소재 A", "유형": "배너(DA)", "노출수": 1000, "클릭수": 10, "비용": 100000}
    ])

# 엑셀 형태의 데이터 에디터
edited_df = st.data_editor(
    st.session_state.df_master, 
    num_rows="dynamic", 
    use_container_width=True,
    column_config={
        "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD", required=True),
        "유형": st.column_config.SelectboxColumn("유형", options=["배너(DA)", "영상(Video)"]),
        "매체": st.column_config.SelectboxColumn("매체", options=["네이버", "카카오", "구글", "메타", "유튜브", "네트워크매체", "인벤", "루리웹", "디시인사이드"])
    }
)

# 데이터 적용 버튼
if st.button("🚀 데이터 분석 및 차트 업데이트", use_container_width=True):
    st.session_state.df_master = edited_df
    st.success("데이터가 성공적으로 반영되었습니다!")

# --- 2. 데이터 가공 및 분석 로직 ---
df = st.session_state.df_master.copy()

if not df.empty:
    # 수치형 변환 및 전처리
    df['날짜'] = pd.to_datetime(df['날짜'])
    for col in ['노출수', '클릭수', '비용']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 지표 계산
    df['CTR'] = (df['클릭수'] / df['노출수'] * 100).fillna(0)
    df['CPC'] = (df['비용'] / df['클릭수']).replace([float('inf')], 0).fillna(0)
    df['CPM'] = (df['비용'] / df['노출수'] * 1000).replace([float('inf')], 0).fillna(0)

    # --- 3. 필터 및 KPI 영역 ---
    st.divider()
    f1, f2, f3 = st.columns([1, 1, 1])
    with f1: v_type = st.pills("📊 유형 필터", ["통합", "배너(DA)", "영상(Video)"], default="통합") 
    with f2: v_media = st.selectbox("🎯 매체 필터", ["전체 매체"] + sorted(df['매체'].unique().tolist()))
    with f3: time_unit = st.segmented_control("📅 시간 단위", ["일", "주", "월"], default="일")

    # 필터링 적용
    plot_df = df.copy()
    if v_type != "통합": plot_df = plot_df[plot_df['유형'] == v_type]
    if v_media != "전체 매체": plot_df = plot_df[plot_df['매체'] == v_media]

    # 시간 단위 그룹화
    if time_unit == "주": plot_df['날짜'] = plot_df['날짜'].dt.to_period('W').apply(lambda r: r.start_time)
    elif time_unit == "월": plot_df['날짜'] = plot_df['날짜'].dt.to_period('M').apply(lambda r: r.start_time)
    
    plot_df = plot_df.groupby(['날짜', '매체', '상품명', '소재명']).agg({'노출수':'sum','클릭수':'sum','비용':'sum'}).reset_index()
    plot_df['CTR'] = (plot_df['클릭수']/plot_df['노출수']*100).fillna(0)
    plot_df['CPC'] = (plot_df['비용']/plot_df['클릭수']).replace([float('inf')], 0).fillna(0)
    plot_df['CPM'] = (plot_df['비용']/plot_df['노출수']*1000).replace([float('inf')], 0).fillna(0)

    # KPI 지표
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 비용", f"₩{int(plot_df['비용'].sum()):,}")
    k2.metric("평균 CTR", f"{plot_df['CTR'].mean():.2f}%")
    k3.metric("평균 CPC", f"₩{int(plot_df['CPC'].mean()):,}")
    k4.metric("평균 CPM", f"₩{int(plot_df['CPM'].mean()):,}")

    # --- 4. 메인 차트 영역 ---
    c_col_l, c_col_r = st.columns([2, 1])
    with c_col_l:
        m_choice = st.radio("표시 지표 선택:", ["CTR", "비용", "클릭수", "CPM"], horizontal=True)
        fig_line = px.line(plot_df, x="날짜", y=m_choice, color="소재명", markers=True, 
                           template="plotly_white", height=400, title=f"{time_unit}별 {m_choice} 추이")
        fig_line.update_xaxes(tickformat="%Y-%m-%d") # 시/분/초 제거
        st.plotly_chart(fig_line, use_container_width=True)
    with c_col_r:
        fig_pie = px.pie(plot_df, values='비용', names='소재명', hole=0.5, template="plotly_white", height=400, title="소재별 비용 비중")
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- 5. 심화 분석 섹션 ---
    st.divider()
    st.subheader("🧐 심화 성과 분석 (Advanced)")
    tab1, tab2 = st.tabs(["📉 매체별 성과 안정성", "🎯 소재별 효율 분포 Map"])

    with tab1:
        fig_box = px.box(df, x="매체", y="CTR", color="매체", points="all", template="plotly_white", height=450, title="매체별 CTR 편차 (안정성 체크)")
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        fig_scatter = px.scatter(plot_df, x="CPM", y="CTR", size="비용", color="소재명", 
                                 hover_data=["매체", "상품명"], text="소재명", template="plotly_white", height=500, title="가성비(CPM) 대비 반응(CTR) 분포")
        fig_scatter.add_hline(y=plot_df['CTR'].mean(), line_dash="dot", annotation_text="평균 CTR")
        fig_scatter.add_vline(x=plot_df['CPM'].mean(), line_dash="dot", annotation_text="평균 CPM")
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.warning("표에 데이터를 입력해 주세요.")