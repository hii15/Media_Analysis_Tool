import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# 1. 페이지 및 스타일 설정
st.set_page_config(page_title="AE 통합 성과 대시보드 PRO", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 20px; color: #1f77b4; font-weight: bold; }
    .stButton>button { border-radius: 8px; font-weight: bold; background-color: #1f77b4; color: white; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 소재별 통합 성과 대시보드")

# --- 1. 데이터 입력 섹션 (메인 상단) ---
st.subheader("📝 데이터 시트 (엑셀 데이터 붙여넣기)")
st.info("💡 엑셀 영역을 복사(Ctrl+C)한 후, 아래 표의 첫 번째 칸을 선택하고 붙여넣기(Ctrl+V) 하세요.")

if 'master_data' not in st.session_state:
    # 샘플 데이터 구조 (사용자가 붙여넣기 편하게 가이드 제공)
    st.session_state.master_data = pd.DataFrame([
        {"날짜": "2025-12-30", "유형": "배너(DA)", "매체": "네이버", "상품명": "GFA", "소재명": "소재 A", "노출수": 1000, "클릭수": 10, "비용": 100000}
    ])

# 엑셀 시트 기능을 하는 데이터 에디터
edited_df = st.data_editor(
    st.session_state.master_data, 
    num_rows="dynamic", 
    use_container_width=True,
    key="editor",
    column_config={
        "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
        "유형": st.column_config.SelectboxColumn("유형", options=["배너(DA)", "영상(Video)"]),
        "매체": st.column_config.SelectboxColumn("매체", options=["네이버", "카카오", "구글", "메타", "유튜브", "네트워크매체", "인벤", "루리웹", "디시인사이드"])
    }
)

if st.button("🚀 데이터 분석 및 차트 업데이트", use_container_width=True):
    st.session_state.master_data = edited_df
    st.rerun()

# --- 2. 분석 및 필터링 (데이터가 있을 때만 실행) ---
df = st.session_state.master_data.copy()

if not df.empty and len(df.columns) >= 8:
    # 데이터 정제
    df['날짜'] = pd.to_datetime(df['날짜'])
    for col in ['노출수', '클릭수', '비용']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 지표 계산
    df['CTR'] = (df['클릭수'] / df['노출수'] * 100).fillna(0)
    df['CPC'] = (df['비용'] / df['클릭수']).replace([float('inf')], 0).fillna(0)
    df['CPM'] = (df['비용'] / df['노출수'] * 1000).replace([float('inf')], 0).fillna(0)

    # 필터 레이아웃
    st.divider()
    f1, f2, f3 = st.columns([1, 1, 1])
    with f1: v_type = st.pills("📊 유형", ["통합", "배너(DA)", "영상(Video)"], default="통합") 
    with f2: v_media = st.selectbox("🎯 매체", ["전체 매체"] + sorted(df['매체'].unique().tolist()))
    with f3: time_unit = st.segmented_control("📅 기간 단위", ["일", "주", "월"], default="일")

    # 필터 적용
    plot_df = df.copy()
    if v_type != "통합": plot_df = plot_df[plot_df['유형'] == v_type]
    if v_media != "전체 매체": plot_df = plot_df[plot_df['매체'] == v_media]

    # 기간 단위별 그룹화
    if time_unit == "주": plot_df['날짜'] = plot_df['날짜'].dt.to_period('W').apply(lambda r: r.start_time)
    elif time_unit == "월": plot_df['날짜'] = plot_df['날짜'].dt.to_period('M').apply(lambda r: r.start_time)
    
    plot_df = plot_df.groupby(['날짜', '매체', '상품명', '소재명']).agg({'노출수':'sum','클릭수':'sum','비용':'sum'}).reset_index()
    plot_df['CTR'] = (plot_df['클릭수']/plot_df['노출수']*100).fillna(0)
    plot_df['CPC'] = (plot_df['비용']/plot_df['클릭수']).replace([float('inf')], 0).fillna(0)
    plot_df['CPM'] = (plot_df['비용']/plot_df['노출수']*1000).replace([float('inf')], 0).fillna(0)

    # 3. KPI 대시보드
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 비용", f"₩{int(plot_df['비용'].sum()):,}")
    k2.metric("평균 CTR", f"{plot_df['CTR'].mean():.2f}%")
    k3.metric("평균 CPC", f"₩{int(plot_df['CPC'].mean()):,}")
    k4.metric("평균 CPM", f"₩{int(plot_df['CPM'].mean()):,}")

    # 4. 트렌드 차트
    c1, c2 = st.columns([2, 1])
    with c1:
        m_choice = st.radio("지표 선택", ["CTR", "비용", "클릭수", "CPM"], horizontal=True)
        fig_line = px.line(plot_df, x="날짜", y=m_choice, color="소재명", markers=True, template="plotly_white", height=400)
        fig_line.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_line, use_container_width=True)
    with c2:
        fig_pie = px.pie(plot_df, values='비용', names='소재명', hole=0.4, template="plotly_white", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- 5. 심화 분석 (Advanced Analytics) ---
    st.divider()
    st.subheader("🧐 심화 성과 분석")
    tab1, tab2 = st.tabs(["📉 매체별 성과 안정성 (Risk)", "🎯 소재별 효율 분석 (Efficiency)"])

    with tab1:
        # 
        fig_box = px.box(df, x="매체", y="CTR", color="매체", points="all", template="plotly_white", height=450, title="매체별 CTR 분포 및 변동성")
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        # 
        fig_scatter = px.scatter(plot_df, x="CPM", y="CTR", size="비용", color="소재명", 
                                 hover_data=["매체", "상품명"], text="소재명", template="plotly_white", height=500, title="가성비(CPM) 대비 반응(CTR) 효율 맵")
        fig_scatter.add_hline(y=plot_df['CTR'].mean(), line_dash="dot", annotation_text="평균 CTR")
        fig_scatter.add_vline(x=plot_df['CPM'].mean(), line_dash="dot", annotation_text="평균 CPM")
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.warning("데이터 시트에 분석할 내용을 입력하거나 엑셀에서 붙여넣어 주세요.")