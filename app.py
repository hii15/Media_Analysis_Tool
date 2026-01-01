import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import re

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="AE 통합 성과 대시보드", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 18px; color: #1f77b4; font-weight: bold; }
    .stButton>button { border-radius: 8px; font-weight: bold; background-color: #1f77b4; color: white; height: 3em; }
    /* 셀 크기 및 표 가독성 향상 */
    div[data-testid="stTable"] { overflow: auto; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 소재별 통합 성과 대시보드")

# --- 1. 데이터 처리 함수 (날짜 자동 변환 및 지표 계산) ---
def process_data(df):
    if df.empty: return df
    
    # 날짜 자동 변환 로직 (20251130 -> 2025-11-30)
    def clean_date(x):
        x = str(x).replace("-", "").replace(".", "").strip()
        if len(x) == 8 and x.isdigit():
            return f"{x[:4]}-{x[4:6]}-{x[6:]}"
        return x

    df['날짜'] = df['날짜'].apply(clean_date)
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    
    # 수치형 변환
    for col in ['노출수', '클릭수', '비용']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 표 우측에 자동 계산 지표 추가
    df['CTR(%)'] = (df['클릭수'] / df['노출수'] * 100).round(2).fillna(0)
    df['CPC'] = (df['비용'] / df['클릭수']).replace([float('inf')], 0).round(0).fillna(0)
    df['CPM'] = (df['비용'] / df['노출수'] * 1000).round(0).fillna(0)
    
    return df

# --- 2. 데이터 입력 (핵심: 에러 방지용 새 키값 사용) ---
st.subheader("📝 데이터 시트 (엑셀 데이터 붙여넣기)")
st.info("💡 '20251130' 처럼 숫자로만 입력해도 날짜가 자동 변환됩니다. 표 오른쪽에서 CTR/CPC/CPM이 자동 계산됩니다.")

# 캐시 충돌 방지를 위한 독립 세션 키
if 'df_final_v4' not in st.session_state:
    st.session_state.df_final_v4 = pd.DataFrame([
        {"날짜": "20251231", "유형": "배너(DA)", "매체": "네이버", "상품명": "GFA", "소재명": "소재 A", "노출수": 1000, "클릭수": 10, "비용": 100000}
    ])

# 데이터 에디터 (셀 크기 및 자동 계산 반영)
raw_edited_df = st.data_editor(
    st.session_state.df_final_v4,
    num_rows="dynamic",
    use_container_width=True,
    key="editor_v4_stable",
    column_config={
        "날짜": st.column_config.TextColumn("날짜 (예: 20251130)", width="medium"),
        "유형": st.column_config.SelectboxColumn("유형", options=["배너(DA)", "영상(Video)"], width="small"),
        "매체": st.column_config.SelectboxColumn("매체", options=["네이버", "카카오", "구글", "메타", "유튜브", "인벤", "루리웹"], width="small"),
        "상품명": st.column_config.TextColumn("상품명", width="medium"),
        "소재명": st.column_config.TextColumn("소재명", width="medium"),
        "노출수": st.column_config.NumberColumn("노출수", format="%d", width="small"),
        "클릭수": st.column_config.NumberColumn("클릭수", format="%d", width="small"),
        "비용": st.column_config.NumberColumn("비용", format="₩%d", width="small")
    }
)

# 데이터 가공 실행
final_df = process_data(raw_edited_df.copy())

if st.button("🚀 데이터 분석 및 대시보드 업데이트", use_container_width=True):
    st.session_state.df_final_v4 = raw_edited_df
    st.rerun()

# --- 3. 분석 시각화 ---
if not final_df.empty and final_df['날짜'].notnull().any():
    st.divider()
    
    # 상단 KPI 요약
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 비용", f"₩{int(final_df['비용'].sum()):,}")
    k2.metric("평균 CTR", f"{final_df['CTR(%)'].mean():.2f}%")
    k3.metric("평균 CPC", f"₩{int(final_df['CPC'].mean()):,}")
    k4.metric("평균 CPM", f"₩{int(final_df['CPM'].mean()):,}")

    # 트렌드 차트
    c1, c2 = st.columns([2, 1])
    with c1:
        m_choice = st.radio("표시 지표", ["CTR(%)", "비용", "클릭수", "CPM"], horizontal=True)
        # 날짜 순 정렬 후 시각화
        chart_df = final_df.sort_values('날짜')
        fig_line = px.line(chart_df, x="날짜", y=m_choice, color="소재명", markers=True, template="plotly_white", height=400)
        st.plotly_chart(fig_line, use_container_width=True)
    with c2:
        fig_pie = px.pie(final_df, values='비용', names='소재명', hole=0.4, template="plotly_white", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- 4. 심화 분석 (Advanced Analytics) ---
    st.divider()
    st.subheader("🧐 심화 분석 (Advanced)")
    t1, t2 = st.tabs(["📉 매체별 성과 안정성", "🎯 소재별 효율 분포(Efficiency Map)"])
    
    with t1:
        fig_box = px.box(final_df, x="매체", y="CTR(%)", color="매체", points="all", template="plotly_white", height=450)
        st.plotly_chart(fig_box, use_container_width=True)
    with t2:
        fig_scatter = px.scatter(final_df, x="CPM", y="CTR(%)", size="비용", color="소재명", 
                                 hover_data=["매체", "상품명"], text="소재명", template="plotly_white", height=500)
        fig_scatter.add_hline(y=final_df['CTR(%)'].mean(), line_dash="dot", annotation_text="평균 CTR")
        fig_scatter.add_vline(x=final_df['CPM'].mean(), line_dash="dot", annotation_text="평균 CPM")
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.warning("표에 정확한 데이터를 입력해 주세요. (날짜 형식 오류 주의)")