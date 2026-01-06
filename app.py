import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="AE 통합 성과 대시보드 PRO", layout="wide")

st.title("🎯 소재별 통합 성과 대시보드")

# --- 데이터 처리 유틸리티 ---
def clean_and_calculate(df):
    if df.empty: 
        return df
    
    new_df = df.copy()

    # 1. 날짜 처리 (에러 방지를 위해 문자열 형식으로 유지)
    def fix_date(x):
        if pd.isna(x) or x == "": return "2025-01-01"
        s = str(x).replace("-", "").replace(".", "").strip()
        if len(s) == 8: return f"{s[:4]}-{s[4:6]}-{s[6:]}"
        elif len(s) == 4: return f"2025-{s[:2]}-{s[2:]}"
        return str(x)

    new_df['날짜'] = new_df['날짜'].apply(fix_date)
    
    # 2. 수치형 변환 및 지표 계산 (정수 및 소수점 강제 지정)
    for col in ['노출수', '클릭수', '비용']:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0).astype(int)

    # CTR, CPC, CPM 계산 (분모가 0인 경우 처리) [cite: 2, 3]
    new_df['CTR(%)'] = (new_df['클릭수'] / new_df['노출수'] * 100).round(2).fillna(0)
    new_df['CPC'] = (new_df['비용'] / new_df['클릭수']).replace([float('inf')], 0).round(0).fillna(0).astype(int)
    new_df['CPM'] = (new_df['비용'] / new_df['노출수'] * 1000).round(0).fillna(0).astype(int)
    
    return new_df

# --- 데이터 저장소 초기화 ---
if 'master_v5' not in st.session_state:
    st.session_state.master_v5 = pd.DataFrame([
        {"날짜": "2025-12-01", "유형": "배너(DA)", "매체": "네이버", "상품명": "GFA", "소재명": "소재 A", 
         "노출수": 1000, "클릭수": 10, "비용": 100000}
    ])

# --- 행 추가 기능 ---
st.subheader("📝 데이터 입력 시트")
if st.button("➕ 7일치 행 추가"):
    try:
        last_date_val = st.session_state.master_v5.iloc[-1]['날짜']
        base_date = datetime.strptime(last_date_val, "%Y-%m-%d")
    except:
        base_date = datetime.now()

    new_rows = []
    for i in range(1, 8):
        new_date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        new_rows.append({"날짜": new_date, "유형": "배너(DA)", "매체": "네이버", "상품명": "", "소재명": "", 
                         "노출수": 0, "클릭수": 0, "비용": 0})
    
    st.session_state.master_v5 = pd.concat([st.session_state.master_v5, pd.DataFrame(new_rows)], ignore_index=True)
    st.rerun()

# --- 메인 데이터 에디터 ---
# 지표가 계산된 데이터 생성 
display_df = clean_and_calculate(st.session_state.master_v5)

# 데이터 에디터 실행
edited_df = st.data_editor(
    display_df,
    num_rows="dynamic",
    use_container_width=True,
    key="editor_v5",
    column_config={
        "날짜": st.column_config.TextColumn("날짜 (YYYY-MM-DD)"),
        "유형": st.column_config.SelectboxColumn("유형", options=["배너(DA)", "영상(Video)"]),
        "매체": st.column_config.SelectboxColumn("매체", options=["네이버", "카카오", "구글", "메타", "유튜브", "인벤", "루리웹"]),
        "노출수": st.column_config.NumberColumn("노출수", format="%d"),
        "클릭수": st.column_config.NumberColumn("클릭수", format="%d"),
        "비용": st.column_config.NumberColumn("비용", format="₩%d"),
        "CTR(%)": st.column_config.NumberColumn("CTR(%)", disabled=True), # 자동 계산 항목 수정 불가 [cite: 6]
        "CPC": st.column_config.NumberColumn("CPC", disabled=True),
        "CPM": st.column_config.NumberColumn("CPM", disabled=True)
    }
)

# 데이터 확정 버튼
if st.button("🚀 분석 데이터로 확정 및 차트 갱신", use_container_width=True):
    save_cols = ["날짜", "유형", "매체", "상품명", "소재명", "노출수", "클릭수", "비용"]
    st.session_state.master_v5 = edited_df[save_cols].copy()
    st.success("데이터가 반영되었습니다!")
    st.rerun()

# --- 시각화 섹션 (final_df 선언 위치 조정) ---
st.divider()
final_df = clean_and_calculate(st.session_state.master_v5) # 변수 선언을 사용 지점보다 위로 배치 

if not final_df.empty:
    # KPI 요약 지표 표시 
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 비용", f"₩{int(final_df['비용'].sum()):,}")
    k2.metric("평균 CTR", f"{final_df['CTR(%)'].mean():.2f}%")
    k3.metric("평균 CPC", f"₩{int(final_df['CPC'].mean()):,}")
    k4.metric("평균 CPM", f"₩{int(final_df['CPM'].mean()):,}")

    # 차트 시각화 [cite: 8]
    c_l, c_r = st.columns([2, 1])
    with c_l:
        m_choice = st.radio("지표 선택", ["CTR(%)", "비용", "클릭수", "CPM"], horizontal=True)
        fig = px.line(final_df.sort_values('날짜'), x="날짜", y=m_choice, color="소재명", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with c_r:
        fig_pie = px.pie(final_df, values='비용', names='소재명', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)