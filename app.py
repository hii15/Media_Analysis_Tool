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

    # 1. 날짜 처리 (문자열로 통일하여 에디터 충돌 방지)
    def fix_date(x):
        if pd.isna(x) or x == "": return None
        s = str(x).replace("-", "").replace(".", "").strip()
        if len(s) == 8: return f"{s[:4]}-{s[4:6]}-{s[6:]}"
        elif len(s) == 4: return f"2025-{s[:2]}-{s[2:]}"
        return str(x)

    new_df['날짜'] = new_df['날짜'].apply(fix_date)
    
    # 지표 계산을 위해 잠시 datetime 변환
    calc_date = pd.to_datetime(new_df['날짜'], errors='coerce')

    # 2. 수치형 변환 및 지표 계산
    for col in ['노출수', '클릭수', '비용']:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0).astype(int) [cite: 2]

    new_df['CTR(%)'] = (new_df['클릭수'] / new_df['노출수'] * 100).round(2).fillna(0) [cite: 2]
    new_df['CPC'] = (new_df['비용'] / new_df['클릭수']).replace([float('inf')], 0).round(0).fillna(0).astype(int) [cite: 2]
    new_df['CPM'] = (new_df['비용'] / new_df['노출수'] * 1000).round(0).fillna(0).astype(int) [cite: 3]
    
    return new_df

# --- 데이터 저장소 초기화 ---
if 'master_v5' not in st.session_state:
    st.session_state.master_v5 = pd.DataFrame([
        {"날짜": "2025-12-01", "유형": "배너(DA)", "매체": "네이버", "상품명": "GFA", "소재명": "소재 A", 
         "노출수": 1000, "클릭수": 10, "비용": 100000}
    ])

# --- 편의 기능: 행 추가 도구 ---
st.subheader("📝 데이터 입력 시트")
c1, c2 = st.columns([1, 4])

with c1:
    if st.button("➕ 7일치 행 추가"):
        try:
            # 마지막 날짜 기준 7일 추가
            last_date_val = st.session_state.master_v5.iloc[-1]['날짜']
            base_date = pd.to_datetime(last_date_val)
        except:
            base_date = datetime.now()

        new_rows = []
        for i in range(1, 8):
            new_date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            new_rows.append({"날짜": new_date, "유형": "배너(DA)", "매체": "네이버", "상품명": "", "소재명": "", 
                             "노출수": 0, "클릭수": 0, "비용": 0})
        
        st.session_state.master_v5 = pd.concat([st.session_state.master_v5, pd.DataFrame(new_rows)], ignore_index=True)
        st.rerun()

# --- 메인 시트 (st.data_editor) ---
# 가공된 데이터 준비
display_df = clean_and_calculate(st.session_state.master_v5)

# 에러 방지를 위해 데이터 타입을 명확히 지정하여 전달
edited_df = st.data_editor(
    display_df,
    num_rows="dynamic",
    use_container_width=True,
    key="editor_v5",
    column_config={
        "날짜": st.column_config.TextColumn("날짜 (YYYY-MM-DD)", help="날짜 형식을 맞춰주세요."),
        "유형": st.column_config.SelectboxColumn("유형", options=["배너(DA)", "영상(Video)"]), [cite: 5]
        "매체": st.column_config.SelectboxColumn("매체", options=["네이버", "카카오", "구글", "메타", "유튜브", "인벤", "루리웹"]), [cite: 5]
        "노출수": st.column_config.NumberColumn("노출수", format="%d"), [cite: 5]
        "클릭수": st.column_config.NumberColumn("클릭수", format="%d"), [cite: 5]
        "비용": st.column_config.NumberColumn("비용", format="₩%d"), [cite: 5]
        "CTR(%)": st.column_config.NumberColumn("CTR(%)", disabled=True), [cite: 6]
        "CPC": st.column_config.NumberColumn("CPC", disabled=True), [cite: 6]
        "CPM": st.column_config.NumberColumn("CPM", disabled=True) [cite: 6]
    }
)

if st.button("🚀 분석 데이터로 확정 및 차트 갱신", use_container_width=True):
    # 저장할 때는 원본 컬럼만 추출
    save_cols = ["날짜", "유형", "매체", "상품명", "소재명", "노출수", "클릭수", "비용"]
    st.session_state.master_v5 = edited_df[save_cols].copy()
    st.success("데이터가 반영되었습니다!")
    st.rerun()

# --- 시각화 섹션 ---
final_df = clean_and_calculate(st.session_state.master_v5)
# 시각화를 위해 날짜 타입 변환
final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')

if not final_df.empty and final_df['날짜'].notnull().any(): [cite: 7]
    st.divider()
    
    # KPI 요약
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 비용", f"₩{int(final_df['비용'].sum()):,}") [cite: 7]
    k2.metric("평균 CTR", f"{final_df['CTR(%)'].mean():.2f}%") [cite: 7]
    k3.metric("평균 CPC", f"₩{int(final_df['CPC'].mean()):,}") [cite: 7]
    k4.metric("평균 CPM", f"₩{int(final_df['CPM'].mean()):,}") [cite: 7]

    # 차트 영역
    c_l, c_r = st.columns([2, 1])
    with c_l:
        m_choice = st.radio("지표 선택", ["CTR(%)", "비용", "클릭수", "CPM"], horizontal=True) [cite: 7]
        fig = px.line(final_df.sort_values('날짜'), x="날짜", y=m_choice, color="소재명", markers=True, template="plotly_white") [cite: 7, 8]
        st.plotly_chart(fig, use_container_width=True)

    with c_r:
        fig_pie = px.pie(final_df, values='비용', names='소재명', hole=0.4) [cite: 8]
        st.plotly_chart(fig_pie, use_container_width=True)