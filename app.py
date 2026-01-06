import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="AE 통합 성과 대시보드 PRO", layout="wide")

st.title("🎯 소재별 통합 성과 대시보드")

# --- 데이터 처리 유틸리티 ---
def clean_and_calculate(df):
    if df.empty: return df
    new_df = df.copy()

    # 날짜 보정: 일단 문자열로 모두 변환하여 에러 방지 [cite: 2]
    def fix_date(x):
        if pd.isna(x) or x == "": return "2025-01-01"
        s = str(x).replace("-", "").replace(".", "").strip()
        if len(s) == 8: return f"{s[:4]}-{s[4:6]}-{s[6:]}"
        elif len(s) == 4: return f"2025-{s[:2]}-{s[2:]}"
        return str(x)

    new_df['날짜'] = new_df['날짜'].apply(fix_date)
    
    # 수치형 변환 [cite: 2]
    for col in ['노출수', '클릭수', '비용']:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0).astype(int)

    # 지표 계산 [cite: 2, 3]
    new_df['CTR(%)'] = (new_df['클릭수'] / new_df['노출수'] * 100).round(2).fillna(0.0)
    new_df['CPC'] = (new_df['비용'] / new_df['클릭수']).replace([float('inf')], 0).round(0).fillna(0).astype(int)
    new_df['CPM'] = (new_df['비용'] / new_df['노출수'] * 1000).round(0).fillna(0).astype(int)
    
    return new_df

# --- 데이터 저장소 초기화 ---
if 'master_v5' not in st.session_state:
    st.session_state.master_v5 = pd.DataFrame([
        {"날짜": "2025-12-01", "유형": "배너(DA)", "매체": "네이버", "상품명": "GFA", "소재명": "소재 A", 
         "노출수": 1000, "클릭수": 10, "비용": 100000}
    ])

# --- 행 추가 도구 ---
st.subheader("📝 데이터 입력 시트")
if st.button("➕ 7일치 행 추가"):
    try:
        last_date_str = str(st.session_state.master_v5.iloc[-1]['날짜'])
        base_date = pd.to_datetime(last_date_str)
    except:
        base_date = datetime.now()

    new_rows = []
    for i in range(1, 8):
        new_date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        new_rows.append({"날짜": new_date, "유형": "배너(DA)", "매체": "네이버", "상품명": "", "소재명": "", 
                         "노출수": 0, "클릭수": 0, "비용": 0})
    
    st.session_state.master_v5 = pd.concat([st.session_state.master_v5, pd.DataFrame(new_rows)], ignore_index=True)
    st.rerun()

# --- [중요] 에러 해결 포인트: 데이터 에디터 전달용 데이터 가공 ---
# display_df를 만들 때 타입을 완전히 고정합니다.
display_df = clean_and_calculate(st.session_state.master_v5)

# Streamlit 에디터의 타입 충돌을 막기 위해 강제 형변환 
display_df['날짜'] = display_df['날짜'].astype(str)
display_df['CTR(%)'] = display_df['CTR(%)'].astype(float)

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
        "CTR(%)": st.column_config.NumberColumn("CTR(%)", disabled=True, format="%.2f%%"),
        "CPC": st.column_config.NumberColumn("CPC", disabled=True, format="₩%d"),
        "CPM": st.column_config.NumberColumn("CPM", disabled=True, format="₩%d")
    }
)

if st.button("🚀 분석 데이터로 확정 및 차트 갱신", use_container_width=True):
    save_cols = ["날짜", "유형", "매체", "상품명", "소재명", "노출수", "클릭수", "비용"]
    st.session_state.master_v5 = edited_df[save_cols].copy()
    st.success("데이터가 반영되었습니다!")
    st.rerun()

# --- 시각화 섹션 ---
final_df = clean_and_calculate(st.session_state.master_v5)
final_df['날짜'] = pd.to_datetime(final_df['날짜']) # 시각화 시에는 다시 날짜형으로 [cite: 7]

if not final_df.empty:
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 비용", f"₩{int(final_df['비용'].sum()):,}")
    k2.metric("평균 CTR", f"{final_df['CTR(%)'].mean():.2f}%")
    k3.metric("평균 CPC", f"₩{int(final_df['CPC'].mean()):,}")
    k4.metric("평균 CPM", f"₩{int(final_df['CPM'].mean()):,}")

    m_choice = st.radio("지표 선택", ["CTR(%)", "비용", "클릭수", "CPM"], horizontal=True)
    fig = px.line(final_df.sort_values('날짜'), x="날짜", y=m_choice, color="소재명", markers=True)
    st.plotly_chart(fig, use_container_width=True)