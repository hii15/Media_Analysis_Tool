import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="In-house 마케팅 성과 및 리스크 분석", layout="wide")
st.title("🎮 광고주 내부 데이터 기반 성과 분석 시스템")
st.caption("매체 성과와 내부 데이터 연동을 위한 통계적 검증 도구")

# --- 데이터 처리 유틸리티 ---
def clean_and_calculate(df):
    if df.empty: return df
    new_df = df.copy()

    def fix_date(x):
        if pd.isna(x) or x == "": return "2025-01-01"
        s = str(x).replace("-", "").replace(".", "").strip()
        if len(s) == 8: return f"{s[:4]}-{s[4:6]}-{s[6:]}"
        elif len(s) == 4: return f"2025-{s[:2]}-{s[2:]}"
        return str(x)

    new_df['날짜'] = new_df['날짜'].apply(fix_date) # [cite: 2]
    
    for col in ['노출수', '클릭수', '비용']:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0).astype(int) # [cite: 3]

    new_df['CTR(%)'] = (new_df['클릭수'] / new_df['노출수'] * 100).round(2).fillna(0.0)
    new_df['CPC'] = (new_df['비용'] / new_df['클릭수']).replace([float('inf')], 0).round(0).fillna(0).astype(int)
    new_df['CPM'] = (new_df['비용'] / new_df['노출수'] * 1000).round(0).fillna(0).astype(int)
    
    return new_df

# --- 데이터 저장소 ---
if 'master_v5' not in st.session_state:
    st.session_state.master_v5 = pd.DataFrame([
        {"날짜": "2025-12-01", "유형": "배너(DA)", "매체": "네이버", "상품명": "GFA", "소재명": "소재 A", 
         "노출수": 1000, "클릭수": 10, "비용": 100000}
    ]) # [cite: 3]

# --- 행 추가 기능 ---
st.subheader("📝 캠페인 데이터 입력")
if st.button("➕ 7일치 행 추가"):
    try:
        last_date_val = st.session_state.master_v5.iloc[-1]['날짜']
        base_date = pd.to_datetime(last_date_val)
    except:
        base_date = datetime.now()
    
    new_rows = []
    for i in range(1, 8):
        new_date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d") # [cite: 4]
        new_rows.append({"날짜": new_date, "유형": "배너(DA)", "매체": "네이버", "상품명": "", "소재명": "", "노출수": 0, "클릭수": 0, "비용": 0})
    st.session_state.master_v5 = pd.concat([st.session_state.master_v5, pd.DataFrame(new_rows)], ignore_index=True)
    st.rerun()

# --- 데이터 에디터 섹션 ---
display_df = clean_and_calculate(st.session_state.master_v5)
display_df['날짜'] = display_df['날짜'].astype(str)

edited_df = st.data_editor(
    display_df,
    num_rows="dynamic",
    use_container_width=True,
    key="editor_v5",
    column_config={
        "날짜": st.column_config.TextColumn("날짜 (YYYY-MM-DD)"),
        "비용": st.column_config.NumberColumn("비용", format="₩%d"), # [cite: 6]
        "CTR(%)": st.column_config.NumberColumn("CTR(%)", disabled=True, format="%.2f%%"),
        "CPC": st.column_config.NumberColumn("CPC", disabled=True, format="₩%d")
    }
)

if st.button("🚀 분석 데이터 확정 및 통계 갱신", use_container_width=True):
    save_cols = ["날짜", "유형", "매체", "상품명", "소재명", "노출수", "클릭수", "비용"]
    st.session_state.master_v5 = edited_df[save_cols].copy() # [cite: 6]
    st.rerun()

# --- 시각화 및 통계 분석 섹션 ---
final_df = clean_and_calculate(st.session_state.master_v5) # [cite: 7]
final_df['날짜'] = pd.to_datetime(final_df['날짜'])

if not final_df.empty:
    st.divider()
    
    # 1. KPI 지표 [cite: 7]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 집행 비용", f"₩{int(final_df['비용'].sum()):,}")
    k2.metric("평균 CTR", f"{final_df['CTR(%)'].mean():.2f}%")
    k3.metric("평균 CPC", f"₩{int(final_df['CPC'].mean()):,}")
    k4.metric("평균 CPM", f"₩{int(final_df['CPM'].mean()):,}")

    # 2. 전문 통계 분석 탭
    st.subheader("📊 전문 통계 분석 및 리스크 평가")
    t_corr, t_vol = st.tabs(["🔗 지표 간 상관관계", "📉 성과 안정성(CV) 리스크"])

    with t_corr:
        # 상관관계 분석: 어떤 지표가 서로 영향을 주는가?
        corr_df = final_df[['노출수', '클릭수', '비용', 'CTR(%)', 'CPC', 'CPM']].corr()
        fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("💡 **In-house 분석 가이드:** CPC와 CTR의 강한 음의 상관관계가 깨진다면, 매체의 피로도가 높거나 타겟팅 최적화가 필요한 시점입니다.")

    with t_vol:
        # 변동성 분석: 성과가 얼마나 예측 가능한가?
        vol_analysis = final_df.groupby('소재명')['CTR(%)'].agg(['mean', 'std']).reset_index()
        vol_analysis['변동계수(CV)'] = (vol_analysis['std'] / vol_analysis['mean'] * 100).round(2).fillna(0)
        
        def get_risk(cv):
            if cv < 20: return "🟢 안정 (확정적 성과)"
            if cv < 50: return "🟡 보통 (주의 관찰)"
            return "🔴 불안정 (리스크 높음)"
        
        vol_analysis['운영 상태'] = vol_analysis['변동계수(CV)'].apply(get_risk)
        st.dataframe(vol_analysis.rename(columns={'mean': '평균 CTR(%)', 'std': '표준편차'}), use_container_width=True)

    # 3. 기본 트렌드 차트 
    st.subheader("📈 매체별 성과 트렌드")
    m_choice = st.selectbox("조회 지표 선택", ["CTR(%)", "비용", "클릭수", "CPC"])
    fig = px.line(final_df.sort_values('날짜'), x="날짜", y=m_choice, color="소재명", markers=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)