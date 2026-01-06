import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="In-house 통합 성과 관리 시스템", layout="wide")
st.title("🎮 광고주용 매체/소재별 통합 성과 대시보드")

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

    new_df['날짜'] = new_df['날짜'].apply(fix_date)
    for col in ['노출수', '클릭수', '비용']:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0).astype(int)
    
    new_df['CTR(%)'] = (new_df['클릭수'] / new_df['노출수'] * 100).round(2).fillna(0.0)
    new_df['CPC'] = (new_df['비용'] / new_df['클릭수']).replace([float('inf')], 0).round(0).fillna(0).astype(int)
    return new_df

# --- 데이터 초기화 ---
if 'master_v6' not in st.session_state:
    st.session_state.master_v6 = pd.DataFrame([
        {"날짜": "2025-12-01", "매체": "네이버", "소재명": "소재 A", "노출수": 1000, "클릭수": 10, "비용": 100000},
        {"날짜": "2025-12-01", "매체": "카카오", "소재명": "소재 B", "노출수": 2000, "클릭수": 30, "비용": 150000}
    ])

# --- UX 개선: 매체별 탭 분리 입력 ---
st.subheader("📝 매체별 데이터 관리")
media_list = ["네이버", "카카오", "구글", "메타", "유튜브"]
tabs = st.tabs(media_list)

# 모든 탭의 편집 결과물을 담을 딕셔너리
updated_data_frames = []

for i, media in enumerate(media_list):
    with tabs[i]:
        # 해당 매체 데이터만 필터링
        media_df = st.session_state.master_v6[st.session_state.master_v6['매체'] == media].copy()
        
        # 만약 해당 매체 데이터가 없으면 빈 행 생성 양식 제공
        if media_df.empty:
            media_df = pd.DataFrame([{"날짜": datetime.now().strftime("%Y-%m-%d"), "매체": media, "소재명": "", "노출수": 0, "클릭수": 0, "비용": 0}])
        
        # 탭별 개별 에디터 (날짜는 문자열로 변환하여 에러 방지)
        media_df['날짜'] = media_df['날짜'].astype(str)
        edited_media_df = st.data_editor(
            media_df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"editor_{media}",
            column_config={"매체": st.column_config.TextColumn("매체", disabled=True)}
        )
        updated_data_frames.append(edited_media_df)

# --- 데이터 통합 저장 ---
if st.button("🚀 모든 매체 데이터 통합 저장 및 분석 갱신", use_container_width=True):
    new_master = pd.concat(updated_data_frames, ignore_index=True)
    st.session_state.master_v6 = new_master
    st.success("전체 매체 데이터가 통합되었습니다!")
    st.rerun()

# --- 통합 시각화 및 통계 분석 ---
final_df = clean_and_calculate(st.session_state.master_v6)
final_df['날짜'] = pd.to_datetime(final_df['날짜'])

if not final_df.empty:
    st.divider()
    st.subheader("📊 통합 분석 리포트")
    
    # 1. 통합 차트 (모든 매체 수치가 섞여서 나옴)
    m_choice = st.selectbox("조회 지표 선택", ["CTR(%)", "비용", "클릭수", "CPC"])
    fig = px.line(final_df.sort_values('날짜'), x="날짜", y=m_choice, color="매체", symbol="소재명", 
                  markers=True, title=f"전체 매체별 {m_choice} 트렌드")
    st.plotly_chart(fig, use_container_width=True)

    # 2. 통계 분석 섹션 (상관관계 & 변동성)
    col1, col2 = st.columns(2)
    with col1:
        st.write("🔗 지표 간 상관관계")
        corr_df = final_df[['노출수', '클릭수', '비용', 'CTR(%)', 'CPC']].corr()
        st.plotly_chart(px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)
    
    with col2:
        st.write("📉 소재별 안정성 점수 (CV)")
        vol = final_df.groupby(['매체', '소재명'])['CTR(%)'].agg(['mean', 'std']).reset_index()
        vol['CV(%)'] = (vol['std'] / vol['mean'] * 100).round(1).fillna(0)
        st.dataframe(vol[['매체', '소재명', 'mean', 'CV(%)']].sort_values('CV(%)'), use_container_width=True)