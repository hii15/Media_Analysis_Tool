import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="In-house 광고 상품별 성과 분석", layout="wide")
st.title("🎮 광고주용 매체/상품별 통합 성과 시스템")

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
if 'master_v7' not in st.session_state:
    st.session_state.master_v7 = pd.DataFrame([
        {"날짜": "2025-12-01", "매체": "네이버", "상품명": "GFA(뉴스)", "소재명": "소재 A", "노출수": 1000, "클릭수": 10, "비용": 100000},
        {"날짜": "2025-12-01", "매체": "네이버", "상품명": "웹툰빅배너", "소재명": "소재 B", "노출수": 2000, "클릭수": 30, "비용": 500000}
    ])

# --- 입력 섹션: 매체 탭 ---
st.subheader("📝 매체/상품별 데이터 입력")
media_list = ["네이버", "카카오", "구글", "메타", "유튜브"]
tabs = st.tabs(media_list)
updated_data_frames = []

for i, media in enumerate(media_list):
    with tabs[i]:
        media_df = st.session_state.master_v7[st.session_state.master_v7['매체'] == media].copy()
        
        if media_df.empty:
            media_df = pd.DataFrame([{"날짜": datetime.now().strftime("%Y-%m-%d"), "매체": media, "상품명": "", "소재명": "", "노출수": 0, "클릭수": 0, "비용": 0}])
        
        media_df['날짜'] = media_df['날짜'].astype(str)
        
        # 에디터에서 상품명을 명확히 입력하도록 설정
        edited_media_df = st.data_editor(
            media_df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"editor_v7_{media}",
            column_config={
                "매체": st.column_config.TextColumn("매체", disabled=True),
                "상품명": st.column_config.TextColumn("상품명 (예: GFA, 웹툰)", help="광고 지면이나 상품명을 구분해서 적어주세요.")
            }
        )
        updated_data_frames.append(edited_media_df)

if st.button("🚀 전체 데이터 저장 및 통합 분석", use_container_width=True):
    st.session_state.master_v7 = pd.concat(updated_data_frames, ignore_index=True)
    st.rerun()

# --- 분석 섹션 ---
final_df = clean_and_calculate(st.session_state.master_v7)
final_df['날짜'] = pd.to_datetime(final_df['날짜'])

if not final_df.empty:
    st.divider()
    
    # 분석 기준 선택 (매체별로 볼지, 상품별로 볼지)
    st.subheader("📊 성과 심층 분석")
    view_option = st.radio("분석 기준", ["매체별", "상품별"], horizontal=True)
    color_target = "매체" if view_option == "매체별" else "상품명"
    
    m_choice = st.selectbox("지표 선택", ["CTR(%)", "비용", "클릭수", "CPC"])
    
    # 트렌드 차트
    fig = px.line(final_df.sort_values('날짜'), x="날짜", y=m_choice, color=color_target, 
                  hover_data=["상품명", "소재명"], markers=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # 상품별 효율성 (TreeMap) - 어떤 상품에 돈을 많이 썼고 효율은 어떤지 시각화
    st.write("🎯 매체/상품별 비용 비중 및 성과")
    fig_tree = px.treemap(final_df, path=['매체', '상품명'], values='비용', 
                          color=m_choice, color_continuous_scale='RdYlGn' if m_choice == 'CTR(%)' else 'RdBu_r')
    st.plotly_chart(fig_tree, use_container_width=True)

    # 상관관계 및 안정성
    c1, c2 = st.columns(2)
    with c1:
        st.write("🔗 지표 상관관계")
        st.plotly_chart(px.imshow(final_df[['노출수', '클릭수', '비용', 'CTR(%)', 'CPC']].corr(), text_auto=True), use_container_width=True)
    with c2:
        st.write("📉 상품별 변동 리스크 (CV)")
        vol = final_df.groupby([color_target])['CTR(%)'].agg(['mean', 'std']).reset_index()
        vol['CV(%)'] = (vol['std'] / vol['mean'] * 100).round(1).fillna(0)
        st.dataframe(vol.sort_values('CV(%)'), use_container_width=True)