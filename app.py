import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re

# 1. 페이지 설정
st.set_page_config(page_title="In-house Marketing BI", layout="wide")

# --- [사이드바] 설정 및 데이터 관리 ---
with st.sidebar:
    st.header("💾 데이터 및 분석 설정")
    # 소재별 자동 채우기 모드 (매체+상품+소재 기준)
    auto_date_mode = st.checkbox("📅 소재별 날짜 자동 채우기", value=True)
    
    st.divider()
    
    # 1. 파일 업로드 로직 (정확한 들여쓰기 적용)
    uploaded_file = st.file_uploader("📂 저장된 CSV 불러오기", type=["csv"])
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            input_df['날짜'] = pd.to_datetime(input_df['날짜'], errors='coerce').dt.date
            
            required_cols = ["날짜", "매체", "상품명", "소재명", "노출수", "클릭수", "비용"]
            if all(col in input_df.columns for col in required_cols):
                if st.button("📥 데이터 적용하기"):
                    st.session_state.db = input_df
                    st.success("데이터를 불러왔습니다.")
                    st.rerun()
            else:
                st.error("CSV 형식이 일치하지 않습니다.")
        except Exception as e:
            st.error(f"로드 중 오류: {e}")

    st.divider()
    n_iterations = st.select_slider("시뮬레이션 반복 횟수", options=[1000, 5000, 10000], value=5000)

st.title("🎯 데이터 기반 마케팅 분석툴")

# --- [유틸리티] 강력한 데이터 정제 함수 ---
def clean_and_process(df_list, auto_date):
    combined = pd.concat(df_list, ignore_index=True)
    if combined.empty:
        return combined
    
    final_chunks = []
    # 매체, 상품명, 소재명을 그룹으로 묶어 날짜 처리
    for keys, group in combined.groupby(['매체', '상품명', '소재명']):
        group = group.reset_index(drop=True)
        
        # 날짜 자동 완성
        if auto_date and not group.empty:
            start_date = pd.to_datetime(group.loc[0, '날짜'], errors='coerce')
            if pd.notnull(start_date):
                group['날짜'] = [start_date + timedelta(days=i) for i in range(len(group))]
        else:
            group['날짜'] = pd.to_datetime(group['날짜'], errors='coerce')
        
        final_chunks.append(group)
    
    df = pd.concat(final_chunks, ignore_index=True)
    df = df.dropna(subset=['날짜'])
    
    # [핵심] 원화 기호, 콤마 등 모든 특수문자 제거 후 숫자로 변환
    for col in ['노출수', '클릭수', '비용']:
        # 숫자와 마침표(.)를 제외한 모든 문자 제거 로직
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    df['CTR(%)'] = (df['클릭수'] / df['노출수'] * 100).round(2).fillna(0.0)
    df['ID'] = "[" + df['매체'].astype(str) + "] " + df['상품명'].astype(str) + "_" + df['소재명'].astype(str)
    return df

# --- [분석] 베이지안 연산 ---
def run_analysis(df, item_a, item_b, iterations):
    res = df.groupby('ID').agg({'클릭수':'sum', '노출수':'sum'})
    a, b = res.loc[item_a], res.loc[item_b]
    samples_a = np.random.beta(max(a['클릭수'], 0)+1, max(a['노출수']-a['클릭수'], 0)+1, iterations)
    samples_b = np.random.beta(max(b['클릭수'], 0)+1, max(b['노출수']-b['클릭수'], 0)+1, iterations)
    return (samples_a > samples_b).mean(), samples_a, samples_b

# --- [데이터] 세션 관리 및 입력부 ---
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame([{
        "날짜": datetime.now().date(), "매체": "네이버", "상품명": "", 
        "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"
    }])

media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]
tabs = st.tabs(media_list)
all_edited_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        curr_df = st.session_state.db[st.session_state.db['매체'] == m].copy()
        if curr_df.empty:
            curr_df = pd.DataFrame([{"날짜": datetime.now().date(), "매체": m, "상품명": "", "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"}])
        
        # 엑셀 복붙 시 충돌 방지를 위해 모두 텍스트 기반으로 처리
        curr_df['날짜'] = pd.to_datetime(curr_df['날짜'], errors='coerce').dt.date

        edited = st.data_editor(
            curr_df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"editor_tab_{m}",
            column_config={
                "날짜": st.column_config.DateColumn("시작일(소재단위 첫줄)"),
                "비용": st.column_config.TextColumn("비용(₩)"),
                "노출수": st.column_config.TextColumn("노출수"),
                "클릭수": st.column_config.TextColumn("클릭수")
            }
        )
        all_edited_data.append(edited)

# --- [실행 버튼] ---
if st.button("🚀 데이터 저장 및 소재별 분석 실행", use_container_width=True):
    try:
        # 데이터 정제 및 날짜 채우기 수행
        st.session_state.db = clean_and_process(all_edited_data, auto_date_mode)
        st.success("데이터가 성공적으로 업데이트되었습니다!")
        st.rerun()
    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")

# --- [리포트] 시각화 ---
final_df = st.session_state.db
if not final_df.empty and 'ID' in final_df.columns and len(final_df['ID'].unique()) >= 2:
    st.divider()
    p_list = sorted(final_df['ID'].unique())
    c1, c2 = st.columns(2)
    with c1: 
        item_a = st.selectbox("기준 상품(A)", p_list, index=0)
    with c2: 
        item_b = st.selectbox("비교 대상(B)", p_list, index=1)
    
    try:
        prob, s_a, s_b = run_analysis(final_df, item_a, item_b, n_iterations)
        
        m1, m2 = st.columns(2)
        m1.metric(f"{item_b} 승리 확률", f"{prob*100:.1f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=s_a, name=item_a, opacity=0.6))
        fig.add_trace(go.Histogram(x=s_b, name=item_b, opacity=0.6))
        fig.update_layout(barmode='overlay', title="소재별 CTR 성과 분포 비교")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"데이터가 부족하여 분석을 시각화할 수 없습니다. (에러: {e})")