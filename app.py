import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re

# 1. 페이지 설정
st.set_page_config(page_title="In-house Marketing BI", layout="wide")

# --- [데이터 정제 함수] ---
def clean_and_process(df_list, auto_date):
    combined = pd.concat(df_list, ignore_index=True)
    if combined.empty:
        return combined
    
    final_chunks = []
    # 매체, 상품명, 소재명을 그룹으로 묶어 날짜 처리
    for keys, group in combined.groupby(['매체', '상품명', '소재명']):
        group = group.reset_index(drop=True)
        if auto_date and not group.empty:
            start_date = pd.to_datetime(group.loc[0, '날짜'], errors='coerce')
            if pd.notnull(start_date):
                group['날짜'] = [start_date + timedelta(days=i) for i in range(len(group))]
        else:
            group['날짜'] = pd.to_datetime(group['날짜'], errors='coerce')
        final_chunks.append(group)
    
    df = pd.concat(final_chunks, ignore_index=True)
    df = df.dropna(subset=['날짜'])
    
    # 원화 기호, 콤마 제거 및 숫자 변환
    for col in ['노출수', '클릭수', '비용']:
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    df['CTR(%)'] = (df['클릭수'] / df['노출수'] * 100).round(2).fillna(0.0)
    df['ID'] = "[" + df['매체'].astype(str) + "] " + df['상품명'].astype(str) + " (" + df['소재명'].astype(str) + ")"
    return df

# --- [사이드바] ---
with st.sidebar:
    st.header("💾 설정")
    auto_date_mode = st.checkbox("📅 소재별 날짜 자동 채우기", value=True)
    n_iterations = st.select_slider("시뮬레이션 반복", options=[1000, 5000, 10000], value=5000)
    
    st.divider()
    # 파일 업로드 (들여쓰기 오류 방지를 위해 단순화)
    uploaded_file = st.file_uploader("📂 CSV 불러오기", type=["csv"])
    if uploaded_file:
        try:
            up_df = pd.read_csv(uploaded_file)
            up_df['날짜'] = pd.to_datetime(up_df['날짜'], errors='coerce').dt.date
            if st.button("📥 데이터 적용"):
                st.session_state.db = up_df
                st.rerun()
        except Exception as e:
            st.error(f"파일 오류: {e}")

st.title("🎯 데이터 기반 마케팅 분석툴")

# --- [세션 초기화] ---
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame([{
        "날짜": datetime.now().date(), "매체": "네이버", "상품명": "상품", 
        "소재명": "소재", "노출수": "0", "클릭수": "0", "비용": "0"
    }])

# --- [입력부] ---
media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]
tabs = st.tabs(media_list)
all_edited_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        curr_df = st.session_state.db[st.session_state.db['매체'] == m].copy()
        if curr_df.empty:
            curr_df = pd.DataFrame([{"날짜": datetime.now().date(), "매체": m, "상품명": "", "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"}])
        
        curr_df['날짜'] = pd.to_datetime(curr_df['날짜'], errors='coerce').dt.date

        edited = st.data_editor(
            curr_df, num_rows="dynamic", use_container_width=True, key=f"ed_{m}",
            column_config={
                "날짜": st.column_config.DateColumn("시작일"),
                "비용": st.column_config.TextColumn("비용(₩)"),
                "노출수": st.column_config.TextColumn("노출수"),
                "클릭수": st.column_config.TextColumn("클릭수")
            }
        )
        all_edited_data.append(edited)

# --- [실행] ---
if st.button("🚀 데이터 저장 및 분석 실행", use_container_width=True):
    try:
        st.session_state.db = clean_and_process(all_edited_data, auto_date_mode)
        st.success("데이터 업데이트 완료!")
        st.rerun()
    except Exception as e:
        st.error(f"실행 오류: {e}")

# --- [리포트] ---
final_df = st.session_state.db
if not final_df.empty and 'ID' in final_df.columns and len(final_df['ID'].unique()) >= 2:
    st.divider()
    p_list = sorted(final_df['ID'].unique())
    c1, c2 = st.columns(2)
    with c1: i_a = st.selectbox("기준 A", p_list, index=0)
    with c2: i_b = st.selectbox("비교 B", p_list, index=1)
    
    res = final_df.groupby('ID').agg({'클릭수':'sum', '노출수':'sum'})
    a, b = res.loc[i_a], res.loc[i_b]
    s_a = np.random.beta(a['클릭수']+1, a['노출수']-a['클릭수']+1, n_iterations)
    s_b = np.random.beta(b['클릭수']+1, b['노출수']-b['클릭수']+1, n_iterations)
    
    st.metric(f"{i_b} 승리 확률", f"{(s_b > s_a).mean()*100:.1f}%")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s_a, name=i_a, opacity=0.6))
    fig.add_trace(go.Histogram(x=s_b, name=i_b, opacity=0.6))
    fig.update_layout(barmode='overlay', title="CTR 성과 사후 분포 비교")
    st.plotly_chart(fig, use_container_width=True)