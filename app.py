import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re

# 1. 페이지 설정
st.set_page_config(page_title="Marketing Intelligence Tool", layout="wide")

# --- [핵심 엔진: 데이터 정제] ---
def process_marketing_data(df_list, auto_date):
    if not df_list: return pd.DataFrame()
    combined = pd.concat(df_list, ignore_index=True)
    
    # 기초 정제: 상품명이 없으면 제외
    combined = combined[combined['상품명'].astype(str).str.strip() != ""]
    if combined.empty: return combined
    
    processed_chunks = []
    for _, group in combined.groupby(['매체', '상품명', '소재명']):
        group = group.reset_index(drop=True)
        
        # 날짜 자동 생성 (20251113 등 모든 형식 대응)
        if auto_date and not group.empty:
            raw_date = str(group.loc[0, '날짜']).strip()
            if len(raw_date) == 8 and raw_date.isdigit():
                raw_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            
            clean_date = raw_date.replace('.', '-').replace('/', '-')
            start_dt = pd.to_datetime(clean_date, errors='coerce')
            
            if pd.notnull(start_dt):
                group['날짜'] = [(start_dt + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(group))]
        
        processed_chunks.append(group)
    
    df = pd.concat(processed_chunks, ignore_index=True)
    
    # 숫자 정제: 콤마, 공백, 특수문자 싹 제거 후 숫자로 변환
    for col in ['노출수', '클릭수', '비용']:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\d]', '', x))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    df['클릭수'] = df[['노출수', '클릭수']].min(axis=1)
    df['CTR(%)'] = np.where(df['노출수'] > 0, (df['클릭수'] / df['노출수'] * 100), 0.0)
    df['ID'] = "[" + df['매체'].astype(str) + "] " + df['상품명'].astype(str) + "_" + df['소재명'].astype(str)
    
    return df

# --- [사이드바] ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    auto_date_mode = st.checkbox("📅 소재별 날짜 자동 생성", value=True)
    n_sim = st.select_slider("🎲 시뮬레이션 정밀도", options=[1000, 5000, 10000], value=5000)

st.title("🎯 통합 마케팅 성과 분석 & 시뮬레이터")

# --- [데이터 관리 및 초기화] ---
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame()

media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]

# --- [입력 섹션] ---
tabs = st.tabs(media_list)
all_editor_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        # 현재 데이터 필터링
        curr = pd.DataFrame()
        if not st.session_state.db.empty:
            curr = st.session_state.db[st.session_state.db['매체'] == m].copy()
        
        if curr.empty:
            curr = pd.DataFrame([{"날짜": datetime.now().strftime("%Y-%m-%d"), "매체": m, "상품명": "", "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"}])
        
        # ⚠️ 핵심: 모든 컬럼의 데이터 타입을 '문자열'로 취급하여 붙여넣기 충돌 방지
        curr = curr.astype(str) 

        edited = st.data_editor(
            curr, 
            num_rows="dynamic", 
            use_container_width=True, 
            key=f"editor_final_{m}", # 키 고정 (충돌 최소화)
            column_config={
                "날짜": st.column_config.TextColumn("날짜"),
                "매체": st.column_config.TextColumn("매체", disabled=True),
                "상품명": st.column_config.TextColumn("상품명"),
                "소재명": st.column_config.TextColumn("소재명"),
                "노출수": st.column_config.TextColumn("노출수"),
                "클릭수": st.column_config.TextColumn("클릭수"),
                "비용": st.column_config.TextColumn("비용")
            }
        )
        all_editor_data.append(edited)

# --- [실행 버튼] ---
col1, col2 = st.columns([4, 1])
with col1:
    if st.button("🚀 데이터 업데이트 및 분석 실행", use_container_width=True):
        try:
            processed = process_marketing_data(all_editor_data, auto_date_mode)
            if not processed.empty:
                st.session_state.db = processed
                st.success("데이터가 성공적으로 업데이트되었습니다!")
                st.rerun()
            else:
                st.warning("분석할 유효한 데이터가 없습니다.")
        except Exception as e:
            st.error(f"오류 발생: {e}")

with col2:
    if st.button("♻️ 전체 초기화", use_container_width=True):
        st.session_state.db = pd.DataFrame()
        st.rerun()

# --- [결과 리포트] ---
df = st.session_state.db
if not df.empty and 'ID' in df.columns:
    ids = sorted(df['ID'].unique())
    if len(ids) >= 2:
        st.divider()
        c1, c2 = st.columns(2)
        with c1: item_a = st.selectbox("기준 소재 (A)", ids, index=0)
        with c2: item_b = st.selectbox("비교 소재 (B)", ids, index=1)

        res = df.groupby('ID').agg({'노출수':'sum', '클릭수':'sum'})
        a, b = res.loc[item_a], res.loc[item_b]

        # 베이지안 시뮬레이션
        s_a = np.random.beta(a['클릭수']+1, max(1, a['노출수']-a['클릭수']+1), n_sim)
        s_b = np.random.beta(b['클릭수']+1, max(1, b['노출수']-b['클릭수']+1), n_sim)
        prob_b_win = (s_b > s_a).mean()
        
        st.subheader("📊 분석 결과")
        m1, m2 = st.columns(2)
        m1.metric(f"{item_b} 승리 확률", f"{prob_b_win*100:.1f}%")
        m2.metric("성과 차이", "확실함" if prob_b_win > 0.95 or prob_b_win < 0.05 else "데이터 부족")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=s_a, name=item_a, opacity=0.6))
        fig.add_trace(go.Histogram(x=s_b, name=item_b, opacity=0.6))
        fig.update_layout(barmode='overlay', title="CTR 사후 분포 비교