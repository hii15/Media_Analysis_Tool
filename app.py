import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re

# 1. 페이지 설정
st.set_page_config(page_title="Marketing Intelligence Tool", layout="wide")

# --- [핵심 엔진: 데이터 정제 및 날짜 자동화] ---
def process_marketing_data(df_list, auto_date):
    if not df_list: return pd.DataFrame()
    combined = pd.concat(df_list, ignore_index=True)
    
    # 1. 기초 정제: 상품명이 비어있는 행은 제외
    combined['상품명'] = combined['상품명'].fillna('')
    combined = combined[combined['상품명'].astype(str).str.strip() != ""]
    if combined.empty: return combined
    
    processed_chunks = []
    for _, group in combined.groupby(['매체', '상품명', '소재명']):
        group = group.reset_index(drop=True)
        
        # 2. 날짜 유연 정제 (20251113 등 대응)
        if auto_date and not group.empty:
            raw_val = str(group.loc[0, '날짜']).strip()
            if len(raw_val) == 8 and raw_val.isdigit():
                raw_val = f"{raw_val[:4]}-{raw_val[4:6]}-{raw_val[6:]}"
            
            clean_date = raw_val.replace('.', '-').replace('/', '-')
            start_dt = pd.to_datetime(clean_date, errors='coerce')
            
            if pd.notnull(start_dt):
                group['날짜'] = [(start_dt + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(group))]
            else:
                group['날짜'] = datetime.now().strftime('%Y-%m-%d')
        
        processed_chunks.append(group)
    
    df = pd.concat(processed_chunks, ignore_index=True)
    
    # 3. 숫자 정밀 정제 (콤마, ₩ 등 제거)
    for col in ['노출수', '클릭수', '비용']:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\d]', '', x))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 4. 논리 오류 보정 및 CTR 계산
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

# --- [데이터 세션 관리] ---
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame()

media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]

# --- [데이터 입력 섹션] ---
tabs = st.tabs(media_list)
all_editor_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        curr = pd.DataFrame()
        if not st.session_state.db.empty:
            curr = st.session_state.db[st.session_state.db['매체'] == m].copy()
        
        if curr.empty:
            curr = pd.DataFrame([{
                "날짜": datetime.now().strftime("%Y-%m-%d"), 
                "매체": m, "상품명": "", "소재명": "", 
                "노출수": "0", "클릭수": "0", "비용": "0"
            }])
        
        # ⚠️ 모든 데이터를 문자열로 처리하여 타입 충돌(Error 1, 2) 완전 차단
        curr = curr.astype(str)

        edited = st.data_editor(
            curr, 
            num_rows="dynamic", 
            use_container_width=True, 
            key=f"editor_v9_{m}",
            column_config={
                "날짜": st.column_config.TextColumn("날짜", help="예: 20251113"),
                "매체": st.column_config.TextColumn("매체", disabled=True),
                "비용": st.column_config.TextColumn("비용(₩)")
            }
        )
        all_editor_data.append(edited)

# --- [버튼 섹션] ---
c_btn1, c_btn2 = st.columns([4, 1])
with c_btn1:
    if st.button("🚀 데이터 업데이트 및 분석 실행", use_container_width=True):
        try:
            processed = process_marketing_data(all_editor_data, auto_date_mode)
            if not processed.empty:
                st.session_state.db = processed
                st.success("데이터 업데이트 완료!")
                st.rerun()
            else:
                st.warning("분석할 유효한 데이터를 입력해주세요.")
        except Exception as e:
            st.error(f"오류 발생: {e}")

with c_btn2:
    if st.button("♻️ 초기화", use_container_width=True):
        st.session_state.db = pd.DataFrame()
        st.rerun()

# --- [분석 리포트] ---
df = st.session_state.db
if not df.empty and 'ID' in df.columns:
    ids = sorted(df['ID'].unique())
    if len(ids) >= 2:
        st.divider()
        st.subheader("📊 소재별 성과 시뮬레이션")
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1: item_a = st.selectbox("기준 소재 (A)", ids, index=0)
        with col_sel2: item_b = st.selectbox("비교 소재 (B)", ids, index=1)

        res = df.groupby('ID').agg({'노출수':'sum', '클릭수':'sum'})
        a, b = res.loc[item_a], res.loc[item_b]

        # 몬테카를로 시뮬레이션
        s_a = np.random.beta(a['클릭수']+1, max(1, a['노출수']-a['클릭수']+1), n_sim)
        s_b = np.random.beta(b['클릭수']+1, max(1, b['노출수']-b['클릭수']+1), n_sim)
        prob_win = (s_b > s_a).mean()

        m1, m2 = st.columns(2)
        m1.metric(f"{item_b} 승리 확률", f"{prob_win*100:.1f}%")
        m2.metric("성과 신뢰도", "매우 높음" if prob_win > 0.95 or prob_win < 0.05 else "데이터 추가 필요")

        # 분포 시각화
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=s_a, name=item_a, opacity=0.6, marker_color='#636EFA'))
        fig.add_trace(go.Histogram(x=s_b, name=item_b, opacity=0.6, marker_color='#EF553B'))
        fig.update_layout(
            barmode='overlay', 
            title="CTR 추정치 분포 비교",
            xaxis_title="추정 CTR",
            yaxis_title="빈도"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 서로 다른 소재를 2개 이상 입력해야 비교 분석이 가능합니다.")