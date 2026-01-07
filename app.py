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
    
    # 1. 기초 정제: 상품명이 없는 행은 과감히 삭제
    combined = combined[combined['상품명'].astype(str).str.strip() != ""]
    if combined.empty: return combined
    
    processed_chunks = []
    for _, group in combined.groupby(['매체', '상품명', '소재명']):
        group = group.reset_index(drop=True)
        
        # 2. 날짜 유연 정제 (20251113, 2025.11.13, 2025-11-13 모두 대응)
        if auto_date and not group.empty:
            raw_date = str(group.loc[0, '날짜']).strip()
            # 숫자로만 된 날짜 (예: 20251113) 처리
            if len(raw_date) == 8 and raw_date.isdigit():
                raw_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            
            clean_date = raw_date.replace('.', '-').replace('/', '-')
            start_dt = pd.to_datetime(clean_date, errors='coerce')
            
            if pd.notnull(start_dt):
                group['날짜'] = [start_dt + timedelta(days=i) for i in range(len(group))]
            else:
                group['날짜'] = datetime.now().date()
        
        processed_chunks.append(group)
    
    df = pd.concat(processed_chunks, ignore_index=True)
    
    # 3. 숫자 정밀 정제 (콤마, ₩, 원화, 소수점 등 제거)
    for col in ['노출수', '클릭수', '비용']:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\d]', '', x))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 4. 논리적 오류 보정 및 CTR 계산
    df['클릭수'] = df[['노출수', '클릭수']].min(axis=1)
    df['CTR(%)'] = np.where(df['노출수'] > 0, (df['클릭수'] / df['노출수'] * 100), 0.0)
    df['ID'] = "[" + df['매체'].astype(str) + "] " + df['상품명'].astype(str) + "_" + df['소재명'].astype(str)
    
    return df

# --- [사이드바] ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    auto_date_mode = st.checkbox("📅 소재별 날짜 자동 생성", value=True, help="첫 줄의 날짜를 기준으로 다음 행들의 날짜를 하루씩 자동 증가시킵니다.")
    n_sim = st.select_slider("🎲 시뮬레이션 정밀도", options=[1000, 5000, 10000], value=5000)

st.title("🎯 통합 마케팅 성과 분석 & 시뮬레이터")

# --- [데이터 관리] ---
media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]

# 에러 방지를 위한 세션 데이터 강제 초기화 함수
def clear_db():
    st.session_state.db = pd.DataFrame()
    st.rerun()

if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame()

# --- [데이터 입력 섹션] ---
tabs = st.tabs(media_list)
all_editor_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        curr = pd.DataFrame()
        if not st.session_state.db.empty:
            curr = st.session_state.db[st.session_state.db['매체'] == m].copy()
        
        if curr.empty:
            curr = pd.DataFrame([{"날짜": datetime.now().strftime("%Y-%m-%d"), "매체": m, "상품명": "", "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"}])
        
        # [수정] column_config를 명확하게 지정하여 데이터 타입 충돌 방지
        edited = st.data_editor(
            curr, 
            num_rows="dynamic", 
            use_container_width=True, 
            key=f"editor_v8_{m}_{len(st.session_state.db)}", # 키를 동적으로 생성하여 캐시 에러 방지
            column_config={
                "날짜": st.column_config.TextColumn("날짜", help="20251113 또는 2025-11-13 형식"),
                "매체": st.column_config.TextColumn("매체", disabled=True),
                "상품명": st.column_config.TextColumn("상품명"),
                "소재명": st.column_config.TextColumn("소재명"),
                "노출수": st.column_config.TextColumn("노출수"),
                "클릭수": st.column_config.TextColumn("클릭수"),
                "비용": st.column_config.TextColumn("비용(₩)")
            }
        )
        all_editor_data.append(edited)

col1, col2 = st.columns([4, 1])
with col1:
    btn_update = st.button("🚀 데이터 업데이트 및 분석 실행", use_container_width=True)
with col2:
    if st.button("♻️ 전체 초기화", use_container_width=True):
        clear_db()

# --- [분석 및 리포트] ---
if btn_update:
    try:
        st.session_state.db = process_marketing_data(all_editor_data, auto_date_mode)
        st.success("데이터 업데이트 성공!")
        st.rerun()
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")

df = st.session_state.db
if not df.empty and 'ID' in df.columns:
    p_list = sorted(df['ID'].unique())
    
    if len(p_list) >= 2:
        st.divider()
        st.subheader("📊 소재별 성과 비교 분석")
        
        c1, c2 = st.columns(2)
        with c1: item_a = st.selectbox("기준 소재 (A)", p_list, index=0)
        with c2: item_b = st.selectbox("비교 소재 (B)", p_list, index=1)

        res = df.groupby('ID').agg({'노출수':'sum', '클릭수':'sum'})
        a, b = res.loc[item_a], res.loc[item_b]

        # 시뮬레이션
        s_a = np.random.beta(a['클릭수']+1, max(1, a['노출수']-a['클릭수']+1), n_sim)
        s_b = np.random.beta(b['클릭수']+1, max(1, b['노출수']-b['클릭수']+1), n_sim)
        
        prob_b_win = (s_b > s_a).mean()
        lift = (s_b.mean() - s_a.mean()) / (s_a.mean() if s_a.mean() > 0 else 1e-9) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{item_b} 승리 확률", f"{prob_b_win*100:.1f}%")
        m2.metric("기대 CTR 개선율", f"{lift:.2f}%")
        m3.metric("신뢰 수준", "확실함" if prob_b_win > 0.95 or prob_b_win < 0.05 else "데이터 더 필요")

        # 분포 그래프
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=s_a, name=item_a, opacity=0.6, marker_color='#636EFA'))
        fig.add_trace(go.Histogram(x=s_b, name=item_b, opacity=0.6, marker_color='#EF553B'))
        fig.update_layout(barmode='overlay', title="CTR 성과 사후 분포", xaxis_title="추정 CTR", yaxis_title="빈도")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 서로 다른 소재 데이터가 2개 이상 필요합니다.")
else:
    st.info("👋 위 테이블에 데이터를 입력하고 '업데이트' 버튼을 눌러주세요.")