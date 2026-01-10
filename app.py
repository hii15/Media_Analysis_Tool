import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

# 1. 페이지 설정
st.set_page_config(page_title="Marketing Intelligence Tool", layout="wide")

# --- [핵심 엔진: 데이터 정제] ---
def process_marketing_data(df_list):
    if not df_list: return pd.DataFrame()
    
    # 각 탭에서 들어온 리스트 결합
    combined = pd.concat(df_list, ignore_index=True)
    
    # 1. 기초 정제: 상품명이 없는 행은 삭제
    combined = combined[combined['상품명'].astype(str).str.strip() != ""]
    if combined.empty: return combined
    
    df = combined.copy()
    
    # 2. 숫자 정밀 정제 (콤마, 원화 기호 등 제거)
    for col in ['노출수', '클릭수', '비용']:
        # 숫자가 아닌 문자는 모두 제거하고 정수로 변환
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\d]', '', x))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 3. 논리적 오류 보정 및 CTR 계산
    df['클릭수'] = df[['노출수', '클릭수']].min(axis=1) # 클릭이 노출보다 클 수 없음
    df['CTR(%)'] = np.where(df['노출수'] > 0, (df['클릭수'] / df['노출수'] * 100), 0.0)
    
    # 4. 고유 ID 생성 (매체 정보는 탭에서 이미 할당됨)
    df['ID'] = "[" + df['매체'] + "] " + df['상품명'] + "_" + df['소재명']
    
    return df

# --- [사이드바] ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    n_sim = st.select_slider("🎲 시뮬레이션 정밀도", options=[1000, 5000, 10000], value=5000)
    st.info("날짜와 매체 열을 제거했습니다. 엑셀 데이터를 자유롭게 붙여넣으세요.")

st.title("🎯 통합 마케팅 성과 분석 & 시뮬레이터")

# --- [데이터 관리] ---
media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]

if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame()

# --- [데이터 입력 섹션] ---
tabs = st.tabs(media_list)
all_editor_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        # 현재 매체의 데이터만 필터링해서 보여줌
        curr = pd.DataFrame()
        if not st.session_state.db.empty:
            curr = st.session_state.db[st.session_state.db['매체'] == m].copy()
            # 화면 표시용에서 '매체'와 'ID', 'CTR' 열은 숨김 (입력 편의)
            curr = curr[['상품명', '소재명', '노출수', '클릭수', '비용']]
        
        if curr.empty:
            curr = pd.DataFrame([{"상품명": "", "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"}])
        
        # 데이터 에디터 (날짜/매체 제외)
        edited = st.data_editor(
            curr.astype(str), 
            num_rows="dynamic", 
            use_container_width=True, 
            key=f"editor_v11_{m}",
            column_config={
                "상품명": st.column_config.TextColumn("상품명 (필수)"),
                "비용": st.column_config.TextColumn("비용(₩)")
            }
        )
        # 입력된 데이터에 해당 탭의 매체명을 강제로 할당
        edited['매체'] = m
        all_editor_data.append(edited)

# --- [버튼 섹션] ---
col1, col2 = st.columns([4, 1])
with col1:
    if st.button("🚀 데이터 업데이트 및 분석 실행", use_container_width=True):
        try:
            st.session_state.db = process_marketing_data(all_editor_data)
            st.success("데이터 업데이트 성공!")
            st.rerun()
        except Exception as e:
            st.error(f"처리 중 오류가 발생했습니다: {e}")

with col2:
    if st.button("♻️ 전체 초기화", use_container_width=True):
        st.session_state.db = pd.DataFrame()
        st.rerun()

# --- [분석 리포트] ---
df = st.session_state.db
if not df.empty and 'ID' in df.columns:
    # 1. 전체 데이터 합산 테이블
    st.subheader("📋 통합 성과 요약")
    summary_table = df.groupby('ID').agg({
        '노출수': 'sum',
        '클릭수': 'sum',
        '비용': 'sum'
    }).reset_index()
    summary_table['CTR(%)'] = (summary_table['클릭수'] / summary_table['노출수'] * 100).fillna(0)
    st.dataframe(summary_table.sort_values('CTR(%)', ascending=False), use_container_width=True)

    # 2. 비교 분석
    p_list = sorted(summary_table['ID'].unique())
    if len(p_list) >= 2:
        st.divider()
        st.subheader("📊 소재별 성과 비교 (Bayesian Simulation)")
        
        c1, c2 = st.columns(2)
        with c1: item_a = st.selectbox("기준 소재 (A)", p_list, index=0)
        with c2: item_b = st.selectbox("비교 소재 (B)", p_list, index=1)

        a = summary_table[summary_table['ID'] == item_a].iloc[0]
        b = summary_table[summary_table['ID'] == item_b].iloc[0]

        # 몬테카를로 시뮬레이션
        s_a = np.random.beta(a['클릭수']+1, max(1, a['노출수']-a['클릭수']+1), n_sim)
        s_b = np.random.beta(b['클릭수']+1, max(1, b['노출수']-b['클릭수']+1), n_sim)
        
        prob_b_win = (s_b > s_a).mean()
        
        m1, m2 = st.columns(2)
        m1.metric(f"{item_b} 승리 확률", f"{prob_b_win*100:.1f}%")
        m2.metric("신뢰 수준", "확실함" if prob_b_win > 0.95 or prob_b_win < 0.05 else "데이터 추가 필요")

        # 분포 그래프
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=s_a, name=item_a, opacity=0.6, marker_color='#636EFA'))
        fig.add_trace(go.Histogram(x=s_b, name=item_b, opacity=0.6, marker_color='#EF553B'))
        fig.update_layout(
            barmode='overlay', 
            title="CTR 성과 사후 분포 비교",
            xaxis_title="추정 CTR",
            yaxis_title="빈도"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 서로 다른 소재 데이터가 2개 이상 필요합니다.")
else:
    st.info("👋 각 매체 탭에 데이터를 입력하고 '업데이트' 버튼을 눌러주세요.")