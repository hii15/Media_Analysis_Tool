import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re

# 1. 페이지 설정
st.set_page_config(page_title="Marketing Intelligence Tool", layout="wide")

# --- [핵심 엔진 1: 데이터 정제 및 날짜 자동화] ---
def process_marketing_data(df_list, auto_date):
    combined = pd.concat(df_list, ignore_index=True)
    if combined.empty: return combined
    
    processed_chunks = []
    # 매체+상품+소재별로 그룹화하여 날짜 및 데이터 정리
    for _, group in combined.groupby(['매체', '상품명', '소재명']):
        group = group.reset_index(drop=True)
        
        # 1. 날짜 자동 완성 (첫 줄 날짜 기준)
        if auto_date and not group.empty:
            raw_val = str(group.loc[0, '날짜']).replace('.', '-').replace(' ', '')
            start_dt = pd.to_datetime(raw_val, errors='coerce')
            if pd.notnull(start_dt):
                group['날짜'] = [start_dt + timedelta(days=i) for i in range(len(group))]
        
        processed_chunks.append(group)
    
    df = pd.concat(processed_chunks, ignore_index=True)
    
    # 2. 숫자 정밀 정제 (₩, 콤마, 공백 제거하여 계산 오류 방지)
    for col in ['노출수', '클릭수', '비용']:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\d]', '', x))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 기본 지표 계산
    df['CTR(%)'] = (df['클릭수'] / df['노출수'] * 100).round(2).fillna(0.0)
    df['ID'] = "[" + df['매체'].astype(str) + "] " + df['상품명'].astype(str) + "_" + df['소재명'].astype(str)
    return df

# --- [사이드바] 설정 ---
with st.sidebar:
    st.header("⚙️ 분석 및 시뮬레이션 설정")
    auto_date_mode = st.checkbox("📅 소재별 날짜 자동 생성", value=True)
    n_sim = st.select_slider("🎲 시뮬레이션 반복 횟수", options=[1000, 5000, 10000, 20000], value=10000)
    st.caption("몬테카를로 및 베이지안 분석의 정밀도를 결정합니다.")

st.title("🎯 통합 마케팅 성과 분석 & 시뮬레이터")

# --- [데이터 세션 및 입력] ---
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame([{"날짜": datetime.now().date(), "매체": "네이버", "상품명": "업데이트", "소재명": "A", "노출수": "0", "클릭수": "0", "비용": "0"}])

media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]
tabs = st.tabs(media_list)
all_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        curr = st.session_state.db[st.session_state.db['매체'] == m].copy()
        if curr.empty:
            curr = pd.DataFrame([{"날짜": datetime.now().date(), "매체": m, "상품명": "", "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"}])
        
        # 텍스트 컬럼으로 설정하여 엑셀 기호(₩) 입력 허용
        edited = st.data_editor(curr, num_rows="dynamic", use_container_width=True, key=f"editor_{m}",
                               column_config={"날짜": st.column_config.TextColumn("날짜(첫줄만)"),
                                              "비용": st.column_config.TextColumn("비용(₩)"),
                                              "노출수": st.column_config.TextColumn("노출수"),
                                              "클릭수": st.column_config.TextColumn("클릭수")})
        all_data.append(edited)

if st.button("🚀 데이터 업데이트 및 시뮬레이션 통합 실행", use_container_width=True):
    try:
        st.session_state.db = process_marketing_data(all_data, auto_date_mode)
        st.success("데이터 정제 및 분석 준비 완료!")
        st.rerun()
    except Exception as e:
        st.error(f"오류 발생: {e}")

# --- [리포트 섹션: 몬테카를로 & 베이지안 통합] ---
df = st.session_state.db
if not df.empty and len(df['ID'].unique()) >= 2:
    st.divider()
    p_list = sorted(df['ID'].unique())
    col_a, col_b = st.columns(2)
    with col_a: item_a = st.selectbox("기준 소재 (A)", p_list, index=0)
    with col_b: item_b = st.selectbox("비교 소재 (B)", p_list, index=1)

    # 데이터 요약
    res = df.groupby('ID').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum'})
    a, b = res.loc[item_a], res.loc[item_b]

    # 1. 몬테카를로 & 베이지안 혼합 시뮬레이션 엔진
    with st.spinner("몬테카를로 시뮬레이션 기동 중..."):
        # 베이지안 사후 분포 샘플링
        s_a = np.random.beta(a['클릭수']+1, a['노출수']-a['클릭수']+1, n_sim)
        s_b = np.random.beta(b['클릭수']+1, b['노출수']-b['클릭수']+1, n_sim)
        
        # 몬테카를로를 통한 우위 확률 계산
        prob_b_win = (s_b > s_a).mean()
        lift = (s_b.mean() - s_a.mean()) / s_a.mean() * 100

    # 2. 성과 비교 대시보드
    st.subheader("📊 시뮬레이션 분석 결과")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{item_b} 승리 확률", f"{prob_b_win*100:.1f}%")
    m2.metric("기대 CTR 개선율", f"{lift:.2f}%")
    m3.metric("신뢰 수준", "매우 높음" if prob_b_win > 0.95 or prob_b_win < 0.05 else "추가 데이터 필요")

    # 3. 분포 시각화 (베이지안 사후 분포)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=s_a, name=item_a, opacity=0.6, marker_color='#636EFA'))
    fig_dist.add_trace(go.Histogram(x=s_b, name=item_b, opacity=0.6, marker_color='#EF553B'))
    fig_dist.update_layout(barmode='overlay', title="CTR 성과 사후 분포 비교 (몬테카를로 시뮬레이션)",
                          xaxis_title="CTR (%)", yaxis_title="샘플 빈도")
    st.plotly_chart(fig_dist, use_container_width=True)

    # 4. 일자별 추이 및 통계
    st.subheader("📈 성과 히스토리")
    trend_df = df[df['ID'].isin([item_a, item_b])]
    fig_line = px.line(trend_df, x='날짜', y='CTR(%)', color='ID', markers=True, title="일자별 CTR 변화 추이")
    st.plotly_chart(fig_line, use_container_width=True)

    # 5. 상세 데이터 테이블
    with st.expander("🔎 상세 데이터 확인"):
        st.dataframe(res.loc[[item_a, item_b]].style.format("{:,}"))