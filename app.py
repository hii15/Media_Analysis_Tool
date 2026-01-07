import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="In-house Marketing BI", layout="wide")

# --- [사이드바] 시뮬레이션 설정 ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    n_iterations = st.select_slider(
        "시뮬레이션 반복 횟수",
        options=[1000, 5000, 10000, 50000, 100000],
        value=10000,
        help="횟수가 많을수록 베이지안 승률 및 예측 분포가 정교해집니다."
    )
    st.info(f"설정된 {n_iterations:,}회 연산은 통계적 수렴을 보장합니다.")

st.title("🎯 마케팅 전략 의사결정 시뮬레이터")

# --- [유틸리티] 데이터 처리 함수 ---
def process_data(df):
    if df.empty: return df
    df = df.copy()
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜'])
    for col in ['노출수', '클릭수', '비용']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df['CTR(%)'] = (df['클릭수'] / df['노출수'] * 100).round(2).fillna(0.0)
    return df

# --- [분석] 베이지안 및 몬테카를로 로직 ---
def run_analysis(df, item_a, item_b, iterations):
    # 베이지안 승률 계산
    res = df.groupby('상품명').agg({'클릭수':'sum', '노출수':'sum'})
    a, b = res.loc[item_a], res.loc[item_b]
    
    samples_a = np.random.beta(a['클릭수']+1, a['노출수']-a['클릭수']+1, iterations)
    samples_b = np.random.beta(b['클릭수']+1, b['노출수']-b['클릭수']+1, iterations)
    
    # 몬테카를로 미래 예측 (상품 B 기준)
    target_ctr = df[df['상품명'] == item_b]['CTR(%)']
    mu, sigma = target_ctr.mean(), target_ctr.std() if target_ctr.std() > 0 else target_ctr.mean()*0.1
    future_sims = np.maximum(0, np.random.normal(mu, sigma, (iterations, 7)))
    
    return (samples_a > samples_b).mean(), samples_a, samples_b, future_sims

# --- [데이터] 세션 관리 및 입력 ---
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame([{"날짜":"2025-01-01","매체":"네이버","상품명":"GFA","소재명":"S1","노출수":10000,"클릭수":100,"비용":500000}])

media_list = ["네이버", "카카오", "구글", "메타", "유튜브"]
tabs = st.tabs(media_list)
all_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        curr_df = st.session_state.db[st.session_state.db['매체'] == m].copy()
        if curr_df.empty: curr_df = pd.DataFrame([{"날짜":datetime.now().strftime("%Y-%m-%d"),"매체":m,"상품명":"","소재명":"","노출수":0,"클릭수":0,"비용":0}])
        edited = st.data_editor(curr_df, num_rows="dynamic", use_container_width=True, key=f"ed_{m}")
        all_data.append(edited)

if st.button("🚀 통합 분석 실행", use_container_width=True):
    st.session_state.db = pd.concat(all_data, ignore_index=True)
    st.rerun()

# --- [리포트] 시각화 분석 ---
final_df = process_data(st.session_state.db)
if not final_df.empty and len(final_df['상품명'].unique()) >= 2:
    st.divider()
    
    # 1. 베이지안 비교
    c1, c2 = st.columns([1, 2])
    with c1:
        p_list = final_df['상품명'].unique()
        item_a = st.selectbox("대조군(A)", p_list, index=0)
        item_b = st.selectbox("실험군(B)", p_list, index=1)
        prob, s_a, s_b, f_sims = run_analysis(final_df, item_a, item_b, n_iterations)
        st.metric(f"{item_b} 승리 확률", f"{prob*100:.1f}%")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=s_a, name=item_a, opacity=0.6))
        fig.add_trace(go.Histogram(x=s_b, name=item_b, opacity=0.6))
        fig.update_layout(barmode='overlay', title="CTR 사후 확률 분포 비교")
        st.plotly_chart(fig, use_container_width=True)

    # 2. 미래 예측
    st.subheader(f"🔮 {item_b} 향후 7일 성과 예측")
    days = [datetime.now() + timedelta(days=i) for i in range(7)]
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=days, y=np.median(f_sims, axis=0), mode='lines+markers', name="예상값"))
    fig_f.add_trace(go.Scatter(x=days, y=np.percentile(f_sims, 95, axis=0), line=dict(width=0), showlegend=False))
    fig_f.add_trace(go.Scatter(x=days, y=np.percentile(f_sims, 5, axis=0), fill='tonexty', line=dict(width=0), name="90% 신뢰구간"))
    st.plotly_chart(fig_f, use_container_width=True)