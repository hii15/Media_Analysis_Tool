import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="In-house Marketing BI", layout="wide")
st.title("🎮 광고 상품별 수명 및 미래 성과 분석 시스템")

# --- 데이터 유틸리티 ---
def clean_and_calculate(df):
    if df.empty: return df
    new_df = df.copy()
    
    # 날짜 처리 (에러 방지를 위해 텍스트 -> 데이트타임 -> 텍스트 변환 과정 관리)
    new_df['날짜'] = pd.to_datetime(new_df['날짜'], errors='coerce')
    new_df = new_df.dropna(subset=['날짜'])
    
    for col in ['노출수', '클릭수', '비용']:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0).astype(int)
    
    new_df['CTR(%)'] = (new_df['클릭수'] / new_df['노출수'] * 100).round(2).fillna(0.0)
    new_df['CPC'] = (new_df['비용'] / new_df['클릭수']).replace([np.inf, -np.inf], 0).round(0).fillna(0).astype(int)
    
    return new_df

# --- [분석 로직] 몬테카를로 및 피로도 ---
def run_monte_carlo(df, iterations=1000):
    if len(df) < 5: return None # 최소 5일치 데이터 필요
    mu = df['CTR(%)'].mean()
    sigma = df['CTR(%)'].std() if df['CTR(%)'].std() > 0 else mu * 0.1
    sims = np.random.normal(mu, sigma, (iterations, 7))
    return np.where(sims < 0, 0, sims)

def analyze_fatigue(df):
    results = []
    for product in df['상품명'].unique():
        p_df = df[df['상품명'] == product].sort_values('날짜')
        if len(p_df) >= 3:
            p_df['Cum_Imp'] = p_df['노출수'].cumsum()
            corr = p_df['Cum_Imp'].corr(p_df['CTR(%)'])
            results.append({"상품명": product, "피로도 지수": round(corr, 2), "평균 CTR": round(p_df['CTR(%)'].mean(), 2)})
    return pd.DataFrame(results)

# --- 데이터 저장소 초기화 ---
if 'master_v10' not in st.session_state:
    # 초기 샘플 데이터
    st.session_state.master_v10 = pd.DataFrame([
        {"날짜": "2025-12-01", "매체": "네이버", "상품명": "GFA(뉴스)", "소재명": "소재A", "노출수": 10000, "클릭수": 120, "비용": 500000}
    ])

# --- [UX] 매체별 입력 탭 ---
media_list = ["네이버", "카카오", "구글", "메타", "유튜브"]
st.subheader("📝 일별 성과 입력")
tabs = st.tabs(media_list)
all_edits = []

for i, media in enumerate(media_list):
    with tabs[i]:
        m_df = st.session_state.master_v10[st.session_state.master_v10['매체'] == media].copy()
        if m_df.empty:
            m_df = pd.DataFrame([{"날짜": datetime.now().strftime("%Y-%m-%d"), "매체": media, "상품명": "", "소재명": "", "노출수": 0, "클릭수": 0, "비용": 0}])
        
        m_df['날짜'] = m_df['날짜'].astype(str)
        edited = st.data_editor(m_df, num_rows="dynamic", use_container_width=True, key=f"editor_{media}")
        all_edits.append(edited)

if st.button("🚀 전체 분석 업데이트", use_container_width=True):
    st.session_state.master_v10 = pd.concat(all_edits, ignore_index=True)
    st.rerun()

# --- [Main] 분석 섹션 ---
final_df = clean_and_calculate(st.session_state.master_v10)

if not final_df.empty:
    st.divider()
    
    # 1. 시각적 성과 분포 (TreeMap)
    st.subheader("💎 매체/상품별 비용 및 효율 비중")
    fig_tree = px.treemap(final_df, path=['매체', '상품명', '소재명'], values='비용', color='CTR(%)', 
                          color_continuous_scale='RdYlGn', title="면적: 비용 / 색상: CTR(%)")
    st.plotly_chart(fig_tree, use_container_width=True)

    # 2. 고도화 분석 (몬테카를로 & 수명 예측)
    st.subheader("🧪 통계적 예측 및 리스크 진단")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        selected_p = st.selectbox("예측 대상 상품 선택", final_df['상품명'].unique())
        p_target_df = final_df[final_df['상품명'] == selected_p]
        sim_data = run_monte_carlo(p_target_df)
        
        if sim_data is not None:
            days = [datetime.now() + timedelta(days=i) for i in range(7)]
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=days, y=np.percentile(sim_data, 95, axis=0), mode='lines', line=dict(width=0), name='상위 5%'))
            fig_sim.add_trace(go.Scatter(x=days, y=np.percentile(sim_data, 5, axis=0), mode='lines', fill='tonexty', line=dict(width=0), name='신뢰구간'))
            fig_sim.add_trace(go.Scatter(x=days, y=np.median(sim_data, axis=0), mode='lines+markers', line=dict(color='red'), name='중간값 예측'))
            fig_sim.update_layout(title=f"{selected_p} 향후 7일 CTR 예측 (90% 신뢰구간)")
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.warning("데이터가 5일 이상 쌓여야 몬테카를로 시뮬레이션이 가능합니다.")

    with c2:
        st.write("📉 **상품별 수명(피로도) 진단**")
        fatigue_res = analyze_fatigue(final_df)
        if not fatigue_res.empty:
            def style_fatigue(v):
                color = 'red' if v < -0.5 else 'orange' if v < 0 else 'green'
                return f'color: {color}; font-weight: bold'
            st.dataframe(fatigue_res.style.applymap(style_fatigue, subset=['피로도 지수']), use_container_width=True)
            st.caption("-1에 가까울수록 노출 대비 효율이 하락하는 '피로' 상태입니다.")
        else:
            st.info("피로도 분석을 위해 데이터가 더 필요합니다.")

    # 3. 예산 증액 시뮬레이션
    st.divider()
    st.subheader("📈 예산 증액 민감도 시뮬레이션")
    if len(final_df) > 3:
        z = np.polyfit(final_df['비용'], final_df['클릭수'], 1)
        p = np.poly1d(z)
        spend_x = np.linspace(final_df['비용'].min(), final_df['비용'].max() * 1.5, 30)
        fig_sens = px.line(x=spend_x, y=p(spend_x), labels={'x':'예상 지출액', 'y':'예상 클릭수'}, 
                           title="비용 투입에 따른 기대 클릭수 증가 곡선")
        st.plotly_chart(fig_sens, use_container_width=True)