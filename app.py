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
def ultra_power_clean(df_list, auto_date):
    # 빈 데이터 제외하고 병합
    valid_dfs = [df for df in df_list if not df.empty]
    if not valid_dfs: return pd.DataFrame()
    combined = pd.concat(valid_dfs, ignore_index=True)
    
    final_chunks = []
    # 매체+상품+소재별 그룹화
    for _, group in combined.groupby(['매체', '상품명', '소재명']):
        group = group.reset_index(drop=True)
        # 날짜 자동 완성 (첫 행 기준)
        if auto_date and not group.empty:
            raw_date = str(group.loc[0, '날짜']).replace('.', '-').replace('/', '-').strip()
            start_date = pd.to_datetime(raw_date, errors='coerce')
            if pd.notnull(start_date):
                group['날짜'] = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(group))]
        final_chunks.append(group)
    
    df = pd.concat(final_chunks, ignore_index=True) if final_chunks else combined
    
    # [핵심] 모든 데이터를 문자열로 취급한 뒤 숫자만 추출
    for col in ['노출수', '클릭수', '비용']:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\d]', '', x))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 지표 계산
    df['CTR(%)'] = (df['클릭수'] / df['노출수'] * 100).round(2).fillna(0.0)
    df['ID'] = "[" + df['매체'].astype(str) + "] " + df['상품명'].astype(str) + "_" + df['소재명'].astype(str)
    return df

# --- [사이드바 설정] ---
with st.sidebar:
    st.header("⚙️ 시뮬레이션 설정")
    auto_date_mode = st.checkbox("📅 소재별 날짜 자동 완성", value=True)
    n_sim = st.select_slider("🎲 반복 횟수(정밀도)", options=[1000, 5000, 10000, 20000], value=10000)

st.title("🎯 마케팅 데이터 통계 분석 & 시뮬레이터")

# --- [데이터 세션 초기화] ---
# 모든 초기값을 '문자열'로 설정하여 Streamlit의 타입 충돌을 원천 차단
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame([{
        "날짜": datetime.now().strftime("%Y-%m-%d"), 
        "매체": "네이버", "상품명": "업데이트", "소재명": "A", 
        "노출수": "0", "클릭수": "0", "비용": "0"
    }])

# --- [입력 탭 구성] ---
media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]
tabs = st.tabs(media_list)
all_input_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        # 세션에서 해당 매체 데이터 추출 및 문자열 강제 변환
        curr = st.session_state.db[st.session_state.db['매체'] == m].copy().astype(str)
        if curr.empty:
            curr = pd.DataFrame([{"날짜": datetime.now().strftime("%Y-%m-%d"), "매체": m, "상품명": "", "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"}])
        
        # 데이터 에디터: 모든 컬럼 설정을 제거하고 기본값(문자열)으로 받음
        edited = st.data_editor(
            curr, num_rows="dynamic", use_container_width=True, key=f"editor_v6_{m}"
        )
        all_input_data.append(edited)

# --- [분석 실행 버튼] ---
if st.button("🚀 데이터 정제 및 시뮬레이션 분석 시작", use_container_width=True):
    try:
        cleaned_df = ultra_power_clean(all_input_data, auto_date_mode)
        # 세션에 저장할 때도 모두 문자열로 변환하여 에러 방지
        st.session_state.db = cleaned_df.astype(str)
        st.success("데이터 정제 완료! 하단 리포트를 확인하세요.")
        st.rerun()
    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {e}")

# --- [리포트 영역] ---
df = st.session_state.db.copy()
# 시뮬레이션을 위해 숫자형으로 일시 변환
if not df.empty and 'ID' in df.columns and len(df['ID'].unique()) >= 2:
    for c in ['노출수', '클릭수', '비용', 'CTR(%)']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    st.divider()
    p_list = sorted(df['ID'].unique())
    c_a, c_b = st.columns(2)
    with c_a: item_a = st.selectbox("기준 소재 (A)", p_list, index=0)
    with c_b: item_b = st.selectbox("비교 소재 (B)", p_list, index=1)

    # 1. 시뮬레이션 엔진 (몬테카를로 + 베이지안)
    res = df.groupby('ID').agg({'노출수':'sum', '클릭수':'sum'})
    a, b = res.loc[item_a], res.loc[item_b]
    
    with st.spinner("통계적 유의성 시뮬레이션 중..."):
        s_a = np.random.beta(a['클릭수']+1, max(a['노출수']-a['클릭수'], 0)+1, n_sim)
        s_b = np.random.beta(b['클릭수']+1, max(b['노출수']-b['클릭수'], 0)+1, n_sim)
        prob_b_win = (s_b > s_a).mean()

    # 2. 결과 시각화
    st.subheader("📊 분석 결과 리포트")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(f"{item_b} 승리 확률", f"{prob_b_win*100:.1f}%")
        st.write("기대 CTR 분포:")
        st.write(f"- {item_a}: {s_a.mean()*100:.2f}%")
        st.write(f"- {item_b}: {s_b.mean()*100:.2f}%")

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=s_a, name=item_a, opacity=0.6, marker_color='#636EFA'))
        fig.add_trace(go.Histogram(x=s_b, name=item_b, opacity=0.6, marker_color='#EF553B'))
        fig.update_layout(barmode='overlay', title="CTR 성과 사후 분포 비교", xaxis_title="CTR (%)")
        st.plotly_chart(fig, use_container_width=True)

    # 3. 일자별 추이
    st.subheader("📈 일자별 CTR 추이")
    trend_df = df[df['ID'].isin([item_a, item_b])]
    fig_line = px.line(trend_df, x='날짜', y='CTR(%)', color='ID', markers=True)
    st.plotly_chart(fig_line, use_container_width=True)