import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re

# 1. 페이지 설정
st.set_page_config(page_title="In-house Marketing BI", layout="wide")

# --- [데이터 정제 로직] ---
def power_clean(df_list, auto_date):
    combined = pd.concat(df_list, ignore_index=True)
    if combined.empty: return combined
    
    final_chunks = []
    # 매체+상품+소재별 그룹화
    for _, group in combined.groupby(['매체', '상품명', '소재명']):
        group = group.reset_index(drop=True)
        
        # 날짜 자동 완성 (첫 행 날짜가 있는 경우만)
        if auto_date and not group.empty:
            # 다양한 날짜 기호(., /)를 하이픈(-)으로 통일
            raw_date = str(group.loc[0, '날짜']).replace('.', '-').replace('/', '-').strip()
            start_date = pd.to_datetime(raw_date, errors='coerce')
            if pd.notnull(start_date):
                group['날짜'] = [(start_date + timedelta(days=i)).date() for i in range(len(group))]
        
        final_chunks.append(group)
    
    df = pd.concat(final_chunks, ignore_index=True)
    
    # [가장 중요한 부분] 숫자 컬럼에서 기호(₩, 콤마, 공백)를 완전히 제거
    for col in ['노출수', '클릭수', '비용']:
        # 숫자가 아닌 모든 것을 빈칸으로 바꿈
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\d]', '', x))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 분석용 ID 및 기본 지표 계산
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
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame([{
        "날짜": datetime.now().strftime("%Y-%m-%d"), "매체": "네이버", "상품명": "상품", "소재명": "소재", "노출수": "0", "클릭수": "0", "비용": "0"
    }])

# --- [입력 탭 구성] ---
media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]
tabs = st.tabs(media_list)
all_input_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        curr = st.session_state.db[st.session_state.db['매체'] == m].copy()
        if curr.empty:
            curr = pd.DataFrame([{"날짜": datetime.now().strftime("%Y-%m-%d"), "매체": m, "상품명": "", "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"}])
        
        # 모든 컬럼을 TextColumn으로 설정하여 엑셀 형식 그대로 수용
        edited = st.data_editor(
            curr, num_rows="dynamic", use_container_width=True, key=f"editor_vfinal_{m}",
            column_config={
                "날짜": st.column_config.TextColumn("날짜(첫줄만 입력)"),
                "매체": st.column_config.TextColumn("매체", disabled=True),
                "상품명": st.column_config.TextColumn("상품명"),
                "소재명": st.column_config.TextColumn("소재명"),
                "노출수": st.column_config.TextColumn("노출수"),
                "클릭수": st.column_config.TextColumn("클릭수"),
                "비용": st.column_config.TextColumn("비용(₩)")
            }
        )
        all_input_data.append(edited)

# --- [분석 실행] ---
if st.button("🚀 데이터 정제 및 시뮬레이션 분석 시작", use_container_width=True):
    try:
        st.session_state.db = power_clean(all_input_data, auto_date_mode)
        st.success("데이터 업데이트 성공! 아래 리포트를 확인하세요.")
        st.rerun()
    except Exception as e:
        st.error(f"실행 중 오류가 발생했습니다. (데이터 형식을 확인하세요): {e}")

# --- [통계 리포트 & 시뮬레이션] ---
df = st.session_state.db
# ID가 생성되었고 최소 2개 이상의 소재가 있을 때 실행
if not df.empty and 'ID' in df.columns and len(df['ID'].unique()) >= 2:
    st.divider()
    p_list = sorted(df['ID'].unique())
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1: item_a = st.selectbox("기준 소재 (A)", p_list, index=0)
    with col_sel2: item_b = st.selectbox("비교 소재 (B)", p_list, index=1)

    # 1. 데이터 집계
    res = df.groupby('ID').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum'})
    a_stats, b_stats = res.loc[item_a], res.loc[item_b]

    # 2. 몬테카를로 시뮬레이션 (베이지안 샘플링)
    with st.spinner("통계적 유의성 계산 중..."):
        # Beta 분포 기반 샘플링
        samples_a = np.random.beta(a_stats['클릭수']+1, a_stats['노출수']-a_stats['클릭수']+1, n_sim)
        samples_b = np.random.beta(b_stats['클릭수']+1, b_stats['노출수']-b_stats['클릭수']+1, n_sim)
        
        prob_b_win = (samples_b > samples_a).mean()
        lift = (samples_b.mean() - samples_a.mean()) / samples_a.mean() * 100

    # 3. 대시보드 출력
    st.subheader("📊 시뮬레이션 리포트")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{item_b} 승리 확률", f"{prob_b_win*100:.1f}%")
    c2.metric("예상 CTR 개선율", f"{lift:.2f}%")
    c3.metric("통계 신뢰도", "신뢰함" if prob_b_win > 0.95 or prob_b_win < 0.05 else "주의 필요")

    # 4. 시각화 (성과 분포)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=samples_a, name=item_a, opacity=0.6, marker_color='#636EFA'))
    fig_dist.add_trace(go.Histogram(x=samples_b, name=item_b, opacity=0.6, marker_color='#EF553B'))
    fig_dist.update_layout(
        barmode='overlay', 
        title="CTR 사후 분포 (어느 쪽 성과가 더 확실한가?)",
        xaxis_title="CTR 수치", yaxis_title="샘플 빈도"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # 5. 일자별 추이
    st.subheader("📈 일자별 CTR 변화")
    trend_df = df[df['ID'].isin([item_a, item_b])]
    fig_line = px.line(trend_df, x='날짜', y='CTR(%)', color='ID', markers=True)
    st.plotly_chart(fig_line, use_container_width=True)