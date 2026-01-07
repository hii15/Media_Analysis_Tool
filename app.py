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
    
    # 필수 컬럼이 하나라도 비어있는 행 삭제
    combined = combined.dropna(subset=['매체', '상품명', '소재명'])
    # 상품명이나 소재명이 공백인 경우 제외
    combined = combined[combined['상품명'].str.strip() != ""]
    
    if combined.empty: return combined
    
    processed_chunks = []
    for _, group in combined.groupby(['매체', '상품명', '소재명']):
        group = group.reset_index(drop=True)
        if auto_date and not group.empty:
            # 날짜 정제
            first_date = str(group.loc[0, '날짜']).replace('.', '-').replace(' ', '')
            start_dt = pd.to_datetime(first_date, errors='coerce')
            if pd.notnull(start_dt):
                group['날짜'] = [start_dt + timedelta(days=i) for i in range(len(group))]
            else:
                group['날짜'] = datetime.now().date()
        processed_chunks.append(group)
    
    df = pd.concat(processed_chunks, ignore_index=True)
    
    # 숫자 정밀 정제
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
    n_sim = st.select_slider("🎲 시뮬레이션 반복 횟수", options=[1000, 5000, 10000, 20000], value=10000)

st.title("🎯 통합 마케팅 성과 분석 & 시뮬레이터")

# --- [데이터 세션 관리] ---
media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]

if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame()

# --- [데이터 입력 섹션: 항상 표시] ---
tabs = st.tabs(media_list)
all_editor_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        # 현재 매체 데이터 가져오기 (없으면 빈 기본 틀 제공)
        curr = pd.DataFrame()
        if not st.session_state.db.empty:
            curr = st.session_state.db[st.session_state.db['매체'] == m].copy()
        
        if curr.empty:
            curr = pd.DataFrame([{"날짜": datetime.now().strftime("%Y-%m-%d"), "매체": m, "상품명": "", "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"}])
        
        # 데이터 에디터 (항상 노출됨)
        edited = st.data_editor(
            curr, 
            num_rows="dynamic", 
            use_container_width=True, 
            key=f"editor_v7_{m}",
            column_config={
                "날짜": st.column_config.TextColumn("날짜(YYYY-MM-DD)"),
                "비용": st.column_config.TextColumn("비용(₩)"),
                "노출수": st.column_config.TextColumn("노출수"),
                "클릭수": st.column_config.TextColumn("클릭수")
            }
        )
        all_editor_data.append(edited)

# --- [업데이트 버튼] ---
if st.button("🚀 데이터 업데이트 및 분석 실행", use_container_width=True):
    try:
        new_df = process_marketing_data(all_editor_data, auto_date_mode)
        if not new_df.empty:
            st.session_state.db = new_df
            st.success("데이터가 업데이트되었습니다!")
            st.rerun()
        else:
            st.warning("분석할 데이터가 없습니다. 상품명과 소재명을 입력했는지 확인해주세요.")
    except Exception as e:
        st.error(f"오류 발생: {e}")

# --- [분석 리포트 섹션: 조건부 노출] ---
df = st.session_state.db

if not df.empty and 'ID' in df.columns:
    # 유효한 ID (상품명 등이 입력된 데이터) 추출
    p_list = sorted(df['ID'].unique())
    
    if len(p_list) >= 2:
        st.divider()
        st.subheader("📊 소재별 성과 비교 분석")
        
        col_a, col_b = st.columns(2)
        with col_a: item_a = st.selectbox("기준 소재 (A)", p_list, index=0)
        with col_b: item_b = st.selectbox("비교 소재 (B)", p_list, index=1)

        res = df.groupby('ID').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum'})
        a, b = res.loc[item_a], res.loc[item_b]

        # 베이지안 시뮬레이션
        with st.spinner("시뮬레이션 분석 중..."):
            s_a = np.random.beta(a['클릭수']+1, max(1, a['노출수']-a['클릭수']+1), n_sim)
            s_b = np.random.beta(b['클릭수']+1, max(1, b['노출수']-b['클릭수']+1), n_sim)
            prob_b_win = (s_b > s_a).mean()
            mean_a = s_a.mean() if s_a.mean() > 0 else 1e-9
            lift = (s_b.mean() - s_a.mean()) / mean_a * 100

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{item_b} 승리 확률", f"{prob_b_win*100:.1f}%")
        m2.metric("기대 CTR 개선율", f"{lift:.2f}%")
        
        conf = "매우 높음" if prob_b_win > 0.95 or prob_b_win < 0.05 else "추가 데이터 필요"
        m3.metric("신뢰도", conf)

        # 그래프 시각화
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=s_a, name=item_a, opacity=0.6, marker_color='#636EFA'))
        fig_dist.add_trace(go.Histogram(x=s_b, name=item_b, opacity=0.6, marker_color='#EF553B'))
        fig_dist.update_layout(barmode='overlay', title="CTR 성과 사후 분포", xaxis_title="추정 CTR", yaxis_title="빈도")
        st.plotly_chart(fig_dist, use_container_width=True)

        trend_df = df[df['ID'].isin([item_a, item_b])]
        fig_line = px.line(trend_df, x='날짜', y='CTR(%)', color='ID', markers=True, title="일자별 CTR 변화 추이")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("💡 서로 다른 '상품명' 혹은 '소재명'을 가진 데이터를 2개 이상 입력하시면 상세 비교 분석이 활성화됩니다.")
else:
    st.info("👋 위 테이블에 광고 성과 데이터를 입력하고 **'업데이트'** 버튼을 클릭하세요. (엑셀 복사-붙여넣기 가능)")