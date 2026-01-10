import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re

st.set_page_config(page_title="Ad Analysis Tool", layout="wide")

# --- [1. 데이터 정제 엔진] ---
def process_marketing_data(df_list, auto_date):
    if not df_list: return pd.DataFrame()
    combined = pd.concat(df_list, ignore_index=True)
    
    # 공백 제거 및 필터링
    combined['상품명'] = combined['상품명'].fillna('').astype(str).str.strip()
    df = combined[combined['상품명'] != ""].copy()
    
    if df.empty: return pd.DataFrame()
    
    # 숫자 정제 (콤마, 특수문자 제거)
    for col in ['노출수', '클릭수', '비용']:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\d]', '', x))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # 날짜 및 ID 생성
    processed_chunks = []
    for _, group in df.groupby(['매체', '상품명', '소재명']):
        group = group.reset_index(drop=True)
        if auto_date:
            raw_date = str(group.loc[0, '날짜']).strip()
            # 20251113 -> 2025-11-13 변환
            if len(raw_date) == 8 and raw_date.isdigit():
                raw_date = f"{raw_date[:4]}-{raw_val[4:6]}-{raw_val[6:]}"
            
            start_dt = pd.to_datetime(raw_date.replace('.', '-'), errors='coerce')
            if pd.notnull(start_dt):
                group['날짜'] = [(start_dt + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(group))]
        processed_chunks.append(group)
    
    final_df = pd.concat(processed_chunks, ignore_index=True)
    final_df['CTR(%)'] = np.where(final_df['노출수'] > 0, (final_df['클릭수'] / final_df['노출수'] * 100), 0.0)
    final_df['ID'] = "[" + final_df['매체'] + "] " + final_df['상품명'] + "_" + final_df['소재명']
    return final_df

# --- [2. 세션 상태 관리] ---
# DB는 분석 결과 저장용, 에디터 데이터는 입력 유지용
if 'db' not in st.session_state: st.session_state.db = pd.DataFrame()
media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]

st.title("🎯 Marketing Analysis Simulator")

# --- [3. 데이터 입력 섹션] ---
tabs = st.tabs(media_list)
all_editor_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        # 에러 방지를 위해 매번 고유한 키를 생성하지 않고 유지 (데이터 유실 방지)
        key = f"input_editor_{m}"
        
        # 초기 데이터 틀
        init_df = pd.DataFrame([{
            "날짜": datetime.now().strftime("%Y-%m-%d"), "매체": m, 
            "상품명": "", "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"
        }])
        
        # 에디터 실행 (유저가 입력한 40행 이상의 데이터가 여기에 담김)
        edited = st.data_editor(
            init_df,
            num_rows="dynamic",
            use_container_width=True,
            key=key
        )
        all_editor_data.append(edited)

# --- [4. 분석 실행] ---
st.divider()
if st.button("🚀 RUN ANALYSIS & SIMULATION", use_container_width=True):
    # 입력된 모든 탭의 데이터를 모아서 처리
    processed = process_marketing_data(all_editor_data, True)
    
    if not processed.empty:
        st.session_state.db = processed
        st.success(f"총 {len(processed)}개의 데이터를 성공적으로 분석했습니다!")
    else:
        st.warning("분석할 유효한 데이터가 없습니다. 상품명을 입력했는지 확인해주세요.")

# --- [5. 분석 리포트 출력] ---
df = st.session_state.db
if not df.empty:
    st.subheader("📋 Overall Results")
    st.dataframe(df, use_container_width=True)

    ids = sorted(df['ID'].unique())
    if len(ids) >= 2:
        st.divider()
        st.subheader("📊 Comparison simulation")
        c1, c2 = st.columns(2)
        with c1: a_id = st.selectbox("Baseline (A)", ids, index=0)
        with c2: b_id = st.selectbox("Comparison (B)", ids, index=1)

        # 시뮬레이션 계산
        res = df.groupby('ID').agg({'노출수':'sum', '클릭수':'sum'})
        a, b = res.loc[a_id], res.loc[b_id]
        
        s_a = np.random.beta(a['클릭수']+1, max(1, a['노출수']-a['클릭수']+1), 5000)
        s_b = np.random.beta(b['클릭수']+1, max(1, b['노출수']-b['클릭수']+1), 5000)
        
        prob = (s_b > s_a).mean()
        
        m1, m2 = st.columns(2)
        m1.metric(f"{b_id} 승리 확률", f"{prob*100:.1f}%")
        m2.metric("신뢰도", "높음" if prob > 0.95 or prob < 0.05 else "데이터 추가 필요")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=s_a, name=a_id, opacity=0.6))
        fig.add_trace(go.Histogram(x=s_b, name=b_id, opacity=0.6))
        fig.update_layout(barmode='overlay', title="CTR Distribution Comparison")
        st.plotly_chart(fig, use_container_width=True)