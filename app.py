import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="In-house Marketing BI", layout="wide")

# --- [사이드바] 데이터 관리 및 설정 ---
with st.sidebar:
    st.header("💾 데이터 관리 (Save/Load)")
    
    # 1. 다운로드 기능
    if 'db' in st.session_state and not st.session_state.db.empty:
        csv = st.session_state.db.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📊 현재 데이터 CSV로 내보내기",
            data=csv,
            file_name=f"marketing_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
        )
    
    st.divider()
    
    # 2. 업로드 기능
    uploaded_file = st.file_uploader("📂 저장된 CSV 파일 불러오기", type=["csv"])
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            # 데이터 정합성 확인 (필수 컬럼 존재 여부)
            required_cols = ["날짜", "매체", "상품명", "소재명", "노출수", "클릭수", "비용"]
            if all(col in input_df.columns for col in required_cols):
                if st.button("📥 데이터 덮어쓰기 적용"):
                    st.session_state.db = input_df
                    st.success("데이터를 성공적으로 불러왔습니다!")
                    st.rerun()
            else:
                st.error("CSV 파일 형식이 일치하지 않습니다.")
        except Exception as e:
            st.error(f"파일을 읽는 중 오류 발생: {e}")

    st.divider()
    st.header("⚙️ 분석 설정")
    n_iterations = st.select_slider("시뮬레이션 반복 횟수", options=[1000, 5000, 10000, 50000], value=10000)

st.title("🎯 데이터 기반 마케팅 분석툴")

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
    res = df.groupby('상품명').agg({'클릭수':'sum', '노출수':'sum'})
    a, b = res.loc[item_a], res.loc[item_b]
    samples_a = np.random.beta(a['클릭수']+1, a['노출수']-a['클릭수']+1, iterations)
    samples_b = np.random.beta(b['클릭수']+1, b['노출수']-b['클릭수']+1, iterations)
    target_ctr = df[df['상품명'] == item_b]['CTR(%)']
    mu, sigma = target_ctr.mean(), target_ctr.std() if target_ctr.std() > 0 else target_ctr.mean()*0.1
    future_sims = np.maximum(0, np.random.normal(mu, sigma, (iterations, 7)))
    return (samples_a > samples_b).mean(), samples_a, samples_b, future_sims

# --- [데이터] 세션 관리 및 입력 ---
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame([{"날짜":"2025-01-01","매체":"네이버","상품명":"GFA","소재명":"S1","노출수":10000,"클릭수":100,"비용":500000}])

media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]
tabs = st.tabs(media_list)
all_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        # 현재 매체에 해당하는 데이터 필터링
        curr_df = st.session_state.db[st.session_state.db['매체'] == m].copy()
        if curr_df.empty:
            curr_df = pd.DataFrame([{"날짜":datetime.now().strftime("%Y-%m-%d"),"매체":m,"상품명":"","소재명":"","노출수":0,"클릭수":0,"비용":0}])
        
        # 날짜 컬럼을 문자열로 변환하여 에디터에서 편집 가능하게 함
        curr_df['날짜'] = curr_df['날짜'].astype(str)
        edited = st.data_editor(curr_df, num_rows="dynamic", use_container_width=True, key=f"ed_{m}")
        all_data.append(edited)

if st.button("🚀 통합 분석 실행 및 데이터 저장", use_container_width=True):
    st.session_state.db = pd.concat(all_data, ignore_index=True)
    st.success("입력한 데이터가 세션에 저장되었습니다.")
    st.rerun()

# --- [리포트] 시각화 분석 (이전과 동일) ---
final_df = process_data(st.session_state.db)
if not final_df.empty and len(final_df['상품명'].unique()) >= 2:
    st.divider()
    # (이후 분석 및 차트 로직...)
    st.info("분석 리포트가 하단에 활성화되었습니다.")