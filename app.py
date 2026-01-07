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
    
    # [신규] 날짜 자동 생성 체크박스
    auto_date_mode = st.checkbox("📅 날짜 자동 생성 모드", value=False, 
                                 help="체크하면 첫 줄 날짜를 기준으로 아래 행들의 날짜를 하루씩 자동으로 채웁니다.")
    
    st.divider()
    
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
            # 날짜 형식 보정
            input_df['날짜'] = pd.to_datetime(input_df['날짜'], errors='coerce').dt.date
            
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
def process_data(df, auto_date):
    if df.empty: return df
    df = df.copy()
    
    if auto_date:
        processed_chunks = []
        for media, group in df.groupby('매체'):
            group = group.reset_index(drop=True)
            first_date = pd.to_datetime(group.loc[0, '날짜'], errors='coerce')
            if pd.notnull(first_date):
                group['날짜'] = [first_date + timedelta(days=i) for i in range(len(group))]
            processed_chunks.append(group)
        df = pd.concat(processed_chunks, ignore_index=True)
    else:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        
    df = df.dropna(subset=['날짜'])
    
    for col in ['노출수', '클릭수', '비용']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    df['CTR(%)'] = (df['클릭수'] / df['노출수'] * 100).round(2).fillna(0.0)
    df['ID'] = "[" + df['매체'] + "] " + df['상품명']
    return df

# --- [분석] 베이지안 및 몬테카를로 로직 ---
def run_analysis(df, item_a, item_b, iterations):
    res = df.groupby('ID').agg({'클릭수':'sum', '노출수':'sum'})
    a, b = res.loc[item_a], res.loc[item_b]
    
    samples_a = np.random.beta(a['클릭수']+1, a['노출수']-a['클릭수']+1, iterations)
    samples_b = np.random.beta(b['클릭수']+1, b['노출수']-b['클릭수']+1, iterations)
    
    target_ctr = df[df['ID'] == item_b]['CTR(%)']
    mu, sigma = target_ctr.mean(), target_ctr.std() if target_ctr.std() > 0 else target_ctr.mean()*0.1
    future_sims = np.maximum(0, np.random.normal(mu, sigma, (iterations, 7)))
    
    return (samples_a > samples_b).mean(), samples_a, samples_b, future_sims

# --- [데이터] 세션 관리 ---
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame([{"날짜":datetime.now().date(),"매체":"네이버","상품명":"GFA","소재명":"S1","노출수":10000,"클릭수":100,"비용":500000}])

media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]
tabs = st.tabs(media_list)
all_data = []

for i, m in enumerate(media_list):
    with tabs[i]:
        curr_df = st.session_state.db[st.session_state.db['매체'] == m].copy()
        curr_df['날짜'] = pd.to_datetime(curr_df['날짜'], errors='coerce')
        
        if curr_df.empty:
            curr_df = pd.DataFrame([{
                "날짜": datetime.now().date(),
                "매체": m, "상품명": "", "소재명": "", "노출수": 0, "클릭수": 0, "비용": 0
            }])
        
        edited = st.data_editor(
            curr_df, 
            num_rows="dynamic", 
            use_container_width=True, 
            key=f"ed_{m}",
            column_config={
                "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD", required=True)
            }
        )
        all_data.append(edited)

if st.button("🚀 통합 분석 실행 및 데이터 저장", use_container_width=True):
    raw_combined = pd.concat(all_data, ignore_index=True)
    st.session_state.db = process_data(raw_combined, auto_date_mode)
    st.success("데이터가 저장되었습니다.")
    st.rerun()

# --- [리포트] ---
final_df = st.session_state.db
if not final_df.empty and 'ID' in final_df.columns and len(final_df['ID'].unique()) >= 2:
    st.divider()
    p_list = sorted(final_df['ID'].unique())
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        item_a = st.selectbox("비교 상품 A", p_list, index=0)
    with col_sel2:
        item_b = st.selectbox("비교 상품 B", p_list, index=1)
        
    prob, s_a, s_b, f_sims = run_analysis(final_df, item_a, item_b, n_iterations)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(f"{item_b} 승리 확률", f"{prob*100:.1f}%")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=s_a, name=item_a, opacity=0.6))
        fig.add_trace(go.Histogram(x=s_b, name=item_b, opacity=0.6))
        fig.update_layout(barmode='overlay', title="CTR 사후 확률 분포 비교")
        st.plotly_chart(fig, use_container_width=True)