import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="AE 데일리 성과 분석", layout="wide")

st.title("📅 데일리 매체 성과 분석")
st.write("날짜별 성과를 기록하고 시각화 추이를 확인하세요.")

# 데이터 저장소 초기화
if 'daily_data' not in st.session_state:
    st.session_state.daily_data = []

# 사이드바 입력창
with st.sidebar:
    st.header("입력창")
    # [개선] 날짜 선택 기능 추가
    target_date = st.date_input("날짜 선택", datetime.now())
    m_name = st.text_input("매체명", "네이버")
    p_name = st.text_input("상품명", "GFA")
    
    col1, col2 = st.columns(2)
    with col1:
        imps = st.number_input("노출수", value=0, step=1000)
        cost = st.number_input("비용", value=0, step=10000)
    with col2:
        clicks = st.number_input("클릭수", value=0, step=10)
    
    if st.button("➕ 데이터 기록", use_container_width=True):
        st.session_state.daily_data.append({
            "날짜": target_date,
            "매체": m_name,
            "상품": p_name,
            "Imps": imps, 
            "Clicks": clicks, 
            "Cost": cost,
            "ID": f"{target_date}_{m_name}_{p_name}"
        })

# 데이터 처리
if st.session_state.daily_data:
    df = pd.DataFrame(st.session_state.daily_data)
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values(by='날짜') # 날짜순 정렬
    
    # 지표 계산
    df['CTR'] = (df['Clicks'] / df['Imps']).fillna(0) * 100
    df['CPC'] = (df['Cost'] / df['Clicks']).replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # --- 데이터 관리 ---
    st.subheader("📊 누적 데이터 내역")
    # 삭제 기능 (날짜와 매체명을 조합해서 선택)
    delete_options = df['ID'].tolist()
    to_delete = st.multiselect("🗑️ 삭제할 데이터(ID) 선택", options=delete_options)
    
    if st.button("선택 삭제"):
        st.session_state.daily_data = [d for d in st.session_state.daily_data if d['ID'] not in to_delete]
        st.rerun()

    st.dataframe(df.drop(columns=['ID']), use_container_width=True)

    # --- [개선] 날짜별 추이 차트 ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    st.subheader("📈 일자별 성과 추이")
    
    # 지표 선택 (CTR을 볼지, CPC를 볼지 선택 가능)
    metric = st.selectbox("확인할 지표를 선택하세요", ["CTR", "CPC", "Cost", "Clicks"])
    
    fig = px.line(df, x="날짜", y=metric, color="매체", markers=True,
                  title=f"날짜별 {metric} 변화 추이",
                  labels={"날짜": "일자", metric: f"{metric} 수치"})
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("데이터를 기록하면 날짜별 성과 그래프가 생성됩니다.")