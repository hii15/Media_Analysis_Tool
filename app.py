import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 페이지 제목
st.title("🎯 매체 성과 측정 분석")
st.write("왼쪽 사이드바에 수치를 입력하고 버튼을 누르세요.")

# 데이터 저장소 (세션 상태)
if 'media_data' not in st.session_state:
    st.session_state.media_data = []

# 사이드바 입력창
with st.sidebar:
    st.header("입력창")
    m_name = st.text_input("매체명", "네이버")
    p_name = st.text_input("상품명", "GFA")
    d_type = st.radio("디바이스", ["MO", "PC"])
    imps = st.number_input("노출수", value=100000)
    clicks = st.number_input("클릭수", value=1000)
    cost = st.number_input("비용", value=1000000)
    
    if st.button("데이터 추가"):
        st.session_state.media_data.append({
            "분석단위": f"{m_name}_{p_name}_{d_type}",
            "Imps": imps, "Clicks": clicks, "Cost": cost
        })

# 데이터가 있을 때만 실행
if st.session_state.media_data:
    df = pd.DataFrame(st.session_state.media_data)
    
    # 지표 계산
    df['CTR'] = (df['Clicks'] / df['Imps']) * 100
    df['CPM'] = (df['Cost'] / df['Imps']) * 1000
    
    # 결과 출력
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 데이터 요약")
        st.dataframe(df)
    
    with col2:
        st.subheader("📈 가성비 차트")
        fig = px.scatter(df, x="CPM", y="CTR", size="Cost", color="분석단위", text="분석단위")
        st.plotly_chart(fig)