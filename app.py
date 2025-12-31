import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 페이지 설정
st.set_page_config(page_title="AE 매체 분석 툴", layout="wide")

st.title("🎯 매체 성과 측정 분석")
st.write("왼쪽 사이드바에 수치를 입력하고 버튼을 누르세요.")

# 데이터 저장소 초기화
if 'media_data' not in st.session_state:
    st.session_state.media_data = []

# 사이드바 입력창
with st.sidebar:
    st.header("입력창")
    m_name = st.text_input("매체명", "네이버")
    p_name = st.text_input("상품명", "GFA")
    d_type = st.radio("디바이스", ["MO", "PC"])
    imps = st.number_input("노출수", value=100000, step=1000)
    clicks = st.number_input("클릭수", value=1000, step=10)
    cost = st.number_input("비용", value=1000000, step=10000)
    
    if st.button("➕ 데이터 추가", use_container_width=True):
        st.session_state.media_data.append({
            "ID": len(st.session_state.media_data), # 삭제를 위한 고유 ID
            "분석단위": f"{m_name}_{p_name}_{d_type}",
            "Imps": imps, "Clicks": clicks, "Cost": cost
        })

# 데이터가 있을 때만 실행
if st.session_state.media_data:
    df = pd.DataFrame(st.session_state.media_data)
    
    # 지표 계산
    df['CTR'] = (df['Clicks'] / df['Imps']) * 100
    df['CPM'] = (df['Cost'] / df['Imps']) * 1000
    
    # --- [개선 1] 데이터 요약 및 삭제 섹션 ---
    st.divider()
    st.subheader("📊 데이터 관리")
    
    # 삭제 기능 추가: 멀티셀렉트로 선택해서 삭제
    delete_options = df['분석단위'].tolist()
    to_delete = st.multiselect("🗑️ 삭제할 데이터를 선택하세요 (중복 가능)", options=delete_options)
    
    if st.button("선택한 데이터 삭제"):
        # 선택되지 않은 데이터만 남기기
        st.session_state.media_data = [d for d in st.session_state.media_data if d['분석단위'] not in to_delete]
        st.rerun()

    # 테이블 출력 (간격을 위해 컨테이너 사용)
    st.dataframe(df[['분석단위', 'Imps', 'Clicks', 'Cost', 'CTR', 'CPM']], use_container_width=True)

    # --- [개선 2] 시각화 섹션 (여유로운 간격 배치) ---
    st.markdown("<br><br>", unsafe_allow_html=True) # 줄바꿈으로 간격 확보
    st.divider()
    
    st.subheader("📈 가성비 차트 분석")
    st.info("차트 종류는 추후 AE님이 원하는 분석 모델에 맞춰 변경 가능합니다.")
    
    # 차트 가독성을 위해 넓게 배치
    fig = px.scatter(df, x="CPM", y="CTR", size="Cost", color="분석단위", 
                     text="분석단위", size_max=40, height=500)
    
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("데이터를 입력하면 분석 리포트가 생성됩니다.")