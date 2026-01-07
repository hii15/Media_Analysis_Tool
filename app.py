import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re

# 1. 페이지 설정 (반드시 코드 최상단에 위치)
st.set_page_config(page_title="Marketing Intelligence Tool", layout="wide")

# --- [핵심 엔진: 데이터 정제 및 날짜 자동화] ---
def process_marketing_data(df_list, auto_date):
    combined = pd.concat(df_list, ignore_index=True)
    if combined.empty: return combined
    
    processed_chunks = []
    for _, group in combined.groupby(['매체', '상품명', '소재명']):
        group = group.reset_index(drop=True)
        
        # 1. 날짜 자동 완성
        if auto_date and not group.empty:
            raw_val = str(group.loc[0, '날짜']).replace('.', '-').replace(' ', '')
            start_dt = pd.to_datetime(raw_val, errors='coerce')
            if pd.notnull(start_dt):
                group['날짜'] = [start_dt + timedelta(days=i) for i in range(len(group))]
        
        processed_chunks.append(group)
    
    # 데이터가 없으면 빈 DF 반환
    if not processed_chunks:
        return combined

    df = pd.concat(processed_chunks, ignore_index=True)
    
    # 2. 숫자 정밀 정제 (특수문자 제거)
    for col in ['노출수', '클릭수', '비용']:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\d]', '', x))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # [안전장치] 클릭수가 노출수보다 크면 노출수와 같게 보정 (논리 에러 방지)
    df['클릭수'] = df[['노출수', '클릭수']].min(axis=1)

    # 기본 지표 계산 (0 나누기 에러 방지)
    df['CTR(%)'] = np.where(df['노출수'] > 0, (df['클릭수'] / df['노출수'] * 100), 0.0)
    df['CTR(%)'] = df['CTR(%)'].round(2)
    
    df['ID'] = "[" + df['매체'].astype(str) + "] " + df['상품명'].astype(str) + "_" + df['소재명'].astype(str)
    return df

# --- [사이드바] 설정 ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    auto_date_mode = st.checkbox("📅 소재별 날짜 자동 생성", value=True)
    n_sim = st.select_slider("🎲 시뮬레이션 반복 횟수", options=[1000, 5000, 10000, 20000], value=10000)
    st.caption("높을수록 분석이 정밀해지지만 속도가 느려질 수 있습니다.")

st.title("🎯 통합 마케팅 성과 분석 & 시뮬레이터")

# --- [데이터 세션 초기화] ---
# 처음 실행 시 에러가 나지 않도록 '빈 깡통' 데이터프레임을 미리 튼튼하게 만들어둡니다.
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame(columns=["날짜", "매체", "상품명", "소재명", "노출수", "클릭수", "비용"])

media_list = ["네이버", "카카오", "구글", "메타", "유튜브", "SOOP", "디시인사이드", "인벤", "루리웹"]
tabs = st.tabs(media_list)
all_data = []

# --- [데이터 입력 에디터 생성] ---
for i, m in enumerate(media_list):
    with tabs[i]:
        # 현재 세션 데이터에서 해당 매체만 필터링
        if not st.session_state.db.empty:
            curr = st.session_state.db[st.session_state.db['매체'] == m].copy()
        else:
            curr = pd.DataFrame()

        # 데이터가 없으면 기본 입력 행 생성
        if curr.empty:
            curr = pd.DataFrame([{"날짜": datetime.now().date(), "매체": m, "상품명": "", "소재명": "", "노출수": "0", "클릭수": "0", "비용": "0"}])
        
        # 엑셀 붙여넣기를 위한 텍스트 컬럼 설정
        edited = st.data_editor(curr, num_rows="dynamic", use_container_width=True, key=f"editor_v6_{m}",
                               column_config={"날짜": st.column_config.TextColumn("날짜(첫줄만)"),
                                             "비용": st.column_config.TextColumn("비용(₩)"),
                                             "노출수": st.column_config.TextColumn("노출수"),
                                             "클릭수": st.column_config.TextColumn("클릭수")})
        all_data.append(edited)

# --- [실행 버튼] ---
if st.button("🚀 데이터 업데이트 및 시뮬레이션 통합 실행", use_container_width=True):
    try:
        st.session_state.db = process_marketing_data(all_data, auto_date_mode)
        st.success("데이터 정제 완료! 아래에서 분석 결과를 확인하세요.")
        st.rerun()
    except Exception as e:
        st.error(f"실행 중 오류 발생: {e}")

# --- [리포트 섹션: 안전 모드 적용] ---
df = st.session_state.db

# 🚨 [핵심 수정] ID 컬럼이 없거나(버튼 안누름), 데이터가 충분하지 않으면 아예 분석 코드를 실행하지 않음
if df.empty or 'ID' not in df.columns:
    st.info("👋 **사용 가이드**: 상단 탭에서 데이터를 입력(또는 엑셀 붙여넣기)한 후, **'데이터 업데이트'** 버튼을 눌러주세요.")

else:
    # 유효한 ID만 추출 (빈 값 제외)
    valid_ids = df[df['ID'].str.len() > 5]['ID'].unique()
    p_list = sorted(valid_ids)

    # 비교 대상이 2개 미만인 경우
    if len(p_list) < 2:
        st.warning("⚠️ **분석 대기 중**: 비교할 소재가 부족합니다. 최소 2개 이상의 서로 다른 소재(상품명/소재명) 데이터를 입력해주세요.")
    
    # 데이터가 충분할 때만 실제 분석 가동
    else:
        st.divider()
        col_a, col_b = st.columns(2)
        
        # 인덱스 에러 방지
        idx_a = 0
        idx_b = 1 if len(p_list) > 1 else 0
        
        with col_a: item_a = st.selectbox("기준 소재 (A)", p_list, index=idx_a)
        with col_b: item_b = st.selectbox("비교 소재 (B)", p_list, index=idx_b)

        # 데이터 집계
        res = df.groupby('ID').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum'})
        
        if item_a in res.index and item_b in res.index:
            a, b = res.loc[item_a], res.loc[item_b]

            # 1. 시뮬레이션 엔진 (ZeroDivision 방지 포함)
            with st.spinner("시뮬레이션 기동 중..."):
                s_a = np.random.beta(a['클릭수']+1, a['노출수']-a['클릭수']+1, n_sim)
                s_b = np.random.beta(b['클릭수']+1, b['노출수']-b['클릭수']+1, n_sim)
                
                prob_b_win = (s_b > s_a).mean()
                mean_a = s_a.mean() if s_a.mean() > 0 else 1e-9 # 0 나누기 방지
                lift = (s_b.mean() - s_a.mean()) / mean_a * 100

            # 2. 결과 대시보드
            st.subheader("📊 시뮬레이션 분석 결과")
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{item_b} 승리 확률", f"{prob_b_win*100:.1f}%")
            m2.metric("기대 CTR 개선율", f"{lift:.2f}%")
            
            # 신뢰도 판단
            confidence_msg = "판단 보류"
            if prob_b_win > 0.95: confidence_msg = "B안 승리 확실 (95%↑)"
            elif prob_b_win < 0.05: confidence_msg = "A안 승리 확실 (95%↑)"
            else: confidence_msg = "추가 데이터 필요 (박빙)"
            m3.metric("신뢰 수준", confidence_msg)

            # 3. 분포 그래프
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=s_a, name=item_a, opacity=0.6, marker_color='#636EFA'))
            fig_dist.add_trace(go.Histogram(x=s_b, name=item_b, opacity=0.6, marker_color='#EF553B'))
            fig_dist.update_layout(barmode='overlay', title="CTR 성과 사후 분포 (몬테카를로)",
                                xaxis_title="추정 CTR", yaxis_title="빈도")
            st.plotly_chart(fig_dist, use_container_width=True)

            # 4. 추이 그래프
            st.subheader("📈 성과 히스토리")
            trend_df = df[df['ID'].isin([item_a, item_b])]
            if not trend_df.empty:
                fig_line = px.line(trend_df, x='날짜', y='CTR(%)', color='ID', markers=True, title="일자별 CTR 변화")
                st.plotly_chart(fig_line, use_container_width=True)