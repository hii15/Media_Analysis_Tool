import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime, timedelta

# 설정
st.set_page_config(page_title="High-Velocity Analytics v26", layout="wide")

# --- [1. 데이터 엔진: 상품/영상 지표 통합] ---
def load_and_process(uploaded_file):
    if uploaded_file.name.endswith('.xlsx'):
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
        df = pd.concat(all_sheets.values(), ignore_index=True)
    else:
        df = pd.read_csv(uploaded_file)
    
    df.columns = [c.strip() for c in df.columns]
    mapping = {
        '날짜': ['날짜', '일자'], '상품': ['상품명', '상품'], '소재': ['소재명', '소재'],
        '노출': ['노출수', '노출'], '클릭': ['클릭수', '클릭'], 
        '조회': ['조회수', '조회', 'View'], '비용': ['비용', '지출']
    }
    
    final_df = pd.DataFrame()
    for k, v in mapping.items():
        for col in v:
            if col in df.columns:
                final_df[k] = df[col]; break
    
    if '조회' not in final_df.columns: final_df['조회'] = 0
    final_df['날짜'] = pd.to_datetime(final_df['날짜'])
    for c in ['노출', '클릭', '조회', '비용']:
        final_df[c] = pd.to_numeric(final_df[c], errors='coerce').fillna(0)
    
    final_df['CTR(%)'] = (final_df['클릭'] / (final_df['노출'] + 1e-9) * 100)
    final_df['VTR(%)'] = (final_df['조회'] / (final_df['노출'] + 1e-9) * 100)
    final_df['ID'] = "[" + final_df['상품'].astype(str) + "] " + final_df['소재'].astype(str)
    return final_df.sort_values('날짜')

# --- [2. 트렌드 엔진: LOESS (단기 추세 최적화)] ---
def get_velocity_trend(data, target_col):
    if len(data) < 5: return None, 0
    
    # 딥러닝 대신 국소 회귀(LOESS)로 단기 흐름 파악
    y = data[target_col].values
    x = np.arange(len(y))
    # frac=0.4는 최근 데이터 비중을 높여 단기 변화에 민감하게 반응하게 함
    filtered = lowess(y, x, frac=0.4)
    
    current_val = filtered[-1, 1]
    prev_val = filtered[-3, 1] if len(filtered) > 3 else filtered[0, 1]
    velocity = (current_val - prev_val) / 2 # 가속도(기울기)
    
    return filtered, velocity

# --- [3. UI 레이아웃] ---
uploaded_file = st.file_uploader("캠페인 데이터를 업로드하세요", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_and_process(uploaded_file)
    ids = sorted(df['ID'].unique())
    tabs = st.tabs(["📊 성과 대시보드", "⚖️ 유의성 진단", "📈 성과 가속도", "🎯 예산 재배분"])

    # --- Tab 1: 팩트 중심 요약 ---
    with tabs[0]:
        st.markdown("### 📊 통합 성과 요약")
        st.caption("집행 기간 내 누적 데이터입니다. 상품별 비용 대비 효율을 평면적으로 비교합니다.")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(df.groupby('상품')['비용'].sum().reset_index(), values='비용', names='상품', hole=0.4, title="상품별 예산 비중"), use_container_width=True)
        with col2:
            metrics = ['CTR(%)']
            if df['조회'].sum() > 0: metrics.append('VTR(%)')
            sel_m = st.selectbox("지표 선택", metrics)
            st.plotly_chart(px.bar(df.groupby('상품')[sel_m].mean().reset_index(), x='상품', y=sel_m, color=sel_m), use_container_width=True)

    # --- Tab 2: 유의성 진단 (조회 지표 대응) ---
    with tabs[1]:
        st.markdown("### ⚖️ 소재 유의성 진단")
        st.caption("**Model**: Beta-Binomial (소량 데이터 최적화)")
        c1, c2 = st.columns(2)
        s1, s2 = c1.selectbox("소재 A", ids, index=0), c2.selectbox("소재 B", ids, index=min(1, len(ids)-1))
        
        # 조회 데이터 유무 확인 후 UI 분기
        v_sum = df[df['ID'].isin([s1, s2])]['조회'].sum()
        mode = st.radio("분석 지표", ["클릭(CTR)", "조회(VTR)"]) if v_sum > 0 else "클릭(CTR)"
        
        t_col, d_col = ('클릭', '노출') if "클릭" in mode else ('조회', '노출')
        
        for s, color in zip([s1, s2], ['#3498db', '#e74c3c']):
            sub = df[df['ID']==s][[t_col, d_col]].sum()
            dist = np.random.beta(sub[t_col]+1, sub[d_col]-sub[t_col]+1, 5000)
            st.plotly_chart(go.Figure(data=[go.Histogram(x=dist, name=s, marker_color=color, opacity=0.6)]), use_container_width=True)

    # --- Tab 3: 가속도 분석 (NeuralProphet 대체) ---
    with tabs[2]:
        st.markdown("### 📈 성과 가속도 분석")
        st.info("딥러닝 대신 LOESS 모델을 사용하여 단기 캠페인의 '상승/하락 흐름'을 포착합니다.")
        sel_id = st.selectbox("분석 대상", ids)
        target_df = df[df['ID']==sel_id]
        
        m_list = ['CTR(%)']
        if target_df['조회'].sum() > 0: m_list.append('VTR(%)')
        sel_m2 = st.selectbox("지표", m_list, key="v_m")
        
        trend_data, velocity = get_velocity_trend(target_df, sel_m2)
        if trend_data is not None:
            st.metric("현재 가속도", f"{velocity:.4f}", delta=f"{'상승' if velocity > 0 else '하락'}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target_df['날짜'], y=target_df[sel_m2], mode='markers', name="실제값"))
            fig.add_trace(go.Scatter(x=target_df['날짜'], y=trend_data[:, 1], name="추세선", line=dict(color='red', width=3)))
            st.plotly_chart(fig, use_container_width=True)

    # --- Tab 4: 실무형 예산 배분 ---
    with tabs[3]:
        st.markdown("### 🎯 가속도 기반 예산 재배분")
        st.caption("**Logic**: 최근 3일 가속도 가중치 + 만원 단위 절삭")
        if st.button("배분안 산출"):
            last_3d = df[df['날짜'] > df['날짜'].max() - timedelta(days=3)]
            res = []
            for i in ids:
                _, v = get_velocity_trend(df[df['ID']==i], 'CTR(%)')
                curr = last_3d[last_3d['ID']==i]['비용'].mean()
                if curr > 0:
                    # 가속도가 양수면 최대 20% 증액, 음수면 최대 20% 감액
                    weight = 1 + np.clip(v * 50, -0.2, 0.2)
                    proposed = round((curr * weight) / 10000) * 10000
                    res.append({'상품소재': i, '현재 일평균': curr, '가속도': v, '제안 예산': proposed})
            
            st.table(pd.DataFrame(res).style.format({'현재 일평균':'{:,.0f}', '제안 예산':'{:,.0f}'}))

# --- 하단 모델 설명 ---
st.markdown("---")
with st.expander("📝 v26 Short-Term Logic Guide"):
    st.markdown("""
    - **유의성 진단**: 베이지안 사후 분포를 사용하여, 데이터가 적은(노출 1,000회 미만) 단기 캠페인에서도 소재 우열을 판별합니다.
    - **가속도(LOESS)**: NeuralProphet이 학습하기엔 데이터가 너무 적으므로, 국소 회귀를 통해 최근 3~5일의 흐름에 민감하게 반응하는 추세선을 그립니다.
    - **예산 배분**: 먼 미래의 예측이 아니라, **"지금 잘 되고 있는가?"**에 집중합니다. 가속도가 붙은 소재에 예산을 집중하며, 모든 수치는 실무 가이드인 **만원 단위**로 제안됩니다.
    """)