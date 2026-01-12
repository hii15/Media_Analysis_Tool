import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime, timedelta

# [UI 설정] 전문가용 용어 배제 및 비즈니스 인터페이스 구성
st.set_page_config(page_title="High-Velocity Product Analytics", layout="wide")

# --- [1. 데이터 엔진: 상품/영상 지표 통합 로직] ---
def load_and_process_comprehensive(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file)
        
        df.columns = [c.strip() for c in df.columns]
        
        # [요청 반영] 매체별이 아닌 상품별 파싱을 위한 매핑
        mapping = {
            '날짜': ['날짜', '일자', 'Date'],
            '상품': ['상품명', '상품', 'Product'],
            '소재': ['소재명', '소재', 'Creative'],
            '노출': ['노출수', '노출', 'Impression'],
            '클릭': ['클릭수', '클릭', 'Click'],
            '조회': ['조회수', '조회', 'View', '조회(View)'],
            '비용': ['비용', '지출', 'Cost']
        }
        
        final_df = pd.DataFrame()
        for k, v in mapping.items():
            for col in v:
                if col in df.columns:
                    final_df[k] = df[col]
                    break
        
        # [요청 반영] 영상 지표(View) 부재 시 자동 생성 및 데이터 정제
        if '조회' not in final_df.columns: final_df['조회'] = 0
        final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
        for c in ['노출', '클릭', '조회', '비용']:
            final_df[c] = pd.to_numeric(final_df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        # [핵심 로직] 상품 중심의 ID 생성
        final_df['CTR(%)'] = (final_df['클릭'] / (final_df['노출'] + 1e-9) * 100)
        final_df['VTR(%)'] = (final_df['조회'] / (final_df['노출'] + 1e-9) * 100)
        final_df['ID'] = "[" + final_df['상품'].astype(str).str.upper() + "] " + final_df['소재'].astype(str)
        
        return final_df.dropna(subset=['날짜']).sort_values('날짜')
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {e}")
        return pd.DataFrame()

# --- [2. 가속도 엔진: 단기 캠페인용 LOESS] ---
def get_velocity_analysis(data, target_col):
    if len(data) < 5: return None, 0
    y = data[target_col].values
    x = np.arange(len(y))
    # 단기 흐름을 잡기 위해 frac(평활도)을 0.4로 고정
    filtered = lowess(y, x, frac=0.4)
    # 최근 3일간의 변화량을 가속도로 정의
    velocity = (filtered[-1, 1] - filtered[-3, 1]) / 2 if len(filtered) > 3 else 0
    return filtered, velocity

# --- [3. 메인 UI 구성] ---
st.title("📦 Product Marketing Intelligence System")

uploaded_file = st.file_uploader("캠페인 데이터(CSV/XLSX)를 업로드하세요", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_and_process_comprehensive(uploaded_file)
    if not df.empty:
        ids = sorted(df['ID'].unique())
        tabs = st.tabs(["📊 통합 성과 요약", "⚖️ 소재 유의성 진단", "📈 성과 가속도 분석", "🎯 예산 재배분 제안"])

        # --- Tab 1: 통합 성과 (상품 중심) ---
        with tabs[0]:
            st.subheader("📊 상품별 성과 요약")
            st.markdown("**(모델 설명)**: 전체 캠페인 기간의 데이터를 상품별로 합산하여 원본 실적을 보여줍니다.")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.pie(df.groupby('상품')['비용'].sum().reset_index(), values='비용', names='상품', hole=0.4, title="상품별 예산 비중"), use_container_width=True)
            with c2:
                metrics = ['CTR(%)']
                if df['조회'].sum() > 0: metrics.append('VTR(%)')
                sel_m = st.selectbox("분석 지표 선택", metrics)
                st.plotly_chart(px.bar(df.groupby('상품')[sel_m].mean().reset_index(), x='상품', y=sel_m, title=f"상품별 평균 {sel_m}"), use_container_width=True)

        # --- Tab 2: 유의성 진단 (영상 지표 대응) ---
        with tabs[1]:
            st.subheader("⚖️ 소재별 유의성 진단")
            st.markdown("**(모델 설명)**: 베이지안(Beta-Binomial) 모델을 통해 소량의 데이터로도 소재 간 우열 확률을 계산합니다.")
            sc1, sc2 = st.columns(2)
            s_a, s_b = sc1.selectbox("소재 A", ids, index=0), sc2.selectbox("소재 B", ids, index=min(1, len(ids)-1))
            
            # [요청 반영] 조회(View)가 있는 소재는 VTR 분석 옵션 제공
            v_check = df[df['ID'].isin([s_a, s_b])]['조회'].sum()
            mode = st.radio("비교 지표", ["클릭(CTR)", "조회(VTR)"]) if v_check > 0 else "클릭(CTR)"
            t_col, d_col = ('클릭', '노출') if "클릭" in mode else ('조회', '노출')

            fig = go.Figure()
            for s, color in zip([s_a, s_b], ['#3498db', '#e74c3c']):
                sub = df[df['ID']==s][[t_col, d_col]].sum()
                dist = np.random.beta(sub[t_col]+1, sub[d_col]-sub[t_col]+1, 5000)
                fig.add_trace(go.Histogram(x=dist, name=s, marker_color=color, opacity=0.6))
            st.plotly_chart(fig, use_container_width=True)

        # --- Tab 3: 가속도 분석 (LOESS 적용) ---
        with tabs[2]:
            st.subheader("📈 성과 가속도 분석")
            st.markdown("**(모델 설명)**: 국소 회귀(LOESS) 모델을 사용하여 단기 캠페인 내 성과의 상승/하락 흐름을 포착합니다.")
            target_id = st.selectbox("분석 상품 선택", ids)
            t_df = df[df['ID']==target_id]
            
            m_opts = ['CTR(%)']
            if t_df['조회'].sum() > 0: m_opts.append('VTR(%)')
            sel_m2 = st.selectbox("분석 지표", m_opts, key="acc_m")
            
            trend, vel = get_velocity_analysis(t_df, sel_m2)
            if trend is not None:
                st.metric("현재 성과 가속도", f"{vel:.4f}", delta=f"{'상승' if vel > 0 else '하락'}")
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=t_df['날짜'], y=t_df[sel_m2], mode='markers', name="실제 실적"))
                fig_acc.add_trace(go.Scatter(x=t_df['날짜'], y=trend[:, 1], name="추세선(LOESS)", line=dict(color='red', width=2)))
                st.plotly_chart(fig_acc, use_container_width=True)

        # --- Tab