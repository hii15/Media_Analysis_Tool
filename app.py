import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime, timedelta

# [UI 설정]
st.set_page_config(page_title="High-Velocity Product Analytics v26.6", layout="wide")

# --- [1. 데이터 엔진: 상품/영상 통합 및 강력한 클리닝] ---
def load_and_clean_data(uploaded_file):
    try:
        # XLSX/CSV 통합 로드
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file)
        
        # 컬럼명 앞뒤 공백 제거
        df.columns = [c.strip() for c in df.columns]
        
        # [요청 반영] 상품 중심 파싱을 위한 매핑
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
        
        # 날짜 정제
        final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
        final_df = final_df.dropna(subset=['날짜'])
        
        # 숫자 데이터 강력 클리닝 (쉼표 및 특수문자 제거 후 float 변환)
        for col in ['노출', '클릭', '조회', '비용']:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
            else:
                final_df[col] = 0
        
        # [요청 반영] 상품명 표준화 및 고유 ID 생성
        final_df['상품'] = final_df['상품'].astype(str).str.upper().str.strip()
        final_df['CTR(%)'] = (final_df['클릭'] / (final_df['노출'] + 1e-9) * 100)
        final_df['VTR(%)'] = (final_df['조회'] / (final_df['노출'] + 1e-9) * 100)
        final_df['ID'] = "[" + final_df['상품'] + "] " + final_df['소재'].astype(str)
        
        return final_df.sort_values('날짜')
    except Exception as e:
        st.error(f"데이터 로드 에러: {e}")
        return pd.DataFrame()

# --- [2. 가속도 엔진: LOESS 기반 추세 분석] ---
def calculate_velocity(data, target_col):
    if len(data) < 5: return None, 0
    y = data[target_col].astype(float).values
    x = np.arange(len(y)).astype(float)
    try:
        filtered = lowess(y, x, frac=0.5)
        velocity = (filtered[-1, 1] - filtered[-3, 1]) / 2 if len(filtered) >= 3 else 0
        return filtered, velocity
    except:
        return None, 0

# --- [3. 메인 UI 및 탭별 가이드 배치] ---
st.title("📦 Product Marketing Intelligence System")

uploaded_file = st.file_uploader("캠페인 데이터를 업로드하세요", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    if not df.empty:
        ids = sorted(df['ID'].unique())
        tabs = st.tabs(["📊 통합 성과 요약", "⚖️ 소재 유의성 진단", "📈 성과 가속도 분석", "🎯 예산 재배분 제안"])

        # --- Tab 1: 요약 ---
        with tabs[0]:
            st.info("### 💡 그래프 읽는 법\n**상품별 누적 실적**을 비교합니다. 왼쪽 파이 차트는 예산 지출 비중을, 오른쪽 막대 차트는 선택한 지표(CTR/VTR)의 효율을 상품별로 합산하여 보여줍니다.")
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df.groupby('상품')['비용'].sum().reset_index(), values='비용', names='상품', hole=0.4, title="상품별 비용 비중"), use_container_width=True)
            
            m_opts = ['CTR(%)']
            if df['조회'].sum() > 0: m_opts.append('VTR(%)')
            sel_m1 = c2.selectbox("성과 지표 선택", m_opts)
            c2.plotly_chart(px.bar(df.groupby('상품')[sel_m1].mean().reset_index(), x='상품', y=sel_m1, title=f"상품별 평균 {sel_m1}"), use_container_width=True)

        # --- Tab 2: 유의성 진단 ---
        with tabs[1]:
            st.info("### 💡 통계 모델: Beta-Binomial Bayesian\n두 소재 중 어떤 것이 성과가 좋은지 **확률적으로 판별**합니다. 히스토그램 곡선이 서로 겹치지 않을수록 결과가 확실하며, 오른쪽으로 치우친 곡선이 승자입니다.")
            sc1, sc2 = st.columns(2)
            s_a, s_b = sc1.selectbox("소재 A", ids, index=0), sc2.selectbox("소재 B", ids, index=min(1, len(ids)-1))
            
            # [요청 반영] 조회(View)가 있는 항목만 분석 분기
            has_view = df[df['ID'].isin([s_a, s_b])]['조회'].sum() > 0
            mode = st.radio("비교 기준", ["CTR(클릭)", "VTR(조회)"]) if has_view else "CTR(클릭)"
            t_col, d_col = ('클릭', '노출') if "CTR" in mode else ('조회', '노출')
            
            fig = go.Figure()
            for s, color in zip([s_a, s_b], ['#3498db', '#e74c3c']):
                sub = df[df['ID']==s][[t_col, d_col]].sum()
                dist = np.random.beta(sub[t_col]+1, sub[d_col]-sub[t_col]+1, 5000)
                fig.add_trace(go.Histogram(x=dist, name=s, marker_color=color, opacity=0.6))
            st.plotly_chart(fig, use_container_width=True)

        # --- Tab 3: 가속도 분석 ---
        with tabs[2]:
            st.info("### 💡 통계 모델: LOESS (Local Regression)\n단기 캠페인의 **상승/하락 흐름**을 읽습니다. 파란 점은 실제 일별 수치이며, 붉은 선은 무작위 변동을 제거한 추세선입니다. 선의 끝이 위를 향하면 가속도가 붙은 상태입니다.")
            target_id = st.selectbox("상품 선택", ids)
            t_df = df[df['ID']==target_id]
            
            m_opts2 = ['CTR(%)']
            if t_df['조회'].sum() > 0: m_opts2.append('VTR(%)')
            sel_m2 = st.selectbox("지표", m_opts2, key="acc_m")
            
            trend, vel = calculate_velocity(t_df, sel_m2)
            if trend is not None:
                st.metric("현재 가속도", f"{vel:.4f}", delta=f"{'상승' if vel > 0 else '하락'}")
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=t_df['날짜'], y=t_df[sel_m2], mode='markers', name="실제 실적"))
                fig_acc.add_trace(go.Scatter(x=t_df['날짜'], y=trend[:, 1], name="추세선(LOESS)", line=dict(color='red', width=2)))
                st.plotly_chart(fig_acc, use_container_width=True)

        # --- Tab 4: 예산 재배분 ---
        with tabs[3]:
            st.info("### 💡 예산 로직: Momentum-Based Reallocation\n최근 3일 성과 가속도에 따라 예산을 증감합니다. 모든 금액은 **10,000원 단위로 절삭**되며, 절삭 후 남는 예산은 성과가 가장 좋은 소재에 합산됩니다.")
            if st.button("예산 최적화 실행"):
                last_3d = df[df['날짜'] > df['날짜'].max() - timedelta(days=3)]
                total_orig_avg = 0
                results = []
                
                for i in ids:
                    curr_avg = last_3d[last_3d['ID']==i]['비용'].mean()
                    if curr_avg > 0:
                        total_orig_avg += curr_avg
                        _, v = calculate_velocity(df[df['ID']==i], 'CTR(%)')
                        weight = 1 + np.clip(v * 20, -0.2, 0.2)
                        # [요청 반영] 만원 단위 절삭
                        proposed = int(round((curr_avg * weight) / 10000) * 10000)
                        results.append({'상품소재': i, '현재일평균': curr_avg, '가속도': v, '제안예산': proposed})
                
                res_df = pd.DataFrame(results)
                if not res_df.empty:
                    # [요청 반영] 절삭 차액 보정 (총액 보존)
                    diff = total_orig_avg - res_df['제안예산'].sum()
                    if abs(diff) >= 10000:
                        res_df.at[res_df['가속도'].idxmax(), '제안예산'] += (diff // 10000) * 10000
                    st.table(res_df.style.format({'현재일평균':'{:,.0f}', '제안예산':'{:,.0f}', '가속도':'{:.4f}'}))