import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime, timedelta

# [UI 설정]
st.set_page_config(page_title="High-Velocity Product Analytics", layout="wide")

# --- [1. 데이터 엔진: 에러 원천 차단 로직] ---
def load_and_process_final(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file)
        
        df.columns = [c.strip() for c in df.columns]
        
        # [요청 반영] 상품 중심 파싱 매핑
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
        
        if '날짜' not in final_df.columns:
            st.error("데이터에서 '날짜' 컬럼을 찾을 수 없습니다.")
            return pd.DataFrame()

        # 데이터 클리닝 (쉼표 및 특수문자 제거)
        final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
        for c in ['노출', '클릭', '조회', '비용']:
            if c in final_df.columns:
                final_df[c] = pd.to_numeric(final_df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
            else:
                final_df[c] = 0
        
        # [핵심] 상품명 대문자 통일 및 ID 생성
        final_df['상품'] = final_df['상품'].astype(str).str.upper().str.strip()
        final_df['CTR(%)'] = (final_df['클릭'] / (final_df['노출'] + 1e-9) * 100)
        final_df['VTR(%)'] = (final_df['조회'] / (final_df['노출'] + 1e-9) * 100)
        final_df['ID'] = "[" + final_df['상품'] + "] " + final_df['소재'].astype(str)
        
        return final_df.dropna(subset=['날짜']).sort_values('날짜')
    except Exception as e:
        st.error(f"파일을 읽는 중 에러가 발생했습니다: {e}")
        return pd.DataFrame()

# --- [2. 가속도 엔진: 에러 방지형 LOESS] ---
def get_velocity_robust(data, target_col):
    if len(data) < 3: return None, 0 # 데이터가 너무 적으면 계산 불가
    
    y = data[target_col].values
    x = np.arange(len(y))
    
    try:
        if len(y) >= 7: # 데이터가 충분할 때만 LOESS 적용
            filtered = lowess(y, x, frac=0.5)
            velocity = (filtered[-1, 1] - filtered[-3, 1]) / 2
            return filtered, velocity
        else: # 데이터가 적으면 단순 선형 기울기 사용
            slope = (y[-1] - y[0]) / len(y)
            return np.column_stack((x, y)), slope
    except:
        return None, 0

# --- [3. UI 레이아웃] ---
st.title("📦 Product Marketing Intelligence System")

uploaded_file = st.file_uploader("캠페인 데이터 업로드 (CSV/XLSX)", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_and_process_final(uploaded_file)
    if not df.empty:
        ids = sorted(df['ID'].unique())
        tabs = st.tabs(["📊 성과 대시보드", "⚖️ 소재 유의성 진단", "📈 성과 가속도 분석", "🎯 예산 재배분 제안"])

        # --- Tab 1: 요약 ---
        with tabs[0]:
            st.subheader("📊 상품별 성과 요약")
            st.caption("모델: 원본 데이터 집계 (Raw Aggregation)")
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df.groupby('상품')['비용'].sum().reset_index(), values='비용', names='상품', hole=0.4, title="상품별 예산 비중"), use_container_width=True)
            
            m_list = ['CTR(%)']
            if df['조회'].sum() > 0: m_list.append('VTR(%)')
            sel_m1 = c2.selectbox("성과 지표 선택", m_list)
            c2.plotly_chart(px.bar(df.groupby('상품')[sel_m1].mean().reset_index(), x='상품', y=sel_m1, title=f"상품별 {sel_m1}"), use_container_width=True)

        # --- Tab 2: 유의성 (영상 대응) ---
        with tabs[1]:
            st.subheader("⚖️ 소재별 유의성 진단")
            st.caption("모델: 베이지안 분포 비교 (Beta-Binomial)")
            sc1, sc2 = st.columns(2)
            s_a, s_b = sc1.selectbox("소재 A", ids, index=0), sc2.selectbox("소재 B", ids, index=min(1, len(ids)-1))
            
            v_sum = df[df['ID'].isin([s_a, s_b])]['조회'].sum()
            mode = st.radio("분석 지표", ["CTR(클릭)", "VTR(조회)"]) if v_sum > 0 else "CTR(클릭)"
            t_c, d_c = ('클릭', '노출') if "CTR" in mode else ('조회', '노출')

            fig = go.Figure()
            for s, color in zip([s_a, s_b], ['#3498db', '#e74c3c']):
                sub = df[df['ID']==s][[t_c, d_c]].sum()
                dist = np.random.beta(sub[t_c]+1, sub[d_c]-sub[t_c]+1, 5000)
                fig.add_trace(go.Histogram(x=dist, name=s, marker_color=color, opacity=0.6))
            st.plotly_chart(fig, use_container_width=True)

        # --- Tab 3: 가속도 ---
        with tabs[2]:
            st.subheader("📈 성과 가속도 분석")
            st.caption("모델: 국소 회귀 (LOESS) - 단기 추세 포착용")
            target_id = st.selectbox("상품 선택", ids)
            t_df = df[df['ID']==target_id]
            
            m_list2 = ['CTR(%)']
            if t_df['조회'].sum() > 0: m_list2.append('VTR(%)')
            sel_m2 = st.selectbox("지표 선택", m_list2, key="acc_m")
            
            trend, vel = get_velocity_robust(t_df, sel_m2)
            if trend is not None:
                st.metric("현재 가속도", f"{vel:.4f}", delta=f"{'상승' if vel > 0 else '하락'}")
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=t_df['날짜'], y=t_df[sel_m2], mode='markers', name="실제값"))
                fig_acc.add_trace(go.Scatter(x=t_df['날짜'], y=trend[:, 1], name="추세선", line=dict(color='red', width=2)))
                st.plotly_chart(fig_acc, use_container_width=True)

        # --- Tab 4: 예산 (만원 절삭) ---
        with tabs[3]:
            st.subheader("🎯 예산 재배분 제안")
            st.caption("로직: 최근 3일 가속도 기반 가중치 부여 및 만원 단위 절삭")
            if st.button("최적 예산안 계산"):
                last_3d = df[df['날짜'] > df['날짜'].max() - timedelta(days=3)]
                total_orig = 0
                results = []
                
                for i in ids:
                    curr = last_3d[last_3d['ID']==i]['비용'].mean()
                    if curr > 0:
                        total_orig += curr
                        _, v = get_velocity_robust(df[df['ID']==i], 'CTR(%)')
                        weight = 1 + np.clip(v * 20, -0.2, 0.2)
                        prop = int(round((curr * weight) / 10000) * 10000)
                        results.append({'ID': i, '현재지출': curr, '가속도': v, '제안예산': prop})
                
                res_df = pd.DataFrame(results)
                if not res_df.empty:
                    # 절삭 차액 보정
                    diff = total_orig - res_df['제안예산'].sum()
                    if abs(diff) >= 10000:
                        res_df.at[res_df['가속도'].idxmax(), '제안예산'] += (diff // 10000) * 10000
                    st.table(res_df.style.format({'현재지출':'{:,.0f}', '제안예산':'{:,.0f}', '가속도':'{:.4f}'}))

# --- 하단 설명 ---
st.markdown("---")
with st.expander("📝 분석 가이드 및 모델 설명"):
    st.write("각 탭별 그래프와 수치는 상품명과 소재명을 결합한 고유 ID를 기반으로 계산됩니다.")
    st.write("모든 예산 제안은 만원 단위로 절삭되며, 성과 가속도가 가장 높은 상품에 잔여 예산이 우선 배정됩니다.")