import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# statsmodels 라이브러리 체크
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

st.set_page_config(page_title="High-Velocity Product Analytics v27.0", layout="wide")

# --- [1. 데이터 엔진: 지표 계산식 확장] ---
def load_and_clean_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file)
        
        df.columns = [c.strip() for c in df.columns]
        mapping = {
            '날짜': ['날짜', '일자'], '상품': ['상품명', '상품'], '소재': ['소재명', '소재'],
            '노출': ['노출수', '노출'], '클릭': ['클릭수', '클릭'], '조회': ['조회수', '조회'], '비용': ['비용', '지출']
        }
        
        final_df = pd.DataFrame()
        for k, v in mapping.items():
            for col in v:
                if col in df.columns: final_df[k] = df[col]; break
        
        final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
        for col in ['노출', '클릭', '조회', '비용']:
            final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
        
        # [상품 중심 전처리 및 단가 지표 계산]
        final_df['상품'] = final_df['상품'].astype(str).str.upper().str.strip()
        final_df['CTR(%)'] = (final_df['클릭'] / (final_df['노출'] + 1e-9) * 100)
        final_df['VTR(%)'] = (final_df['조회'] / (final_df['노출'] + 1e-9) * 100)
        final_df['CPC'] = (final_df['비용'] / (final_df['클릭'] + 1e-9))
        final_df['CPM'] = (final_df['비용'] / (final_df['노출'] + 1e-9) * 1000)
        final_df['ID'] = "[" + final_df['상품'] + "] " + final_df['소재'].astype(str)
        
        return final_df.dropna(subset=['날짜']).sort_values('날짜')
    except Exception as e:
        st.error(f"데이터 로드 에러: {e}"); return pd.DataFrame()

# --- [2. 가속도 및 경고 로직] ---
def get_vel_with_alert(data, target_col):
    if len(data) < 5: return None, 0, "데이터 부족"
    y, x = data[target_col].astype(float).values, np.arange(len(data)).astype(float)
    
    if HAS_STATSMODELS:
        try:
            f = lowess(y, x, frac=0.5)
            v = (f[-1, 1] - f[-3, 1]) / 2 if len(f) >= 3 else 0
            # [논리적 임계치 적용]
            if v < -0.01: status = "🔴 교체 검토 (급락)"
            elif v < 0: status = "🟡 주의 (하락세 시작)"
            else: status = "🟢 양호 (상승/유지)"
            return f, v, status
        except: pass
    return None, 0, "계산 불가"

# --- [3. UI 레이아웃] ---
uploaded_file = st.file_uploader("캠페인 데이터를 업로드하세요", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    if not df.empty:
        ids = sorted(df['ID'].unique())
        tabs = st.tabs(["📊 성과 대시보드", "⚖️ 소재 유의성 진단", "📈 성과 가속도 분석", "🎯 예산 재배분 제안"])

        with tabs[0]:
            st.info("**[가이드]** 상품별 물량 비중과 효율 단가를 비교합니다. 좌측에서 노출/클릭/비용 비중을 선택하여 상품별 점유율을 확인하세요.")
            c1, c2 = st.columns(2)
            # 좌측 원형 그래프 (지표 선택 추가)
            pie_m = c1.selectbox("비중 지표 선택", ["비용", "노출", "클릭"])
            c1.plotly_chart(px.pie(df.groupby('상품')[pie_m].sum().reset_index(), values=pie_m, names='상품', hole=0.4, title=f"상품별 {pie_m} 총합 비중"), use_container_width=True)
            # 우측 막대 그래프 (단가 지표 추가)
            bar_m = c2.selectbox("효율 지표 선택", ['CTR(%)', 'CPC', 'CPM', 'VTR(%)'])
            c2.plotly_chart(px.bar(df.groupby('상품')[bar_m].mean().reset_index(), x='상품', y=bar_m, title=f"상품별 평균 {bar_m}"), use_container_width=True)

        with tabs[1]:
            st.info("**[가이드]** 소재 간 우열을 베이지안 확률로 계산합니다. 곡선이 겹치지 않을수록 우열이 명확합니다.")
            sc1, sc2 = st.columns(2)
            s_a, s_b = sc1.selectbox("소재 A", ids, index=0), sc2.selectbox("소재 B", ids, index=min(1, len(ids)-1))
            mode = st.radio("비교 지표", ["CTR(클릭)", "VTR(조회)"]) if df['조회'].sum() > 0 else "CTR(클릭)"
            t_col, d_col = ('클릭', '노출') if "CTR" in mode else ('조회', '노출')
            fig = go.Figure()
            for s, color in zip([s_a, s_b], ['#3498db', '#e74c3c']):
                sub = df[df['ID']==s][[t_col, d_col]].sum()
                dist = np.random.beta(sub[t_col]+1, sub[d_col]-sub[t_col]+1, 5000)
                fig.add_trace(go.Histogram(x=dist, name=s, marker_color=color, opacity=0.6))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            st.info("### 📈 가속도(Velocity)란?\n성과가 정점을 찍고 내려오는 **'피로도'**를 감지하는 지표입니다. 0보다 작아지면 소재가 타겟에게 질리기 시작했다는 신호입니다.")
            target_id = st.selectbox("상품 선택", ids)
            t_df = df[df['ID']==target_id]
            sel_m2 = st.selectbox("분석 지표", ['CTR(%)', 'VTR(%)'] if t_df['조회'].sum() > 0 else ['CTR(%)'])
            
            trend, vel, status = get_vel_with_alert(t_df, sel_m2)
            if trend is not None:
                c_v1, c_v2 = st.columns(2)
                c_v1.metric("현재 가속도", f"{vel:.4f}")
                c_v2.subheader(f"진단 결과: {status}") # 가속도 기반 상태 표시
                
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=t_df['날짜'], y=t_df[sel_m2], mode='markers', name="실제값"))
                fig_acc.add_trace(go.Scatter(x=t_df['날짜'], y=trend[:, 1], name="추세선(LOESS)", line=dict(color='red', width=2)))
                st.plotly_chart(fig_acc, use_container_width=True)

        with tabs[3]:
            st.info("**[가이드]** 가속도에 따라 예산을 재배분합니다. 모든 금액은 **만원 단위로 절삭**되며 잔액은 1위 상품에 합산됩니다.")
            if st.button("최적 예산안 산출"):
                last_3d = df[df['날짜'] > df['날짜'].max() - timedelta(days=3)]
                total_orig, results = 0, []
                for i in ids:
                    curr = last_3d[last_3d['ID']==i]['비용'].mean()
                    if curr > 0:
                        total_orig += curr
                        _, v, _ = get_vel_with_alert(df[df['ID']==i], 'CTR(%)')
                        # 가속도 가중치 적용 (±20% 범위 내)
                        prop = int(round((curr * (1 + np.clip(v * 20, -0.2, 0.2))) / 10000) * 10000)
                        results.append({'ID': i, '현재평균': curr, '가속도': v, '제안예산': prop})
                res_df = pd.DataFrame(results)
                if not res_df.empty:
                    diff = total_orig - res_df['제안예산'].sum()
                    if abs(diff) >= 10000: res_df.at[res_df['가속도'].idxmax(), '제안예산'] += (diff // 10000) * 10000
                    st.table(res_df.style.format({'현재평균':'{:,.0f}', '제안예산':'{:,.0f}', '가속도':'{:.4f}'}))