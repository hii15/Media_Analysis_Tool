import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# 라이브러리 체크
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

st.set_page_config(page_title="Product Marketing Intelligence v28.0", layout="wide")

# --- [1. 데이터 엔진] ---
def load_and_clean_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file)
        
        df.columns = [c.strip() for c in df.columns]
        mapping = {'날짜':['날짜','일자'], '상품':['상품명','상품'], '소재':['소재명','소재'],
                   '노출':['노출수','노출'], '클릭':['클릭수','클릭'], '조회':['조회수','조회'], '비용':['비용','지출']}
        
        final_df = pd.DataFrame()
        for k, v in mapping.items():
            for col in v:
                if col in df.columns: final_df[k] = df[col]; break
        
        final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
        for col in ['노출', '클릭', '조회', '비용']:
            final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
        
        final_df['상품'] = final_df['상품'].astype(str).str.upper().str.strip()
        final_df['CTR(%)'] = (final_df['클릭'] / (final_df['노출'] + 1e-9) * 100)
        final_df['VTR(%)'] = (final_df['조회'] / (final_df['노출'] + 1e-9) * 100)
        final_df['CPC'] = (final_df['비용'] / (final_df['클릭'] + 1e-9))
        final_df['CPM'] = (final_df['비용'] / (final_df['노출'] + 1e-9) * 1000)
        final_df['ID'] = "[" + final_df['상품'] + "] " + final_df['소재'].astype(str)
        
        return final_df.dropna(subset=['날짜']).sort_values('날짜')
    except Exception as e:
        st.error(f"데이터 로드 에러: {e}"); return pd.DataFrame()

# --- [2. 핵심 분석 로직 (가속도 및 경고)] ---
def get_vel_with_alert(data, target_col):
    if len(data) < 5: return None, 0, "데이터 부족"
    y, x = data[target_col].astype(float).values, np.arange(len(data)).astype(float)
    if HAS_STATSMODELS:
        try:
            f = lowess(y, x, frac=0.5)
            v = (f[-1, 1] - f[-3, 1]) / 2 if len(f) >= 3 else 0
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
        tabs = st.tabs(["📊 성과 대시보드", "⚖️ 소재 유의성 진단", "📈 성과 가속도 분석", "🎯 예산 재배분 제안", "🧪 사후 검증(Backtest)"])

        # [기존 탭들은 v27.0 로직 유지]
        with tabs[0]:
            st.info("**[가이드]** 상품별 물량 비중과 효율 단가를 비교합니다.")
            c1, c2 = st.columns(2)
            pie_m = c1.selectbox("비중 지표 선택", ["비용", "노출", "클릭"])
            c1.plotly_chart(px.pie(df.groupby('상품')[pie_m].sum().reset_index(), values=pie_m, names='상품', hole=0.4), use_container_width=True)
            bar_m = c2.selectbox("효율 지표 선택", ['CTR(%)', 'CPC', 'CPM', 'VTR(%)'])
            c2.plotly_chart(px.bar(df.groupby('상품')[bar_m].mean().reset_index(), x='상품', y=bar_m), use_container_width=True)

        with tabs[1]:
            st.info("**[가이드]** 베이지안 확률 기반 소재 우열 진단")
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
            st.info("**[가이드]** 가속도가 0보다 작아지면 소재 피로도가 시작된 신호입니다.")
            target_id = st.selectbox("상품 선택", ids)
            t_df = df[df['ID']==target_id]
            sel_m2 = st.selectbox("분석 지표", ['CTR(%)', 'VTR(%)'] if t_df['조회'].sum() > 0 else ['CTR(%)'])
            trend, vel, status = get_vel_with_alert(t_df, sel_m2)
            if trend is not None:
                st.metric("현재 가속도", f"{vel:.4f}", delta=status)
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=t_df['날짜'], y=t_df[sel_m2], mode='markers', name="실제값"))
                fig_acc.add_trace(go.Scatter(x=t_df['날짜'], y=trend[:, 1], name="추세선", line=dict(color='red')))
                st.plotly_chart(fig_acc, use_container_width=True)

        with tabs[3]:
            st.info("**[가이드]** 가속도 기반 만원 단위 절삭 예산안")
            if st.button("예산안 계산"):
                last_3d = df[df['날짜'] > df['날짜'].max() - timedelta(days=3)]
                results = []
                for i in ids:
                    curr = last_3d[last_3d['ID']==i]['비용'].mean()
                    if curr > 0:
                        _, v, _ = get_vel_with_alert(df[df['ID']==i], 'CTR(%)')
                        prop = int(round((curr * (1 + np.clip(v * 20, -0.2, 0.2))) / 10000) * 10000)
                        results.append({'ID': i, '현재평균': curr, '가속도': v, '제안예산': prop})
                res_df = pd.DataFrame(results)
                st.table(res_df.style.format({'현재평균':'{:,.0f}', '제안예산':'{:,.0f}', '가속도':'{:.4f}'}))

        # --- [신규 Tab 5: 사후 검증 로직 통합] ---
        with tabs[4]:
            st.info("### 🕵️ 가속도 모델의 예측력 검증 (Backtesting)\n전체 데이터를 절반으로 나눠, **전반기의 가속도**가 **후반기의 실제 성과 변화**를 얼마나 맞혔는지 측정합니다.")
            
            # 시간순 분할
            min_d, max_d = df['날짜'].min(), df['날짜'].max()
            mid_d = min_d + (max_d - min_d) / 2
            train_df = df[df['날짜'] <= mid_d]
            test_df = df[df['날짜'] > mid_d]
            
            bt_list = []
            for i in ids:
                tr_sub, te_sub = train_df[train_df['ID']==i], test_df[test_df['ID']==i]
                if len(tr_sub) >= 5 and len(te_sub) >= 3:
                    _, v, _ = get_vel_with_alert(tr_sub, 'CTR(%)')
                    actual_diff = te_sub['CTR(%)'].mean() - tr_sub['CTR(%)'].mean()
                    # 예측 적중 논리: (가속도 + 성과 +) OR (가속도 - 성과 -)
                    is_hit = "✅ 적중" if v * actual_diff > 0 else "❌ 빗나감"
                    bt_list.append({'상품소재': i, '전반기 가속도': v, '후반기 성과변화': actual_diff, '결과': is_hit})
            
            if bt_list:
                bt_df = pd.DataFrame(bt_list)
                h_rate = (bt_df['결과'] == "✅ 적중").mean() * 100
                
                c_bt1, c_bt2 = st.columns([1, 2])
                c_bt1.metric("모델 적중률", f"{h_rate:.1f}%")
                c_bt1.write(f"**학습 기간**: {min_d.date()} ~ {mid_d.date()}")
                c_bt1.write(f"**검증 기간**: {(mid_d+timedelta(days=1)).date()} ~ {max_d.date()}")
                
                fig_bt = px.scatter(bt_df, x='전반기 가속도', y='후반기 성과변화', color='결과', 
                                    hover_name='상품소재', title="예측(가속도) vs 실제 성과 변화")
                fig_bt.add_hline(y=0, line_dash="dash"); fig_bt.add_vline(x=0, line_dash="dash")
                c_bt2.plotly_chart(fig_bt, use_container_width=True)
                st.table(bt_df.style.format({'전반기 가속도':'{:.4f}', '후반기 성과변화':'{:.4f}'}))
            else:
                st.warning("사후 검증을 수행하기에 데이터 기간이 너무 짧습니다. (최소 10일 이상의 데이터 권장)")