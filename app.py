import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import HuberRegressor
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="Marketing Data Science Pro", layout="wide")

# --- [엔진: 데이터 통합 및 정제] ---
def clean_and_process(df):
    col_map_patterns = {
        '날짜': ['날짜', '일자', 'Date', 'Day', '일시'],
        '매체': ['매체', '채널', 'Media', 'Channel', 'Platform'],
        '상품명': ['상품명', '상품', 'Product', 'Campaign'],
        '소재명': ['소재명', '소재', 'Creative', 'AdName', 'Content'],
        '노출수': ['노출수', '노출', 'Imp', 'Impression'],
        '클릭수': ['클릭수', '클릭'],
        '비용': ['비용', '지출', 'Cost', 'Spend']
    }
    
    final_df = pd.DataFrame()
    for std_key, patterns in col_map_patterns.items():
        found_col = None
        for actual_col in df.columns:
            clean_actual = str(actual_col).strip().replace(" ", "").replace("_", "")
            if any(p in clean_actual for p in patterns):
                found_col = actual_col
                break
        if found_col:
            final_df[std_key] = df[found_col]
        else:
            return pd.DataFrame(), std_key

    final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
    for col in ['노출수', '클릭수', '비용']:
        final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    
    final_df['CTR(%)'] = np.where(final_df['노출수'] > 0, (final_df['클릭수'] / final_df['노출수'] * 100), 0.0)
    final_df['CPC'] = np.where(final_df['클릭수'] > 0, (final_df['비용'] / final_df['클릭수']), 0.0)
    final_df['CPM'] = np.where(final_df['노출수'] > 0, (final_df['비용'] / final_df['노출수'] * 1000), 0.0)
    final_df['ID'] = "[" + final_df['상품명'].astype(str) + "] " + final_df['소재명'].astype(str)
    
    return final_df.dropna(subset=['날짜']), None

def ml_forecast_advanced(data):
    if len(data) < 7: return None, None, None
    y = data['CTR(%)'].values
    x = np.arange(len(y)).reshape(-1, 1)
    
    model = HuberRegressor()
    try:
        model.fit(x, y)
        future_x = np.arange(len(y), len(y) + 7).reshape(-1, 1)
        forecast = model.predict(future_x)
        future_dates = [data['날짜'].max() + timedelta(days=i) for i in range(1, 8)]
        
        y_pred = model.predict(x)
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        reliability = 1 - (rmse / (np.mean(y) + 1e-6))
        return future_dates, forecast, max(0, min(reliability, 1))
    except:
        return None, None, None

# --- [UI 메인] ---
st.title("📊 마케팅 데이터 과학 통합 대시보드")

uploaded_file = st.file_uploader("파일을 업로드하세요 (xlsx, csv)", type=['xlsx', 'csv'])

if uploaded_file:
    all_dfs = []
    if uploaded_file.name.endswith('xlsx'):
        xl = pd.ExcelFile(uploaded_file)
        for sheet in xl.sheet_names:
            temp_df = pd.read_excel(uploaded_file, sheet_name=sheet)
            processed, _ = clean_and_process(temp_df)
            if not processed.empty: all_dfs.append(processed)
    else:
        try: raw_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        except: raw_df = pd.read_csv(uploaded_file, encoding='cp949')
        processed, _ = clean_and_process(raw_df)
        if not processed.empty: all_dfs.append(processed)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        ids = sorted(full_df['ID'].unique())
        
        tab1, tab2, tab3, tab4 = st.tabs(["💎 상품 요약", "🔍 전체 성과", "⚖️ 베이지안 진단", "📈 수명 예측 & 모델 가이드"])

        # TAB 1 & 2 로직 (이전과 동일)
        with tab1:
            st.header("🏢 상품별 통합 성과")
            p_sum = full_df.groupby('상품명').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum', 'CTR(%)':'mean'}).reset_index()
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(p_sum, values='비용', names='상품명', title="상품별 예산 비중"), use_container_width=True)
            p_sum['효율성'] = (p_sum['CTR(%)'] / (p_sum['비용'] / p_sum['노출수'].replace(0, 1))).fillna(0)
            c2.plotly_chart(px.bar(p_sum, x='상품명', y='효율성', title="예산 효율성 가이드"), use_container_width=True)

        with tab2:
            st.header("🔍 모든 소재 성과 리포트")
            sum_df = full_df.groupby(['ID', '매체']).agg({'노출수':'sum', '클릭수':'sum', '비용':'sum'}).reset_index()
            sum_df['CTR(%)'] = (sum_df['클릭수'] / sum_df['노출수'] * 100).fillna(0)
            sum_df['CPC'] = (sum_df['비용'] / sum_df['클릭수']).replace([np.inf, -np.inf], 0).fillna(0)
            sum_df['CPM'] = (sum_df['비용'] / sum_df['노출수'] * 1000).replace([np.inf, -np.inf], 0).fillna(0)
            st.dataframe(sum_df.style.background_gradient(cmap='Blues', subset=['CTR(%)']).format({'비용':'{:,.0f}', 'CPC':'{:,.1f}', 'CPM':'{:,.1f}', 'CTR(%)':'{:.2f}%'}), use_container_width=True)

        # TAB 3: 베이지안 (그래프 설명 보강)
        with tab3:
            st.header("⚖️ 소재간 베이지안 우열 진단")
            st.markdown("""
            **📊 그래프 독해 가이드**
            - **X축(예측 CTR):** 오른쪽으로 갈수록 성과가 좋음을 의미합니다.
            - **Y축(확률 밀도):** 산이 높을수록 해당 데이터에 대한 확신이 강함을 의미합니다.
            - **두 산의 거리:** 산이 서로 겹치지 않고 멀수록, A와 B의 성과 차이는 우연이 아닌 '실력'일 가능성이 높습니다.
            """)
            c_sel1, c_sel2 = st.columns(2)
            sel_a = c_sel1.selectbox("기준 소재 (A)", ids, index=0, key="b_a")
            sel_b = c_sel2.selectbox("비교 소재 (B)", ids, index=min(1, len(ids)-1), key="b_b")
            
            df_a, df_b = full_df[full_df['ID']==sel_a], full_df[full_df['ID']==sel_b]
            s_a, s_b = df_a[['노출수','클릭수']].sum(), df_b[['노출수','클릭수']].sum()
            
            if s_a['노출수'] > 100 and s_b['노출수'] > 100:
                dist_a = np.random.beta(s_a['클릭수']+1, s_a['노출수']-s_a['클릭수']+1, 10000)
                dist_b = np.random.beta(s_b['클릭수']+1, s_b['노출수']-s_b['클릭수']+1, 10000)
                prob_b_win = (dist_b > dist_a).mean()
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=dist_a, name=f"A: {sel_a}", marker_color='blue', opacity=0.5))
                fig_dist.add_trace(go.Histogram(x=dist_b, name=f"B: {sel_b}", marker_color='red', opacity=0.5))
                st.plotly_chart(fig_dist, use_container_width=True)
                winner = sel_b if prob_b_win > 0.5 else sel_a
                win_p = prob_b_win if prob_b_win > 0.5 else 1 - prob_b_win
                st.success(f"🏆 최종 진단: **[{winner}]** 소재가 더 우수할 확률이 **{win_p*100:.1f}%**입니다.")
            else:
                st.warning("노출 데이터가 100회 미만입니다. 더 많은 데이터가 쌓인 후 분석하세요.")

        # TAB 4: 수명 예측 & 모델 상세 가이드 (고도화 통합)
        with tab4:
            st.header("📈 머신러닝 수명 예측 및 모델 진단")
            sel_target = st.selectbox("분석 대상 선택", ids, key="ml_target_v10")
            target_df = full_df[full_df['ID']==sel_target].sort_values('날짜')
            
            if len(target_df) >= 7:
                f_dates, f_vals, rel_score = ml_forecast_advanced(target_df)
                
                # 1. 메인 그래프
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(x=target_df['날짜'], y=target_df['CTR(%)'], name="실적", line=dict(color='#1f77b4')))
                fig_ml.add_trace(go.Scatter(x=f_dates, y=f_vals, name="7일 예측", line=dict(dash='dash', color='#d62728')))
                avg_ctr = target_df['CTR(%)'].mean()
                fig_ml.add_hline(y=avg_ctr * 0.8, line_dash="dot", line_color="orange", annotation_text="교체 권장선")
                st.plotly_chart(fig_ml, use_container_width=True)
                
                # 2. 지표 요약
                curr_ctr, pred_ctr = target_df['CTR(%)'].iloc[-1], f_vals[-1]
                diff_pct = (pred_ctr - curr_ctr) / curr_ctr * 100
                c_m1, c_m2, c_m3 = st.columns(3)
                c_m1.metric("현재 CTR", f"{curr_ctr:.2f}%")
                c_m2.metric("7일 후 예측", f"{pred_ctr:.2f}%", f"{diff_pct:.1f}%")
                c_m3.metric("모델 신뢰도", f"{rel_score*100:.1f}%")
                
                # 3. AI 진단 리포트
                st.divider()
                if diff_pct < -10:
                    st.error(f"🔴 **진단: 소재 피로도 발생** - 현재 하락 추세가 뚜렷합니다. 예측 신뢰도는 {rel_score*100:.1f}%이며, 교체를 권장합니다.")
                elif diff_pct > 10:
                    st.success(f"🟢 **진단: 성과 유지/상승** - 현재 수명이 충분히 남은 상태입니다. 운영 유지가 가능합니다.")
                else:
                    st.warning(f"🟡 **진단: 정체기 진입** - 성과가 박스권에 갇혀 있습니다. 신규 소재를 준비할 시점입니다.")

                # 4. 모델 가이드 (비교표 추가)
                st.subheader("🤖 데이터 과학 모델 가이드")
                st.write("현재 대시보드는 **Huber Regression**을 사용하여 이상치에 강한 수명 예측을 제공하고 있습니다.")
                
                m_compare = {
                    "모델명": ["Huber (현재)", "Prophet", "CausalImpact", "MMM (구글 Meridian)"],
                    "특징": ["이상치에 강함", "요일/계절성 반영", "광고 효과의 인과 분석", "전체 매체 기여도 분석"],
                    "필요 데이터": ["매체 데이터", "매체 데이터", "매체 + 실험 데이터", "매체 + 게임 BI 매출 데이터"]
                }
                st.table(m_compare)
                
                if rel_score < 0.6:
                    st.info("💡 **알림:** 현재 신뢰도가 낮게 나오는 이유는 데이터의 요일별 변동이 크기 때문일 수 있습니다. 웹툰 지면처럼 고정된 지면일수록 신뢰도는 높아지는 경향이 있습니다.")
            else:
                st.warning("최소 7일 이상의 데이터가 필요합니다.")