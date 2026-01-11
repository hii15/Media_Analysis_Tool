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
        '클릭수': ['클릭수', '클릭', 'Click'],
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
    final_df['ID'] = "[" + final_df['상품명'].astype(str) + "] " + final_df['소재명'].astype(str)
    
    return final_df.dropna(subset=['날짜']), None

def ml_forecast(data):
    if len(data) < 7: return None, None
    y = data['CTR(%)'].values
    x = np.arange(len(y)).reshape(-1, 1)
    model = HuberRegressor()
    model.fit(x, y)
    forecast = model.predict(np.arange(len(y), len(y) + 7).reshape(-1, 1))
    future_dates = [data['날짜'].max() + timedelta(days=i) for i in range(1, 8)]
    return future_dates, forecast

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
        
        # 탭 분리 구성
        tab1, tab2, tab3 = st.tabs(["💎 상품별 요약 & 예산 최적화", "⚖️ 베이지안 승률 분석", "📈 머신러닝 성과 예측"])

        # --- TAB 1: 상품별 요약 및 예산 최적화 ---
        with tab1:
            st.header("🏢 상품별 통합 성과 및 예산 가이드")
            p_sum = full_df.groupby('상품명').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum', 'CTR(%)':'mean'}).reset_index()
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.plotly_chart(px.pie(p_sum, values='비용', names='상품명', title="상품별 예산 비중"), use_container_width=True)
            with col_p2:
                # 예산 최적화 로직 (성과 대비 효율성)
                p_sum['효율성점수'] = (p_sum['CTR(%)'] / (p_sum['비용'] / p_sum['노출수'])).fillna(0)
                st.plotly_chart(px.bar(p_sum, x='상품명', y='효율성점수', title="상품별 예산 효율성 가이드 (높을수록 증액 권장)"), use_container_width=True)
            
            st.subheader("💡 예산 분배 전략 가이드")
            top_p = p_sum.sort_values('효율성점수', ascending=False).iloc[0]['상품명']
            st.info(f"현재 데이터 기준, **[{top_p}]** 상품의 비용 대비 클릭 전환 효율이 가장 높습니다. 해당 상품으로의 예산 점유율 확대를 검토하세요.")

        # --- TAB 2: 베이지안 승률 분석 ---
        with tab2:
            st.header("⚖️ 소재간 베이지안 우열 진단")
            ids = sorted(full_df['ID'].unique())
            c1, c2 = st.columns(2)
            sel_a = c1.selectbox("기준 소재 (A)", ids, index=0, key="b_a")
            sel_b = c2.selectbox("비교 소재 (B)", ids, index=min(1, len(ids)-1), key="b_b")
            
            df_a, df_b = full_df[full_df['ID']==sel_a], full_df[full_df['ID']==sel_b]
            s_a, s_b = df_a[['노출수','클릭수']].sum(), df_b[['노출수','클릭수']].sum()
            
            # 신뢰도 필터링 기능
            if s_a['노출수'] < 100 or s_b['노출수'] < 100:
                st.warning("⚠️ 노출 데이터가 너무 적어 통계적 신뢰도가 낮습니다. (최소 100회 이상 권장)")
            
            dist_a = np.random.beta(s_a['클릭수']+1, s_a['노출수']-s_a['클릭수']+1, 10000)
            dist_b = np.random.beta(s_b['클릭수']+1, s_b['노출수']-s_b['클릭수']+1, 10000)
            prob_b_win = (dist_b > dist_a).mean()
            
            # 논리적 근거 시각화: CTR 분포도
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=dist_a, name=f"{sel_a} 분포", marker_color='blue', opacity=0.6))
            fig_dist.add_trace(go.Histogram(x=dist_b, name=f"{sel_b} 분포", marker_color='red', opacity=0.6))
            fig_dist.update_layout(title="CTR 확률 분포 비교 (두 그래프가 멀수록 결과가 확실함)", barmode='overlay')
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.success(f"**결과 분석:** {sel_b} 소재가 {sel_a}보다 우수할 확률은 **{prob_b_win*100:.1f}%**입니다.")

        # --- TAB 3: 머신러닝 성과 예측 ---
        with tab3:
            st.header("📈 머신러닝 기반 수명 예측 및 추세")
            sel_target = st.selectbox("예측 대상 소재 선택", ids, key="ml_target")
            target_df = full_df[full_df['ID']==sel_target].sort_values('날짜')
            
            if len(target_df) >= 7:
                f_dates, f_vals = ml_forecast(target_df)
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(x=target_df['날짜'], y=target_df['CTR(%)'], name="실제 CTR"))
                fig_ml.add_trace(go.Scatter(x=f_dates, y=f_vals, name="머신러닝 예측(추세)", line=dict(dash='dash', color='red')))
                st.plotly_chart(fig_ml, use_container_width=True)
                
                # 수명 진단 로직
                last_ctr = target_df['CTR(%)'].iloc[-1]
                pred_ctr = f_vals[-1]
                if pred_ctr < last_ctr * 0.8:
                    st.error(f"🚨 **소재 피로도 경보:** 7일 내 성과가 {((1-pred_ctr/last_ctr)*100):.1f}% 하락할 것으로 예측됩니다. 소재 교체를 준비하세요.")
                else:
                    st.success("✅ **성과 유지 중:** 소재의 수명이 충분히 남은 것으로 판단됩니다.")
            else:
                st.warning("데이터가 7일 이상 축적되어야 머신러닝 추세 분석이 가능합니다.")

    else:
        st.error("데이터를 불러올 수 없습니다. 컬럼명을 확인해 주세요.")