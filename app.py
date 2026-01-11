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
    
    # 지표 계산 (CTR, CPC, CPM)
    final_df['CTR(%)'] = np.where(final_df['노출수'] > 0, (final_df['클릭수'] / final_df['노출수'] * 100), 0.0)
    final_df['CPC'] = np.where(final_df['클릭수'] > 0, (final_df['비용'] / final_df['클릭수']), 0.0)
    final_df['CPM'] = np.where(final_df['노출수'] > 0, (final_df['비용'] / final_df['노출수'] * 1000), 0.0)
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
st.title("📊 마케팅 데이터 통합 대시보드")

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
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "💎 상품별 요약 & 예산 가이드", 
            "🔍 전체 소재 성과 리포트", 
            "⚖️ 소재간 베이지안 진단", 
            "📈 머신러닝 수명 예측"
        ])

        # --- TAB 1: 상품별 요약 ---
        with tab1:
            st.header("🏢 상품별 통합 성과")
            p_sum = full_df.groupby('상품명').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum', 'CTR(%)':'mean'}).reset_index()
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.plotly_chart(px.pie(p_sum, values='비용', names='상품명', title="상품별 예산 비중"), use_container_width=True)
            with col_p2:
                p_sum['효율성점수'] = (p_sum['CTR(%)'] / (p_sum['비용'] / p_sum['노출수'])).fillna(0)
                st.plotly_chart(px.bar(p_sum, x='상품명', y='효율성점수', title="상품별 예산 효율성 (높을수록 증액 권장)"), use_container_width=True)

        # --- TAB 2: 전체 소재 성과 리포트 (통합 뷰) ---
        with tab2:
            st.header("🔍 모든 상품/소재 성과 일람")
            total_summary = full_df.groupby(['ID', '매체']).agg({
                '노출수': 'sum', '클릭수': 'sum', '비용': 'sum'
            }).reset_index()
            
            # 파생 지표 재계산
            total_summary['CTR(%)'] = (total_summary['클릭수'] / total_summary['노출수'] * 100).fillna(0)
            total_summary['CPC'] = (total_summary['비용'] / total_summary['클릭수']).replace([np.inf, -np.inf], 0).fillna(0)
            total_summary['CPM'] = (total_summary['비용'] / total_summary['노출수'] * 1000).replace([np.inf, -np.inf], 0).fillna(0)
            
            st.dataframe(
                total_summary.style.background_gradient(cmap='Blues', subset=['CTR(%)'])
                .format({'비용': '{:,.0f}', 'CPC': '{:,.1f}', 'CPM': '{:,.1f}', 'CTR(%)': '{:.2f}%'}),
                use_container_width=True
            )
            st.caption("💡 모든 시트의 데이터를 통합한 결과입니다. CTR이 높고 CPC/CPM이 낮은 소재를 발굴하세요.")

        # --- TAB 3: 베이지안 승률 분석 ---
        with tab3:
            st.header("⚖️ 소재간 베이지안 우열 진단")
            c_sel1, c_sel2 = st.columns(2)
            sel_a = c_sel1.selectbox("기준 소재 (A)", ids, index=0, key="b_a")
            sel_b = c_sel2.selectbox("비교 소재 (B)", ids, index=min(1, len(ids)-1), key="b_b")
            
            df_a, df_b = full_df[full_df['ID']==sel_a], full_df[full_df['ID']==sel_b]
            s_a, s_b = df_a[['노출수','클릭수']].sum(), df_b[['노출수','클릭수']].sum()
            
            dist_a = np.random.beta(s_a['클릭수']+1, s_a['노출수']-s_a['클릭수']+1, 10000)
            dist_b = np.random.beta(s_b['클릭수']+1, s_b['노출수']-s_b['클릭수']+1, 10000)
            
            prob_b_win = (dist_b > dist_a).mean()
            prob_a_win = 1 - prob_b_win
            winner, winner_prob, loser = (sel_a, prob_a_win, sel_b) if prob_a_win > prob_b_win else (sel_b, prob_b_win, sel_a)

            st.plotly_chart(go.Figure(data=[
                go.Histogram(x=dist_a, name=f"A: {sel_a}", opacity=0.5, marker_color='blue'),
                go.Histogram(x=dist_b, name=f"B: {sel_b}", opacity=0.5, marker_color='red')
            ]).update_layout(barmode='overlay', title="CTR 확률 분포 대조"), use_container_width=True)
            
            st.success(f"🏆 **최종 진단:** **[{winner}]** 소재가 **[{loser}]**보다 우수할 확률이 **{winner_prob*100:.1f}%**입니다.")

        # --- TAB 4: 머신러닝 성과 예측 ---
        with tab4:
            st.header("📈 성과 추세 및 미래 수명 예측")
            sel_target = st.selectbox("분석 대상 선택", ids, key="ml_target")
            target_df = full_df[full_df['ID']==sel_target].sort_values('날짜')
            
            if len(target_df) >= 7:
                f_dates, f_vals = ml_forecast(target_df)
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(x=target_df['날짜'], y=target_df['CTR(%)'], name="현재 실적", line=dict(color='black')))
                fig_ml.add_trace(go.Scatter(x=f_dates, y=f_vals, name="7일 뒤 예측", line=dict(dash='dash', color='red')))
                st.plotly_chart(fig_ml, use_container_width=True)
                
                curr_ctr, pred_ctr = target_df['CTR(%)'].iloc[-1], f_vals[-1]
                if pred_ctr < curr_ctr * 0.85:
                    st.error(f"🚨 **피로도 주의:** 성과가 {(1-pred_ctr/curr_ctr)*100:.1f}% 하락할 것으로 보입니다. 소재 교체를 검토하세요.")
                else:
                    st.success("✅ **추세 양호:** 현재 소재의 성과 흐름이 안정적입니다.")
            else:
                st.warning("예측을 위해 7일 이상의 데이터가 필요합니다.")

    else:
        st.error("데이터 로드 실패. 파일 형식 및 컬럼명을 확인하세요.")