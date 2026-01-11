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
        # 숫자가 아닌 문자(콤마 등) 제거 후 수치화
        final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    
    # 지표 계산
    final_df['CTR(%)'] = np.where(final_df['노출수'] > 0, (final_df['클릭수'] / final_df['노출수'] * 100), 0.0)
    final_df['ID'] = "[" + final_df['상품명'].astype(str) + "] " + final_df['소재명'].astype(str)
    
    return final_df.dropna(subset=['날짜']), None

def ml_forecast(data):
    if len(data) < 5: return None, None
    y = data['CTR(%)'].values
    x = np.arange(len(y)).reshape(-1, 1)
    model = HuberRegressor()
    try:
        model.fit(x, y)
        forecast = model.predict(np.arange(len(y), len(y) + 7).reshape(-1, 1))
        future_dates = [data['날짜'].max() + timedelta(days=i) for i in range(1, 8)]
        return future_dates, forecast
    except:
        return None, None

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
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "💎 상품별 요약 & 예산 가이드", 
            "🔍 전체 소재 성과 리포트", 
            "⚖️ 소재간 베이지안 진단", 
            "📈 머신러닝 수명 예측"
        ])

        with tab1:
            st.header("🏢 상품별 통합 성과")
            p_sum = full_df.groupby('상품명').agg({'노출수':'sum', '클릭수':'sum', '비용':'sum'}).reset_index()
            p_sum['CTR(%)'] = (p_sum['클릭수'] / p_sum['노출수'] * 100).fillna(0)
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.plotly_chart(px.pie(p_sum, values='비용', names='상품명', title="상품별 예산 비중"), use_container_width=True)
            with col_p2:
                # 효율성 점수: 클릭당 비용의 역수 개념 활용
                p_sum['효율성점수'] = (p_sum['CTR(%)'] / (p_sum['비용'] / p_sum['노출수'].replace(0, 1))).fillna(0)
                st.plotly_chart(px.bar(p_sum, x='상품명', y='효율성점수', title="상품별 예산 효율성"), use_container_width=True)

        with tab2:
            st.header("🔍 모든 상품/소재 성과 일람")
            total_summary = full_df.groupby(['ID', '매체']).agg({'노출수': 'sum', '클릭수': 'sum', '비용': 'sum'}).reset_index()
            total_summary['CTR(%)'] = (total_summary['클릭수'] / total_summary['노출수'] * 100).fillna(0)
            total_summary['CPC'] = (total_summary['비용'] / total_summary['클릭수']).replace([np.inf, -np.inf], 0).fillna(0)
            total_summary['CPM'] = (total_summary['비용'] / total_summary['노출수'] * 1000).replace([np.inf, -np.inf], 0).fillna(0)
            
            # 스타일 에러 방지를 위한 예외 처리형 출력
            try:
                st.dataframe(
                    total_summary.style.background_gradient(cmap='Blues', subset=['CTR(%)'])
                    .format({'비용': '{:,.0f}', 'CPC': '{:,.1f}', 'CPM': '{:,.1f}', 'CTR(%)': '{:.2f}%'}),
                    use_container_width=True
                )
            except:
                st.dataframe(total_summary, use_container_width=True) # 스타일 오류 시 일반 표 출력

        with tab3:
            st.header("⚖️ 소재간 베이지안 우열 진단")
            c_sel1, c_sel2 = st.columns(2)
            sel_a = c_sel1.selectbox("기준 소재 (A)", ids, index=0, key="b_a")
            sel_b = c_sel2.selectbox("비교 소재 (B)", ids, index=min(1, len(ids)-1), key="b_b")
            
            df_a, df_b = full_df[full_df['ID']==sel_a], full_df[full_df['ID']==sel_b]
            s_a, s_b = df_a[['노출수','클릭수']].sum(), df_b[['노출수','클릭수']].sum()
            
            if s_a['노출수'] > 0 and s_b['노출수'] > 0:
                dist_a = np.random.beta(s_a['클릭수']+1, s_a['노출수']-s_a['클릭수']+1, 10000)
                dist_b = np.random.beta(s_b['클릭수']+1, s_b['노출수']-s_b['클릭수']+1, 10000)
                prob_b_win = (dist_b > dist_a).mean()
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=dist_a, name=f"A: {sel_a}", marker_color='blue', opacity=0.5))
                fig_dist.add_trace(go.Histogram(x=dist_b, name=f"B: {sel_b}", marker_color='red', opacity=0.5))
                st.plotly_chart(fig_dist, use_container_width=True)
                
                winner = sel_b if prob_b_win > 0.5 else sel_a
                win_p = prob_b_win if prob_b_win > 0.5 else 1 - prob_b_win
                st.success(f"🏆 최종 진단: **[{winner}]** 소재가 우수할 확률이 **{win_p*100:.1f}%**입니다.")
            else:
                st.warning("선택한 소재의 노출 데이터가 없습니다.")

        with tab4:
            st.header("📈 머신러닝 수명 예측")
            sel_target = st.selectbox("분석 대상 선택", ids, key="ml_target")
            target_df = full_df[full_df['ID']==sel_target].sort_values('날짜')
            
            f_dates, f_vals = ml_forecast(target_df)
            if f_dates is not None:
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(x=target_df['날짜'], y=target_df['CTR(%)'], name="현재 실적"))
                fig_ml.add_trace(go.Scatter(x=f_dates, y=f_vals, name="7일 예측", line=dict(dash='dash', color='red')))
                st.plotly_chart(fig_ml, use_container_width=True)
            else:
                st.warning("예측을 위한 데이터가 부족합니다 (최소 5일 이상 권장).")
    else:
        st.error("데이터를 찾을 수 없습니다.")