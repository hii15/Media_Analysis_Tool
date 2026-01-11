import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from datetime import datetime, timedelta
import logging

# 시스템 설정
logging.getLogger('prophet').setLevel(logging.WARNING)
st.set_page_config(page_title="Product Marketing Intelligence", layout="wide")

# --- [1. 데이터 로드 및 상품 단위 통합] ---
def load_and_standardize(uploaded_file):
    if uploaded_file.name.endswith('.xlsx'):
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
        df = pd.concat(all_sheets.values(), ignore_index=True)
    else:
        df = pd.read_csv(uploaded_file)
    
    # 공백 제거 및 컬럼 매핑
    df.columns = [c.strip() for c in df.columns]
    mapping = {
        '날짜': ['날짜', '일자', 'Date'],
        '상품': ['상품명', '상품', 'Product'],
        '소재': ['소재명', '소재', 'Creative'],
        '노출': ['노출수', '노출', 'Impression'],
        '클릭': ['클릭수', '클릭', 'Click'],
        '비용': ['비용', '지출', 'Cost']
    }
    
    final_df = pd.DataFrame()
    for k, v in mapping.items():
        for col in v:
            if col in df.columns:
                final_df[k] = df[col]
                break
    
    final_df['날짜'] = pd.to_datetime(final_df['날짜'])
    for c in ['노출', '클릭', '비용']:
        final_df[c] = pd.to_numeric(final_df[c], errors='coerce').fillna(0)
    
    final_df['CTR(%)'] = (final_df['클릭'] / (final_df['노출'] + 1e-9) * 100)
    # 파싱 기준: 상품명과 소재명을 결합한 고유 ID 생성
    final_df['ID'] = "[" + final_df['상품'].astype(str) + "] " + final_df['소재'].astype(str)
    return final_df.dropna(subset=['날짜'])

# --- [2. 트렌드 예측 및 적합도 계산] ---
def get_trend_analysis(data):
    # 최소 14일 이상의 데이터 확보 및 변동성 확인
    if len(data) < 14 or data['CTR(%)'].std() < 0.01:
        return None, 0, 0
    
    try:
        df = data.groupby('날짜')['CTR(%)'].mean().reset_index().rename(columns={'날짜':'ds', 'CTR(%)':'y'})
        m = Prophet(interval_width=0.8, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(df)
        
        future = m.make_future_dataframe(periods=7)
        forecast = m.predict(future)
        
        # 모델 적합도 (R-Squared)
        y_true = df['y'].values
        y_pred = forecast.iloc[:len(y_true)]['yhat'].values
        r2 = 1 - (np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-9))
        
        # 최근 7일 추세 기울기
        slope = (forecast['yhat'].values[-1] - forecast['yhat'].values[-7]) / 7
        return forecast, slope, max(0, min(r2, 0.99))
    except:
        return None, 0, 0

# --- [3. 만원 단위 예산 배분 알고리즘] ---
def optimize_budget_rounded(base_df, total_budget):
    # 1. 성과 비례 가중치 계산 (기울기 기반)
    # 기울기가 높을수록(성과 상승 중) 더 많은 예산 배정
    base_df['weight'] = base_df['추세'].apply(lambda x: 1 + (x * 10) if x > 0 else 1 + (x * 5))
    base_df['weight'] = base_df['weight'].clip(lower=0.5) # 최소 유지비율 50%
    
    # 2. 1차 제안가 계산
    raw_proposal = base_df['현재지출'] * base_df['weight']
    
    # 3. 만원 단위 절삭 (실무 최적화)
    base_df['제안예산'] = (raw_proposal / 10000).round() * 10000
    
    # 4. 절삭 후 발생하는 차액(Residual) 처리
    current_total = base_df['제안예산'].sum()
    diff = total_budget - current_total
    
    if abs(diff) >= 10000:
        # 성과가 가장 좋은(기울기가 높은) 상품에 차액 몰아주기
        best_idx = base_df['추세'].idxmax()
        # 차액을 만원 단위로 보정하여 가산
        base_df.at[best_idx, '제안예산'] += (diff // 10000) * 10000
        
    return base_df

# --- [4. UI 메인 레이아웃] ---
st.title("📦 Product Marketing Analytics System")

uploaded_file = st.file_uploader("분석용 데이터를 업로드하세요 (Excel/CSV)", type=['csv', 'xlsx'])

if uploaded_file:
    full_df = load_and_standardize(uploaded_file)
    ids = sorted(full_df['ID'].unique())
    
    tabs = st.tabs(["📊 성과 대시보드", "⚖️ 성과 유의성 검정", "📈 트렌드 분석", "🎯 예산 재배분"])

    with tabs[0]:
        # 통합 데이터 시각화 (팩트 중심)
        st.markdown("### 전체 상품 집계 데이터")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(full_df.groupby('상품')['비용'].sum().reset_index(), 
                                   values='비용', names='상품', hole=0.4, title="상품별 비용 집행 비중"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(full_df.groupby('상품')['CTR(%)'].mean().reset_index(), 
                                   x='상품', y='CTR(%)', title="상품별 평균 CTR (%)"), use_container_width=True)

    with tabs[1]:
        st.markdown("### 소재별 승률 분석")
        # (기존 베이지안 비교 로직 유지 - 주석처리된 ID 기반 파싱)
        sc1, sc2 = st.columns(2)
        sel_a = sc1.selectbox("소재 A 선택", ids, index=0)
        sel_b = sc2.selectbox("소재 B 선택", ids, index=min(1, len(ids)-1))
        
        s_a = full_df[full_df['ID']==sel_a][['노출','클릭']].sum(numeric_only=True)
        s_b = full_df[full_df['ID']==sel_b][['노출','클릭']].sum(numeric_only=True)
        
        dist_a = np.random.beta(s_a['클릭']+1, s_a['노출']-s_a['클릭']+1, 5000)
        dist_b = np.random.beta(s_b['클릭']+1, s_b['노출']-s_b['클릭']+1, 5000)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=dist_a, name=sel_a, opacity=0.6, marker_color='#3498db'))
        fig.add_trace(go.Histogram(x=dist_b, name=sel_b, opacity=0.6, marker_color='#e74c3c'))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("### 시계열 트렌드 예측")
        sel_target = st.selectbox("분석 대상 선택", ids)
        f_data, f_slope, r2 = get_trend_analysis(full_df[full_df['ID']==sel_target])
        
        if f_data is not None:
            st.metric("예측 모델 적합도", f"{r2*100:.1f}%")
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=full_df[full_df['ID']==sel_target]['날짜'], y=full_df[full_df['ID']==sel_target]['CTR(%)'], mode='markers', name="실측값"))
            fig_f.add_trace(go.Scatter(x=f_data['ds'], y=f_data['yhat'], name="추세 예측", line=dict(color='red', dash='dash')))
            fig_f.add_trace(go.Scatter(x=f_data['ds'], y=f_data['yhat_upper'], line=dict(width=0), showlegend=False))
            fig_f.add_trace(go.Scatter(x=f_data['ds'], y=f_data['yhat_lower'], fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name="예측 범위"))
            st.plotly_chart(fig_f, use_container_width=True)
        else:
            st.warning("데이터가 불충분하거나(14일 미만) 수치 변동이 없어 예측이 불가능합니다.")

    with tabs[3]:
        st.markdown("### 만원 단위 예산 재배분안")
        st.info("최근 7일간의 일평균 지출액을 기준으로 성과 추세를 반영하여 제안합니다.")
        
        if st.button("🚀 최적 배분 계산"):
            last_7d = full_df[full_df['날짜'] > full_df['날짜'].max() - timedelta(days=7)]
            
            analysis_list = []
            for i in ids:
                target_data = full_df[full_df['ID']==i]
                _, slope, _ = get_trend_analysis(target_data)
                recent_avg_spend = last_7d[last_7d['ID']==i]['비용'].mean()
                if recent_avg_spend > 0:
                    analysis_list.append({'ID': i, '현재지출': recent_avg_spend, '추세': slope})
            
            ana_df = pd.DataFrame(analysis_list)
            if not ana_df.empty:
                result_df = optimize_budget_rounded(ana_df, ana_df['현재지출'].sum())
                
                # 결과 테이블 정제
                result_df['조정액'] = result_df['제안예산'] - result_df['현재지출']
                display_df = result_df[['ID', '현재지출', '제안예산', '조정액', '추세']]
                
                st.dataframe(display_df.style.format({
                    '현재지출': '{:,.0f}', '제안예산': '{:,.0f}', '조정액': '{:+,.0f}', '추세': '{:.4f}'
                }))
            else:
                st.error("최근 7일간의 지출 데이터가 있는 상품이 없습니다.")

# --- 각 탭별 모델 설명 (하단 배치) ---
st.markdown("---")
with st.expander("🛠️ 시스템 분석 가이드"):
    st.markdown("""
    - **성과 요약**: 상품명 열을 기준으로 데이터 시트를 통합하여 원본 수치를 집계합니다.
    - **유의성 검정**: 베이지안 통계(Beta-Binomial)를 통해 노출량 대비 클릭 성과의 안정성을 검증합니다.
    - **트렌드 분석**: Prophet 라이브러리를 통해 데이터의 요일별 특성과 주기성을 파악합니다. 적합도가 100%에 가깝게 나오는 경우는 시계열적 변동이 없는 평탄한 데이터일 때 발생하며, 이 경우 예측 신뢰도는 낮게 평가됩니다.
    - **예산 재배분**: 최근 일주일 지출을 베이스라인으로 성과 기울기에 따라 가중치를 부여하며, 모든 제안가는 실무 편의를 위해 **10,000원 단위로 절삭** 및 보정됩니다.
    """)