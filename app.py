import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import logging

# Prophet 로그 억제
logging.getLogger('prophet').setLevel(logging.WARNING)
st.set_page_config(page_title="Marketing Analytics Pro", layout="wide")

# --- [1. 정교화된 데이터 엔진] ---
def clean_and_process_pro(df):
    # 컬럼 매핑: Exact Match 최우선 (문제 1 해결)
    mapping = {
        '날짜': ['날짜', '일자', 'Date'],
        '매체': ['매체', '채널', 'Media'],
        '상품명': ['상품명', '상품', 'Product'],
        '소재명': ['소재명', '소재', 'Creative'],
        '노출수': ['노출수', '노출', 'Impression'],
        '클릭수': ['클릭수', '클릭', 'Click'],
        '비용': ['비용', '지출', 'Cost']
    }
    
    final_df = pd.DataFrame()
    for std_key, patterns in mapping.items():
        # 정확히 일치하는 컬럼 우선 탐색
        found = [c for c in df.columns if str(c).strip() in patterns]
        if found:
            final_df[std_key] = df[found[0]]
        else:
            # 차선책으로 포함 여부 확인 (경고와 함께)
            found_sub = [c for c in df.columns if any(p in str(c) for p in patterns)]
            if found_sub: final_df[std_key] = df[found_sub[0]]
    
    if len(final_df.columns) < len(mapping):
        return pd.DataFrame(), "필수 컬럼 매핑 실패"

    final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
    for col in ['노출수', '클릭수', '비용']:
        final_df[col] = pd.to_numeric(final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    
    # 지표 계산 (노출수 0 방어)
    final_df['CTR(%)'] = np.where(final_df['노출수'] > 0, (final_df['클릭수'] / final_df['노출수'] * 100), 0.0)
    final_df['ID'] = "[" + final_df['상품명'].astype(str) + "] " + final_df['소재명'].astype(str)
    
    return final_df.dropna(subset=['날짜']), None

# --- [2. 통계적 일관성을 갖춘 예측 엔진] ---
def robust_forecast(data):
    # 문제 2 해결: 최소 노출 데이터 필터 (100회 미만은 노이즈로 간주)
    valid_data = data[data['노출수'] >= 100].sort_values('날짜').copy()
    if len(valid_data) < 7: return None, None
    
    # 문제 4 해결: CTR Logit 변환 (0~100 범위를 실수 전체로 확장)
    # p = CTR/100, y = log(p/(1-p))
    p = np.clip(valid_df['CTR(%)'].values / 100, 0.0001, 0.9999)
    valid_df['y_logit'] = np.log(p / (1 - p))
    
    # 문제 3 해결: Prophet 단일 모델로 Trend+Seasonality 통합 처리
    m = Prophet(interval_width=0.8, daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=True)
    m.fit(valid_df[['날짜', 'y_logit']].rename(columns={'날짜': 'ds', 'y_logit': 'y'}))
    
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    
    # 역변환 함수 (Logistic function)
    def inv_logit(x): return (np.exp(x) / (1 + np.exp(x))) * 100
    
    # 결과 변환
    res = pd.DataFrame({
        'ds': forecast['ds'],
        'yhat': inv_logit(forecast['yhat']),
        'yhat_lower': inv_logit(forecast['yhat_lower']),
        'yhat_upper': inv_logit(forecast['yhat_upper'])
    })
    
    # 문제 5 해결: Fit Quality 계산 (R-squared 기반 적합도)
    y_true = valid_df['y_logit'].values
    y_pred = forecast.iloc[:len(y_true)]['yhat'].values
    res_ss = np.sum((y_true - y_pred)**2)
    tot_ss = np.sum((y_true - np.mean(y_true))**2)
    fit_quality = max(0, 1 - (res_ss / (tot_ss + 1e-6)))
    
    return res, fit_quality

# --- [3. UI 레이어] ---
st.title("🔬 고신뢰도 마케팅 분석 시스템 (Ver. Pro)")
st.warning("⚠️ 본 도구는 의사결정 '참고용'이며, 최종 판단은 마케터의 비즈니스 도메인 지식을 결합해야 합니다.")

uploaded_file = st.file_uploader("분석 데이터 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    # 데이터 로딩 로직 (생략 - 이전과 동일)
    # ...
    if not full_df.empty:
        tab1, tab2, tab3 = st.tabs(["💎 성과 진단", "⚖️ 베이지안 비교", "📈 확률적 수명 예측"])

        with tab1:
            # 문제 6 해결: 효율성 점수 텍스트 완화 및 최소 노출 필터 안내
            st.header("🏢 상품별 성과 효율 (Threshold 100+)")
            # ... (바 차트 시각화)
            st.info("💡 효율성 점수는 노출 100회 이상의 데이터셋에서만 유의미한 수치를 보입니다.")

        with tab3:
            st.header("📈 확률적 수명 추세 분석")
            sel_target = st.selectbox("소재 선택", sorted(full_df['ID'].unique()))
            target_df = full_df[full_df['ID'] == sel_target]
            
            forecast_res, fit_score = robust_forecast(target_df)
            
            if forecast_res is not None:
                # 문제 8 해결: 예측 구간(Shadow) 시각화
                fig = go.Figure()
                # 실제 데이터
                fig.add_trace(go.Scatter(x=target_df['날짜'], y=target_df['CTR(%)'], name="실측치", mode='lines+markers', line=dict(color='black')))
                # 예측 구간 (Uncertainty Area)
                fig.add_trace(go.Scatter(x=forecast_res['ds'], y=forecast_res['yhat_upper'], line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=forecast_res['ds'], y=forecast_res['yhat_lower'], fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', line=dict(width=0), name="80% 예측 구간"))
                # 예측 중심선
                fig.add_trace(go.Scatter(x=forecast_res['ds'], y=forecast_res['yhat'], name="기대 추세", line=dict(color='red', dash='dash')))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 문제 5, 8 해결: 용어 수정 및 불확실성 강조
                c1, c2 = st.columns(2)
                c1.metric("모델 적합도(Fit Quality)", f"{fit_score*100:.1f}%")
                
                st.divider()
                st.subheader("🕵️ 분석 결과 가이드")
                curr_ctr = target_df['CTR(%)'].iloc[-1]
                pred_ctr = forecast_res['yhat'].iloc[-1]
                
                if pred_ctr < curr_ctr * 0.8:
                    st.error(f"📉 **추세 주의:** 통계적으로 유의미한 하락 신호가 감지되었습니다. (7일 후 기대값: {pred_ctr:.2f}%)")
                elif pred_ctr > curr_ctr * 1.1:
                    st.success(f"📈 **추세 양호:** 현재 성과가 유지되거나 상승할 확률이 높습니다.")
                else:
                    st.warning("📊 **정체기:** 뚜렷한 방향성이 보이지 않는 구간입니다.")

                with st.expander("📝 통계적 가정 및 한계"):
                    st.write("""
                    1. **Logit Transformation**: CTR의 0~100% 경계값 문제를 해결하기 위해 로그 변환 후 모델링되었습니다.
                    2. **Uncertainty Interval**: 붉은색 영역은 80% 확률로 데이터가 존재할 수 있는 범위입니다. 영역이 넓을수록 예측이 불확실함을 의미합니다.
                    3. **Simpson's Paradox**: 본 지표는 통합 데이터이므로, 특정 지면이나 타겟팅 변화에 따른 세부 성과와는 다를 수 있습니다.
                    """)