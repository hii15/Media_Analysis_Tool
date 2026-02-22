import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="마케팅 통합 분석", layout="wide")
st.title("🎮 마케팅 통합 분석 시스템 v2")
st.markdown("**Bayesian 통계 기반 성과 분석 & MMP 연동 ROAS/CPI 의사결정 지원**")
st.markdown("---")

# ─────────────────────────────────────────────
# 데이터 로드 함수
# ─────────────────────────────────────────────

def load_and_clean_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file, sep='\t' if uploaded_file.name.endswith('.tsv') else ',')

        df.columns = [c.strip() for c in df.columns]
        mapping = {
            '날짜': ['날짜', '일자', 'date'],
            '매체': ['매체', 'media', 'channel'],
            '상품': ['상품명', '상품', 'product', 'app'],
            '소재': ['소재명', '소재', 'material', 'creative', 'ad_name'],
            '노출': ['노출수', '노출', 'impressions'],
            '클릭': ['클릭수', '클릭', 'clicks'],
            '비용': ['비용', '지출', 'cost', 'spend'],
        }

        final_df = pd.DataFrame()
        for k, v in mapping.items():
            for col in v:
                if col in df.columns:
                    final_df[k] = df[col]
                    break

        final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')
        for col in ['노출', '클릭', '비용']:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(
                    final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                    errors='coerce'
                ).fillna(0)

        final_df['CTR(%)'] = final_df['클릭'] / (final_df['노출'] + 1e-9) * 100
        final_df['CPC'] = final_df['비용'] / (final_df['클릭'] + 1e-9)
        final_df['ID'] = "[" + final_df['상품'].astype(str) + "] " + final_df['소재'].astype(str)

        return final_df.dropna(subset=['날짜']).sort_values(['ID', '날짜'])
    except Exception as e:
        st.error(f"광고 데이터 로드 실패: {e}")
        return pd.DataFrame()


def load_mmp_data(uploaded_file):
    """
    MMP CSV 컬럼명 자동 매핑
    지원: Appsflyer, Adjust, Singular, 커스텀
    """
    try:
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file, sep='\t' if uploaded_file.name.endswith('.tsv') else ',')

        df.columns = [c.strip() for c in df.columns]

        mapping = {
            '날짜':        ['날짜', '일자', 'date', 'Day', 'Install Date'],
            '매체':        ['매체', 'media', 'channel', 'Media Source', 'Network', 'partner'],
            '소재':        ['소재', '소재명', 'creative', 'ad_name', 'Ad', 'Creative', 'material'],
            '상품':        ['상품', '앱', 'app', 'product', '상품명'],
            '설치':        ['설치', '설치수', 'installs', 'Installs', 'install'],
            '이벤트수':    ['이벤트수', '이벤트', 'events', 'conversions', 'key_events',
                           'af_purchase', 'purchase', 'event_count'],
            '매출':        ['매출', '수익', 'revenue', 'Revenue', 'af_revenue', 'ltv_revenue'],
            'D1잔존율':    ['D1잔존율', 'd1_retention', 'D1 Retention', 'retention_day_1'],
            'D7잔존율':    ['D7잔존율', 'd7_retention', 'D7 Retention', 'retention_day_7'],
        }

        final_df = pd.DataFrame()
        matched_cols = {}
        for k, candidates in mapping.items():
            for col in candidates:
                if col in df.columns:
                    final_df[k] = df[col]
                    matched_cols[k] = col
                    break

        if '날짜' not in final_df.columns:
            st.error("MMP 데이터에 날짜 컬럼을 찾을 수 없습니다.")
            return pd.DataFrame(), {}

        final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')

        numeric_cols = ['설치', '이벤트수', '매출', 'D1잔존율', 'D7잔존율']
        for col in numeric_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(
                    final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                    errors='coerce'
                ).fillna(0)

        if '상품' not in final_df.columns:
            final_df['상품'] = 'unknown'
        if '소재' not in final_df.columns:
            final_df['소재'] = 'unknown'

        final_df['ID'] = "[" + final_df['상품'].astype(str) + "] " + final_df['소재'].astype(str)

        return final_df.dropna(subset=['날짜']), matched_cols
    except Exception as e:
        st.error(f"MMP 데이터 로드 실패: {e}")
        return pd.DataFrame(), {}


def merge_ad_mmp(ad_df, mmp_df):
    """
    광고 + MMP 데이터 조인
    조인 키: 날짜 × 매체 × 소재 (ID)
    """
    join_keys = ['날짜', 'ID']
    if '매체' in ad_df.columns and '매체' in mmp_df.columns:
        join_keys = ['날짜', '매체', 'ID']

    mmp_agg_cols = ['설치', '이벤트수', '매출', 'D1잔존율', 'D7잔존율']
    available_mmp_cols = [c for c in mmp_agg_cols if c in mmp_df.columns]

    # MMP 집계 (날짜×ID 기준)
    agg_dict = {c: 'sum' for c in available_mmp_cols if c not in ['D1잔존율', 'D7잔존율']}
    if 'D1잔존율' in available_mmp_cols:
        agg_dict['D1잔존율'] = 'mean'
    if 'D7잔존율' in available_mmp_cols:
        agg_dict['D7잔존율'] = 'mean'

    mmp_grouped = mmp_df.groupby(['날짜', 'ID']).agg(agg_dict).reset_index()

    merged = pd.merge(ad_df, mmp_grouped, on=['날짜', 'ID'], how='left')

    # 핵심 지표 계산
    if '설치' in merged.columns:
        merged['설치'] = merged['설치'].fillna(0)
        merged['CPI'] = merged['비용'] / (merged['설치'] + 1e-9)
        merged['IPM'] = merged['설치'] / (merged['노출'] + 1e-9) * 1000
        merged['Install_CVR(%)'] = merged['설치'] / (merged['클릭'] + 1e-9) * 100

    if '이벤트수' in merged.columns:
        merged['이벤트수'] = merged['이벤트수'].fillna(0)
        merged['CPA'] = merged['비용'] / (merged['이벤트수'] + 1e-9)
        if '설치' in merged.columns:
            merged['Event_Rate(%)'] = merged['이벤트수'] / (merged['설치'] + 1e-9) * 100

    if '매출' in merged.columns:
        merged['매출'] = merged['매출'].fillna(0)
        merged['ROAS(%)'] = merged['매출'] / (merged['비용'] + 1e-9) * 100

    return merged


# ─────────────────────────────────────────────
# Bayesian 분석 (복합 스코어 지원)
# ─────────────────────────────────────────────

def analyze_empirical_bayes(df, benchmark_df=None, use_manual_prior=False,
                             score_weights=None):
    """
    score_weights: {'ctr': float, 'cvr': float, 'roas': float}
    None이면 CTR 단독 분석
    """
    if score_weights is None:
        score_weights = {'ctr': 1.0, 'cvr': 0.0, 'roas': 0.0}

    global_ctr = df['클릭'].sum() / (df['노출'].sum() + 1e-9)
    id_stats = df.groupby('ID').agg({
        '클릭': 'sum', '노출': 'sum', '비용': 'sum', '매체': 'first'
    })

    # MMP 지표 집계
    extra_cols = {}
    for col in ['설치', '이벤트수', '매출']:
        if col in df.columns:
            extra_cols[col] = df.groupby('ID')[col].sum()

    id_stats = id_stats.join(pd.DataFrame(extra_cols))
    id_ctrs = id_stats['클릭'] / (id_stats['노출'] + 1e-9)

    agg = id_stats.reset_index()
    agg['raw_ctr'] = id_ctrs.values

    # Prior 설정
    if use_manual_prior and benchmark_df is not None:
        benchmark_dict = benchmark_df.set_index('매체')['업계평균CTR(%)'].to_dict()
        strength_dict = benchmark_df.set_index('매체')['Prior강도'].to_dict()

        for idx, row in agg.iterrows():
            media = row['매체']
            if media in benchmark_dict:
                prior_ctr = benchmark_dict[media] / 100
                prior_strength = strength_dict[media]
                alpha_0 = prior_ctr * prior_strength
                beta_0 = (1 - prior_ctr) * prior_strength
            else:
                alpha_0, beta_0 = 1, 99

            agg.loc[idx, 'post_alpha'] = alpha_0 + row['클릭']
            agg.loc[idx, 'post_beta'] = beta_0 + (row['노출'] - row['클릭'])
            agg.loc[idx, 'alpha_0'] = alpha_0
            agg.loc[idx, 'beta_0'] = beta_0
    else:
        var_ctr = max(id_ctrs.var(), 1e-7)
        kappa = (global_ctr * (1 - global_ctr) / var_ctr) - 1
        kappa = np.clip(kappa, 10, 1000)
        alpha_0 = global_ctr * kappa
        beta_0 = (1 - global_ctr) * kappa

        agg['post_alpha'] = alpha_0 + agg['클릭']
        agg['post_beta'] = beta_0 + (agg['노출'] - agg['클릭'])
        agg['alpha_0'] = alpha_0
        agg['beta_0'] = beta_0

    agg['exp_ctr'] = agg['post_alpha'] / (agg['post_alpha'] + agg['post_beta'])

    # 복합 스코어 계산
    scores = np.zeros(len(agg))
    w_total = sum(score_weights.values()) + 1e-9

    # CTR 기여
    if score_weights.get('ctr', 0) > 0:
        ctr_norm = (agg['exp_ctr'] - agg['exp_ctr'].min()) / (agg['exp_ctr'].max() - agg['exp_ctr'].min() + 1e-9)
        scores += score_weights['ctr'] / w_total * ctr_norm.values

    # CVR (Install CVR) 기여
    if score_weights.get('cvr', 0) > 0 and '설치' in agg.columns:
        cvr = agg['설치'] / (agg['클릭'] + 1e-9)
        cvr_norm = (cvr - cvr.min()) / (cvr.max() - cvr.min() + 1e-9)
        scores += score_weights['cvr'] / w_total * cvr_norm.values

    # ROAS 기여
    if score_weights.get('roas', 0) > 0 and '매출' in agg.columns:
        roas = agg['매출'] / (agg['비용'] + 1e-9)
        roas_norm = (roas - roas.min()) / (roas.max() - roas.min() + 1e-9)
        scores += score_weights['roas'] / w_total * roas_norm.values

    agg['composite_score'] = scores

    # Bayesian 최고 확률 (CTR 기준)
    samples = np.random.beta(
        agg['post_alpha'].values[:, None],
        agg['post_beta'].values[:, None],
        size=(len(agg), 5000)
    )
    agg['prob_is_best'] = np.bincount(
        np.argmax(samples, axis=0),
        minlength=len(agg)
    ) / 5000

    # 최근 7일 평균 비용
    max_date = df['날짜'].max()
    last_7d = df[df['날짜'] >= max_date - timedelta(days=6)]
    last_costs = last_7d.groupby('ID')['비용'].sum() / 7
    agg = agg.merge(last_costs.rename('avg_cost_7d'), on='ID', how='left').fillna(0)

    return agg


# ─────────────────────────────────────────────
# CUSUM / 조기경고 함수 (기존 유지)
# ─────────────────────────────────────────────

def get_binomial_cusum(clicks, imps, p0):
    p1 = np.clip(p0 * 0.85, 1e-6, 1 - 1e-6)
    p0 = np.clip(p0, 1e-6, 1 - 1e-6)
    llr = clicks * np.log(p1 / p0) + (imps - clicks) * np.log((1 - p1) / (1 - p0))
    s = 0
    cusum = []
    for val in llr:
        s = min(0, s + val)
        cusum.append(s)
    return np.array(cusum)


def get_adaptive_threshold(p0, daily_impressions):
    base_h = -8.0
    ctr_factor = 0.6 if p0 < 0.005 else (0.8 if p0 < 0.01 else (1.0 if p0 < 0.02 else 1.2))
    volume_factor = 1.5 if daily_impressions > 5000000 else (1.2 if daily_impressions > 1000000 else 1.0)
    return base_h * ctr_factor * volume_factor


def get_confidence_level(material, df):
    mat_id = material['ID']
    mat_data = df[df['ID'] == mat_id]
    data_score = 1 if material['노출'] > 1000000 else (0.5 if material['노출'] > 100000 else 0)
    if len(mat_data) >= 7:
        daily_ctr_std = mat_data['CTR(%)'].std()
        stability_score = 1 if daily_ctr_std < material['exp_ctr'] * 50 else (
            0.5 if daily_ctr_std < material['exp_ctr'] * 100 else 0)
    else:
        stability_score = 0
    total_score = (data_score + stability_score) / 2
    if total_score >= 0.7:
        return "🟢 높음", "충분한 데이터와 안정적 패턴"
    elif total_score >= 0.4:
        return "🟡 보통", "추가 관찰 권장"
    else:
        return "🔴 낮음", "데이터 부족 또는 변동성 높음"


# ─────────────────────────────────────────────
# 사이드바 설정
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ 분석 설정")

    st.markdown("### 📊 Prior 설정 방식")
    prior_mode = st.radio(
        "Prior 설정",
        ["자동 (데이터 기반)", "수동 (벤치마크 기반)"],
        help="자동: 현재 데이터로 Prior 추정 / 수동: 업계 벤치마크 입력"
    )

    benchmark_df = None
    if prior_mode == "수동 (벤치마크 기반)":
        st.markdown("### 📋 매체별 벤치마크 입력")
        if 'benchmark_data' not in st.session_state:
            st.session_state.benchmark_data = pd.DataFrame({
                '매체': ['네이버 GFA', '유튜브', 'GDN', '페이스북'],
                '업계평균CTR(%)': [0.8, 2.5, 0.3, 1.2],
                'Prior강도': [100, 100, 100, 100]
            })
        edited_benchmark = st.data_editor(
            st.session_state.benchmark_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                '매체': st.column_config.TextColumn("매체명"),
                '업계평균CTR(%)': st.column_config.NumberColumn("업계 평균 CTR (%)", min_value=0.0, max_value=10.0, format="%.2f"),
                'Prior강도': st.column_config.NumberColumn("Prior 강도", min_value=10, max_value=1000)
            }
        )
        st.session_state.benchmark_data = edited_benchmark
        benchmark_df = edited_benchmark

    st.markdown("---")
    st.markdown("### 🎯 KPI 목표 설정")
    target_cpi = st.number_input("목표 CPI (원)", min_value=0, value=3000, step=500)
    target_roas = st.number_input("목표 ROAS (%)", min_value=0, value=300, step=50)
    target_cpa = st.number_input("목표 CPA (원)", min_value=0, value=10000, step=1000)

    st.markdown("---")
    st.markdown("### ⚖️ 복합 스코어 가중치")
    st.caption("Bayesian 분석에서 소재 순위 산정 기준")
    w_ctr  = st.slider("CTR 가중치",  0.0, 1.0, 0.4, 0.1)
    w_cvr  = st.slider("CVR 가중치",  0.0, 1.0, 0.3, 0.1)
    w_roas = st.slider("ROAS 가중치", 0.0, 1.0, 0.3, 0.1)
    score_weights = {'ctr': w_ctr, 'cvr': w_cvr, 'roas': w_roas}


# ─────────────────────────────────────────────
# 파일 업로드
# ─────────────────────────────────────────────

st.markdown("### 📂 데이터 업로드")
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.markdown("**① 광고 데이터** (필수)")
    uploaded_ad = st.file_uploader(
        "노출/클릭/비용 데이터",
        type=['csv', 'xlsx', 'tsv'],
        key="ad_upload"
    )

with col_up2:
    st.markdown("**② MMP 데이터** (선택 — 설치/매출 포함)")
    uploaded_mmp = st.file_uploader(
        "MMP 리포트 (Appsflyer/Adjust/Singular 등)",
        type=['csv', 'xlsx', 'tsv'],
        key="mmp_upload"
    )

    with st.expander("📋 MMP 파일 스펙 안내"):
        st.markdown("""
        **필수 컬럼:** 날짜, 매체 또는 소재  
        **선택 컬럼:** 설치수, 이벤트수, 매출, D1잔존율, D7잔존율

        | MMP | 날짜 컬럼 | 설치 컬럼 | 매출 컬럼 |
        |-----|-----------|-----------|-----------|
        | Appsflyer | Date | Installs | Revenue |
        | Adjust | Day | Installs | Revenue |
        | Singular | Date | Installs | Revenue |
        | 커스텀 | 날짜/일자 | 설치/설치수 | 매출/수익 |
        """)

st.markdown("---")

# ─────────────────────────────────────────────
# 메인 분석 실행
# ─────────────────────────────────────────────

if uploaded_ad:
    df_ad = load_and_clean_data(uploaded_ad)
    has_mmp = False
    df_merged = df_ad.copy()

    if uploaded_mmp:
        df_mmp, matched_cols = load_mmp_data(uploaded_mmp)
        if not df_mmp.empty:
            df_merged = merge_ad_mmp(df_ad, df_mmp)
            has_mmp = True
            st.success(f"✅ MMP 데이터 연동 완료 | 매핑된 컬럼: {list(matched_cols.values())}")
        else:
            st.warning("MMP 데이터 로드 실패. 광고 데이터만으로 분석합니다.")
    else:
        st.info("💡 MMP 데이터를 업로드하면 CPI/ROAS/퍼널 분석이 활성화됩니다.")

    if not df_ad.empty:
        use_manual_prior = (prior_mode == "수동 (벤치마크 기반)")
        res_agg = analyze_empirical_bayes(
            df_merged, benchmark_df, use_manual_prior, score_weights
        )
        ids = sorted(df_merged['ID'].unique())

        # ─────────────────────────────────────────────
        # 탭 구성
        # ─────────────────────────────────────────────
        tab_labels = [
            "📋 주간 체크리스트",
            "📊 성과 대시보드",
            "🧬 Bayesian 분석",
            "⏰ 조기 경고",
            "📉 CUSUM 모니터링",
        ]
        if has_mmp:
            tab_labels += [
                "🔽 퍼널 분석",
                "💰 ROAS/CPI 비교",
                "👤 유저 품질",
                "🧮 예산 시뮬레이터",
            ]

        tabs = st.tabs(tab_labels)

        # ──────────────────────────────────────────
        # TAB 0 : 주간 체크리스트
        # ──────────────────────────────────────────
        with tabs[0]:
            st.markdown("## 📋 주간 의사결정 체크리스트")
            st.markdown(f"**분석 기준일: {df_merged['날짜'].max().strftime('%Y년 %m월 %d일')}**")
            st.markdown("---")

            today = df_merged['날짜'].max()
            this_week_start = today - timedelta(days=6)
            last_week_start = this_week_start - timedelta(days=7)
            last_week_end   = this_week_start - timedelta(days=1)

            this_week = df_merged[df_merged['날짜'] >= this_week_start]
            last_week = df_merged[(df_merged['날짜'] >= last_week_start) & (df_merged['날짜'] <= last_week_end)]

            st.markdown("### 🚨 즉시 조치 필요")
            critical_items = []

            for _, mat in res_agg.iterrows():
                mat_id = mat['ID']

                # CTR 급락
                mat_tw = this_week[this_week['ID'] == mat_id]['CTR(%)'].mean()
                mat_lw = last_week[last_week['ID'] == mat_id]['CTR(%)'].mean()
                if mat_lw > 0 and (mat_tw - mat_lw) / mat_lw < -0.3:
                    critical_items.append({
                        '소재': mat_id, '문제': f"CTR {abs((mat_tw-mat_lw)/mat_lw)*100:.0f}% 급락",
                        '이번주': f"{mat_tw:.2f}%", '지난주': f"{mat_lw:.2f}%",
                        '액션': '소재 교체 또는 타겟 재설정', '우선순위': 1
                    })

                # 비용 집중 & 클릭 저조
                mat_cost = this_week[this_week['ID'] == mat_id]['비용'].sum()
                total_cost = this_week['비용'].sum()
                cost_share = mat_cost / total_cost if total_cost > 0 else 0
                mat_clicks = this_week[this_week['ID'] == mat_id]['클릭'].sum()
                total_clicks = this_week['클릭'].sum()
                click_share = mat_clicks / total_clicks if total_clicks > 0 else 0
                if cost_share > 0.4 and click_share < 0.3:
                    critical_items.append({
                        '소재': mat_id, '문제': f"비용 {cost_share*100:.0f}%, 클릭 {click_share*100:.0f}%",
                        '이번주': f"{mat_cost:,.0f}원", '지난주': '-',
                        '액션': '예산 재분배 또는 입찰가 조정', '우선순위': 1
                    })

                # MMP: CPI 목표 초과
                if has_mmp and '설치' in this_week.columns and target_cpi > 0:
                    mat_inst = this_week[this_week['ID'] == mat_id]['설치'].sum()
                    mat_cpi  = mat_cost / (mat_inst + 1e-9)
                    if mat_inst > 10 and mat_cpi > target_cpi * 1.5:
                        critical_items.append({
                            '소재': mat_id, '문제': f"CPI {mat_cpi:,.0f}원 (목표 {target_cpi:,}원의 {mat_cpi/target_cpi*100:.0f}%)",
                            '이번주': f"설치 {mat_inst:.0f}개", '지난주': '-',
                            '액션': '입찰가 인하 또는 타겟 범위 축소', '우선순위': 1
                        })

                # MMP: ROAS 목표 미달
                if has_mmp and '매출' in this_week.columns and target_roas > 0:
                    mat_rev  = this_week[this_week['ID'] == mat_id]['매출'].sum()
                    mat_roas = mat_rev / (mat_cost + 1e-9) * 100
                    if mat_cost > 10000 and mat_roas < target_roas * 0.7:
                        critical_items.append({
                            '소재': mat_id, '문제': f"ROAS {mat_roas:.0f}% (목표 {target_roas}%의 {mat_roas/target_roas*100:.0f}%)",
                            '이번주': f"매출 {mat_rev:,.0f}원", '지난주': '-',
                            '액션': '소재 품질 점검 또는 랜딩페이지 확인', '우선순위': 1
                        })

            if critical_items:
                st.error(f"⚠️ {len(critical_items)}건의 긴급 이슈")
                for idx, item in enumerate(critical_items, 1):
                    with st.expander(f"🔴 [{idx}] {item['소재']}: {item['문제']}", expanded=True):
                        c1, c2 = st.columns(2)
                        c1.metric("이번주", item['이번주'])
                        c2.metric("지난주", item['지난주'])
                        st.warning(f"**권장 액션:** {item['액션']}")
            else:
                st.success("✅ 긴급 조치 필요한 항목 없음")

            st.markdown("---")
            st.markdown("### 💡 개선 기회")

            opportunities = []
            material_perf = this_week.groupby('ID').agg({'CTR(%)': 'mean', '비용': 'sum', '클릭': 'sum'}).reset_index()

            if len(material_perf) > 0:
                best = material_perf.loc[material_perf['CTR(%)'].idxmax()]
                if best['비용'] / (this_week['비용'].sum() + 1e-9) < 0.4:
                    opportunities.append({
                        '기회': f"🟢 고성과 소재 '{best['ID']}' 증액 기회",
                        '근거': f"CTR {best['CTR(%)']:.2f}%로 1위, 예산 점유율 {best['비용']/this_week['비용'].sum()*100:.0f}%",
                        '제안': "10~20% 점진 증액 후 3일 모니터링"
                    })

            if has_mmp and 'ROAS(%)' in this_week.columns:
                roas_by_id = this_week.groupby('ID').apply(
                    lambda x: x['매출'].sum() / (x['비용'].sum() + 1e-9) * 100
                )
                if len(roas_by_id) > 0:
                    best_roas_id = roas_by_id.idxmax()
                    best_roas_val = roas_by_id.max()
                    if best_roas_val > target_roas * 1.3:
                        opportunities.append({
                            '기회': f"💰 고ROAS 소재 '{best_roas_id}' 추가 증액",
                            '근거': f"ROAS {best_roas_val:.0f}% (목표 대비 {best_roas_val/target_roas*100:.0f}%)",
                            '제안': "예산 20~30% 추가 투입 검토"
                        })

            media_div = this_week.groupby('매체')['비용'].sum()
            if len(media_div) > 0 and (media_div / media_div.sum()).max() > 0.6:
                opportunities.append({
                    '기회': f"📱 매체 다각화 필요 ({media_div.idxmax()} 편중)",
                    '근거': f"단일 매체 의존도 {media_div.max()/media_div.sum()*100:.0f}%",
                    '제안': "타 매체 소규모 테스트 시작"
                })

            if opportunities:
                for idx, opp in enumerate(opportunities, 1):
                    with st.expander(f"💡 [{idx}] {opp['기회']}", expanded=False):
                        st.info(f"**근거:** {opp['근거']}")
                        st.success(f"**제안:** {opp['제안']}")
            else:
                st.info("추가 개선 기회 없음 (현상 유지)")

            st.markdown("---")
            st.markdown("### 📊 이번주 성과 요약")

            kpi_cols = st.columns(4 if not has_mmp else 6)
            tw_cost   = this_week['비용'].sum()
            lw_cost   = last_week['비용'].sum()
            tw_clicks = this_week['클릭'].sum()
            lw_clicks = last_week['클릭'].sum()
            tw_ctr    = tw_clicks / (this_week['노출'].sum() + 1e-9) * 100
            lw_ctr    = lw_clicks / (last_week['노출'].sum() + 1e-9) * 100
            tw_cpc    = tw_cost / (tw_clicks + 1e-9)
            lw_cpc    = lw_cost / (lw_clicks + 1e-9)

            kpi_cols[0].metric("총 지출",   f"{tw_cost:,.0f}원",  f"{(tw_cost-lw_cost)/lw_cost*100:+.1f}%" if lw_cost > 0 else "N/A")
            kpi_cols[1].metric("총 클릭",   f"{tw_clicks:,}회",   f"{(tw_clicks-lw_clicks)/lw_clicks*100:+.1f}%" if lw_clicks > 0 else "N/A")
            kpi_cols[2].metric("평균 CTR",  f"{tw_ctr:.2f}%",     f"{tw_ctr-lw_ctr:+.2f}%p")
            kpi_cols[3].metric("평균 CPC",  f"{tw_cpc:,.0f}원",   f"{tw_cpc-lw_cpc:+.0f}원")

            if has_mmp:
                if '설치' in this_week.columns:
                    tw_inst = this_week['설치'].sum()
                    tw_cpi  = tw_cost / (tw_inst + 1e-9)
                    kpi_cols[4].metric("총 설치",   f"{tw_inst:,.0f}개", "")
                    # kpi_cols[4].metric("CPI",  f"{tw_cpi:,.0f}원",  f"목표 {target_cpi:,}원")
                if '매출' in this_week.columns:
                    tw_rev  = this_week['매출'].sum()
                    tw_roas = tw_rev / (tw_cost + 1e-9) * 100
                    kpi_cols[5].metric("ROAS", f"{tw_roas:.0f}%",  f"목표 {target_roas}%")

        # ──────────────────────────────────────────
        # TAB 1 : 성과 대시보드
        # ──────────────────────────────────────────
        with tabs[1]:
            st.markdown("### 📊 성과 대시보드")

            global_ctr = df_merged['클릭'].sum() / (df_merged['노출'].sum() + 1e-9)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("전체 평균 CTR",  f"{global_ctr*100:.2f}%")
            m2.metric("분석 기간",       f"{(df_merged['날짜'].max()-df_merged['날짜'].min()).days}일")
            m3.metric("총 소재 수",      len(ids))
            m4.metric("총 집행 비용",    f"{df_merged['비용'].sum():,.0f}원")

            if has_mmp:
                m5, m6, m7, m8 = st.columns(4)
                if '설치' in df_merged.columns:
                    total_inst = df_merged['설치'].sum()
                    avg_cpi    = df_merged['비용'].sum() / (total_inst + 1e-9)
                    m5.metric("총 설치",    f"{total_inst:,.0f}개")
                    m6.metric("평균 CPI",   f"{avg_cpi:,.0f}원",
                               delta=f"목표 {target_cpi:,}원",
                               delta_color="inverse" if avg_cpi > target_cpi else "normal")
                if '매출' in df_merged.columns:
                    total_rev  = df_merged['매출'].sum()
                    total_roas = total_rev / (df_merged['비용'].sum() + 1e-9) * 100
                    m7.metric("총 매출",    f"{total_rev:,.0f}원")
                    m8.metric("전체 ROAS",  f"{total_roas:.0f}%",
                               delta=f"목표 {target_roas}%",
                               delta_color="normal" if total_roas >= target_roas else "inverse")

            st.markdown("---")
            st.markdown("### 🏆 소재별 최고 성과 확률 (Bayesian CTR)")
            fig_prob = px.bar(
                res_agg.sort_values('prob_is_best', ascending=True),
                x='prob_is_best', y='ID', orientation='h',
                text=res_agg.sort_values('prob_is_best', ascending=True)['prob_is_best'].apply(lambda x: f"{x*100:.1f}%")
            )
            fig_prob.update_xaxes(tickformat='.0%', title='최고 성과 확률')
            fig_prob.update_yaxes(title='')
            fig_prob.update_traces(textposition='outside')
            st.plotly_chart(fig_prob, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📈 소재별 상세 성과")

            disp_cols = ['ID', 'raw_ctr', 'exp_ctr', '노출', '클릭', '비용', 'prob_is_best', 'avg_cost_7d']
            disp_rename = {
                'ID': '소재', 'raw_ctr': '원본CTR(%)', 'exp_ctr': '보정CTR(%)',
                '노출': '노출수', '클릭': '클릭수', '비용': '비용',
                'prob_is_best': '최고확률(%)', 'avg_cost_7d': '일평균비용'
            }

            if has_mmp:
                for c in ['설치', 'CPI', 'ROAS(%)']:
                    if c in res_agg.columns:
                        disp_cols.append(c)

            display_df = res_agg[[c for c in disp_cols if c in res_agg.columns]].copy()
            display_df['raw_ctr'] = display_df['raw_ctr'] * 100
            display_df['exp_ctr'] = display_df['exp_ctr'] * 100
            display_df['prob_is_best'] = display_df['prob_is_best'] * 100
            display_df = display_df.rename(columns=disp_rename)

            fmt = {
                '원본CTR(%)': '{:.2f}', '보정CTR(%)': '{:.2f}',
                '노출수': '{:,.0f}', '클릭수': '{:,.0f}',
                '비용': '{:,.0f}', '최고확률(%)': '{:.1f}', '일평균비용': '{:,.0f}',
            }
            if '설치' in display_df.columns:
                fmt['설치'] = '{:,.0f}'
            if 'CPI' in display_df.columns:
                fmt['CPI'] = '{:,.0f}'
            if 'ROAS(%)' in display_df.columns:
                fmt['ROAS(%)'] = '{:.1f}'

            st.dataframe(
                display_df.style.format(fmt).background_gradient(subset=['보정CTR(%)'], cmap='RdYlGn'),
                use_container_width=True
            )

            st.markdown("---")
            st.markdown("### 📊 CTR 추이")
            daily_ctr = df_merged.groupby(['날짜', 'ID']).agg({'클릭': 'sum', '노출': 'sum'}).reset_index()
            daily_ctr['CTR'] = daily_ctr['클릭'] / daily_ctr['노출'] * 100
            fig_trend = px.line(daily_ctr, x='날짜', y='CTR', color='ID', markers=True)
            fig_trend.update_layout(yaxis_title='CTR (%)', xaxis_title='')
            st.plotly_chart(fig_trend, use_container_width=True)

        # ──────────────────────────────────────────
        # TAB 2 : Bayesian 분석
        # ──────────────────────────────────────────
        with tabs[2]:
            st.markdown("### 🧬 Bayesian 분석 상세")

            # 복합 스코어 시각화
            if has_mmp and (w_cvr > 0 or w_roas > 0):
                st.markdown("#### 🏅 복합 스코어 순위")
                st.caption(f"가중치 — CTR: {w_ctr}, CVR: {w_cvr}, ROAS: {w_roas}")
                fig_score = px.bar(
                    res_agg.sort_values('composite_score', ascending=True),
                    x='composite_score', y='ID', orientation='h',
                    color='composite_score', color_continuous_scale='RdYlGn',
                    text=res_agg.sort_values('composite_score', ascending=True)['composite_score'].apply(lambda x: f"{x:.2f}")
                )
                fig_score.update_xaxes(title='복합 스코어 (0~1)')
                fig_score.update_yaxes(title='')
                fig_score.update_traces(textposition='outside')
                fig_score.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_score, use_container_width=True)
                st.markdown("---")

            st.markdown("#### Prior 설정 현황")
            if use_manual_prior:
                st.success("✅ 수동 설정 모드 (벤치마크 기반)")
                prior_summary = res_agg[['ID', '매체', 'alpha_0', 'beta_0']].copy()
                prior_summary['Prior_CTR(%)'] = prior_summary['alpha_0'] / (prior_summary['alpha_0'] + prior_summary['beta_0']) * 100
                prior_summary['Prior_강도'] = prior_summary['alpha_0'] + prior_summary['beta_0']
                st.dataframe(
                    prior_summary[['ID', '매체', 'Prior_CTR(%)', 'Prior_강도']].style.format({'Prior_CTR(%)': '{:.2f}', 'Prior_강도': '{:.0f}'}),
                    use_container_width=True
                )
            else:
                st.info("ℹ️ 자동 설정 모드 (데이터 기반)")
                alpha_0 = res_agg['alpha_0'].iloc[0]
                beta_0  = res_agg['beta_0'].iloc[0]
                kappa   = alpha_0 + beta_0
                c1, c2, c3 = st.columns(3)
                c1.metric("Prior α₀", f"{alpha_0:.1f}")
                c2.metric("Prior β₀", f"{beta_0:.1f}")
                c3.metric("κ (Kappa)", f"{kappa:.1f}")

            st.markdown("---")
            st.markdown("#### Posterior 분포")
            fig_post = go.Figure()
            colors = px.colors.qualitative.Set2
            for idx, (_, row) in enumerate(res_agg.iterrows()):
                x = np.linspace(0, 0.05, 500)
                y = beta_dist.pdf(x, row['post_alpha'], row['post_beta'])
                fig_post.add_trace(go.Scatter(
                    x=x*100, y=y, name=row['ID'],
                    mode='lines', fill='tozeroy', opacity=0.6,
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
            fig_post.update_layout(
                title="소재별 실제 CTR 분포 (Posterior)",
                xaxis_title="CTR (%)", yaxis_title="확률 밀도", height=450
            )
            st.plotly_chart(fig_post, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 신뢰도 평가")
            conf_data = []
            for _, mat in res_agg.iterrows():
                lvl, reason = get_confidence_level(mat, df_merged)
                conf_data.append({'소재': mat['ID'], '신뢰도': lvl, '이유': reason,
                                   '노출수': mat['노출'], '데이터일수': len(df_merged[df_merged['ID'] == mat['ID']])})
            st.dataframe(pd.DataFrame(conf_data).style.format({'노출수': '{:,.0f}'}), use_container_width=True)

        # ──────────────────────────────────────────
        # TAB 3 : 조기경고 (기존 유지)
        # ──────────────────────────────────────────
        with tabs[3]:
            st.markdown("### ⏰ 소재 피로도 조기 경고")
            st.markdown("선형 회귀로 CTR 추세를 분석, 교체 시점을 조기 예측합니다.")
            st.markdown("---")

            for mat_id in ids:
                mat_data = df_merged[df_merged['ID'] == mat_id].sort_values('날짜')
                if len(mat_data) < 5:
                    st.warning(f"**{mat_id}**: 데이터 부족 (최소 5일 필요)")
                    continue

                X = np.arange(len(mat_data)).reshape(-1, 1)
                y = mat_data['CTR(%)'].values
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                current_ctr = y[-1]

                if slope < -0.001:
                    days_left = max(0, int((current_ctr - current_ctr * 0.5) / abs(slope)))
                    if days_left == 0:
                        lifespan_status = "⚠️ 즉시 교체 검토"
                    elif days_left <= 3:
                        lifespan_status = f"🔴 긴급 (추정 D-{days_left})"
                    elif days_left <= 7:
                        lifespan_status = f"🟡 주의 (추정 D-{days_left})"
                    else:
                        lifespan_status = f"🟢 안정 (추정 D-{days_left})"
                else:
                    lifespan_status = "✅ 하락 추세 없음"
                    days_left = None

                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f"**{mat_id}**")
                    st.markdown(f"**상태:** {lifespan_status}")
                    st.markdown(f"현재 CTR: {current_ctr:.2f}% | 일평균 변화: {slope:.4f}%p")
                with c2:
                    fig_mini = go.Figure()
                    fig_mini.add_trace(go.Scatter(x=mat_data['날짜'], y=y, mode='lines+markers', name='실제'))
                    fig_mini.add_trace(go.Scatter(x=mat_data['날짜'], y=model.predict(X),
                                                   mode='lines', name='추세', line=dict(dash='dash', color='red')))
                    fig_mini.update_layout(height=200, showlegend=False, margin=dict(l=0,r=0,t=0,b=0), yaxis_title='CTR(%)')
                    st.plotly_chart(fig_mini, use_container_width=True)
                st.markdown("---")

        # ──────────────────────────────────────────
        # TAB 4 : CUSUM (기존 유지 + CPI 이중 감지)
        # ──────────────────────────────────────────
        with tabs[4]:
            st.markdown("### 📉 CUSUM 이상 감지")
            st.markdown("기준 성과 대비 누적 이탈도를 추적하여 성과 하락을 조기 감지합니다.")
            st.markdown("---")

            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                selected_material = st.selectbox("소재 선택", ids, key="cusum_sel")
            with c2:
                cusum_metric = st.radio("감지 지표", ["CTR", "CPI"] if has_mmp and '설치' in df_merged.columns else ["CTR"])
            with c3:
                threshold_mode = st.radio("임계값", ["자동", "수동"])

            sub = df_merged[df_merged['ID'] == selected_material].sort_values('날짜')

            if cusum_metric == "CTR":
                clicks_arr = sub['클릭'].values
                imps_arr   = sub['노출'].values
                p0_val     = sub.head(7)['클릭'].sum() / (sub.head(7)['노출'].sum() + 1e-9) if len(sub) >= 7 else sub['클릭'].sum() / (sub['노출'].sum() + 1e-9)
                avg_daily_imp = sub['노출'].mean()
                h_threshold   = get_adaptive_threshold(p0_val, avg_daily_imp) if threshold_mode == "자동" else st.slider("임계값(h)", -20.0, -3.0, -8.0, 0.5)
                cusum_vals    = get_binomial_cusum(clicks_arr, imps_arr, p0_val)
                y_label       = "CUSUM (CTR)"
                p0_label      = f"기준 CTR: {p0_val*100:.2f}%"
            else:
                # CPI 기반 CUSUM (비용/설치 비율)
                cpi_series = sub['비용'] / (sub['설치'] + 1e-9)
                p0_cpi     = cpi_series.head(7).mean() if len(sub) >= 7 else cpi_series.mean()
                # CPI 상승 감지: 정규화 후 이항 CUSUM 근사
                norm_cpi   = (cpi_series - p0_cpi) / (p0_cpi + 1e-9)
                s = 0; cusum_vals = []
                for v in norm_cpi:
                    s = min(0, s - v)  # CPI 상승이면 음수
                    cusum_vals.append(s)
                cusum_vals = np.array(cusum_vals)
                h_threshold = -1.5 if threshold_mode == "자동" else st.slider("임계값(h)", -5.0, -0.5, -1.5, 0.1)
                y_label     = "CUSUM (CPI 상승 감지)"
                p0_label    = f"기준 CPI: {p0_cpi:,.0f}원"

            col1, col2, col3 = st.columns(3)
            col1.metric("기준 지표", p0_label)
            col2.metric("감지 임계값 (h)", f"{h_threshold:.2f}")
            col3.metric("현재 CUSUM", f"{cusum_vals[-1]:.2f}")

            fig_cusum = go.Figure()
            fig_cusum.add_trace(go.Scatter(x=sub['날짜'], y=cusum_vals, mode='lines+markers',
                                            name='CUSUM', line=dict(color='blue', width=2)))
            fig_cusum.add_hline(y=h_threshold, line_dash="dash", line_color="red", annotation_text="임계값")
            fig_cusum.update_layout(title=f"{selected_material} — {y_label}",
                                     xaxis_title="날짜", yaxis_title=y_label, height=400)
            st.plotly_chart(fig_cusum, use_container_width=True)

            if cusum_vals[-1] < h_threshold:
                delta = abs(cusum_vals[-1] - h_threshold)
                severity = "🔴 심각" if delta > abs(h_threshold) * 2 else "🟡 경계"
                st.error(f"⚠️ **성과 하락 감지** (CUSUM: {cusum_vals[-1]:.2f} < 임계값: {h_threshold:.2f})")
                st.markdown(f"**심각도:** {severity}")
                first_breach = np.where(cusum_vals < h_threshold)[0]
                if len(first_breach) > 0:
                    st.markdown(f"**최초 감지일:** {sub.iloc[first_breach[0]]['날짜'].strftime('%Y-%m-%d')}")
            else:
                st.success(f"✅ 정상 범위 (CUSUM: {cusum_vals[-1]:.2f})")

        # ──────────────────────────────────────────
        # TAB 5 : 퍼널 분석 (MMP 전용)
        # ──────────────────────────────────────────
        if has_mmp:
            with tabs[5]:
                st.markdown("### 🔽 퍼널 분석")
                st.markdown("노출 → 클릭 → 설치 → 이벤트 전 단계 낙수율을 소재별로 비교합니다.")
                st.markdown("---")

                funnel_agg = df_merged.groupby('ID').agg(
                    노출=('노출', 'sum'),
                    클릭=('클릭', 'sum'),
                    설치=('설치', 'sum') if '설치' in df_merged.columns else ('클릭', 'count'),
                    이벤트=('이벤트수', 'sum') if '이벤트수' in df_merged.columns else ('클릭', 'count'),
                    비용=('비용', 'sum'),
                ).reset_index()

                # 전환율 계산
                funnel_agg['CTR(%)']         = funnel_agg['클릭'] / (funnel_agg['노출'] + 1e-9) * 100
                if '설치' in df_merged.columns:
                    funnel_agg['Install_CVR(%)'] = funnel_agg['설치'] / (funnel_agg['클릭'] + 1e-9) * 100
                    funnel_agg['IPM']            = funnel_agg['설치'] / (funnel_agg['노출'] + 1e-9) * 1000
                    funnel_agg['CPI']            = funnel_agg['비용'] / (funnel_agg['설치'] + 1e-9)
                if '이벤트수' in df_merged.columns and '설치' in df_merged.columns:
                    funnel_agg['Event_Rate(%)']  = funnel_agg['이벤트'] / (funnel_agg['설치'] + 1e-9) * 100
                    funnel_agg['CPA']            = funnel_agg['비용'] / (funnel_agg['이벤트'] + 1e-9)

                # 소재 선택
                sel_ids = st.multiselect("비교할 소재 선택 (최대 5개)", ids, default=ids[:min(5, len(ids))])

                if sel_ids:
                    sub_funnel = funnel_agg[funnel_agg['ID'].isin(sel_ids)]

                    # 퍼널 차트
                    st.markdown("#### 📊 전환율 히트맵")
                    heatmap_cols = ['CTR(%)']
                    if '설치' in df_merged.columns:
                        heatmap_cols += ['Install_CVR(%)', 'IPM']
                    if '이벤트수' in df_merged.columns:
                        heatmap_cols.append('Event_Rate(%)')

                    heatmap_df = sub_funnel.set_index('ID')[heatmap_cols]
                    fig_heatmap = px.imshow(
                        heatmap_df.values,
                        x=heatmap_cols,
                        y=heatmap_df.index.tolist(),
                        color_continuous_scale='RdYlGn',
                        aspect='auto',
                        text_auto='.2f'
                    )
                    fig_heatmap.update_layout(height=300 + len(sel_ids) * 40,
                                              xaxis_title='', yaxis_title='')
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                    # 단계별 절대량 퍼널
                    st.markdown("#### 🌊 퍼널 단계별 볼륨")
                    funnel_stages = ['노출', '클릭']
                    if '설치' in df_merged.columns:
                        funnel_stages.append('설치')
                    if '이벤트수' in df_merged.columns:
                        funnel_stages.append('이벤트')

                    for mat_id in sel_ids:
                        row = sub_funnel[sub_funnel['ID'] == mat_id].iloc[0]
                        vals = [row[c] for c in funnel_stages if c in row.index]
                        fig_f = go.Figure(go.Funnel(
                            y=funnel_stages[:len(vals)],
                            x=vals,
                            textinfo="value+percent initial"
                        ))
                        fig_f.update_layout(title=mat_id, height=280, margin=dict(l=0,r=0,t=40,b=0))
                        st.plotly_chart(fig_f, use_container_width=True)

                    # 상세 테이블
                    st.markdown("#### 📋 퍼널 상세 수치")
                    disp_cols_f = ['ID'] + [c for c in ['노출', '클릭', '설치', '이벤트', 'CTR(%)',
                                                           'Install_CVR(%)', 'IPM', 'Event_Rate(%)',
                                                           'CPI', 'CPA'] if c in sub_funnel.columns]
                    fmt_f = {c: '{:,.0f}' for c in ['노출', '클릭', '설치', '이벤트', 'CPI', 'CPA', 'IPM']}
                    fmt_f.update({c: '{:.2f}' for c in ['CTR(%)', 'Install_CVR(%)', 'Event_Rate(%)']})
                    st.dataframe(
                        sub_funnel[disp_cols_f].style.format(fmt_f).background_gradient(subset=['CTR(%)'], cmap='RdYlGn'),
                        use_container_width=True
                    )

            # ──────────────────────────────────────────
            # TAB 6 : ROAS/CPI 비교
            # ──────────────────────────────────────────
            with tabs[6]:
                st.markdown("### 💰 ROAS/CPI 소재별 비교")
                st.markdown("---")

                roas_cpi_agg = df_merged.groupby('ID').agg(
                    비용=('비용', 'sum'),
                    설치=('설치', 'sum') if '설치' in df_merged.columns else ('클릭', 'count'),
                    매출=('매출', 'sum') if '매출' in df_merged.columns else ('비용', 'count'),
                ).reset_index()

                if '설치' in df_merged.columns:
                    roas_cpi_agg['CPI'] = roas_cpi_agg['비용'] / (roas_cpi_agg['설치'] + 1e-9)
                    roas_cpi_agg['CPI_달성률(%)'] = target_cpi / (roas_cpi_agg['CPI'] + 1e-9) * 100

                if '매출' in df_merged.columns:
                    roas_cpi_agg['ROAS(%)'] = roas_cpi_agg['매출'] / (roas_cpi_agg['비용'] + 1e-9) * 100
                    roas_cpi_agg['ROAS_달성률(%)'] = roas_cpi_agg['ROAS(%)'] / target_roas * 100

                # CPI 비교 차트
                if '설치' in df_merged.columns:
                    st.markdown("#### 📊 소재별 CPI vs 목표")
                    fig_cpi = go.Figure()
                    fig_cpi.add_trace(go.Bar(
                        x=roas_cpi_agg['ID'], y=roas_cpi_agg['CPI'],
                        marker_color=['#2ecc71' if v <= target_cpi else '#e74c3c' for v in roas_cpi_agg['CPI']],
                        name='실제 CPI'
                    ))
                    fig_cpi.add_hline(y=target_cpi, line_dash="dash", line_color="blue",
                                       annotation_text=f"목표 CPI {target_cpi:,}원")
                    fig_cpi.update_layout(yaxis_title='CPI (원)', xaxis_title='', height=380)
                    st.plotly_chart(fig_cpi, use_container_width=True)

                # ROAS 비교 차트
                if '매출' in df_merged.columns:
                    st.markdown("#### 📊 소재별 ROAS vs 목표")
                    fig_roas = go.Figure()
                    fig_roas.add_trace(go.Bar(
                        x=roas_cpi_agg['ID'], y=roas_cpi_agg['ROAS(%)'],
                        marker_color=['#2ecc71' if v >= target_roas else '#e74c3c' for v in roas_cpi_agg['ROAS(%)']],
                        name='실제 ROAS'
                    ))
                    fig_roas.add_hline(y=target_roas, line_dash="dash", line_color="blue",
                                        annotation_text=f"목표 ROAS {target_roas}%")
                    fig_roas.update_layout(yaxis_title='ROAS (%)', xaxis_title='', height=380)
                    st.plotly_chart(fig_roas, use_container_width=True)

                # CPI × ROAS 산점도
                if '설치' in df_merged.columns and '매출' in df_merged.columns:
                    st.markdown("#### 🎯 CPI × ROAS 포지셔닝 맵")
                    fig_scatter = px.scatter(
                        roas_cpi_agg, x='CPI', y='ROAS(%)', text='ID',
                        size='비용', color='ROAS(%)',
                        color_continuous_scale='RdYlGn',
                        labels={'CPI': 'CPI (원) ← 낮을수록 좋음', 'ROAS(%)': 'ROAS (%) → 높을수록 좋음'}
                    )
                    fig_scatter.add_vline(x=target_cpi, line_dash="dash", line_color="gray",
                                           annotation_text=f"목표 CPI")
                    fig_scatter.add_hline(y=target_roas, line_dash="dash", line_color="gray",
                                           annotation_text=f"목표 ROAS")
                    fig_scatter.update_traces(textposition='top center')
                    fig_scatter.update_layout(height=450, coloraxis_showscale=False)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    st.caption("✅ 좌상단(낮은 CPI + 높은 ROAS): 최우선 투자 대상")

                # 일별 ROAS/CPI 추이
                st.markdown("#### 📈 일별 추이")
                daily_col = 'ROAS(%)' if '매출' in df_merged.columns else 'CPI'
                if 'ROAS(%)' in df_merged.columns or 'CPI' in df_merged.columns:
                    daily_roas_cpi = df_merged.groupby(['날짜', 'ID']).apply(
                        lambda x: x['매출'].sum() / (x['비용'].sum() + 1e-9) * 100
                        if '매출' in df_merged.columns
                        else x['비용'].sum() / (x['설치'].sum() + 1e-9)
                    ).reset_index(name=daily_col)
                    fig_daily = px.line(daily_roas_cpi, x='날짜', y=daily_col, color='ID', markers=True)
                    if '매출' in df_merged.columns:
                        fig_daily.add_hline(y=target_roas, line_dash="dash", line_color="red",
                                             annotation_text=f"목표 {target_roas}%")
                    fig_daily.update_layout(height=380)
                    st.plotly_chart(fig_daily, use_container_width=True)

                # 요약 테이블
                st.markdown("#### 📋 수익성 요약")
                perf_cols = ['ID', '비용', '설치', 'CPI', 'CPI_달성률(%)', '매출', 'ROAS(%)', 'ROAS_달성률(%)']
                avail_cols = [c for c in perf_cols if c in roas_cpi_agg.columns]
                fmt_roas = {c: '{:,.0f}' for c in ['비용', '설치', 'CPI', '매출'] if c in roas_cpi_agg.columns}
                fmt_roas.update({c: '{:.1f}' for c in ['ROAS(%)', 'CPI_달성률(%)', 'ROAS_달성률(%)'] if c in roas_cpi_agg.columns})

                grad_col = 'ROAS(%)' if 'ROAS(%)' in roas_cpi_agg.columns else ('CPI_달성률(%)' if 'CPI_달성률(%)' in roas_cpi_agg.columns else None)
                styled = roas_cpi_agg[avail_cols].style.format(fmt_roas)
                if grad_col:
                    styled = styled.background_gradient(subset=[grad_col], cmap='RdYlGn')
                st.dataframe(styled, use_container_width=True)

            # ──────────────────────────────────────────
            # TAB 7 : 유저 품질
            # ──────────────────────────────────────────
            with tabs[7]:
                st.markdown("### 👤 유저 품질 분석")
                st.markdown("소재별로 획득한 유저의 질 — 잔존율, 이벤트 전환율, LTV를 비교합니다.")
                st.markdown("---")

                quality_agg = df_merged.groupby('ID').agg(
                    설치=('설치', 'sum') if '설치' in df_merged.columns else ('클릭', 'count'),
                    이벤트수=('이벤트수', 'sum') if '이벤트수' in df_merged.columns else ('클릭', 'count'),
                    매출=('매출', 'sum') if '매출' in df_merged.columns else ('비용', 'count'),
                    비용=('비용', 'sum'),
                ).reset_index()

                if '설치' in df_merged.columns and '이벤트수' in df_merged.columns:
                    quality_agg['Event_Rate(%)'] = quality_agg['이벤트수'] / (quality_agg['설치'] + 1e-9) * 100
                if '설치' in df_merged.columns and '매출' in df_merged.columns:
                    quality_agg['LTV_per_Install'] = quality_agg['매출'] / (quality_agg['설치'] + 1e-9)

                has_retention = any(c in df_merged.columns for c in ['D1잔존율', 'D7잔존율'])

                # 잔존율 히트맵
                if has_retention:
                    st.markdown("#### 📊 D1/D7 잔존율 비교")
                    ret_cols = [c for c in ['D1잔존율', 'D7잔존율'] if c in df_merged.columns]
                    ret_agg  = df_merged.groupby('ID')[ret_cols].mean().reset_index()

                    fig_ret = go.Figure()
                    for col in ret_cols:
                        fig_ret.add_trace(go.Bar(name=col, x=ret_agg['ID'], y=ret_agg[col]))
                    fig_ret.update_layout(barmode='group', yaxis_title='잔존율 (%)', height=380)
                    st.plotly_chart(fig_ret, use_container_width=True)

                # 이벤트 전환율
                if 'Event_Rate(%)' in quality_agg.columns:
                    st.markdown("#### 📊 설치 후 핵심 이벤트 전환율")
                    fig_evt = px.bar(
                        quality_agg.sort_values('Event_Rate(%)', ascending=True),
                        x='Event_Rate(%)', y='ID', orientation='h',
                        color='Event_Rate(%)', color_continuous_scale='Blues',
                        text=quality_agg.sort_values('Event_Rate(%)', ascending=True)['Event_Rate(%)'].apply(lambda x: f"{x:.1f}%")
                    )
                    fig_evt.update_traces(textposition='outside')
                    fig_evt.update_layout(height=350, coloraxis_showscale=False)
                    st.plotly_chart(fig_evt, use_container_width=True)

                # LTV per Install
                if 'LTV_per_Install' in quality_agg.columns:
                    st.markdown("#### 💎 설치당 매출 (LTV Proxy)")
                    fig_ltv = px.bar(
                        quality_agg.sort_values('LTV_per_Install', ascending=True),
                        x='LTV_per_Install', y='ID', orientation='h',
                        color='LTV_per_Install', color_continuous_scale='Greens',
                        text=quality_agg.sort_values('LTV_per_Install', ascending=True)['LTV_per_Install'].apply(lambda x: f"{x:,.0f}원")
                    )
                    fig_ltv.update_traces(textposition='outside')
                    fig_ltv.update_layout(height=350, coloraxis_showscale=False)
                    st.plotly_chart(fig_ltv, use_container_width=True)

                # 유저 품질 종합 테이블
                st.markdown("#### 📋 유저 품질 종합")
                q_cols = ['ID', '설치', '이벤트수', '매출', 'Event_Rate(%)', 'LTV_per_Install']
                if has_retention:
                    ret_table = df_merged.groupby('ID')[[c for c in ['D1잔존율', 'D7잔존율'] if c in df_merged.columns]].mean().reset_index()
                    quality_agg = quality_agg.merge(ret_table, on='ID', how='left')
                    q_cols += [c for c in ['D1잔존율', 'D7잔존율'] if c in df_merged.columns]

                avail_q = [c for c in q_cols if c in quality_agg.columns]
                fmt_q = {c: '{:,.0f}' for c in ['설치', '이벤트수', '매출', 'LTV_per_Install'] if c in quality_agg.columns}
                fmt_q.update({c: '{:.1f}' for c in ['Event_Rate(%)', 'D1잔존율', 'D7잔존율'] if c in quality_agg.columns})
                st.dataframe(quality_agg[avail_q].style.format(fmt_q), use_container_width=True)

            # ──────────────────────────────────────────
            # TAB 8 : 예산 시뮬레이터
            # ──────────────────────────────────────────
            with tabs[8]:
                st.markdown("### 🧮 예산 시뮬레이터")
                st.markdown("목표 CPI/ROAS를 기준으로 소재별 최적 예산 배분을 추천합니다.")
                st.markdown("---")

                sim_agg = df_merged.groupby('ID').agg(
                    비용=('비용', 'sum'),
                    설치=('설치', 'sum') if '설치' in df_merged.columns else ('클릭', 'count'),
                    매출=('매출', 'sum') if '매출' in df_merged.columns else ('비용', 'count'),
                    클릭=('클릭', 'sum'),
                    노출=('노출', 'sum'),
                ).reset_index()

                sim_agg['CPI']     = sim_agg['비용'] / (sim_agg['설치'] + 1e-9)
                sim_agg['ROAS(%)'] = sim_agg['매출'] / (sim_agg['비용'] + 1e-9) * 100 if '매출' in df_merged.columns else 0.0
                sim_agg['CTR(%)']  = sim_agg['클릭'] / (sim_agg['노출'] + 1e-9) * 100

                # 목표 설정
                st.markdown("#### ⚙️ 시뮬레이션 파라미터")
                s1, s2, s3 = st.columns(3)
                total_budget  = s1.number_input("총 예산 (원)", min_value=100000, value=int(df_merged['비용'].sum()), step=100000)
                sim_target_cpi  = s2.number_input("목표 CPI (원) [시뮬]", min_value=0, value=target_cpi, step=500)
                sim_target_roas = s3.number_input("목표 ROAS (%) [시뮬]", min_value=0, value=target_roas, step=50)

                st.markdown("#### 🏅 추천 배분 방식 선택")
                alloc_mode = st.radio(
                    "배분 기준",
                    ["CPI 성과 비례", "ROAS 성과 비례", "복합 스코어 비례"],
                    horizontal=True
                )

                # 스코어 산출
                if alloc_mode == "CPI 성과 비례":
                    inv_cpi = 1 / (sim_agg['CPI'] + 1e-9)
                    sim_agg['alloc_score'] = inv_cpi / inv_cpi.sum()
                elif alloc_mode == "ROAS 성과 비례":
                    roas_pos = np.clip(sim_agg['ROAS(%)'], 0, None)
                    sim_agg['alloc_score'] = (roas_pos + 1e-9) / (roas_pos.sum() + 1e-9)
                else:
                    sim_agg['alloc_score'] = res_agg.set_index('ID')['composite_score'].reindex(sim_agg['ID']).fillna(1/len(sim_agg)).values
                    sim_agg['alloc_score'] = sim_agg['alloc_score'] / (sim_agg['alloc_score'].sum() + 1e-9)

                sim_agg['추천_예산'] = sim_agg['alloc_score'] * total_budget

                # 목표 달성 예측
                sim_agg['예상_설치'] = sim_agg['추천_예산'] / (sim_agg['CPI'] + 1e-9)
                sim_agg['예상_매출'] = sim_agg['추천_예산'] * sim_agg['ROAS(%)'] / 100 if '매출' in df_merged.columns else 0

                # 시각화
                st.markdown("#### 💰 추천 예산 배분")
                fig_alloc = px.pie(
                    sim_agg, values='추천_예산', names='ID',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_alloc.update_traces(textinfo='label+percent')
                fig_alloc.update_layout(height=400)
                st.plotly_chart(fig_alloc, use_container_width=True)

                # 현재 vs 추천 비교
                st.markdown("#### 📊 현재 vs 추천 예산 비교")
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(name='현재 예산', x=sim_agg['ID'], y=sim_agg['비용']))
                fig_compare.add_trace(go.Bar(name='추천 예산', x=sim_agg['ID'], y=sim_agg['추천_예산']))
                fig_compare.update_layout(barmode='group', yaxis_title='예산 (원)', height=380)
                st.plotly_chart(fig_compare, use_container_width=True)

                # 예상 성과
                st.markdown("#### 🎯 배분 시 예상 성과")
                pred_inst  = sim_agg['예상_설치'].sum()
                pred_rev   = sim_agg['예상_매출'].sum()
                pred_cpi   = total_budget / (pred_inst + 1e-9)
                pred_roas  = pred_rev / (total_budget + 1e-9) * 100

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("예상 총 설치",   f"{pred_inst:,.0f}개")
                p2.metric("예상 평균 CPI",  f"{pred_cpi:,.0f}원",
                           delta=f"목표 {sim_target_cpi:,}원",
                           delta_color="normal" if pred_cpi <= sim_target_cpi else "inverse")
                if '매출' in df_merged.columns:
                    p3.metric("예상 총 매출",   f"{pred_rev:,.0f}원")
                    p4.metric("예상 ROAS",      f"{pred_roas:.0f}%",
                               delta=f"목표 {sim_target_roas}%",
                               delta_color="normal" if pred_roas >= sim_target_roas else "inverse")

                # 상세 테이블
                st.markdown("#### 📋 소재별 예산 배분 상세")
                sim_display = sim_agg[['ID', '비용', '추천_예산', 'alloc_score', 'CPI', 'ROAS(%)', '예상_설치', '예상_매출']].copy()
                sim_display.columns = ['소재', '현재예산', '추천예산', '배분비중', 'CPI', 'ROAS(%)', '예상설치', '예상매출']
                fmt_sim = {'현재예산': '{:,.0f}', '추천예산': '{:,.0f}', '배분비중': '{:.1%}',
                            'CPI': '{:,.0f}', 'ROAS(%)': '{:.1f}', '예상설치': '{:,.0f}', '예상매출': '{:,.0f}'}
                st.dataframe(
                    sim_display.style.format(fmt_sim).background_gradient(subset=['배분비중'], cmap='Blues'),
                    use_container_width=True
                )

                st.caption("⚠️ 예상 성과는 과거 성과 기반 선형 추정치이며, 실제 결과와 다를 수 있습니다.")

        # ──────────────────────────────────────────
        # 데이터 한계 안내
        # ──────────────────────────────────────────
        st.markdown("---")
        with st.expander("🔍 현재 데이터로 답할 수 없는 질문", expanded=False):
            st.markdown("""
            ### ❌ 현재 데이터의 한계

            **1. 인과 관계 추정 불가**
            - "예산 2배 증액 시 설치 몇 개 증가?"는 선형 가정 기반 추정만 가능
            - 필요: 과거 예산 변경 실험 데이터 (A/B 테스트)

            **2. 장기 LTV 예측 불가**
            - 현재 매출 = 단기 수익, 진짜 LTV는 6~12개월 코호트 필요
            - 지금은 "설치당 단기 매출"만 측정 가능

            **3. 외부 요인 미반영**
            - 시즌성, 경쟁사 입찰, 플랫폼 알고리즘 변화 미통제
            - CUSUM 이상 감지 시 외부 요인 별도 확인 필요

            **4. 어트리뷰션 윈도우**
            - 설치~이벤트 사이 시간 차로 단기 지표 과소 측정 가능

            ### ✅ 이 시스템으로 답할 수 있는 질문

            - 지금 당장 어떤 소재에 예산을 더 써야 하나?
            - 어떤 소재가 목표 CPI/ROAS를 초과/미달 중인가?
            - 어떤 소재의 유저 품질이 가장 좋은가?
            - 성과 하락이 시작된 소재는 어디인가?
            """)
    else:
        st.warning("광고 데이터를 로드할 수 없습니다. 파일 형식을 확인해주세요.")

else:
    st.markdown("### 📋 시스템 소개")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### ✨ 핵심 기능 (v2)

        **광고 데이터만 있을 때**
        - Bayesian CTR 분석 (Prior 자동/수동)
        - 소재 피로도 조기 경고 (선형 회귀)
        - CUSUM 이상 감지
        - 주간 의사결정 체크리스트

        **MMP 데이터 추가 시 활성화**
        - 🔽 퍼널 분석 (노출 → 클릭 → 설치 → 이벤트)
        - 💰 ROAS/CPI 소재별 비교 + 목표 대비 달성률
        - 👤 유저 품질 (D1/D7 잔존율, Event Rate, LTV)
        - 🧮 예산 시뮬레이터 (최적 배분 추천)
        """)
    with c2:
        st.markdown("""
        #### 📂 데이터 파일 형식

        **광고 데이터 필수 컬럼**
        ```
        날짜, 매체, 상품, 소재, 노출, 클릭, 비용
        ```

        **MMP 데이터 선택 컬럼**
        ```
        날짜, 매체, 소재 (조인 키)
        설치수, 이벤트수, 매출     (수익성)
        D1잔존율, D7잔존율         (유저 품질)
        ```

        **지원 MMP:** Appsflyer · Adjust · Singular · 커스텀 CSV
        """)

    st.markdown("---")
    st.caption("💡 Tip: 사이드바에서 목표 CPI/ROAS와 복합 스코어 가중치를 설정하세요")