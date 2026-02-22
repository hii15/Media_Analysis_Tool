import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="마케팅 통합 분석", layout="wide")
st.title("🎮 마케팅 통합 분석 시스템")
st.markdown("**Bayesian 통계 기반 성과 분석 & ROAS/CPI 의사결정 지원**")
st.markdown("---")


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_data(uploaded_file):
    """
    MMP 단일 파일 로드 + 컬럼명 자동 매핑
    지원: Appsflyer, Adjust, Singular, 커스텀 CSV/XLSX/TSV
    """
    try:
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            sep = '\t' if uploaded_file.name.endswith('.tsv') else ','
            df = pd.read_csv(uploaded_file, sep=sep)

        df.columns = [c.strip() for c in df.columns]

        mapping = {
            '날짜':     ['날짜', '일자', 'date', 'Date', 'Day'],
            '매체':     ['매체', 'media', 'channel', 'Media Source', 'Network', 'partner', 'Channel'],
            '상품':     ['상품', '상품명', 'app', 'product', 'App', 'Product'],
            '소재':     ['소재', '소재명', 'creative', 'Creative', 'ad_name', 'Ad', 'material'],
            '노출':     ['노출', '노출수', 'impressions', 'Impressions'],
            '클릭':     ['클릭', '클릭수', 'clicks', 'Clicks'],
            '비용':     ['비용', '지출', 'cost', 'Cost', 'spend', 'Spend'],
            '설치':     ['설치', '설치수', 'installs', 'Installs', 'install'],
            '이벤트수': ['이벤트수', '이벤트', 'events', 'conversions', 'key_events',
                        'af_purchase', 'purchase', 'event_count', 'Events'],
            '매출':     ['매출', '수익', 'revenue', 'Revenue', 'af_revenue', 'ltv_revenue'],
            'D1잔존율': ['D1잔존율', 'd1_retention', 'D1 Retention', 'retention_day_1'],
            'D7잔존율': ['D7잔존율', 'd7_retention', 'D7 Retention', 'retention_day_7'],
        }

        final_df = pd.DataFrame()
        matched = {}
        for k, candidates in mapping.items():
            for col in candidates:
                if col in df.columns:
                    final_df[k] = df[col]
                    matched[k] = col
                    break

        required = ['날짜', '노출', '클릭', '비용']
        missing = [c for c in required if c not in final_df.columns]
        if missing:
            st.error(f"필수 컬럼을 찾을 수 없습니다: {missing}\n원본 컬럼: {list(df.columns)}")
            return pd.DataFrame(), {}

        final_df['날짜'] = pd.to_datetime(final_df['날짜'], errors='coerce')

        numeric_cols = ['노출', '클릭', '비용', '설치', '이벤트수', '매출', 'D1잔존율', 'D7잔존율']
        for col in numeric_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(
                    final_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                    errors='coerce'
                ).fillna(0)

        if '상품' not in final_df.columns:
            final_df['상품'] = '상품미상'
        if '소재' not in final_df.columns:
            final_df['소재'] = '소재미상'

        final_df['ID']     = "[" + final_df['상품'].astype(str) + "] " + final_df['소재'].astype(str)
        final_df['CTR(%)'] = final_df['클릭'] / (final_df['노출'] + 1e-9) * 100
        final_df['CPC']    = final_df['비용'] / (final_df['클릭'] + 1e-9)

        if '설치' in final_df.columns:
            final_df['CPI']            = final_df['비용'] / (final_df['설치'] + 1e-9)
            final_df['IPM']            = final_df['설치'] / (final_df['노출'] + 1e-9) * 1000
            final_df['Install_CVR(%)'] = final_df['설치'] / (final_df['클릭'] + 1e-9) * 100

        if '이벤트수' in final_df.columns:
            final_df['CPA'] = final_df['비용'] / (final_df['이벤트수'] + 1e-9)
            if '설치' in final_df.columns:
                final_df['Event_Rate(%)'] = final_df['이벤트수'] / (final_df['설치'] + 1e-9) * 100

        if '매출' in final_df.columns:
            final_df['ROAS(%)'] = final_df['매출'] / (final_df['비용'] + 1e-9) * 100

        return final_df.dropna(subset=['날짜']).sort_values(['ID', '날짜']), matched

    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame(), {}


# ─────────────────────────────────────────────
# Bayesian 분석 — CTR 전용 Empirical Bayes
# ─────────────────────────────────────────────

def analyze_empirical_bayes(df):
    global_ctr = df['클릭'].sum() / (df['노출'].sum() + 1e-9)

    agg_dict = {'클릭': 'sum', '노출': 'sum', '비용': 'sum'}
    if '매체' in df.columns:
        agg_dict['매체'] = 'first'
    for col in ['설치', '이벤트수', '매출']:
        if col in df.columns:
            agg_dict[col] = 'sum'

    id_stats = df.groupby('ID').agg(agg_dict).reset_index()
    id_ctrs  = id_stats['클릭'] / (id_stats['노출'] + 1e-9)

    var_ctr = max(id_ctrs.var(), 1e-7)
    kappa   = np.clip((global_ctr * (1 - global_ctr) / var_ctr) - 1, 10, 1000)
    alpha_0 = global_ctr * kappa
    beta_0  = (1 - global_ctr) * kappa

    id_stats['alpha_0']    = alpha_0
    id_stats['beta_0']     = beta_0
    id_stats['post_alpha'] = alpha_0 + id_stats['클릭']
    id_stats['post_beta']  = beta_0  + (id_stats['노출'] - id_stats['클릭'])
    id_stats['raw_ctr']    = id_ctrs.values
    id_stats['exp_ctr']    = id_stats['post_alpha'] / (id_stats['post_alpha'] + id_stats['post_beta'])

    samples = np.random.beta(
        id_stats['post_alpha'].values[:, None],
        id_stats['post_beta'].values[:, None],
        size=(len(id_stats), 5000)
    )
    id_stats['prob_is_best'] = np.bincount(
        np.argmax(samples, axis=0), minlength=len(id_stats)
    ) / 5000

    max_date   = df['날짜'].max()
    last_costs = df[df['날짜'] >= max_date - timedelta(days=6)].groupby('ID')['비용'].sum() / 7
    id_stats   = id_stats.merge(last_costs.rename('avg_cost_7d'), on='ID', how='left').fillna(0)

    return id_stats, alpha_0, beta_0, kappa


# ─────────────────────────────────────────────
# CUSUM / 조기경고 유틸
# ─────────────────────────────────────────────

def get_binomial_cusum(clicks, imps, p0):
    p1 = np.clip(p0 * 0.85, 1e-6, 1 - 1e-6)
    p0 = np.clip(p0, 1e-6, 1 - 1e-6)
    llr = clicks * np.log(p1 / p0) + (imps - clicks) * np.log((1 - p1) / (1 - p0))
    s, cusum = 0, []
    for v in llr:
        s = min(0, s + v)
        cusum.append(s)
    return np.array(cusum)


def get_adaptive_threshold(p0, daily_imp):
    ctr_f = 0.6 if p0 < 0.005 else (0.8 if p0 < 0.01 else (1.0 if p0 < 0.02 else 1.2))
    vol_f = 1.5 if daily_imp > 5_000_000 else (1.2 if daily_imp > 1_000_000 else 1.0)
    return -8.0 * ctr_f * vol_f


def get_confidence_level(material, df):
    mat_data   = df[df['ID'] == material['ID']]
    data_score = 1 if material['노출'] > 1_000_000 else (0.5 if material['노출'] > 100_000 else 0)
    if len(mat_data) >= 7:
        std  = mat_data['CTR(%)'].std()
        stab = 1 if std < material['exp_ctr'] * 50 else (0.5 if std < material['exp_ctr'] * 100 else 0)
    else:
        stab = 0
    score = (data_score + stab) / 2
    if score >= 0.7: return "🟢 높음", "충분한 데이터와 안정적 패턴"
    if score >= 0.4: return "🟡 보통", "추가 관찰 권장"
    return "🔴 낮음", "데이터 부족 또는 변동성 높음"


# ─────────────────────────────────────────────
# 사이드바 — KPI 목표만
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ 분석 설정")
    st.markdown("### 🎯 KPI 목표")
    target_cpi  = st.number_input("목표 CPI (원)",  min_value=0, value=3000,  step=500)
    target_roas = st.number_input("목표 ROAS (%)",  min_value=0, value=300,   step=50)
    target_cpa  = st.number_input("목표 CPA (원)",  min_value=0, value=10000, step=1000)
    st.markdown("---")
    st.caption("""
    **Prior 설정:** 자동 (Empirical Bayes)
    업로드 데이터 전체 분포로 Prior를 추정합니다.
    소재 수·데이터 기간에 따라 자동 조정됩니다.
    """)


# ─────────────────────────────────────────────
# 파일 업로드
# ─────────────────────────────────────────────

st.markdown("### 📂 MMP 데이터 업로드")
uploaded_file = st.file_uploader(
    "노출/클릭/비용/설치/매출 통합 리포트 (CSV · XLSX · TSV)",
    type=['csv', 'xlsx', 'tsv']
)

with st.expander("📋 파일 컬럼 스펙 안내"):
    st.markdown("""
    | 구분 | 컬럼명 (한글 · 영문 모두 자동 인식) |
    |------|--------------------------------------|
    | **필수** | 날짜, 노출, 클릭, 비용 |
    | **권장** | 매체, 상품, 소재 |
    | **MMP** | 설치(Installs), 이벤트수(Events), 매출(Revenue) |
    | **품질** | D1잔존율, D7잔존율 |

    **지원 MMP:** Appsflyer · Adjust · Singular · 커스텀
    """)

st.markdown("---")


# ─────────────────────────────────────────────
# 메인 분석
# ─────────────────────────────────────────────

if uploaded_file:
    df, matched_cols = load_data(uploaded_file)

    if not df.empty:
        has_install   = '설치'    in df.columns
        has_event     = '이벤트수' in df.columns
        has_revenue   = '매출'    in df.columns
        has_retention = any(c in df.columns for c in ['D1잔존율', 'D7잔존율'])

        st.success(f"✅ 데이터 로드 완료 | 인식된 컬럼: {list(matched_cols.values())}")

        res_agg, alpha_0, beta_0, kappa = analyze_empirical_bayes(df)
        ids = sorted(df['ID'].unique())

        tabs = st.tabs([
            "📋 주간 체크리스트",
            "📊 성과 대시보드",
            "🧬 Bayesian 분석",
            "⏰ 조기 경고",
            "📉 CUSUM 모니터링",
            "🔽 퍼널 분석",
            "💰 ROAS/CPI 비교",
            "👤 유저 품질",
            "🧮 예산 시뮬레이터",
        ])

        # ── TAB 0 : 주간 체크리스트 ──────────────────
        with tabs[0]:
            st.markdown("## 📋 주간 의사결정 체크리스트")
            st.markdown(f"**기준일: {df['날짜'].max().strftime('%Y년 %m월 %d일')}**")
            st.markdown("---")

            today     = df['날짜'].max()
            tw_start  = today - timedelta(days=6)
            lw_start  = tw_start - timedelta(days=7)
            lw_end    = tw_start - timedelta(days=1)
            this_week = df[df['날짜'] >= tw_start]
            last_week = df[(df['날짜'] >= lw_start) & (df['날짜'] <= lw_end)]

            st.markdown("### 🚨 즉시 조치 필요")
            critical = []

            for _, mat in res_agg.iterrows():
                mid    = mat['ID']
                tw_sub = this_week[this_week['ID'] == mid]
                lw_sub = last_week[last_week['ID'] == mid]

                tw_ctr = tw_sub['CTR(%)'].mean()
                lw_ctr = lw_sub['CTR(%)'].mean()
                if lw_ctr > 0 and (tw_ctr - lw_ctr) / lw_ctr < -0.3:
                    critical.append({'소재': mid, '문제': f"CTR {abs((tw_ctr-lw_ctr)/lw_ctr)*100:.0f}% 급락",
                                     '이번주': f"{tw_ctr:.2f}%", '지난주': f"{lw_ctr:.2f}%",
                                     '액션': '소재 교체 또는 타겟 재설정'})

                mat_cost   = tw_sub['비용'].sum()
                cost_share  = mat_cost / (this_week['비용'].sum() + 1e-9)
                click_share = tw_sub['클릭'].sum() / (this_week['클릭'].sum() + 1e-9)
                if cost_share > 0.4 and click_share < 0.3:
                    critical.append({'소재': mid, '문제': f"비용 {cost_share*100:.0f}% 집중, 클릭 {click_share*100:.0f}%",
                                     '이번주': f"{mat_cost:,.0f}원", '지난주': '-',
                                     '액션': '예산 재분배 또는 입찰가 조정'})

                if has_install and target_cpi > 0:
                    inst = tw_sub['설치'].sum()
                    cpi  = mat_cost / (inst + 1e-9)
                    if inst > 10 and cpi > target_cpi * 1.5:
                        critical.append({'소재': mid, '문제': f"CPI {cpi:,.0f}원 (목표의 {cpi/target_cpi*100:.0f}%)",
                                         '이번주': f"설치 {inst:.0f}개", '지난주': '-',
                                         '액션': '입찰가 인하 또는 타겟 범위 축소'})

                if has_revenue and target_roas > 0:
                    rev  = tw_sub['매출'].sum()
                    roas = rev / (mat_cost + 1e-9) * 100
                    if mat_cost > 10000 and roas < target_roas * 0.7:
                        critical.append({'소재': mid, '문제': f"ROAS {roas:.0f}% (목표의 {roas/target_roas*100:.0f}%)",
                                         '이번주': f"매출 {rev:,.0f}원", '지난주': '-',
                                         '액션': '소재 품질 점검 또는 랜딩페이지 확인'})

            if critical:
                st.error(f"⚠️ {len(critical)}건 긴급 이슈")
                for i, item in enumerate(critical, 1):
                    with st.expander(f"🔴 [{i}] {item['소재']}: {item['문제']}", expanded=True):
                        c1, c2 = st.columns(2)
                        c1.metric("이번주", item['이번주'])
                        c2.metric("지난주", item['지난주'])
                        st.warning(f"**권장 액션:** {item['액션']}")
            else:
                st.success("✅ 긴급 조치 필요 항목 없음")

            st.markdown("---")
            st.markdown("### 💡 개선 기회")
            opps = []

            mat_perf = this_week.groupby('ID').agg(CTR=('CTR(%)', 'mean'), 비용=('비용', 'sum')).reset_index()
            if len(mat_perf):
                best = mat_perf.loc[mat_perf['CTR'].idxmax()]
                if best['비용'] / (this_week['비용'].sum() + 1e-9) < 0.4:
                    opps.append({'기회': f"🟢 고성과 소재 '{best['ID']}' 증액",
                                  '근거': f"CTR {best['CTR']:.2f}%로 1위, 예산 점유율 {best['비용']/this_week['비용'].sum()*100:.0f}%",
                                  '제안': "10~20% 점진 증액 후 3일 모니터링"})

            if has_revenue:
                roas_by = this_week.groupby('ID').apply(
                    lambda x: x['매출'].sum() / (x['비용'].sum() + 1e-9) * 100)
                if len(roas_by):
                    bid, bval = roas_by.idxmax(), roas_by.max()
                    if bval > target_roas * 1.3:
                        opps.append({'기회': f"💰 고ROAS 소재 '{bid}' 추가 증액",
                                      '근거': f"ROAS {bval:.0f}% (목표 대비 {bval/target_roas*100:.0f}%)",
                                      '제안': "예산 20~30% 추가 투입 검토"})

            if '매체' in df.columns:
                med_div = this_week.groupby('매체')['비용'].sum()
                if len(med_div) and (med_div / med_div.sum()).max() > 0.6:
                    opps.append({'기회': f"📱 매체 다각화 ({med_div.idxmax()} 편중)",
                                  '근거': f"단일 매체 의존도 {med_div.max()/med_div.sum()*100:.0f}%",
                                  '제안': "타 매체 소규모 테스트 시작"})

            if opps:
                for i, o in enumerate(opps, 1):
                    with st.expander(f"💡 [{i}] {o['기회']}", expanded=False):
                        st.info(f"**근거:** {o['근거']}")
                        st.success(f"**제안:** {o['제안']}")
            else:
                st.info("추가 개선 기회 없음 (현상 유지)")

            st.markdown("---")
            st.markdown("### 📊 이번주 성과 요약")
            n_cols = 4 + (1 if has_install else 0) + (1 if has_revenue else 0)
            kpi_c  = st.columns(n_cols)

            tw_cost   = this_week['비용'].sum()
            lw_cost   = last_week['비용'].sum()
            tw_clicks = this_week['클릭'].sum()
            lw_clicks = last_week['클릭'].sum()
            tw_ctr_   = tw_clicks / (this_week['노출'].sum() + 1e-9) * 100
            lw_ctr_   = lw_clicks / (last_week['노출'].sum() + 1e-9) * 100
            tw_cpc_   = tw_cost / (tw_clicks + 1e-9)
            lw_cpc_   = lw_cost / (lw_clicks + 1e-9)

            kpi_c[0].metric("총 지출",  f"{tw_cost:,.0f}원",  f"{(tw_cost-lw_cost)/lw_cost*100:+.1f}%" if lw_cost > 0 else "N/A")
            kpi_c[1].metric("총 클릭",  f"{tw_clicks:,}회",   f"{(tw_clicks-lw_clicks)/lw_clicks*100:+.1f}%" if lw_clicks > 0 else "N/A")
            kpi_c[2].metric("평균 CTR", f"{tw_ctr_:.2f}%",    f"{tw_ctr_-lw_ctr_:+.2f}%p")
            kpi_c[3].metric("평균 CPC", f"{tw_cpc_:,.0f}원",  f"{tw_cpc_-lw_cpc_:+.0f}원")
            idx = 4
            if has_install:
                tw_inst = this_week['설치'].sum()
                tw_cpi_ = tw_cost / (tw_inst + 1e-9)
                kpi_c[idx].metric("평균 CPI", f"{tw_cpi_:,.0f}원", f"목표 {target_cpi:,}원",
                                   delta_color="normal" if tw_cpi_ <= target_cpi else "inverse")
                idx += 1
            if has_revenue:
                tw_roas_ = this_week['매출'].sum() / (tw_cost + 1e-9) * 100
                kpi_c[idx].metric("ROAS", f"{tw_roas_:.0f}%", f"목표 {target_roas}%",
                                   delta_color="normal" if tw_roas_ >= target_roas else "inverse")

        # ── TAB 1 : 성과 대시보드 ─────────────────────
        with tabs[1]:
            st.markdown("### 📊 성과 대시보드")

            global_ctr = df['클릭'].sum() / (df['노출'].sum() + 1e-9)
            m = st.columns(4)
            m[0].metric("전체 평균 CTR", f"{global_ctr*100:.2f}%")
            m[1].metric("분석 기간",     f"{(df['날짜'].max()-df['날짜'].min()).days}일")
            m[2].metric("총 소재 수",    len(ids))
            m[3].metric("총 집행 비용",  f"{df['비용'].sum():,.0f}원")

            if has_install or has_revenue:
                m2 = st.columns(4)
                ci = 0
                if has_install:
                    avg_cpi = df['비용'].sum() / (df['설치'].sum() + 1e-9)
                    m2[ci].metric("총 설치",    f"{df['설치'].sum():,.0f}개")
                    m2[ci+1].metric("평균 CPI", f"{avg_cpi:,.0f}원",
                                     delta=f"목표 {target_cpi:,}원",
                                     delta_color="normal" if avg_cpi <= target_cpi else "inverse")
                    ci += 2
                if has_revenue:
                    total_roas = df['매출'].sum() / (df['비용'].sum() + 1e-9) * 100
                    m2[ci].metric("총 매출",     f"{df['매출'].sum():,.0f}원")
                    m2[ci+1].metric("전체 ROAS", f"{total_roas:.0f}%",
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
            disp_cols   = ['ID', 'raw_ctr', 'exp_ctr', '노출', '클릭', '비용', 'prob_is_best', 'avg_cost_7d']
            disp_rename = {'ID': '소재', 'raw_ctr': '원본CTR(%)', 'exp_ctr': '보정CTR(%)',
                           '노출': '노출수', '클릭': '클릭수', '비용': '비용',
                           'prob_is_best': '최고확률(%)', 'avg_cost_7d': '일평균비용'}
            for c in ['설치', 'CPI', 'ROAS(%)']:
                if c in res_agg.columns:
                    disp_cols.append(c)

            disp_df = res_agg[[c for c in disp_cols if c in res_agg.columns]].copy()
            disp_df['raw_ctr']      *= 100
            disp_df['exp_ctr']      *= 100
            disp_df['prob_is_best'] *= 100
            disp_df = disp_df.rename(columns=disp_rename)

            fmt = {'원본CTR(%)': '{:.2f}', '보정CTR(%)': '{:.2f}', '노출수': '{:,.0f}',
                   '클릭수': '{:,.0f}', '비용': '{:,.0f}', '최고확률(%)': '{:.1f}', '일평균비용': '{:,.0f}'}
            for c, f in [('설치', '{:,.0f}'), ('CPI', '{:,.0f}'), ('ROAS(%)', '{:.1f}')]:
                if c in disp_df.columns:
                    fmt[c] = f

            st.dataframe(
                disp_df.style.format(fmt).background_gradient(subset=['보정CTR(%)'], cmap='RdYlGn'),
                use_container_width=True
            )

            st.markdown("---")
            st.markdown("### 📊 CTR 일별 추이")
            daily_ctr = df.groupby(['날짜', 'ID']).agg(클릭=('클릭', 'sum'), 노출=('노출', 'sum')).reset_index()
            daily_ctr['CTR'] = daily_ctr['클릭'] / daily_ctr['노출'] * 100
            fig_t = px.line(daily_ctr, x='날짜', y='CTR', color='ID', markers=True)
            fig_t.update_layout(yaxis_title='CTR (%)', xaxis_title='')
            st.plotly_chart(fig_t, use_container_width=True)

        # ── TAB 2 : Bayesian 분석 ─────────────────────
        with tabs[2]:
            st.markdown("### 🧬 Bayesian 분석 상세")

            st.markdown("#### Prior 설정 (Empirical Bayes — 자동)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prior α₀", f"{alpha_0:.1f}")
            c2.metric("Prior β₀", f"{beta_0:.1f}")
            c3.metric("κ (강도)", f"{kappa:.1f}")
            st.caption(
                f"Prior CTR: {alpha_0/(alpha_0+beta_0)*100:.2f}%  |  "
                f"κ={kappa:.0f} → 가상 노출 {kappa*10000:,.0f}회 상당  |  "
                f"소재 {len(ids)}개 · {(df['날짜'].max()-df['날짜'].min()).days}일 데이터 기반 자동 추정"
            )

            st.markdown("---")
            st.markdown("#### Posterior 분포 (실제 CTR 추정)")
            fig_post = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, (_, row) in enumerate(res_agg.iterrows()):
                x = np.linspace(0, 0.05, 500)
                y = beta_dist.pdf(x, row['post_alpha'], row['post_beta'])
                fig_post.add_trace(go.Scatter(
                    x=x*100, y=y, name=row['ID'],
                    mode='lines', fill='tozeroy', opacity=0.6,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            fig_post.update_layout(
                title="소재별 실제 CTR 분포 (Posterior)",
                xaxis_title="CTR (%)", yaxis_title="확률 밀도", height=450
            )
            st.plotly_chart(fig_post, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 신뢰도 평가")
            conf_rows = []
            for _, mat in res_agg.iterrows():
                lvl, reason = get_confidence_level(mat, df)
                conf_rows.append({'소재': mat['ID'], '신뢰도': lvl, '이유': reason,
                                   '노출수': mat['노출'],
                                   '데이터일수': len(df[df['ID'] == mat['ID']])})
            st.dataframe(
                pd.DataFrame(conf_rows).style.format({'노출수': '{:,.0f}'}),
                use_container_width=True
            )

        # ── TAB 3 : 조기경고 ──────────────────────────
        with tabs[3]:
            st.markdown("### ⏰ 소재 피로도 조기 경고")
            st.markdown("선형 회귀로 CTR 추세를 분석해 교체 시점을 조기 예측합니다.")
            st.markdown("---")

            for mid in ids:
                mat_data = df[df['ID'] == mid].sort_values('날짜')
                if len(mat_data) < 5:
                    st.warning(f"**{mid}**: 데이터 부족 (최소 5일 필요)")
                    continue

                X     = np.arange(len(mat_data)).reshape(-1, 1)
                y     = mat_data['CTR(%)'].values
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                cur   = y[-1]

                if slope < -0.001:
                    dl     = max(0, int((cur - cur * 0.5) / abs(slope)))
                    status = ("⚠️ 즉시 교체 검토" if dl == 0 else
                              f"🔴 긴급 (D-{dl})" if dl <= 3 else
                              f"🟡 주의 (D-{dl})" if dl <= 7 else
                              f"🟢 안정 (D-{dl})")
                else:
                    status = "✅ 하락 추세 없음"

                co1, co2 = st.columns([2, 1])
                with co1:
                    st.markdown(f"**{mid}**  |  **{status}**")
                    st.markdown(f"현재 CTR: {cur:.2f}%  |  일평균 변화: {slope:.4f}%p")
                with co2:
                    fig_m = go.Figure()
                    fig_m.add_trace(go.Scatter(x=mat_data['날짜'], y=y, mode='lines+markers', name='실제'))
                    fig_m.add_trace(go.Scatter(x=mat_data['날짜'], y=model.predict(X),
                                               mode='lines', name='추세', line=dict(dash='dash', color='red')))
                    fig_m.update_layout(height=200, showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig_m, use_container_width=True)
                st.markdown("---")

        # ── TAB 4 : CUSUM ─────────────────────────────
        with tabs[4]:
            st.markdown("### 📉 CUSUM 이상 감지")
            st.markdown("기준 성과 대비 누적 이탈도를 추적해 구조적 하락을 감지합니다.")
            st.markdown("---")

            cu1, cu2, cu3 = st.columns([2, 1, 1])
            sel_mat      = cu1.selectbox("소재 선택", ids)
            cusum_metric = cu2.radio("감지 지표", ["CTR", "CPI"] if has_install else ["CTR"])
            th_mode      = cu3.radio("임계값", ["자동", "수동"])

            sub = df[df['ID'] == sel_mat].sort_values('날짜')

            if cusum_metric == "CTR":
                p0     = (sub.head(7)['클릭'].sum() / (sub.head(7)['노출'].sum() + 1e-9)
                          if len(sub) >= 7 else sub['클릭'].sum() / (sub['노출'].sum() + 1e-9))
                h      = (get_adaptive_threshold(p0, sub['노출'].mean()) if th_mode == "자동"
                          else st.slider("임계값(h)", -20.0, -3.0, -8.0, 0.5))
                cv     = get_binomial_cusum(sub['클릭'].values, sub['노출'].values, p0)
                p0_lbl = f"기준 CTR: {p0*100:.2f}%"
                y_lbl  = "CUSUM (CTR)"
            else:
                cpi_s  = sub['비용'] / (sub['설치'] + 1e-9)
                p0_cpi = cpi_s.head(7).mean() if len(sub) >= 7 else cpi_s.mean()
                norm   = (cpi_s - p0_cpi) / (p0_cpi + 1e-9)
                s, cv  = 0, []
                for v in norm:
                    s = min(0, s - v)
                    cv.append(s)
                cv     = np.array(cv)
                h      = (-1.5 if th_mode == "자동" else st.slider("임계값(h)", -5.0, -0.5, -1.5, 0.1))
                p0_lbl = f"기준 CPI: {p0_cpi:,.0f}원"
                y_lbl  = "CUSUM (CPI 상승 감지)"

            r1, r2, r3 = st.columns(3)
            r1.metric("기준 지표",   p0_lbl)
            r2.metric("감지 임계값", f"{h:.2f}")
            r3.metric("현재 CUSUM",  f"{cv[-1]:.2f}")

            fig_cs = go.Figure()
            fig_cs.add_trace(go.Scatter(x=sub['날짜'], y=cv, mode='lines+markers',
                                         line=dict(color='blue', width=2), name='CUSUM'))
            fig_cs.add_hline(y=h, line_dash="dash", line_color="red", annotation_text="임계값")
            fig_cs.update_layout(title=f"{sel_mat} — {y_lbl}",
                                  xaxis_title="날짜", yaxis_title=y_lbl, height=400)
            st.plotly_chart(fig_cs, use_container_width=True)

            if cv[-1] < h:
                delta    = abs(cv[-1] - h)
                severity = "🔴 심각" if delta > abs(h) * 2 else "🟡 경계"
                st.error(f"⚠️ **성과 하락 감지** (CUSUM {cv[-1]:.2f} < 임계값 {h:.2f})")
                st.markdown(f"**심각도:** {severity}")
                breach = np.where(cv < h)[0]
                if len(breach):
                    st.markdown(f"**최초 감지일:** {sub.iloc[breach[0]]['날짜'].strftime('%Y-%m-%d')}")
            else:
                st.success(f"✅ 정상 범위 (CUSUM {cv[-1]:.2f})")

        # ── TAB 5 : 퍼널 분석 ─────────────────────────
        with tabs[5]:
            st.markdown("### 🔽 퍼널 분석")

            if not has_install:
                st.info("설치(Installs) 컬럼이 없습니다. MMP 데이터에 설치 수를 포함해주세요.")
            else:
                st.markdown("노출 → 클릭 → 설치 → 이벤트 단계별 낙수율을 소재별로 비교합니다.")
                st.markdown("---")

                agg_dict_f = {'노출': ('노출', 'sum'), '클릭': ('클릭', 'sum'), '비용': ('비용', 'sum'),
                               '설치': ('설치', 'sum')}
                if has_event:
                    agg_dict_f['이벤트'] = ('이벤트수', 'sum')
                funnel_agg = df.groupby('ID').agg(**agg_dict_f).reset_index()

                funnel_agg['CTR(%)']         = funnel_agg['클릭'] / (funnel_agg['노출'] + 1e-9) * 100
                funnel_agg['Install_CVR(%)'] = funnel_agg['설치'] / (funnel_agg['클릭'] + 1e-9) * 100
                funnel_agg['IPM']            = funnel_agg['설치'] / (funnel_agg['노출'] + 1e-9) * 1000
                funnel_agg['CPI']            = funnel_agg['비용'] / (funnel_agg['설치'] + 1e-9)
                if has_event:
                    funnel_agg['Event_Rate(%)'] = funnel_agg['이벤트'] / (funnel_agg['설치'] + 1e-9) * 100
                    funnel_agg['CPA']            = funnel_agg['비용'] / (funnel_agg['이벤트'] + 1e-9)

                sel_ids = st.multiselect("소재 선택", ids, default=ids[:min(5, len(ids))])

                if sel_ids:
                    sf = funnel_agg[funnel_agg['ID'].isin(sel_ids)]

                    hm_cols = ['CTR(%)', 'Install_CVR(%)', 'IPM'] + (['Event_Rate(%)'] if has_event else [])
                    hm_df   = sf.set_index('ID')[hm_cols]
                    fig_hm  = px.imshow(hm_df.values, x=hm_cols, y=hm_df.index.tolist(),
                                         color_continuous_scale='RdYlGn', aspect='auto', text_auto='.2f')
                    fig_hm.update_layout(height=300 + len(sel_ids)*40, title="전환율 히트맵")
                    st.plotly_chart(fig_hm, use_container_width=True)

                    st.markdown("#### 🌊 소재별 퍼널 볼륨")
                    stages = ['노출', '클릭', '설치'] + (['이벤트'] if has_event else [])
                    fcols  = st.columns(min(len(sel_ids), 3))
                    for i, mid in enumerate(sel_ids):
                        row  = sf[sf['ID'] == mid].iloc[0]
                        vals = [row[c] for c in stages if c in row.index]
                        fig_f = go.Figure(go.Funnel(y=stages[:len(vals)], x=vals,
                                                     textinfo="value+percent initial"))
                        fig_f.update_layout(title=mid, height=280, margin=dict(l=0,r=0,t=40,b=0))
                        fcols[i % len(fcols)].plotly_chart(fig_f, use_container_width=True)

                    t_cols = ['ID', '노출', '클릭', '설치', 'CTR(%)', 'Install_CVR(%)', 'IPM', 'CPI']
                    if has_event:
                        t_cols += ['이벤트', 'Event_Rate(%)', 'CPA']
                    t_cols = [c for c in t_cols if c in sf.columns]
                    fmt_f  = {c: '{:,.0f}' for c in ['노출','클릭','설치','이벤트','CPI','CPA','IPM']}
                    fmt_f.update({c: '{:.2f}' for c in ['CTR(%)','Install_CVR(%)','Event_Rate(%)']})
                    st.dataframe(sf[t_cols].style.format(fmt_f)
                                   .background_gradient(subset=['CTR(%)'], cmap='RdYlGn'),
                                 use_container_width=True)

        # ── TAB 6 : ROAS/CPI 비교 ─────────────────────
        with tabs[6]:
            st.markdown("### 💰 ROAS/CPI 소재별 비교")

            if not has_install and not has_revenue:
                st.info("설치(Installs) 또는 매출(Revenue) 컬럼이 없습니다.")
            else:
                st.markdown("---")
                agg_dict_r = {'비용': ('비용', 'sum')}
                if has_install: agg_dict_r['설치'] = ('설치', 'sum')
                if has_revenue: agg_dict_r['매출'] = ('매출', 'sum')
                rc_agg = df.groupby('ID').agg(**agg_dict_r).reset_index()

                if has_install:
                    rc_agg['CPI']         = rc_agg['비용'] / (rc_agg['설치'] + 1e-9)
                    rc_agg['CPI달성률(%)'] = target_cpi / (rc_agg['CPI'] + 1e-9) * 100
                if has_revenue:
                    rc_agg['ROAS(%)']      = rc_agg['매출'] / (rc_agg['비용'] + 1e-9) * 100
                    rc_agg['ROAS달성률(%)'] = rc_agg['ROAS(%)'] / target_roas * 100

                if has_install:
                    st.markdown("#### 📊 소재별 CPI vs 목표")
                    fig_cpi = go.Figure()
                    fig_cpi.add_trace(go.Bar(
                        x=rc_agg['ID'], y=rc_agg['CPI'],
                        marker_color=['#2ecc71' if v <= target_cpi else '#e74c3c' for v in rc_agg['CPI']],
                    ))
                    fig_cpi.add_hline(y=target_cpi, line_dash="dash", line_color="blue",
                                       annotation_text=f"목표 CPI {target_cpi:,}원")
                    fig_cpi.update_layout(yaxis_title='CPI (원)', height=360)
                    st.plotly_chart(fig_cpi, use_container_width=True)

                if has_revenue:
                    st.markdown("#### 📊 소재별 ROAS vs 목표")
                    fig_roas = go.Figure()
                    fig_roas.add_trace(go.Bar(
                        x=rc_agg['ID'], y=rc_agg['ROAS(%)'],
                        marker_color=['#2ecc71' if v >= target_roas else '#e74c3c' for v in rc_agg['ROAS(%)']],
                    ))
                    fig_roas.add_hline(y=target_roas, line_dash="dash", line_color="blue",
                                        annotation_text=f"목표 ROAS {target_roas}%")
                    fig_roas.update_layout(yaxis_title='ROAS (%)', height=360)
                    st.plotly_chart(fig_roas, use_container_width=True)

                if has_install and has_revenue:
                    st.markdown("#### 🎯 CPI × ROAS 포지셔닝 맵")
                    fig_sc = px.scatter(
                        rc_agg, x='CPI', y='ROAS(%)', text='ID', size='비용',
                        color='ROAS(%)', color_continuous_scale='RdYlGn',
                        labels={'CPI': 'CPI (원) ← 낮을수록 좋음', 'ROAS(%)': 'ROAS (%) → 높을수록 좋음'}
                    )
                    fig_sc.add_vline(x=target_cpi,  line_dash="dash", line_color="gray", annotation_text="목표 CPI")
                    fig_sc.add_hline(y=target_roas, line_dash="dash", line_color="gray", annotation_text="목표 ROAS")
                    fig_sc.update_traces(textposition='top center')
                    fig_sc.update_layout(height=450, coloraxis_showscale=False)
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.caption("✅ 좌상단 (낮은 CPI + 높은 ROAS): 최우선 투자 대상")

                if has_revenue:
                    st.markdown("#### 📈 일별 ROAS 추이")
                    daily_r = df.groupby(['날짜', 'ID']).apply(
                        lambda x: x['매출'].sum() / (x['비용'].sum() + 1e-9) * 100
                    ).reset_index(name='ROAS(%)')
                    fig_dr = px.line(daily_r, x='날짜', y='ROAS(%)', color='ID', markers=True)
                    fig_dr.add_hline(y=target_roas, line_dash="dash", line_color="red",
                                      annotation_text=f"목표 {target_roas}%")
                    fig_dr.update_layout(height=360)
                    st.plotly_chart(fig_dr, use_container_width=True)

                st.markdown("#### 📋 수익성 요약")
                s_cols = ['ID', '비용'] + \
                         (['설치', 'CPI', 'CPI달성률(%)'] if has_install else []) + \
                         (['매출', 'ROAS(%)', 'ROAS달성률(%)'] if has_revenue else [])
                s_cols = [c for c in s_cols if c in rc_agg.columns]
                fmt_s  = {c: '{:,.0f}' for c in ['비용','설치','CPI','매출'] if c in rc_agg.columns}
                fmt_s.update({c: '{:.1f}' for c in ['ROAS(%)','CPI달성률(%)','ROAS달성률(%)'] if c in rc_agg.columns})
                grad   = 'ROAS(%)' if has_revenue else ('CPI달성률(%)' if has_install else None)
                styled = rc_agg[s_cols].style.format(fmt_s)
                if grad:
                    styled = styled.background_gradient(subset=[grad], cmap='RdYlGn')
                st.dataframe(styled, use_container_width=True)

        # ── TAB 7 : 유저 품질 ─────────────────────────
        with tabs[7]:
            st.markdown("### 👤 유저 품질 분석")

            if not has_install:
                st.info("설치(Installs) 컬럼이 없습니다.")
            else:
                st.markdown("설치된 유저의 질 — 이벤트 전환율, LTV, 잔존율을 소재별로 비교합니다.")
                st.markdown("---")

                agg_dict_q = {'설치': ('설치', 'sum'), '비용': ('비용', 'sum')}
                if has_event:   agg_dict_q['이벤트수'] = ('이벤트수', 'sum')
                if has_revenue: agg_dict_q['매출']     = ('매출',     'sum')
                q_agg = df.groupby('ID').agg(**agg_dict_q).reset_index()

                if has_event:
                    q_agg['Event_Rate(%)'] = q_agg['이벤트수'] / (q_agg['설치'] + 1e-9) * 100
                if has_revenue:
                    q_agg['LTV_per_Install'] = q_agg['매출'] / (q_agg['설치'] + 1e-9)

                if has_retention:
                    ret_cols = [c for c in ['D1잔존율', 'D7잔존율'] if c in df.columns]
                    ret_agg  = df.groupby('ID')[ret_cols].mean().reset_index()
                    st.markdown("#### 📊 D1/D7 잔존율")
                    fig_ret = go.Figure()
                    for rc in ret_cols:
                        fig_ret.add_trace(go.Bar(name=rc, x=ret_agg['ID'], y=ret_agg[rc]))
                    fig_ret.update_layout(barmode='group', yaxis_title='잔존율 (%)', height=360)
                    st.plotly_chart(fig_ret, use_container_width=True)

                if has_event:
                    st.markdown("#### 📊 설치 후 핵심 이벤트 전환율")
                    fig_ev = px.bar(
                        q_agg.sort_values('Event_Rate(%)', ascending=True),
                        x='Event_Rate(%)', y='ID', orientation='h',
                        color='Event_Rate(%)', color_continuous_scale='Blues',
                        text=q_agg.sort_values('Event_Rate(%)', ascending=True)['Event_Rate(%)'].apply(lambda x: f"{x:.1f}%")
                    )
                    fig_ev.update_traces(textposition='outside')
                    fig_ev.update_layout(height=350, coloraxis_showscale=False)
                    st.plotly_chart(fig_ev, use_container_width=True)

                if has_revenue:
                    st.markdown("#### 💎 설치당 매출 (LTV Proxy)")
                    fig_ltv = px.bar(
                        q_agg.sort_values('LTV_per_Install', ascending=True),
                        x='LTV_per_Install', y='ID', orientation='h',
                        color='LTV_per_Install', color_continuous_scale='Greens',
                        text=q_agg.sort_values('LTV_per_Install', ascending=True)['LTV_per_Install'].apply(lambda x: f"{x:,.0f}원")
                    )
                    fig_ltv.update_traces(textposition='outside')
                    fig_ltv.update_layout(height=350, coloraxis_showscale=False)
                    st.plotly_chart(fig_ltv, use_container_width=True)

                st.markdown("#### 📋 유저 품질 종합")
                qc = ['ID', '설치'] + \
                     (['이벤트수', 'Event_Rate(%)'] if has_event   else []) + \
                     (['매출', 'LTV_per_Install']  if has_revenue else [])
                if has_retention:
                    q_agg = q_agg.merge(ret_agg, on='ID', how='left')
                    qc   += ret_cols
                qc    = [c for c in qc if c in q_agg.columns]
                fmt_q = {c: '{:,.0f}' for c in ['설치','이벤트수','매출','LTV_per_Install'] if c in q_agg.columns}
                fmt_q.update({c: '{:.1f}' for c in ['Event_Rate(%)','D1잔존율','D7잔존율'] if c in q_agg.columns})
                st.dataframe(q_agg[qc].style.format(fmt_q), use_container_width=True)

        # ── TAB 8 : 예산 시뮬레이터 ───────────────────
        with tabs[8]:
            st.markdown("### 🧮 예산 시뮬레이터")

            if not has_install and not has_revenue:
                st.info("설치 또는 매출 데이터가 없습니다.")
            else:
                st.markdown("목표 CPI/ROAS 기준으로 소재별 최적 예산 배분을 추천합니다.")
                st.markdown("---")

                agg_dict_s = {'비용': ('비용', 'sum'), '클릭': ('클릭', 'sum'), '노출': ('노출', 'sum')}
                if has_install: agg_dict_s['설치'] = ('설치', 'sum')
                if has_revenue: agg_dict_s['매출'] = ('매출', 'sum')
                sim = df.groupby('ID').agg(**agg_dict_s).reset_index()

                if has_install: sim['CPI']    = sim['비용'] / (sim['설치'] + 1e-9)
                if has_revenue: sim['ROAS(%)'] = sim['매출'] / (sim['비용'] + 1e-9) * 100

                s1, s2, s3 = st.columns(3)
                total_bud = s1.number_input("총 예산 (원)", min_value=100_000,
                                             value=int(df['비용'].sum()), step=100_000)
                sim_cpi   = s2.number_input("목표 CPI (원)", min_value=0, value=target_cpi, step=500)
                sim_roas  = s3.number_input("목표 ROAS (%)", min_value=0, value=target_roas, step=50)

                alloc_opts = []
                if has_install: alloc_opts.append("CPI 성과 비례 (낮은 CPI → 더 많이)")
                if has_revenue: alloc_opts.append("ROAS 성과 비례 (높은 ROAS → 더 많이)")
                alloc_mode = st.radio("배분 기준", alloc_opts, horizontal=True)

                if "CPI" in alloc_mode:
                    inv = 1 / (sim['CPI'] + 1e-9)
                    sim['alloc_score'] = inv / inv.sum()
                else:
                    rp = np.clip(sim['ROAS(%)'], 0, None)
                    sim['alloc_score'] = (rp + 1e-9) / (rp.sum() + 1e-9)

                sim['추천_예산'] = sim['alloc_score'] * total_bud
                if has_install: sim['예상_설치'] = sim['추천_예산'] / (sim['CPI'] + 1e-9)
                if has_revenue: sim['예상_매출'] = sim['추천_예산'] * sim['ROAS(%)'] / 100

                st.markdown("#### 💰 추천 예산 배분")
                fig_pie = px.pie(sim, values='추천_예산', names='ID', hole=0.4,
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                fig_pie.update_traces(textinfo='label+percent')
                fig_pie.update_layout(height=380)
                st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("#### 📊 현재 vs 추천 예산")
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(name='현재', x=sim['ID'], y=sim['비용']))
                fig_cmp.add_trace(go.Bar(name='추천', x=sim['ID'], y=sim['추천_예산']))
                fig_cmp.update_layout(barmode='group', yaxis_title='예산 (원)', height=360)
                st.plotly_chart(fig_cmp, use_container_width=True)

                st.markdown("#### 🎯 예상 성과")
                pc = st.columns(4)
                if has_install:
                    pred_inst = sim['예상_설치'].sum()
                    pred_cpi_ = total_bud / (pred_inst + 1e-9)
                    pc[0].metric("예상 총 설치",  f"{pred_inst:,.0f}개")
                    pc[1].metric("예상 평균 CPI", f"{pred_cpi_:,.0f}원",
                                  delta=f"목표 {sim_cpi:,}원",
                                  delta_color="normal" if pred_cpi_ <= sim_cpi else "inverse")
                if has_revenue:
                    pred_rev   = sim['예상_매출'].sum()
                    pred_roas_ = pred_rev / (total_bud + 1e-9) * 100
                    pc[2].metric("예상 총 매출",  f"{pred_rev:,.0f}원")
                    pc[3].metric("예상 ROAS",     f"{pred_roas_:.0f}%",
                                  delta=f"목표 {sim_roas}%",
                                  delta_color="normal" if pred_roas_ >= sim_roas else "inverse")

                st.markdown("#### 📋 소재별 배분 상세")
                sd_cols = ['ID', '비용', '추천_예산', 'alloc_score'] + \
                          (['CPI', '예상_설치']    if has_install else []) + \
                          (['ROAS(%)', '예상_매출'] if has_revenue else [])
                sd_cols = [c for c in sd_cols if c in sim.columns]
                sd      = sim[sd_cols].rename(columns={
                    'ID': '소재', '비용': '현재예산', '추천_예산': '추천예산',
                    'alloc_score': '배분비중', '예상_설치': '예상설치', '예상_매출': '예상매출'
                })
                fmt_sd = {'현재예산': '{:,.0f}', '추천예산': '{:,.0f}', '배분비중': '{:.1%}',
                           'CPI': '{:,.0f}', 'ROAS(%)': '{:.1f}', '예상설치': '{:,.0f}', '예상매출': '{:,.0f}'}
                fmt_sd = {k: v for k, v in fmt_sd.items() if k in sd.columns}
                st.dataframe(
                    sd.style.format(fmt_sd).background_gradient(subset=['배분비중'], cmap='Blues'),
                    use_container_width=True
                )
                st.caption("⚠️ 예상 성과는 과거 성과 기반 선형 추정치입니다.")

        st.markdown("---")
        with st.expander("🔍 이 툴로 답할 수 없는 질문", expanded=False):
            st.markdown("""
            **인과 관계 추정 불가** — 예산 변경 효과는 A/B 테스트 없이 선형 추정만 가능
            **장기 LTV 불가** — 현재 매출은 단기 수익, 진짜 LTV는 6~12개월 코호트 필요
            **외부 요인 미반영** — 시즌성·경쟁사 입찰·알고리즘 변화 미통제
            **어트리뷰션 윈도우** — 설치~이벤트 시간차로 단기 지표 과소 측정 가능

            이 시스템의 포지셔닝: "완벽한 예측"이 아닌 **"지금 당장 행동할 것을 찾는 조기 경보 시스템"**
            """)
    else:
        st.warning("데이터를 로드할 수 없습니다. 파일 형식과 컬럼명을 확인해주세요.")

else:
    st.markdown("### 📋 시스템 소개")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### ✨ 기능 목록

        **항상 활성화**
        - Bayesian CTR 분석 (Empirical Bayes 자동 Prior)
        - 소재 피로도 조기 경고 (선형 회귀)
        - CUSUM 이상 감지
        - 주간 의사결정 체크리스트

        **설치(Installs) 컬럼 있을 때**
        - 퍼널 분석, CPI 비교, 예산 시뮬레이터

        **매출(Revenue) 컬럼 있을 때**
        - ROAS 비교, LTV 분석, 예산 시뮬레이터

        **잔존율(D1/D7) 컬럼 있을 때**
        - 유저 품질 잔존율 차트
        """)
    with c2:
        st.markdown("""
        #### 📂 파일 컬럼 가이드

        **필수**
        ```
        날짜, 노출, 클릭, 비용
        ```
        **MMP 지표 (있으면 자동 인식)**
        ```
        매체, 상품, 소재
        설치 / Installs
        이벤트수 / Events / conversions
        매출 / Revenue
        D1잔존율 / D7잔존율
        ```
        **지원 MMP**
        Appsflyer · Adjust · Singular · 커스텀 CSV
        """)

    st.markdown("---")
    st.caption("💡 MMP 리포트 파일 하나만 업로드하면 바로 분석이 시작됩니다.")