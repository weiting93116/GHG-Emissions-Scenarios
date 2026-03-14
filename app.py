"""
溫室氣體排放預測系統 v4
改進清單（相較 v3）：
1. ADF + KPSS 雙重檢定決定 d，強制非負序列 d≤1
   文獻：Kwiatkowski et al. (1992) JoE；Hyndman & Athanasopoulos (2021) ch.9
2. 三模型競爭：log-ARIMA / ETS / Holt damped trend
   以 OOS RMSE 選模（非跨空間 AIC 比較）
3. Holt damped trend：趨勢隨時間衰減，防止長期外推發散
   文獻：Gardner & McKenzie (1985) Mgmt Sci
4. MC 改用歷史 bootstrap，不假設 σ 倍率
   文獻：Efron & Tibshirani (1993)
5. AD-EF 折年率動態從最後資料年計算
6. 土地匯外推加飽和上限（物理約束）
7. ETS 下界只保護 ≥0（移除任意 0.05 倍率）
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import os, io, json, math, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── CORS ─────────────────────────────────────────────────
_raw_origins  = os.environ.get('ALLOWED_ORIGINS', '*')
_allowed_list = [o.strip() for o in _raw_origins.split(',') if o.strip()]
_allow_all    = (_raw_origins.strip() == '*') or not _allowed_list

def _cors_origin(origin):
    if not origin: return '*'
    if _allow_all: return '*'
    if origin in _allowed_list: return origin
    return None

@app.after_request
def apply_cors(response):
    origin = request.headers.get('Origin', '').strip()
    allowed = _cors_origin(origin)
    if allowed:
        response.headers['Access-Control-Allow-Origin']  = allowed
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        response.headers['Access-Control-Max-Age']       = '86400'
        response.headers['Vary']                         = 'Origin'
    return response

@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        origin = request.headers.get('Origin', '').strip()
        allowed = _cors_origin(origin)
        resp = app.make_response('')
        resp.status_code = 204
        if allowed:
            resp.headers['Access-Control-Allow-Origin']  = allowed
            resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
            resp.headers['Access-Control-Max-Age']       = '86400'
            resp.headers['Vary']                         = 'Origin'
        return resp

# ── JSON 工具 ─────────────────────────────────────────────
def nan_to_none(obj):
    if isinstance(obj, dict):  return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [nan_to_none(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj

def safe_json(data, status=200):
    return app.response_class(
        response=json.dumps(nan_to_none(data), ensure_ascii=False),
        status=status, mimetype='application/json')

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    tb = traceback.format_exc()
    print(f"UNHANDLED: {tb}")
    return safe_json({"error": str(e), "trace": tb[-500:]}, 500)

# ── 數值清洗 ──────────────────────────────────────────────
_IPCC_NA = frozenset([
    'NE','NA','N/A','NO','IE','C','NO,IE','NE,IE','',
    'NOT ESTIMATED','NOT OCCURRING','INCLUDED ELSEWHERE','CONFIDENTIAL'
])

def clean_numeric(val):
    if val is None: return np.nan
    s = str(val).strip().replace('"','').replace(' ','').replace('\u00a0','')
    if s.startswith('(') and s.endswith(')'): s = '-' + s[1:-1]
    s = s.replace(',','').replace('-','') if s == '-' else s.replace(',','')
    if s.upper() in _IPCC_NA or s.strip() in ('','-'): return np.nan
    try: return float(s)
    except: return np.nan

def clean_df(df):
    df = df.copy()
    for col in df.columns:
        df[col] = (pd.to_numeric(df[col], errors='coerce') if col == 'year'
                   else df[col].apply(clean_numeric))
    return df

# ── 欄位自動偵測 ──────────────────────────────────────────
def detect_columns(df):
    mapping = {}
    cl = {c: c.lower().replace(' ','_') for c in df.columns}
    patterns = {
        "year":["year","年份","年度"], "co2":["co2_value","co2","二氧化碳"],
        "ch4":["ch4_value","ch4","甲烷"], "n2o":["n2o_value","n2o","氧化亞氮"],
        "total":["total_ghg_emission_value","total_ghg","total","總排放","合計"],
        "land":["co2_absorption_value","absorption","land","土地匯","lulucf"],
        "net":["net_ghg_emission_value","net_ghg","net","淨排放"],
        "energy":["energy","能源"], "industry":["industry","工業"],
        "agri":["agri","農業"], "waste":["waste","waste_ghg","廢棄物"],
        "hfc":["hfcs_value","hfc","hfcs","氫氟碳化物"],
        "pfc":["pfcs_value","pfc","pfcs","全氟碳化物"],
        "sf6":["sf6_value","sf6","六氟化硫"], "nf3":["nf3_value","nf3","三氟化氮"],
    }
    for key, cands in patterns.items():
        for oc, ol in cl.items():
            if any(ol == p or ol.startswith(p) for p in cands):
                if key not in mapping: mapping[key] = oc
    return mapping

# ══════════════════════════════════════════════════════════
# ADF + KPSS 雙重檢定決定 d
# 文獻：
#   Kwiatkowski et al. (1992) Journal of Econometrics 54, 159-178
#   Hyndman & Athanasopoulos (2021) Forecasting: P&P ch.9
#   Nelson & Plosser (1982) JME — I(2) 在環境序列極罕見
# ══════════════════════════════════════════════════════════
def determine_d(series):
    """
    ADF（H0：有單位根）與 KPSS（H0：平穩）雙重檢定
    只有兩者一致才改變 d，不一致時保守取 d=1
    對非負有界序列強制 d<=1：
      - log 轉換已壓縮趨勢強度
      - d=2 在非負序列無物理意義
      - 參照 Hyndman & Athanasopoulos (2021)
    回傳 (d, reason_str, test_detail_dict)
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    s = series[~np.isnan(series)].astype(float)
    n = len(s)
    test_detail = {}

    try:
        adf_stat, adf_p = adfuller(s, autolag='AIC')[:2]
        kpss_stat, kpss_p = kpss(s, regression='c', nlags='auto')[:2]
        test_detail['adf_orig']  = {'stat': round(float(adf_stat),4), 'p': round(float(adf_p),4)}
        test_detail['kpss_orig'] = {'stat': round(float(kpss_stat),4),'p': round(float(kpss_p),4)}
    except Exception as e:
        return 1, f"ADF/KPSS 檢定失敗（{str(e)[:40]}），保守取 d=1", test_detail

    adf_nonstat  = adf_p  > 0.05
    kpss_nonstat = kpss_p < 0.05

    if not adf_nonstat and not kpss_nonstat:
        reason = (f"ADF p={adf_p:.3f}（無單位根）且 KPSS p={kpss_p:.3f}（平穩），"
                  f"原序列 I(0)，d=0")
        return 0, reason, test_detail

    ds = np.diff(s)
    try:
        adf_stat2, adf_p2 = adfuller(ds, autolag='AIC')[:2]
        kpss_stat2, kpss_p2 = kpss(ds, regression='c', nlags='auto')[:2]
        test_detail['adf_diff1']  = {'stat': round(float(adf_stat2),4), 'p': round(float(adf_p2),4)}
        test_detail['kpss_diff1'] = {'stat': round(float(kpss_stat2),4),'p': round(float(kpss_p2),4)}
    except Exception:
        return 1, "一階差分後 ADF/KPSS 失敗，保守取 d=1", test_detail

    if adf_p2 <= 0.05 and kpss_p2 >= 0.05:
        reason = (f"原序列 ADF p={adf_p:.3f}，KPSS p={kpss_p:.3f}（不平穩）；"
                  f"一階差分後 ADF p={adf_p2:.3f}，KPSS p={kpss_p2:.3f}（平穩），d=1")
        return 1, reason, test_detail

    # 一階差分後仍不平穩，但對非負序列強制上限 d=1
    reason = (f"一階差分後 ADF p={adf_p2:.3f}，KPSS p={kpss_p2:.3f}，"
              f"檢定顯示可能需 d=2，但依非負序列物理約束與小樣本限制（n={n}），"
              f"強制取 d=1（Hyndman & Athanasopoulos, 2021）")
    return 1, reason, test_detail


# ── pmdarima ─────────────────────────────────────────────
try:
    from pmdarima import auto_arima as _pm_auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

# ══════════════════════════════════════════════════════════
# Auto-ARIMA 選階
# ══════════════════════════════════════════════════════════
def select_arima_order(series, max_p=3, max_q=3):
    s = series[~np.isnan(series)]; n = len(s)
    if n < 30:   max_p, max_q = min(max_p,1), min(max_q,1)
    elif n < 40: max_p, max_q = min(max_p,2), min(max_q,2)

    d, d_reason, d_tests = determine_d(s)
    warning = None; method = "pmdarima"

    if PMDARIMA_AVAILABLE:
        try:
            model = _pm_auto_arima(
                s, start_p=0, max_p=max_p, start_q=0, max_q=max_q,
                d=d, seasonal=False, information_criterion='bic',
                stepwise=True, error_action='ignore',
                suppress_warnings=True, n_jobs=1)
            best_p, best_d, best_q = model.order
            best_bic = model.bic()
            tbl = _build_aic_table(s, best_d, max_p, max_q, n)
            if n < 35 and best_p + best_q > 2:
                warning = f"⚠️ 樣本數僅 {n} 筆，已採用 BIC 準則抑制過擬合"
            exp = build_exp(best_p, best_d, best_q, d_reason, d_tests, n,
                            method="pmdarima·BIC + ADF/KPSS")
            return {"p":best_p,"d":best_d,"q":best_q,"aic":round(best_bic,2),
                    "d_reason":d_reason,"d_tests":d_tests,"warning":warning,
                    "explanation":exp,"aic_table":tbl,"sample_size":n,"engine":"pmdarima"}
        except Exception as e:
            warning = f"⚠️ pmdarima 失敗（{str(e)[:60]}），切換手工 BIC"
            method  = "fallback·BIC"

    sd = s.copy().astype(float)
    for _ in range(d): sd = np.diff(sd)
    best_aic, best_p, best_q = np.inf, 0, 0; tbl = []
    for p in range(max_p+1):
        for q in range(max_q+1):
            try:
                if p > 0 and len(sd) > p+1:
                    X = np.column_stack([sd[p-i-1:len(sd)-i-1] for i in range(p)]
                                        + [np.ones(len(sd)-p)])
                    y = sd[p:]
                    if X.shape[0] < X.shape[1]+2: continue
                    coef,_,_,_ = np.linalg.lstsq(X, y, rcond=None)
                    resid = y - X@coef; sig2 = np.var(resid); k = p+q+1
                else:
                    resid = sd-np.mean(sd); sig2 = np.var(resid); k = 1
                if sig2 <= 0: continue
                m = len(sd)-p
                if m < 2: continue
                penalty = k*np.log(m) if n<40 else 2*k
                ll = -0.5*m*np.log(2*np.pi*sig2) - 0.5*m
                score = penalty - 2*ll
                tbl.append({"p":p,"d":d,"q":q,"AIC":round(score,2)})
                if score < best_aic: best_aic=score; best_p,best_q=p,q
            except: continue
    exp = build_exp(best_p, d, best_q, d_reason, d_tests, n, method=method)
    return {"p":best_p,"d":d,"q":best_q,"aic":round(best_aic,2),
            "d_reason":d_reason,"d_tests":d_tests,"warning":warning,
            "explanation":exp,"aic_table":sorted(tbl,key=lambda x:x["AIC"])[:12],
            "sample_size":n,"engine":"fallback"}


def _build_aic_table(s, d, max_p, max_q, n):
    sd = s.copy().astype(float)
    for _ in range(d): sd = np.diff(sd)
    tbl = []
    for p in range(max_p+1):
        for q in range(max_q+1):
            try:
                if p > 0 and len(sd) > p+1:
                    X = np.column_stack([sd[p-i-1:len(sd)-i-1] for i in range(p)]
                                        + [np.ones(len(sd)-p)])
                    y = sd[p:]
                    if X.shape[0] < X.shape[1]+2: continue
                    coef,_,_,_ = np.linalg.lstsq(X,y,rcond=None)
                    resid=y-X@coef; sig2=np.var(resid); k=p+q+1
                else:
                    resid=sd-np.mean(sd); sig2=np.var(resid); k=1
                if sig2<=0: continue
                m=len(sd)-p
                if m<2: continue
                penalty=k*np.log(m) if n<40 else 2*k
                ll=-0.5*m*np.log(2*np.pi*sig2)-0.5*m
                tbl.append({"p":p,"d":d,"q":q,"AIC":round(penalty-2*ll,2)})
            except: continue
    return sorted(tbl,key=lambda x:x["AIC"])[:12]


def build_exp(p, d, q, d_reason, d_tests, n, method="BIC"):
    DE = {
        0:"**d=0**：ADF 與 KPSS 雙重檢定一致確認原序列已平穩，無需差分。",
        1:f"**d=1**：{d_reason}。",
    }
    PE = {
        0:"**p=0**：PACF 無顯著截尾，無 AR 自回歸結構。",
        1:"**p=1**：PACF 在 lag=1 截尾，具一年慣性。",
        2:"**p=2**：PACF 在 lag=2 截尾，具景氣循環慣性。",
        3:"**p=3**：三期自回歸，小樣本需注意過擬合。",
    }
    QE = {
        0:"**q=0**：ACF 無顯著截尾，殘差無移動平均結構。",
        1:"**q=1**：ACF 在 lag=1 截尾，衝擊效果持續一年後消退。",
        2:"**q=2**：衝擊效果延續兩年。",
        3:"**q=3**：衝擊效果延續三年。",
    }
    adf_orig  = d_tests.get('adf_orig',  {})
    kpss_orig = d_tests.get('kpss_orig', {})
    adf_d1    = d_tests.get('adf_diff1', {})
    kpss_d1   = d_tests.get('kpss_diff1',{})
    test_str = (
        f"\n\n> 🔬 **d 決定依據（ADF + KPSS 雙重檢定）**\n\n"
        f"> 原序列：ADF p={adf_orig.get('p','—')}，KPSS p={kpss_orig.get('p','—')}\n\n"
        + (f"> 一階差分：ADF p={adf_d1.get('p','—')}，KPSS p={kpss_d1.get('p','—')}\n\n"
           if adf_d1 else "")
        + f"> 文獻：Kwiatkowski et al. (1992) JoE；Hyndman & Athanasopoulos (2021) ch.9"
    )
    engine_note = f"\n\n> ⚙️ **p, q 選階引擎：{method}**"
    small_note  = (f"\n\n> 📌 **小樣本保護**：n={n}，BIC 懲罰加重。" if n<40 else "")
    return {
        "p": PE.get(p, f"p={p}"),
        "d": DE.get(d, f"d={d}"),
        "q": QE.get(q, f"q={q}"),
        "summary": (f"由 **{method}** 選定 **ARIMA({p},{d},{q})**，"
                    f"d 由 ADF+KPSS 雙重檢定決定。"
                    f"{test_str}{engine_note}{small_note}"),
        "adf_reason": d_reason,
    }


# ══════════════════════════════════════════════════════════
# 三個預測模型
# ══════════════════════════════════════════════════════════

def _fit_log_arima(series, order, steps):
    """log-ARIMA，文獻：Box & Jenkins (1976)"""
    from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
    s_min = float(np.nanmin(series))
    epsilon = max(1.0, abs(s_min)/1000) if s_min<=0 else 0.0
    log_s = np.log(series + epsilon)
    p, d, q = order[0], min(order[1], 1), order[2]
    model = SM_ARIMA(log_s, order=(p,d,q)).fit(method_kwargs={"warn_convergence":False})
    fc_obj = model.get_forecast(steps=steps)
    log_mu = fc_obj.predicted_mean.values
    log_ci = fc_obj.conf_int(alpha=0.05).values
    sigma2 = float(model.params.get("sigma2", np.var(model.resid)))
    fc_mean = np.maximum(np.exp(log_mu + sigma2/2) - epsilon, 0)
    fc_up   = np.maximum(np.exp(log_ci[:,1]) - epsilon, 0)
    fc_lo   = np.maximum(np.exp(log_ci[:,0]) - epsilon, 0)
    fitted  = np.exp(np.array(model.fittedvalues, dtype=float)) - epsilon
    if len(fitted) < len(series):
        fitted = np.concatenate([[float(series[0])], fitted])
    return {
        "forecast":  [round(float(v),2) for v in fc_mean],
        "upper95":   [round(float(v),2) for v in fc_up],
        "lower95":   [round(float(v),2) for v in fc_lo],
        "sigma":     round(float(np.sqrt(sigma2)),4),
        "aic":       round(float(model.aic),4),
        "model_obj": model,
        "in_sample": np.array(fitted[:len(series)], dtype=float),
        "order":     (p,d,q),
    }


def _fit_ets(series, steps):
    """
    ETS，文獻：Hyndman & Khandakar (2008) JSS
    下界只保護 >=0（排放量為非負物理量）
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    best_aic, best_result = np.inf, None
    for error in ['add']:
        for trend in [None, 'add']:
            for damped in ([False] if trend is None else [False, True]):
                try:
                    m = ETSModel(series, error=error, trend=trend,
                                 damped_trend=damped, seasonal=None).fit(
                        disp=False, maxiter=200)
                    if m.aic < best_aic: best_aic=m.aic; best_result=m
                except: continue
    if best_result is None: raise ValueError("ETS 全組合失敗")
    fc_obj = best_result.get_forecast(steps)
    fc_mu  = np.maximum(fc_obj.predicted_mean.values, 0)
    fc_ci  = fc_obj.conf_int(alpha=0.05).values
    fc_up  = np.maximum(fc_ci[:,1], fc_mu)
    fc_lo  = np.maximum(fc_ci[:,0], 0)
    ets_spec = (f"ETS({best_result.model.error_type},"
                f"{best_result.model.trend_type or 'N'},N)"
                + (" damped" if getattr(best_result.model,'damped_trend',False) else ""))
    return {
        "forecast":  [round(float(v),2) for v in fc_mu],
        "upper95":   [round(float(v),2) for v in fc_up],
        "lower95":   [round(float(v),2) for v in fc_lo],
        "sigma":     round(float(np.std(best_result.resid)),4),
        "aic":       round(float(best_aic),4),
        "model_obj": best_result,
        "in_sample": best_result.fittedvalues,
        "ets_spec":  ets_spec,
    }


def _fit_holt(series, steps):
    """
    Holt 雙指數平滑（damped_trend=True）
    趨勢隨時間衰減，防止長期外推線性發散
    文獻：
      Gardner & McKenzie (1985) Management Science 31(10), 1237-1246
      Hyndman & Athanasopoulos (2021) ch.8 — 長期預測建議使用 damped
    """
    from statsmodels.tsa.holtwinters import Holt
    model = Holt(series, damped_trend=True).fit(optimized=True, remove_bias=True)
    fc = model.forecast(steps)
    resid = model.resid
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else float(np.std(resid))
    ci_width = 1.96 * sigma * np.sqrt(np.arange(1, steps+1))
    fc_mean = np.maximum(fc.values, 0)
    fc_up   = np.maximum(fc.values + ci_width, 0)
    fc_lo   = np.maximum(fc.values - ci_width, 0)
    fitted  = model.fittedvalues.values
    if len(fitted) < len(series):
        fitted = np.concatenate([[float(series[0])], fitted])
    return {
        "forecast":   [round(float(v),2) for v in fc_mean],
        "upper95":    [round(float(v),2) for v in fc_up],
        "lower95":    [round(float(v),2) for v in fc_lo],
        "sigma":      round(sigma, 4),
        "aic":        round(float(model.aic), 4),
        "model_obj":  model,
        "in_sample":  np.array(fitted[:len(series)], dtype=float),
        "holt_spec":  "Holt(damped=True)",
        "alpha":      round(float(model.params.get('smoothing_level', np.nan)), 4),
        "beta":       round(float(model.params.get('smoothing_trend', np.nan)), 4),
        "phi":        round(float(model.params.get('damping_trend', np.nan)), 4),
    }


def _model_validation(series, in_sample, model_type="arima"):
    """MAPE/RMSE/MAE/R²/Ljung-Box，文獻：Hyndman & Koehler (2006) IJF"""
    y    = series.astype(float)
    yhat = np.array(in_sample, dtype=float)
    min_n = min(len(y), len(yhat))
    y, yhat = y[-min_n:], yhat[-min_n:]
    resid  = y - yhat
    ss_res = np.sum(resid**2); ss_tot = np.sum((y-np.mean(y))**2)
    nonzero = y != 0
    mape = float(np.mean(np.abs(resid[nonzero]/y[nonzero]))*100) if nonzero.any() else None
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae  = float(np.mean(np.abs(resid)))
    r2   = float(1-ss_res/ss_tot) if ss_tot > 0 else None
    lb_lag = max(1, min(10, len(resid)//5))
    try:
        import statsmodels.stats.diagnostic as sm_diag
        lb = sm_diag.acorr_ljungbox(resid, lags=[lb_lag], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[0])
        lb_pval = float(lb["lb_pvalue"].iloc[0])
        lb_pass = lb_pval > 0.05
    except: lb_stat=lb_pval=lb_pass=None
    return {
        "mape":    round(mape,4) if mape is not None else None,
        "rmse":    round(rmse,2), "mae": round(mae,2),
        "r2":      round(r2,4)   if r2   is not None else None,
        "lb_stat": round(lb_stat,4) if lb_stat is not None else None,
        "lb_pval": round(lb_pval,4) if lb_pval is not None else None,
        "lb_pass": lb_pass, "lb_lag": lb_lag, "n": len(y),
    }


# ══════════════════════════════════════════════════════════
# 核心選模：三模型競爭，OOS RMSE 選最佳
# 文獻：
#   Tashman (2000) IJF — Rolling origin evaluation
#   Hyndman & Koehler (2006) IJF — OOS accuracy measures
# ══════════════════════════════════════════════════════════
def select_best_model(series, order, steps):
    """
    三模型競爭：log-ARIMA / ETS / Holt(damped)
    選模標準：OOS RMSE（rolling origin 1 步預測）
    樣本太少（n<15）時退回 AIC（附警告）
    """
    results = {}; errors = {}
    p, d, q = order
    log_order = (p, min(d,1), q)

    for key, fn, arg in [
        ('log_arima', _fit_log_arima, log_order),
        ('ets',       _fit_ets,       None),
        ('holt',      _fit_holt,      None),
    ]:
        try:
            results[key] = fn(series, arg, steps) if arg is not None else fn(series, steps)
        except Exception as e:
            errors[key] = str(e)
            print(f"[{key} error] {e}")

    if not results:
        fc = _arima_fallback(series, order, steps)
        fc.update({'fit_errors':errors,'best_model':'fallback',
                   'used_order':order,'validation':{},'oos_rmse':{}})
        return fc

    # ── OOS RMSE 選模 ──
    s_clean = series[~np.isnan(series)].astype(float)
    n = len(s_clean)
    holdout_n = min(3, max(2, n//8)) if n >= 15 else None
    oos_rmse  = {}
    selection_method = "OOS RMSE（rolling origin）"

    if holdout_n and len(results) > 1:
        for key in results:
            errs = []
            for i in range(holdout_n):
                train  = s_clean[:n - holdout_n + i]
                actual = s_clean[n - holdout_n + i]
                try:
                    if key == 'log_arima': r = _fit_log_arima(train, log_order, 1)
                    elif key == 'ets':     r = _fit_ets(train, 1)
                    else:                  r = _fit_holt(train, 1)
                    errs.append((float(r['forecast'][0]) - actual)**2)
                except: errs.append(np.nan)
            rmse_v = (float(np.sqrt(np.nanmean(errs)))
                      if not all(np.isnan(errs)) else np.inf)
            oos_rmse[key] = round(rmse_v, 2)
        best_key = min(oos_rmse, key=lambda k: oos_rmse[k])
    else:
        best_key = min(results, key=lambda k: results[k]['aic'])
        selection_method = "AIC（樣本數不足以做 OOS，跨空間比較僅供參考）"

    best = results[best_key]
    in_s = best.get('in_sample')
    val  = {}
    if in_s is not None and len(in_s) > 0:
        try: val = _model_validation(series, in_s, best_key)
        except Exception as e: print(f"[validation error] {e}")

    print(f"[select_best_model] best={best_key}, oos_rmse={oos_rmse}, errors={errors}")

    return {
        "forecast":         best['forecast'],
        "upper95":          best['upper95'],
        "lower95":          best['lower95'],
        "sigma":            best['sigma'],
        "best_model":       best_key,
        "model_aic":        best['aic'],
        "arima_aic":        results['log_arima']['aic'] if 'log_arima' in results else None,
        "ets_aic":          results['ets']['aic']       if 'ets'       in results else None,
        "holt_aic":         results['holt']['aic']      if 'holt'      in results else None,
        "ets_spec":         results['ets'].get('ets_spec','ETS') if 'ets' in results else None,
        "holt_spec":        results['holt'].get('holt_spec','Holt(damped=True)') if 'holt' in results else None,
        "holt_params":      {"alpha":results['holt'].get('alpha'),
                             "beta": results['holt'].get('beta'),
                             "phi":  results['holt'].get('phi')} if 'holt' in results else {},
        "validation":       val,
        "fit_errors":       errors if errors else None,
        "used_order":       log_order,
        "oos_rmse":         oos_rmse,
        "selection_method": selection_method,
    }


def _arima_fallback(series, order, steps):
    """純手工 AR 預測，所有模型失敗時使用"""
    p, d, q = order
    s = series[~np.isnan(series)].astype(float)
    sd = s.copy()
    for _ in range(min(d,1)): sd = np.diff(sd)
    n = len(sd); ms = np.mean(sd); ar = np.zeros(p)
    if p > 0 and n > p:
        X = np.column_stack([sd[p-i-1:n-i-1] for i in range(p)])
        try: ar,_,_,_ = np.linalg.lstsq(X, sd[p:], rcond=None)
        except: pass
    resid = (sd[p:] - np.array([np.dot(ar, sd[i:i+p][::-1]) for i in range(p,n)])
             if p > 0 and n > p else sd - ms)
    sigma = np.std(resid); ext = list(sd); fd = []
    for _ in range(steps):
        ap = np.dot(ar, ext[-p:][::-1]) if p > 0 else ms
        fd.append(ap); ext.append(ap)
    preds = []
    for i, f in enumerate(fd):
        if d == 0: preds.append(f)
        else: preds.append((s[-1] if i==0 else preds[-1]) + f)
    preds = np.array(preds); sq = np.sqrt(np.arange(1, steps+1))
    return {
        "forecast": [round(float(v),2) for v in np.maximum(preds,0)],
        "upper95":  [round(float(v),2) for v in preds+1.96*sigma*sq],
        "lower95":  [round(float(v),2) for v in np.maximum(preds-1.96*sigma*sq,0)],
        "sigma":    round(float(sigma),4),
    }


# ══════════════════════════════════════════════════════════
# Hold-out OOS + DM 檢定（三模型版）
# ══════════════════════════════════════════════════════════
def holdout_validation(series, order, holdout=5):
    s = series[~np.isnan(series)].astype(float); n = len(s)
    if n < holdout+10: holdout = max(3, n//5)
    log_order = (order[0], min(order[1],1), order[2])
    arima_errs, ets_errs, holt_errs, actuals = [], [], [], []
    train_end = n - holdout
    for i in range(holdout):
        train = s[:train_end+i]; actual = s[train_end+i]; actuals.append(actual)
        for key, fn, arg in [('arima',_fit_log_arima,log_order),
                              ('ets',_fit_ets,None),('holt',_fit_holt,None)]:
            try:
                r = fn(train, arg, 1) if arg is not None else fn(train, 1)
                err = float(r['forecast'][0]) - actual
            except: err = np.nan
            if key=='arima': arima_errs.append(err)
            elif key=='ets': ets_errs.append(err)
            else:            holt_errs.append(err)

    def _met(errs, acts):
        e=np.array(errs,dtype=float); a=np.array(acts,dtype=float); nz=a!=0
        mape=float(np.nanmean(np.abs(e[nz]/a[nz]))*100) if nz.any() else None
        return {"mape":round(mape,4) if mape else None,
                "rmse":round(float(np.sqrt(np.nanmean(e**2))),2),
                "mae": round(float(np.nanmean(np.abs(e))),2)}

    return {"holdout_n":holdout,
            "log_arima":_met(arima_errs,actuals),
            "ets":      _met(ets_errs,  actuals),
            "holt":     _met(holt_errs, actuals),
            "arima_errors":[round(float(e),2) if not np.isnan(e) else None for e in arima_errs],
            "ets_errors":  [round(float(e),2) if not np.isnan(e) else None for e in ets_errs],
            "holt_errors": [round(float(e),2) if not np.isnan(e) else None for e in holt_errs]}


def diebold_mariano_test(series, order, holdout=5):
    """DM 檢定，文獻：Diebold & Mariano (1995) JBES；Harvey et al. (1997) IJF"""
    from scipy import stats as sp_stats
    oos = holdout_validation(series, order, holdout)

    def _dm(e1_list, e2_list, label=""):
        e1=np.array(e1_list,dtype=float); e2=np.array(e2_list,dtype=float)
        valid=~(np.isnan(e1)|np.isnan(e2)); e1,e2=e1[valid],e2[valid]; h=len(e1)
        if h < 3: return {"error":f"有效配對不足（{h}筆）","h":h}
        dv=e1**2-e2**2; d_bar=np.mean(dv); se_d=np.std(dv,ddof=1)/np.sqrt(h)
        if se_d < 1e-10:
            return {"dm_stat":0.0,"dm_pval":1.0,"hln_stat":0.0,"hln_pval":1.0,
                    "h":h,"conclusion":"兩模型預測完全相同"}
        dm_stat=float(d_bar/se_d); dm_pval=float(2*(1-sp_stats.norm.cdf(abs(dm_stat))))
        hln_cf=np.sqrt((h+1-2+(1/h))/h); hln_stat=float(dm_stat*hln_cf)
        hln_pval=float(2*(1-sp_stats.t.cdf(abs(hln_stat),df=h-1)))
        conclusion=(f"p={hln_pval:.4f} < 0.05，{'前者' if d_bar<0 else '後者'}顯著較優"
                    if hln_pval<0.05
                    else f"p={hln_pval:.4f} ≥ 0.05，兩模型無顯著差異")
        return {"h":h,"dm_stat":round(dm_stat,4),"dm_pval":round(dm_pval,4),
                "hln_stat":round(hln_stat,4),"hln_pval":round(hln_pval,4),
                "d_bar":round(float(d_bar),4),"conclusion":conclusion}

    return {
        "arima_vs_ets":  _dm(oos['arima_errors'],oos['ets_errors']),
        "arima_vs_holt": _dm(oos['arima_errors'],oos['holt_errors']),
        "ets_vs_holt":   _dm(oos['ets_errors'],  oos['holt_errors']),
        "oos": oos,
        "reference":"Diebold & Mariano (1995) JBES; Harvey et al. (1997) IJF",
    }


# ══════════════════════════════════════════════════════════
# 蒙地卡羅：歷史 bootstrap（無分布假設）
# 文獻：Efron & Tibshirani (1993) An Introduction to the Bootstrap
# ══════════════════════════════════════════════════════════
def monte_carlo_bootstrap(series, steps, n_sim=500, seed=42):
    """
    從歷史年變動率 bootstrap 抽樣（有放回）
    不假設任何 sigma 倍率或分布形狀
    不確定性完全來自資料本身
    """
    rng = np.random.default_rng(seed)
    s   = series[~np.isnan(series)].astype(float)
    annual_rates = np.diff(s) / s[:-1]
    base = float(s[-1])
    all_paths = np.zeros((n_sim, steps))
    for sim_i in range(n_sim):
        sampled = rng.choice(annual_rates, size=steps, replace=True)
        v = base
        for t, r in enumerate(sampled):
            v = max(v*(1+r), 0.0); all_paths[sim_i,t] = v
    return {
        "p5":  [round(float(v),1) for v in np.percentile(all_paths, 5,  axis=0)],
        "p25": [round(float(v),1) for v in np.percentile(all_paths, 25, axis=0)],
        "p50": [round(float(v),1) for v in np.percentile(all_paths, 50, axis=0)],
        "p75": [round(float(v),1) for v in np.percentile(all_paths, 75, axis=0)],
        "p95": [round(float(v),1) for v in np.percentile(all_paths, 95, axis=0)],
        "n_sim": n_sim,
        "n_hist_rates": len(annual_rates),
        "method": "historical bootstrap（歷史年變動率 bootstrap，無分布假設）",
        "reference": "Efron & Tibshirani (1993) An Introduction to the Bootstrap",
    }


# ══════════════════════════════════════════════════════════
# AD-EF 情境：動態折年率
# ══════════════════════════════════════════════════════════
def adef_scenarios(base_val, steps, params, bau_cagr=None,
                   last_year=None, base_2005=None):
    """
    AD-EF 三情境
    NDC 2030 / 淨零 2050 折年率從最後資料年動態計算
    當 base_2005 可取得時，折年率有明確數學依據（官方目標倒推）
    文獻：Kaya (1990)；Ang & Zhang (2000) Energy Policy
    """
    gdp=params.get("gdp",0.0); pop=params.get("pop",0.0)
    eff=params.get("eff",0.0); re =params.get("re", 0.0)
    ela=params.get("elasticity",0.0)
    exog_ad=gdp*ela+pop*0.35; exog_ef=eff+re*0.012
    last_year = last_year or 2023

    # NDC 2030 折年率
    if base_2005 and last_year < 2030 and base_val > 0:
        target_2030 = base_2005 * 0.76
        policy_rate = float((target_2030/base_val)**(1/(2030-last_year)) - 1)
        policy_note = (f"{policy_rate*100:+.2f}%/yr"
                       f"（動態：{last_year}→2030，NDC -24% vs 2005）")
    else:
        policy_rate = -0.016
        policy_note = "-1.60%/yr（預設，Taiwan NDC Update 2022）"

    # NDC 淨零 2050 折年率
    if base_2005 and last_year < 2050 and base_val > 0:
        target_2050 = base_2005 * 0.10
        ndc_rate = float((target_2050/base_val)**(1/(2050-last_year)) - 1)
        ndc_note  = (f"{ndc_rate*100:+.2f}%/yr"
                     f"（動態：{last_year}→2050，淨零 -90% vs 2005）")
    else:
        ndc_rate = -0.045
        ndc_note = "-4.50%/yr（預設，國發會 2050 淨零路徑）"

    scenarios = {
        "bau":    {"base_rate": float(bau_cagr) if bau_cagr is not None else 0.004,
                   "label":"基準情境 BAU","color":"#f59e0b",
                   "citation":"環境部排放清冊 (2024)；CAGR 由資料 2005-2019 計算",
                   "rate_note":(f"{float(bau_cagr)*100:+.2f}%/yr（資料 CAGR 2005-2019）"
                                if bau_cagr is not None else "+0.40%/yr（預設值）")},
        "policy": {"base_rate": policy_rate,
                   "label":"積極政策情境（NDC 2030）","color":"#38bdf8",
                   "citation":"Taiwan NDC Update (2022), UNFCCC Submission",
                   "rate_note":policy_note},
        "ndc":    {"base_rate": ndc_rate,
                   "label":"NDC 淨零情境（2050）","color":"#00e5c0",
                   "citation":"台灣 2050 淨零排放路徑 (2022)；IPCC AR6 WG3 C1",
                   "rate_note":ndc_note},
    }
    result = {}
    for key, sc in scenarios.items():
        net_rate = sc["base_rate"] + exog_ad - exog_ef
        vals=[]; v=base_val
        for _ in range(steps):
            v=max(v*(1+net_rate),0.0); vals.append(round(v,2))
        result[key]={
            "values":vals,"label":sc["label"],"color":sc["color"],
            "citation":sc["citation"],"rate_note":sc["rate_note"],
            "net_rate":round(net_rate*100,3),
        }
    return result


# ══════════════════════════════════════════════════════════
# 自動方法論段落
# ══════════════════════════════════════════════════════════
def generate_methods_text(ts, orr, fc, dm_result, mc_result,
                          scenarios, hy, za_result=None,
                          sigma_data=None, bau_cagr=None):
    p,d,q=orr['p'],orr['d'],orr['q']; n=len(ts); y_start=hy[0]; y_end=hy[-1]
    best_m=fc.get('best_model','log_arima'); val=fc.get('validation',{})
    used=fc.get('used_order',(p,min(d,1),q)); up,ud,uq=used
    sel_meth=fc.get('selection_method','OOS RMSE'); oos_rmse=fc.get('oos_rmse',{})
    d_reason=orr.get('d_reason','ADF+KPSS'); d_tests=orr.get('d_tests',{})
    adf_p =d_tests.get('adf_orig', {}).get('p','—')
    kpss_p=d_tests.get('kpss_orig',{}).get('p','—')
    model_map={"log_arima":f"log-ARIMA({up},{ud},{uq})",
               "ets":fc.get('ets_spec','ETS'),"holt":fc.get('holt_spec','Holt(damped=True)'),
               "fallback":"ARIMA fallback"}
    model_zh=model_en=model_map.get(best_m,best_m)
    mape_str=f"{val.get('mape','N/A')}%" if val.get('mape') else "N/A"
    rmse_str=f"{val.get('rmse','N/A'):,} kt" if val.get('rmse') else "N/A"
    lb_str=(f"Q({val.get('lb_lag',10)})={val.get('lb_stat','N/A')}, "
            f"p={val.get('lb_pval','N/A')}")
    oos_str=", ".join([f"{k}={v}" for k,v in oos_rmse.items()]) if oos_rmse else "N/A"
    mc_method=mc_result.get('method','bootstrap') if mc_result else 'N/A'
    n_sim=mc_result.get('n_sim',500) if mc_result else 0
    n_hist_r=mc_result.get('n_hist_rates',0) if mc_result else 0

    dm_str_en=dm_str_zh="DM test not available"
    if dm_result and 'arima_vs_ets' in dm_result:
        oos=dm_result.get('oos',{}); la=oos.get('log_arima',{})
        et=oos.get('ets',{}); ho=oos.get('holt',{})
        ave=dm_result.get('arima_vs_ets',{})
        dm_str_en=(f"DM test (HLN) comparing 3 models. OOS MAPE: "
                   f"log-ARIMA={la.get('mape','—')}%, ETS={et.get('mape','—')}%, "
                   f"Holt={ho.get('mape','—')}%. ARIMA vs ETS: {ave.get('conclusion','—')}.")
        dm_str_zh=(f"三模型 DM 檢定。OOS MAPE：log-ARIMA={la.get('mape','—')}%，"
                   f"ETS={et.get('mape','—')}%，Holt={ho.get('mape','—')}%。"
                   f"ARIMA vs ETS：{ave.get('conclusion','—')}。")

    za_str_en=za_str_zh=""
    if za_result and za_result.get('skipped'):
        za_str_en=f"ZA test skipped: {za_result.get('reason','')}."
        za_str_zh=f"ZA 檢定未執行：{za_result.get('reason','')}。"
    elif za_result and not za_result.get('error'):
        za_str_en=(f"Zivot-Andrews (1992) test: stat={za_result['za_stat']}, "
                   f"p={za_result['za_pval']}, breakpoint={za_result['bp_year']}. "
                   f"{za_result['conclusion']}.")
        za_str_zh=(f"ZA 統計量={za_result['za_stat']}，p={za_result['za_pval']}，"
                   f"斷點={za_result['bp_year']}年。{za_result['conclusion']}。")

    sc_b=scenarios.get('bau',{}); sc_p=scenarios.get('policy',{}); sc_n=scenarios.get('ndc',{})

    en_text=f"""3. Methodology

3.1 Data
Annual GHG emissions ({y_start}-{y_end}, n={n}, kt CO2e).

3.2 Stationarity and Differencing Order
d determined by joint ADF-KPSS testing (Kwiatkowski et al., 1992;
Hyndman & Athanasopoulos, 2021, ch.9).
Original series: ADF p={adf_p}, KPSS p={kpss_p}. {d_reason}
For non-negative bounded series, d is capped at 1 (I(2) has no physical
basis for emission quantities).

3.3 Model Selection
Three competing models: (1) log-ARIMA({up},{ud},{uq}),
(2) {fc.get('ets_spec','ETS')}, (3) {fc.get('holt_spec','Holt(damped=True)')}.
Holt damped-trend prevents long-run linear extrapolation
(Gardner & McKenzie, 1985; Hyndman & Athanasopoulos, 2021 ch.8).
Selection: {sel_meth} (OOS RMSE: {oos_str}). Selected: {model_en}.
Note: AIC is not directly comparable across models operating in different
spaces (Hyndman & Koehler, 2006); OOS RMSE is the primary criterion.

3.3b Structural Break Test
{za_str_en}

3.4 In-Sample Fit
MAPE={mape_str}, RMSE={rmse_str}. Ljung-Box: {lb_str}.

3.5 Out-of-Sample Validation
{dm_str_en}

3.6 Scenario Analysis (Kaya / AD-EF Framework)
BAU: {sc_b.get('rate_note','—')} ({sc_b.get('citation','—')}).
Active Policy (NDC 2030): {sc_p.get('rate_note','—')} ({sc_p.get('citation','—')}).
Net-Zero (NDC 2050): {sc_n.get('rate_note','—')} ({sc_n.get('citation','—')}).
Rates dynamically computed from last data year ({y_end}) to target years.

3.7 Uncertainty Quantification
Monte Carlo ({n_sim:,} iterations, seed=42) using historical bootstrap
resampling ({n_hist_r} observed annual rates, with replacement).
No distributional assumptions. Results: 5th-95th percentile bands.
Reference: Efron & Tibshirani (1993).

References
Diebold & Mariano (1995) JBES; Efron & Tibshirani (1993) Bootstrap;
Gardner & McKenzie (1985) Mgmt Sci 31(10); Harvey et al. (1997) IJF;
Hyndman & Athanasopoulos (2021); Hyndman & Khandakar (2008) JSS;
Hyndman & Koehler (2006) IJF; Kaya (1990);
Kwiatkowski et al. (1992) JoE; Ljung & Box (1978) Biometrika;
Tashman (2000) IJF; Zivot & Andrews (1992) JBES."""

    zh_text=f"""三、研究方法

（一）資料
{y_start}-{y_end} 年 GHG 排放清冊（n={n}，kt CO₂e）。

（二）差分階數決定
本研究採 ADF-KPSS 雙重檢定決定 d（Kwiatkowski et al., 1992；
Hyndman & Athanasopoulos, 2021, ch.9）。
原序列：ADF p={adf_p}，KPSS p={kpss_p}。{d_reason}。
非負有界序列 d 上限設為 1（I(2) 對排放量無物理意義）。

（三）模型選擇
競爭模型：(1) log-ARIMA({up},{ud},{uq})，
(2) {fc.get('ets_spec','ETS')}，
(3) {fc.get('holt_spec','Holt(damped=True)')}。
Holt damped trend 使趨勢隨時間衰減，防止長期外推線性發散
（Gardner & McKenzie, 1985；Hyndman & Athanasopoulos, 2021 ch.8）。
選模標準：{sel_meth}（OOS RMSE：{oos_str}）。選用：{model_zh}。
注意：log-ARIMA AIC 在 log 空間計算，與其他模型不可直接比較
（Hyndman & Koehler, 2006），本研究以 OOS RMSE 為主要準則。

（三之一）結構斷點檢定
{za_str_zh}

（四）樣本內配適度
MAPE={mape_str}，RMSE={rmse_str}。Ljung-Box：{lb_str}。

（五）樣本外驗證
{dm_str_zh}

（六）情境分析（Kaya / AD-EF 框架）
BAU：{sc_b.get('rate_note','—')}（{sc_b.get('citation','—')}）。
積極政策（NDC 2030）：{sc_p.get('rate_note','—')}（{sc_p.get('citation','—')}）。
NDC 淨零（2050）：{sc_n.get('rate_note','—')}（{sc_n.get('citation','—')}）。
折年率均從最後資料年（{y_end}年）動態計算至目標年。

（七）不確定性量化
蒙地卡羅模擬（{n_sim:,}次，seed=42），採歷史年變動率 bootstrap 抽樣
（{n_hist_r} 個觀測年變動率，有放回），不假設任何分布形狀。
結果以 5th-95th 百分位區間呈現（Efron & Tibshirani, 1993）。"""

    return {"en":en_text,"zh":zh_text}


# ══════════════════════════════════════════════════════════
# Zivot-Andrews 結構斷點檢定
# ══════════════════════════════════════════════════════════
def zivot_andrews_test(series, years):
    try:
        from statsmodels.tsa.stattools import zivot_andrews
        s=series[~np.isnan(series)].astype(float)
        za_stat,za_pval,za_cvdict,za_bplag,za_bpidx=zivot_andrews(
            s,trim=0.15,maxlag=None,regression='ct',autolag='AIC')
        bp_year=int(years[za_bpidx]) if za_bpidx<len(years) else None
        cv_5pct=za_cvdict.get('5%',None)
        if za_pval<0.05:
            conclusion=(f"p={za_pval:.4f} < 0.05，拒絕含斷點單位根，"
                        f"序列在 {bp_year} 年斷點前後均平穩")
            arima_note=f"建議加入 {bp_year} 年虛擬變數"
        else:
            conclusion=(f"p={za_pval:.4f} ≥ 0.05，未拒絕含斷點單位根，"
                        f"ARIMA 差分設定合理")
            arima_note="現有差分設定已充分處理趨勢"
        return {"za_stat":round(float(za_stat),4),"za_pval":round(float(za_pval),4),
                "bp_year":bp_year,"bp_lag":int(za_bplag),
                "cv_5pct":round(float(cv_5pct),3) if cv_5pct else None,
                "conclusion":conclusion,"arima_note":arima_note,
                "reference":"Zivot & Andrews (1992) JBES 10(3)"}
    except ImportError:
        return {"error":"statsmodels.tsa.stattools.zivot_andrews 不可用"}
    except Exception as e:
        return {"error":f"ZA 檢定失敗：{str(e)[:80]}"}


# ── 診斷數據 ──────────────────────────────────────────────
def _acf_values(series, nlags=20):
    s=np.array(series,dtype=float); n=len(s); mu=np.mean(s); c0=np.var(s)
    if c0==0: return [1.0]+[0.0]*nlags
    return [1.0]+[float(np.mean((s[:n-k]-mu)*(s[k:]-mu))/c0) for k in range(1,nlags+1)]

def _pacf_values(series, nlags=20):
    acf=_acf_values(series,nlags); pacf=[1.0,acf[1]]; phi={1:[acf[1]]}
    for k in range(2,nlags+1):
        prev=phi[k-1]
        num=acf[k]-sum(prev[j]*acf[k-1-j] for j in range(k-1))
        den=1.0-sum(prev[j]*acf[j+1] for j in range(k-1))
        pk=num/den if abs(den)>1e-10 else 0.0
        phi[k]=[prev[j]-pk*prev[k-2-j] for j in range(k-1)]+[pk]
        pacf.append(float(pk))
    return pacf

def _conf_band(n,nlags):
    cb=1.96/np.sqrt(n); return [round(cb,4)]*(nlags+1)

def compute_diagnostics(series, order, steps=None):
    p,d,q=order; s=np.array(series[~np.isnan(series)],dtype=float); n=len(s)
    nlags=min(20,n//2-1)
    sd=s.copy()
    for _ in range(min(d,1)): sd=np.diff(sd)
    ar=np.zeros(p)
    if p>0 and len(sd)>p:
        X=np.column_stack([sd[p-i-1:len(sd)-i-1] for i in range(p)])
        try: ar,_,_,_=np.linalg.lstsq(X,sd[p:],rcond=None)
        except: pass
    fitted=(np.array([np.dot(ar,sd[i:i+p][::-1]) for i in range(p,len(sd))])
            if p>0 else np.full(len(sd),np.mean(sd)))
    resid=sd[p:]-fitted if p>0 else sd-np.mean(sd)
    sr=np.std(resid) if np.std(resid)>0 else 1.0; std_resid=resid/sr
    diff1=np.diff(s).tolist(); diff2=np.diff(np.diff(s)).tolist() if len(s)>2 else []
    return {
        "residuals":    [round(float(v),4) for v in std_resid],
        "resid_acf":    [round(v,4) for v in _acf_values(std_resid,nlags)],
        "resid_conf":   _conf_band(len(std_resid),nlags),
        "orig_acf":     [round(v,4) for v in _acf_values(s,nlags)],
        "orig_pacf":    [round(v,4) for v in _pacf_values(s,nlags)],
        "orig_conf":    _conf_band(n,nlags),
        "diff1_series": [round(float(v),2) for v in diff1],
        "diff1_acf":    ([round(v,4) for v in _acf_values(np.array(diff1),nlags)]
                         if len(diff1)>nlags else []),
        "diff2_series": [round(float(v),2) for v in diff2],
        "diff2_acf":    ([round(v,4) for v in _acf_values(np.array(diff2),nlags)]
                         if len(diff2)>nlags else []),
        "orig_series":  [round(float(v),2) for v in s.tolist()],
        "nlags":        nlags,
    }


# ── 讀檔 ─────────────────────────────────────────────────
def read_file(f):
    raw=f.read(); fn=f.filename.lower()
    if fn.endswith('.csv'):
        for enc in ['utf-8-sig','utf-8','big5','cp950']:
            try: return pd.read_csv(io.BytesIO(raw),encoding=enc,dtype=str)
            except: continue
        raise ValueError("CSV 編碼解析失敗")
    elif fn.endswith(('.xlsx','.xls')):
        return pd.read_excel(io.BytesIO(raw),dtype=str)
    raise ValueError("僅支援 CSV 或 Excel")

def _load_and_prep(req):
    df=read_file(req.files['file'])
    cm={"year":req.form.get("col_year",""),"total":req.form.get("col_total",""),
        "co2":req.form.get("col_co2",""),"ch4":req.form.get("col_ch4",""),
        "n2o":req.form.get("col_n2o",""),"land":req.form.get("col_land",""),
        "net":req.form.get("col_net",""),"energy":req.form.get("col_energy",""),
        "industry":req.form.get("col_industry",""),"agri":req.form.get("col_agri","")}
    rename={orig:std for std,orig in cm.items() if orig and orig in df.columns}
    detected=detect_columns(df)
    for std,orig in detected.items():
        if std not in rename.values() and orig not in rename and orig in df.columns:
            rename[orig]=std
    df=df.rename(columns=rename)
    if 'year' not in df.columns: df=df.rename(columns={df.columns[0]:'year'})
    df=clean_df(df)
    df=df.dropna(subset=['year']).sort_values('year').reset_index(drop=True)
    df['year']=df['year'].astype(int)
    if 'total' not in df.columns:
        gc=[c for c in ['co2','ch4','n2o','hfc','pfc','sf6','nf3'] if c in df.columns]
        if gc: df['total']=df[gc].sum(axis=1,min_count=1)
    if 'total' not in df.columns: raise ValueError("找不到總排放量欄位")
    dfc=df.dropna(subset=['total']).copy()
    if len(dfc)<5: raise ValueError(f"有效數據不足（{len(dfc)} 筆）")
    return dfc

def _get_adef_params(req):
    return {"gdp":float(req.form.get("adef_gdp",0.0)),
            "elasticity":float(req.form.get("adef_elasticity",0.0)),
            "pop":float(req.form.get("adef_pop",0.0)),
            "eff":float(req.form.get("adef_eff",0.0)),
            "re": float(req.form.get("adef_re", 0.0))}


# ══════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return safe_json({"status":"ok","service":"GHG Forecast API v4"})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return safe_json({"error":"未收到檔案"},400)
    try:
        df=read_file(request.files['file']); detected=detect_columns(df)
        dfc=clean_df(df)
        preview=dfc.head(5).where(pd.notnull(dfc),None).to_dict(orient='records')
        return safe_json({"columns":list(df.columns),"detected":detected,
                          "preview":preview,"rows":len(df)})
    except Exception as e: return safe_json({"error":str(e)},400)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files: return safe_json({"error":"未收到檔案"},400)
    try: dfc=_load_and_prep(request)
    except Exception as e: return safe_json({"error":str(e)},400)

    ts=dfc['total'].values.astype(float); hy=dfc['year'].tolist(); ly=hy[-1]
    steps=2050-ly
    if steps<=0: return safe_json({"error":f"資料已涵蓋至 {ly} 年"},400)

    # BAU CAGR
    try:
        ref_years=np.array(hy); ref_ts=ts.copy()
        mask=(ref_years>=2005)&(ref_years<=2019)
        if mask.sum()>=5:
            bau_cagr=float((ref_ts[mask][-1]/ref_ts[mask][0])**(1/(mask.sum()-1))-1)
        elif len(ts)>=5:
            bau_cagr=float((ts[-1]/ts[0])**(1/(len(ts)-1))-1)
        else:
            bau_cagr=0.0
        sigma_data=float(np.std(np.diff(ts)/ts[:-1],ddof=1))
    except:
        bau_cagr=0.004; sigma_data=0.008

    # 2005 基準值
    mask_2005=np.array(hy)==2005
    base_2005_val=float(ts[mask_2005][0]) if mask_2005.any() else None

    # ARIMA 選階
    orr=select_arima_order(ts); p,d,q=orr['p'],orr['d'],orr['q']

    # 三模型競爭
    fc=select_best_model(ts,(p,d,q),steps)
    fy=list(range(ly+1,2051))

    # AD-EF 情境
    params=_get_adef_params(request)
    scenarios=adef_scenarios(ts[-1],steps,params,
                              bau_cagr=bau_cagr,last_year=ly,
                              base_2005=base_2005_val)

    # DM 檢定
    n_ts=len(ts)
    if n_ts<20:
        dm_result={"skipped":True,"reason":f"樣本數 {n_ts} 筆（<20）"}
    else:
        holdout_n=3 if n_ts<30 else min(5,n_ts//6)
        try: dm_result=diebold_mariano_test(ts,(p,d,q),holdout=holdout_n)
        except Exception as e: dm_result={"error":str(e)}

    # Zivot-Andrews
    if n_ts<20:
        za_result={"skipped":True,"reason":f"樣本數 {n_ts} 筆（<20）"}
    else:
        try: za_result=zivot_andrews_test(ts,np.array(hy))
        except Exception as e: za_result={"error":str(e)}

    # 蒙地卡羅 bootstrap
    try:
        mc_result=monte_carlo_bootstrap(ts,steps,n_sim=500)
        mc_result['bau']={"p50":mc_result['p50'],
                          "p5": mc_result['p5'],
                          "p95":mc_result['p95']}
    except Exception as e:
        mc_result={"error":str(e)}

    # 方法論段落
    try:
        methods_text=generate_methods_text(
            ts,orr,fc,dm_result,mc_result,scenarios,hy,
            za_result=za_result,sigma_data=sigma_data,bau_cagr=bau_cagr)
    except Exception as e:
        methods_text={"error":str(e)}

    # 氣體種類預測
    GAS_MIN_N=10; GAS_ETS_MIN_N=25; gas_results={}
    for col in ['co2','ch4','n2o','hfc','pfc','sf6','nf3']:
        if col in dfc.columns and not dfc[col].isna().all():
            s=dfc[col].dropna().values.astype(float); n_gas=len(s)
            if n_gas<GAS_MIN_N:
                gas_results[col]={"skipped":True,
                    "reason":f"樣本數 {n_gas} 筆（<{GAS_MIN_N}）",
                    "history":[round(float(v),2) for v in s]}
                continue
            try:
                g_ord=(min(p,1),min(d,1),0)
                g=(select_best_model(s,g_ord,steps) if n_gas>=GAS_ETS_MIN_N
                   else _fit_log_arima(s,g_ord,steps))
                gas_results[col]={
                    "history":  [round(float(v),2) for v in s],
                    "forecast": [round(float(v),2) for v in g['forecast']],
                    "upper95":  [round(float(v),2) for v in g['upper95']],
                    "lower95":  [round(float(v),2) for v in g['lower95']],
                    "best_model":g.get('best_model','log_arima'),"n":n_gas}
            except Exception as e:
                gas_results[col]={"error":str(e)[:80],"n":n_gas}

    # 部門分解
    SECTORS={'energy':{'label':'能源部門','color':'#f97316'},
             'industry':{'label':'工業製程','color':'#a78bfa'},
             'agri':{'label':'農業部門','color':'#4ade80'},
             'waste':{'label':'廢棄物部門','color':'#fb923c'}}
    sector_results={}
    for scol,smeta in SECTORS.items():
        if scol not in dfc.columns or dfc[scol].isna().all(): continue
        sv=dfc[scol].dropna().values.astype(float); n_sv=len(sv)
        if n_sv<10:
            sector_results[scol]={"label":smeta["label"],"color":smeta["color"],
                "skipped":True,"reason":f"樣本數 {n_sv} 筆（<10）"}
            continue
        try:
            sg=select_best_model(sv,(min(p,1),min(d,1),0),steps)
            sector_results[scol]={
                'label':smeta['label'],'color':smeta['color'],
                'history':[round(float(v),2) for v in sv],
                'hist_years':[int(dfc.loc[dfc[scol].notna(),'year'].values[i])
                              for i in range(len(sv))],
                'forecast':[round(float(v),2) for v in sg['forecast']],
                'upper95': [round(float(v),2) for v in sg['upper95']],
                'lower95': [round(float(v),2) for v in sg['lower95']]}
        except: pass

    # 歷史表
    hist_tbl=[]
    for _,row in dfc.iterrows():
        r={"year":int(row['year'])}
        for c in ['energy','industry','agri','waste','land','total','net',
                  'co2','ch4','n2o','hfc','pfc','sf6','nf3']:
            v=row.get(c,None)
            r[c]=round(float(v),2) if v is not None and pd.notna(v) else None
        hist_tbl.append(r)

    # 土地匯外推（加飽和上限）
    land_vals=dfc['land'].dropna().values if 'land' in dfc.columns else []
    if len(land_vals)>=2:
        n_fit=min(len(land_vals),10)
        slope,_=np.polyfit(np.arange(n_fit),land_vals[-n_fit:].astype(float),1)
        # 飽和上限：負值（吸收）最多增加 50%，物理約束
        land_floor=(float(land_vals[-1])*1.5 if land_vals[-1]<0
                    else float(land_vals[-1])*0.5)
        fc_land_series=[max(float(land_vals[-1])+slope*(i+1),land_floor)
                        for i in range(steps)]
    elif len(land_vals)==1:
        slope=0.0; fc_land_series=[float(land_vals[0])]*steps
    else:
        slope=None; fc_land_series=[None]*steps

    fc_tbl=[]
    for i,yr in enumerate(fy):
        fc_total=round(fc['forecast'][i],2)
        fc_land_i=round(fc_land_series[i],2) if fc_land_series[i] is not None else None
        fc_net=round(fc_total+fc_land_i,2) if fc_land_i is not None else None
        fc_tbl.append({"year":yr,"total":fc_total,
                        "upper95":round(fc['upper95'][i],2),
                        "lower95":round(fc['lower95'][i],2),
                        "land":fc_land_i,"net":fc_net,
                        **{c:None for c in ['energy','industry','agri','waste',
                                            'co2','ch4','n2o']}})

    used_order=fc.get('used_order',(p,min(d,1),q))

    return safe_json({
        "status":"ok",
        "hist_years":hy,"hist_total":[round(float(v),2) for v in ts],
        "fc_years":fy,
        "fc_total":[round(float(v),2) for v in fc['forecast']],
        "fc_upper":[round(float(v),2) for v in fc['upper95']],
        "fc_lower":[round(float(v),2) for v in fc['lower95']],
        "sigma":fc['sigma'],
        "arima_order":{"p":p,"d":d,"q":q},
        "arima_explanation":orr['explanation'],
        "aic_table":orr['aic_table'],
        "d_reason":orr.get('d_reason',''),
        "d_tests": orr.get('d_tests',{}),
        "sample_size":orr['sample_size'],
        "warning":orr['warning'],
        "fc_net":[r["net"] for r in fc_tbl],
        "fc_land_series":[r["land"] for r in fc_tbl],
        "fc_land_slope":round(float(slope),2) if slope is not None else None,
        "sector_results":sector_results,"gas_results":gas_results,
        "history_table":hist_tbl,"forecast_table":fc_tbl,
        "dm_result":dm_result,"za_result":za_result,
        "bau_cagr":round(float(bau_cagr)*100,3),
        "sigma_data":round(float(sigma_data)*100,3),
        "mc_result":mc_result,"methods_text":methods_text,
        "model_info":{
            "best_model":       fc.get('best_model'),
            "arima_order":      {"p":used_order[0],"d":used_order[1],"q":used_order[2]},
            "arima_aic":        fc.get('arima_aic'),
            "ets_aic":          fc.get('ets_aic'),
            "holt_aic":         fc.get('holt_aic'),
            "model_aic":        fc.get('model_aic'),
            "ets_spec":         fc.get('ets_spec'),
            "holt_spec":        fc.get('holt_spec'),
            "holt_params":      fc.get('holt_params',{}),
            "validation":       fc.get('validation'),
            "fit_errors":       fc.get('fit_errors'),
            "oos_rmse":         fc.get('oos_rmse',{}),
            "selection_method": fc.get('selection_method',''),
        },
        "diagnostics":compute_diagnostics(ts,(p,d,q)),
        "scenarios":scenarios,
    })


@app.route('/api/scenarios', methods=['POST'])
def scenarios_only():
    if 'file' not in request.files: return safe_json({"error":"未收到檔案"},400)
    try: dfc=_load_and_prep(request)
    except Exception as e: return safe_json({"error":str(e)},400)
    ts=dfc['total'].values.astype(float); hy=dfc['year'].tolist(); ly=hy[-1]
    steps=2050-ly; params=_get_adef_params(request)
    try:
        ref_years=np.array(hy); ref_ts=ts.copy()
        mask=(ref_years>=2005)&(ref_years<=2019)
        bau_cagr=(float((ref_ts[mask][-1]/ref_ts[mask][0])**(1/(mask.sum()-1))-1)
                  if mask.sum()>=5
                  else float((ts[-1]/ts[0])**(1/(len(ts)-1))-1) if len(ts)>=5 else 0.0)
        sigma_data=float(np.std(np.diff(ts)/ts[:-1],ddof=1)) if len(ts)>2 else 0.008
        mask_2005=np.array(hy)==2005
        base_2005_val=float(ts[mask_2005][0]) if mask_2005.any() else None
    except:
        bau_cagr=0.004; sigma_data=0.008; base_2005_val=None
    scenarios=adef_scenarios(ts[-1],steps,params,
                              bau_cagr=bau_cagr,last_year=ly,
                              base_2005=base_2005_val)
    try: mc_result=monte_carlo_bootstrap(ts,steps,n_sim=500)
    except: mc_result=None
    return safe_json({"scenarios":scenarios,"mc_result":mc_result,
                      "bau_cagr":round(bau_cagr*100,3),
                      "sigma_data":round(sigma_data*100,3)})


@app.route('/api/health')
def api_health():
    return safe_json({"status":"running","version":"v4",
                      "improvements":["ADF+KPSS dual test for d",
                                      "3-model: log-ARIMA/ETS/Holt-damped",
                                      "OOS RMSE model selection",
                                      "Historical bootstrap MC",
                                      "Dynamic scenario rates"],
                      "allowed_origins":_allowed_list if not _allow_all else "*"})


if __name__=='__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=os.environ.get('RENDER') is None)
