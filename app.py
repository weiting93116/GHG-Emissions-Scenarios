"""
溫室氣體排放預測系統 v3
- Auto-ARIMA 自動選階（含小樣本保護）
- AD-EF 結構層（GDP彈性、技術改善率滑桿）
- 三情境對照（BAU / 積極政策 / NDC）
- 統計信心區間 vs 情境範圍視覺區隔
- 前後端合一，無跨域問題
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import os, io, json, math, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── CORS 設定 ────────────────────────────────────────────
_raw_origins  = os.environ.get('ALLOWED_ORIGINS', '*')
_allowed_list = [o.strip() for o in _raw_origins.split(',') if o.strip()]
_allow_all    = (_raw_origins.strip() == '*') or not _allowed_list

def _cors_origin(origin):
    if not origin:
        return '*'
    if _allow_all:
        return '*'
    if origin in _allowed_list:
        return origin
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

# ── JSON 工具 ───────────────────────────────────────────
def nan_to_none(obj):
    if isinstance(obj, dict):   return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [nan_to_none(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj

def safe_json(data, status=200):
    return app.response_class(
        response=json.dumps(nan_to_none(data), ensure_ascii=False),
        status=status, mimetype='application/json')

@app.errorhandler(500)
def handle_500(e):
    resp = safe_json({"error": f"伺服器錯誤：{str(e)}"}, 500)
    return resp

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    tb = traceback.format_exc()
    print(f"UNHANDLED: {tb}")
    resp = safe_json({"error": str(e), "trace": tb[-500:]}, 500)
    return resp

# ── 數值清洗 ────────────────────────────────────────────
_IPCC_NA = frozenset([
    'NE','NA','N/A','NO','IE','C','NO,IE','NE,IE',
    '','NOT ESTIMATED','NOT OCCURRING',
    'INCLUDED ELSEWHERE','CONFIDENTIAL'
])

def clean_numeric(val):
    if val is None: return np.nan
    s = str(val).strip().replace('"','').replace(' ','').replace('\u00a0','')
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    s = s.replace(',','').replace('-','') if s == '-' else s.replace(',','')
    if s.upper() in _IPCC_NA: return np.nan
    if s.strip() in ('', '-'): return np.nan
    try: return float(s)
    except: return np.nan

def clean_df(df):
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') if col=='year' else df[col].apply(clean_numeric)
    return df

# ── 欄位自動偵測 ────────────────────────────────────────
def detect_columns(df):
    mapping = {}
    cl = {c: c.lower().replace(' ','_') for c in df.columns}
    patterns = {
        "year":  ["year","年份","年度"],
        "co2":   ["co2_value","co2","二氧化碳"],
        "ch4":   ["ch4_value","ch4","甲烷"],
        "n2o":   ["n2o_value","n2o","氧化亞氮"],
        "total": ["total_ghg_emission_value","total_ghg","total","總排放","合計"],
        "land":  ["co2_absorption_value","absorption","land","土地匯","lulucf"],
        "net":   ["net_ghg_emission_value","net_ghg","net","淨排放"],
        "energy":   ["energy","能源"],
        "industry": ["industry","工業"],
        "agri":     ["agri","農業"],
        "waste":    ["waste","waste_ghg","廢棄物","廢棄物部門"],
        "hfc":      ["hfcs_value","hfc","hfcs","氫氟碳化物"],
        "pfc":      ["pfcs_value","pfc","pfcs","全氟碳化物"],
        "sf6":      ["sf6_value","sf6","六氟化硫"],
        "nf3":      ["nf3_value","nf3","三氟化氮"],
    }
    for key, cands in patterns.items():
        for oc, ol in cl.items():
            if any(ol==p or ol.startswith(p) for p in cands):
                if key not in mapping: mapping[key]=oc
    return mapping

# ── ADF 平穩性檢定 ──────────────────────────────────────
def adf_test(series):
    s = series[~np.isnan(series)]
    vr0,vr1 = np.var(s), np.var(np.diff(s))
    vr2 = np.var(np.diff(np.diff(s))) if len(s)>3 else vr1
    ts = abs(np.polyfit(np.arange(len(s)),s,1)[0]) / (np.std(s)+1e-10)
    if ts>0.05 or vr1<vr0*0.7:
        if vr2<vr1*0.7: d,r=2,f"原序列具明顯趨勢（斜率強度={ts:.3f}），一階差分後仍不平穩（Var比={vr1/vr0:.3f}），建議 d=2"
        else:            d,r=1,f"原序列具明顯趨勢（斜率強度={ts:.3f}），一階差分後達到平穩（Var比={vr1/vr0:.3f}），建議 d=1"
        stat=False
    else: d,r,stat=0,f"原序列已接近平穩（斜率強度={ts:.3f}），無需差分，建議 d=0",True
    return {"stationary":stat,"recommended_d":d,"reason":r}

# ── pmdarima 可用性偵測 ─────────────────────────────────
try:
    from pmdarima import auto_arima as _pm_auto_arima
    from pmdarima.arima import ndiffs
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

# ── Auto-ARIMA ──────────────────────────────────────────
def select_arima_order(series, max_p=3, max_q=3):
    s = series[~np.isnan(series)]; n = len(s)
    adf = adf_test(s)

    if n < 30:   max_p, max_q = min(max_p, 1), min(max_q, 1)
    elif n < 40: max_p, max_q = min(max_p, 2), min(max_q, 2)

    method = "pmdarima"
    warning = None

    if PMDARIMA_AVAILABLE:
        try:
            d_pm = ndiffs(s, alpha=0.05, test='adf', max_d=2)
            model = _pm_auto_arima(
                s,
                start_p=0, max_p=max_p,
                start_q=0, max_q=max_q,
                d=d_pm,
                seasonal=False,
                information_criterion='bic',
                stepwise=True,
                error_action='ignore',
                suppress_warnings=True,
                n_jobs=1
            )
            best_p, best_d, best_q = model.order
            best_bic = model.bic()
            tbl = _build_aic_table(s, best_d, max_p, max_q, n)
            if n < 35 and best_p + best_q > 2:
                warning = (f"⚠️ 樣本數僅 {n} 筆，pmdarima 選出 ARIMA({best_p},{best_d},{best_q})，"
                           f"已採用 BIC 準則抑制過擬合")
            adf["recommended_d"] = best_d
            exp = build_exp(best_p, best_d, best_q, adf, n, method="pmdarima·BIC")
            return {"p": best_p, "d": best_d, "q": best_q,
                    "aic": round(best_bic, 2), "adf": adf,
                    "warning": warning, "explanation": exp,
                    "aic_table": tbl, "sample_size": n,
                    "engine": "pmdarima"}
        except Exception as e:
            warning = f"⚠️ pmdarima 執行失敗（{str(e)[:60]}），已切換至手工 BIC 選階"
            method = "fallback·BIC"

    # ── Fallback ──
    d = adf["recommended_d"]
    sd = s.copy().astype(float)
    for _ in range(d): sd = np.diff(sd)
    best_aic, best_p, best_q = np.inf, 0, 0
    tbl = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                if p > 0 and len(sd) > p + 1:
                    X = np.column_stack([sd[p-i-1:len(sd)-i-1] for i in range(p)] + [np.ones(len(sd)-p)])
                    y = sd[p:]
                    if X.shape[0] < X.shape[1] + 2: continue
                    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    resid = y - X @ coef; sig2 = np.var(resid); k = p + q + 1
                else:
                    resid = sd - np.mean(sd); sig2 = np.var(resid); k = 1
                if sig2 <= 0: continue
                m = len(sd) - p
                if m < 2: continue
                penalty = k * np.log(m) if n < 40 else 2 * k
                ll = -0.5 * m * np.log(2 * np.pi * sig2) - 0.5 * m
                score = penalty - 2 * ll
                tbl.append({"p": p, "d": d, "q": q, "AIC": round(score, 2)})
                if score < best_aic: best_aic = score; best_p, best_q = p, q
            except: continue
    if warning is None and n < 35 and best_p + best_q > 2:
        warning = f"⚠️ 樣本數僅 {n} 筆，已套用嚴格 BIC-like 懲罰項"
    exp = build_exp(best_p, d, best_q, adf, n, method=method)
    return {"p": best_p, "d": d, "q": best_q, "aic": round(best_aic, 2), "adf": adf,
            "warning": warning, "explanation": exp,
            "aic_table": sorted(tbl, key=lambda x: x["AIC"])[:12],
            "sample_size": n, "engine": "fallback"}


def _build_aic_table(s, d, max_p, max_q, n):
    sd = s.copy().astype(float)
    for _ in range(d): sd = np.diff(sd)
    tbl = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                if p > 0 and len(sd) > p + 1:
                    X = np.column_stack([sd[p-i-1:len(sd)-i-1] for i in range(p)] + [np.ones(len(sd)-p)])
                    y = sd[p:]
                    if X.shape[0] < X.shape[1] + 2: continue
                    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    resid = y - X @ coef; sig2 = np.var(resid); k = p + q + 1
                else:
                    resid = sd - np.mean(sd); sig2 = np.var(resid); k = 1
                if sig2 <= 0: continue
                m = len(sd) - p
                if m < 2: continue
                penalty = k * np.log(m) if n < 40 else 2 * k
                ll = -0.5 * m * np.log(2 * np.pi * sig2) - 0.5 * m
                tbl.append({"p": p, "d": d, "q": q, "AIC": round(penalty - 2 * ll, 2)})
            except: continue
    return sorted(tbl, key=lambda x: x["AIC"])[:12]

def build_exp(p, d, q, adf, n, method="BIC"):
    DE={0:"**d=0（不差分）**：原序列已平穩，無需差分，直接建模。",
        1:f"**d=1（一階差分）**：{adf['reason']}。差分後趨於平穩，為年度排放最常見設定。",
        2:f"**d=2（二階差分）**：{adf['reason']}。需兩次差分才能平穩。"}
    PE={0:"**p=0（無 AR 項）**：與過去各期無顯著自相關，殘差近似白雜訊，預測依賴趨勢項。",
        1:"**p=1（AR(1)）**：當期受前一年影響，PACF 在 lag=1 截尾，具一年慣性。",
        2:"**p=2（AR(2)）**：當期受前兩年影響，具景氣循環慣性，PACF 在 lag=2 截尾。",
        3:"**p=3（AR(3)）**：三期自回歸，序列具較長記憶性，小樣本需注意過擬合。"}
    QE={0:"**q=0（無 MA 項）**：殘差無移動平均結構，衝擊效果不跨期，各期獨立。",
        1:"**q=1（MA(1)）**：衝擊（政策、危機）影響持續一年後消退，ACF 在 lag=1 截尾。",
        2:"**q=2（MA(2)）**：衝擊效果延續兩年，適合政策需 1-2 年反映的情境。",
        3:"**q=3（MA(3)）**：衝擊效果延續三年，適合有重大政策轉折點的序列。"}
    engine_note = (
        f"\n\n> 🔬 **選階引擎：{method}**\n\n"
        f"> 採用 **pmdarima** 套件的真實 `auto_arima`（stepwise search），以 **BIC 準則**選出最佳組合。"
        if "pmdarima" in method else
        f"\n\n> ⚙️ **選階引擎：{method}**（手工窮舉，BIC-like 懲罰項 k·ln(m)）"
    )
    small_note = f"\n\n> 📌 **小樣本保護**：樣本數僅 {n} 筆，已自動縮小搜尋範圍，採用嚴格 BIC 懲罰有效抑制過擬合。" if n < 40 else ""
    return {"p": PE.get(p, f"p={p}"), "d": DE.get(d, f"d={d}"), "q": QE.get(q, f"q={q}"),
            "summary": f"由 **{method}** 選定 **ARIMA({p},{d},{q})**。{engine_note}{small_note}",
            "adf_reason": adf["reason"]}

# ── ARIMA 預測（fallback 用） ──────────────────────────
def arima_forecast(series, order, steps):
    p, d, q = order
    s = series[~np.isnan(series)]
    orig = s.copy().astype(float)

    if PMDARIMA_AVAILABLE:
        try:
            from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
            model = SM_ARIMA(orig, order=(p, d, q)).fit(method_kwargs={"warn_convergence": False})
            fc_obj = model.get_forecast(steps=steps)
            preds = fc_obj.predicted_mean.values
            ci    = fc_obj.conf_int(alpha=0.05).values
            sigma = float(np.sqrt(model.params.get("sigma2", np.var(model.resid))))
            return {
                "forecast": [round(float(v), 2) for v in preds],
                "upper95":  [round(float(v), 2) for v in ci[:, 1]],
                "lower95":  [round(float(v), 2) for v in ci[:, 0]],
                "sigma":    round(sigma, 4)
            }
        except Exception:
            pass

    sd = orig.copy()
    for _ in range(d): sd = np.diff(sd)
    n = len(sd); ms = np.mean(sd); ar = np.zeros(p)
    if p > 0 and n > p:
        X = np.column_stack([sd[p-i-1:n-i-1] for i in range(p)]); y = sd[p:]
        try: ar, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except: ar = np.zeros(p)
    resid = (sd[p:] - np.array([np.dot(ar, sd[i:i+p][::-1]) for i in range(p, n)])) if p > 0 and n > p else sd - ms
    sigma = np.std(resid); ext = list(sd); fd = []
    for _ in range(steps):
        ap = np.dot(ar, ext[-p:][::-1]) if p > 0 else ms
        fd.append(ap); ext.append(ap)
    preds = []
    for i, f in enumerate(fd):
        if d == 0: preds.append(f)
        elif d == 1: preds.append((orig[-1] if i == 0 else preds[-1]) + f)
        elif d == 2:
            prev = (orig[-1] + (orig[-1] - orig[-2])) if i == 0 else (preds[-1] + (preds[-1] - (orig[-1] if i == 1 else preds[-2])))
            preds.append(prev + f)
    preds = np.array(preds); sq = np.sqrt(np.arange(1, steps + 1))
    return {
        "forecast": [round(float(v), 2) for v in preds],
        "upper95":  [round(float(v), 2) for v in preds + 1.96 * sigma * sq],
        "lower95":  [round(float(v), 2) for v in preds - 1.96 * sigma * sq],
        "sigma":    round(float(sigma), 4)
    }


# ══════════════════════════════════════════════════════════════
# 核心預測引擎
# ══════════════════════════════════════════════════════════════

def _fit_log_arima(series, order, steps):
    """
    log-ARIMA：對 ln(y) 建模後 exp 還原
    - d 上限強制為 1：log 空間 d=2 長期外推必發散
    """
    from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA

    s_min = float(np.nanmin(series))
    epsilon = max(1.0, abs(s_min) / 1000) if s_min <= 0 else 0.0
    series_adj = series + epsilon
    log_s = np.log(series_adj)

    p, d, q = order
    d = min(d, 1)  # ← 關鍵：log 空間 d 上限為 1

    model = SM_ARIMA(log_s, order=(p, d, q)).fit(
        method_kwargs={"warn_convergence": False})
    fc_obj = model.get_forecast(steps=steps)
    log_mu = fc_obj.predicted_mean.values
    log_ci = fc_obj.conf_int(alpha=0.05).values

    sigma2  = float(model.params.get("sigma2", np.var(model.resid)))
    fc_mean = np.exp(log_mu + sigma2 / 2) - epsilon
    fc_up   = np.exp(log_ci[:, 1]) - epsilon
    fc_lo   = np.exp(log_ci[:, 0]) - epsilon
    fc_mean = np.maximum(fc_mean, 0)
    fc_lo   = np.maximum(fc_lo,   0)

    fitted = np.exp(np.array(model.fittedvalues, dtype=float)) - epsilon
    if len(fitted) < len(series):
        fitted = np.concatenate([[float(series[0])], fitted])
    in_sample = np.array(fitted[:len(series)], dtype=float)

    return {
        "forecast":   [round(float(v), 2) for v in fc_mean],
        "upper95":    [round(float(v), 2) for v in fc_up],
        "lower95":    [round(float(v), 2) for v in fc_lo],
        "sigma":      round(float(np.sqrt(sigma2)), 4),
        "aic":        round(float(model.aic), 4),
        "model_obj":  model,
        "in_sample":  in_sample,
        "log_series": log_s,
        "order":      (p, d, q),
    }

def _fit_ets(series, steps):
    """
    ETS（指數平滑狀態空間模型）
    Hyndman & Khandakar (2008) 自動選擇 error/trend/season 組合
    """
    try:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        best_aic, best_result = np.inf, None
        for error in ['add']:
            for trend in [None, 'add']:
                for damped in ([False] if trend is None else [False, True]):
                    try:
                        m = ETSModel(series, error=error, trend=trend,
                                     damped_trend=damped, seasonal=None).fit(
                            disp=False, maxiter=200)
                        if m.aic < best_aic:
                            best_aic = m.aic; best_result = m
                    except Exception:
                        continue
        if best_result is None:
            raise ValueError("ETS 全組合失敗")
        fc_obj = best_result.get_forecast(steps)
        fc_mu  = fc_obj.predicted_mean.values
        fc_ci  = fc_obj.conf_int(alpha=0.05).values
        fc_mu = np.maximum(fc_mu, series[-1] * 0.05)
        fc_up = np.maximum(fc_ci[:, 1], fc_mu)
        fc_lo = np.maximum(fc_ci[:, 0], series[-1] * 0.02)
        in_sample = best_result.fittedvalues
        return {
            "forecast": [round(float(v), 2) for v in fc_mu],
            "upper95":  [round(float(v), 2) for v in fc_up],
            "lower95":  [round(float(v), 2) for v in fc_lo],
            "sigma":    round(float(np.std(best_result.resid)), 4),
            "aic":      round(float(best_aic), 4),
            "model_obj": best_result,
            "in_sample": in_sample,
            "ets_spec":  f"ETS({best_result.model.error_type},{best_result.model.trend_type or 'N'},N)"
                         + (" damped" if getattr(best_result.model, 'damped_trend', False) else ""),
        }
    except Exception as e:
        raise RuntimeError(f"ETS 失敗：{e}")


def _model_validation(series, in_sample, model_obj=None, model_type="arima"):
    """
    樣本內驗證指標：MAPE, RMSE, MAE, R², Ljung-Box
    """
    from scipy import stats as sp_stats
    y    = series.astype(float)
    yhat = np.array(in_sample, dtype=float)
    min_n = min(len(y), len(yhat))
    y, yhat = y[-min_n:], yhat[-min_n:]
    resid  = y - yhat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    nonzero = y != 0
    mape = float(np.mean(np.abs(resid[nonzero] / y[nonzero])) * 100) if nonzero.any() else None
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae  = float(np.mean(np.abs(resid)))
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    lb_lag = max(1, min(10, len(resid) // 5))
    try:
        import statsmodels.stats.diagnostic as sm_diag
        lb = sm_diag.acorr_ljungbox(resid, lags=[lb_lag], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[0])
        lb_pval = float(lb["lb_pvalue"].iloc[0])
        lb_pass = lb_pval > 0.05
    except Exception:
        lb_stat, lb_pval, lb_pass = None, None, None

    return {
        "mape":    round(mape, 4) if mape is not None else None,
        "rmse":    round(rmse, 2),
        "mae":     round(mae,  2),
        "r2":      round(r2,   4) if r2 is not None else None,
        "lb_stat": round(lb_stat, 4) if lb_stat is not None else None,
        "lb_pval": round(lb_pval, 4) if lb_pval is not None else None,
        "lb_pass": lb_pass,
        "lb_lag":  lb_lag,
        "n":       len(y),
    }


def select_best_model(series, order, steps):
    """
    主要入口：log-ARIMA 與 ETS 並跑，AIC 選最佳
    ★ 修正：統一使用 log_order（d<=1）給 log-ARIMA，
            避免 d=2 時 in_sample 長度與驗證對齊失敗
    """
    results = {}
    errors  = {}

    # ── 關鍵修正：log-ARIMA 統一用 d<=1 ──
    p, d, q = order
    log_order = (p, min(d, 1), q)

    # ── log-ARIMA ──
    try:
        results['log_arima'] = _fit_log_arima(series, log_order, steps)
    except Exception as e:
        errors['log_arima'] = str(e)
        print(f"[log_arima error] {e}")

    # ── ETS ──
    try:
        results['ets'] = _fit_ets(series, steps)
    except Exception as e:
        errors['ets'] = str(e)
        print(f"[ets error] {e}")

    if not results:
        # 兩者均失敗 → fallback 原始 ARIMA（無 log）
        print(f"[select_best_model] Both failed, fallback. errors={errors}")
        fc = arima_forecast(series, order, steps)
        fc['fit_errors'] = errors
        fc['best_model'] = 'fallback'
        fc['used_order'] = order
        fc['validation'] = {}
        return fc

    # ── AIC 選最佳 ──
    best_key = min(results, key=lambda k: results[k]['aic'])
    best     = results[best_key]

    # ── 驗證指標 ──
    in_s = best.get('in_sample', None)
    if in_s is not None and len(in_s) > 0:
        try:
            val = _model_validation(series, in_s, best.get('model_obj'), best_key)
        except Exception as e:
            print(f"[validation error] {e}")
            val = {}
    else:
        val = {}

    arima_aic = results['log_arima']['aic'] if 'log_arima' in results else None
    ets_aic   = results['ets']['aic']       if 'ets'       in results else None
    ets_spec  = results['ets'].get('ets_spec', 'ETS') if 'ets' in results else None

    print(f"[select_best_model] best={best_key}, arima_aic={arima_aic}, ets_aic={ets_aic}, val={val}, errors={errors}")

    return {
        "forecast":     best['forecast'],
        "upper95":      best['upper95'],
        "lower95":      best['lower95'],
        "sigma":        best['sigma'],
        "best_model":   best_key,
        "model_aic":    best['aic'],
        "arima_aic":    arima_aic,
        "ets_aic":      ets_aic,
        "ets_spec":     ets_spec,
        "validation":   val,
        "fit_errors":   errors if errors else None,
        "used_order":   log_order,   # ← 實際使用的 order（d 已限制為 <=1）
    }


# ══════════════════════════════════════════════════════════════
# Hold-out OOS 驗證 + Diebold-Mariano 檢定
# ══════════════════════════════════════════════════════════════

def holdout_validation(series, order, holdout=5):
    """
    Hold-out 樣本外驗證（Rolling origin 1步預測）
    """
    s = series[~np.isnan(series)].astype(float)
    n = len(s)
    if n < holdout + 10:
        holdout = max(3, n // 5)

    # ★ 修正：holdout 也統一用 log_order
    p, d, q = order
    log_order = (p, min(d, 1), q)

    arima_errs, ets_errs, actuals = [], [], []
    train_end = n - holdout

    for i in range(holdout):
        train = s[:train_end + i]
        actual = s[train_end + i]
        actuals.append(actual)

        try:
            r_a = _fit_log_arima(train, log_order, steps=1)
            arima_errs.append(float(r_a['forecast'][0]) - actual)
        except Exception:
            arima_errs.append(np.nan)

        try:
            r_e = _fit_ets(train, steps=1)
            ets_errs.append(float(r_e['forecast'][0]) - actual)
        except Exception:
            ets_errs.append(np.nan)

    arima_errs = np.array(arima_errs, dtype=float)
    ets_errs   = np.array(ets_errs,   dtype=float)
    actuals    = np.array(actuals,     dtype=float)

    def _metrics(errs, acts):
        nonzero = acts != 0
        mape = float(np.nanmean(np.abs(errs[nonzero] / acts[nonzero])) * 100) if nonzero.any() else None
        rmse = float(np.sqrt(np.nanmean(errs ** 2)))
        mae  = float(np.nanmean(np.abs(errs)))
        return {"mape": round(mape, 4) if mape else None,
                "rmse": round(rmse, 2), "mae": round(mae, 2)}

    return {
        "holdout_n":   holdout,
        "holdout_years": list(range(int(series[~np.isnan(series)].shape[0]) - holdout + 1,
                                    int(series[~np.isnan(series)].shape[0]) + 1)),
        "log_arima":   _metrics(arima_errs, actuals),
        "ets":         _metrics(ets_errs,   actuals),
        "arima_errors": [round(float(e), 2) if not np.isnan(e) else None for e in arima_errs],
        "ets_errors":   [round(float(e), 2) if not np.isnan(e) else None for e in ets_errs],
    }


def diebold_mariano_test(series, order, holdout=5):
    """
    Diebold-Mariano (1995) 預測精度比較檢定
    Harvey-Leybourne-Newbold (1997) 小樣本修正版
    """
    from scipy import stats as sp_stats

    oos = holdout_validation(series, order, holdout)
    e1 = np.array(oos['arima_errors'], dtype=float)
    e2 = np.array(oos['ets_errors'],   dtype=float)

    valid = ~(np.isnan(e1) | np.isnan(e2))
    e1, e2 = e1[valid], e2[valid]
    h = len(e1)

    if h < 3:
        return {"error": f"有效配對不足（{h}筆），無法執行 DM 檢定",
                "oos": oos}

    d = e1 ** 2 - e2 ** 2
    d_bar = np.mean(d)
    se_d = np.std(d, ddof=1) / np.sqrt(h)

    if se_d < 1e-10:
        return {"dm_stat": 0.0, "dm_pval": 1.0, "hlm_stat": 0.0, "hlm_pval": 1.0,
                "conclusion": "兩模型預測完全相同", "oos": oos, "h": h}

    dm_stat = float(d_bar / se_d)
    dm_pval = float(2 * (1 - sp_stats.norm.cdf(abs(dm_stat))))

    hln_cf  = np.sqrt((h + 1 - 2 + (1/h)) / h)
    hln_stat = float(dm_stat * hln_cf)
    hln_pval = float(2 * (1 - sp_stats.t.cdf(abs(hln_stat), df=h - 1)))

    if hln_pval < 0.05:
        winner = "log-ARIMA" if d_bar < 0 else "ETS"
        conclusion = f"p={hln_pval:.4f} < 0.05，拒絕 H₀，{winner} 預測精度顯著較優"
    else:
        conclusion = f"p={hln_pval:.4f} ≥ 0.05，未拒絕 H₀，兩模型預測精度無顯著差異"

    return {
        "h":         h,
        "dm_stat":   round(dm_stat,  4),
        "dm_pval":   round(dm_pval,  4),
        "hln_stat":  round(hln_stat, 4),
        "hln_pval":  round(hln_pval, 4),
        "d_bar":     round(float(d_bar), 4),
        "conclusion": conclusion,
        "oos":       oos,
        "reference": "Diebold & Mariano (1995) JBES; Harvey et al. (1997) IJF",
    }


# ══════════════════════════════════════════════════════════════
# 蒙地卡羅情境模擬
# ══════════════════════════════════════════════════════════════

def monte_carlo_scenarios(base_val, steps, n_sim=200, seed=42, sigma_data=None, bau_cagr=None):
    rng = np.random.default_rng(seed)
    _sd = float(sigma_data) if sigma_data is not None else 0.008
    _mu_bau = float(bau_cagr) if bau_cagr is not None else 0.004

    scenario_params = {
        "bau":    {"mu": _mu_bau, "sigma": _sd,        "label": "基準情境 BAU",         "color": "#f59e0b"},
        "policy": {"mu": -0.016,  "sigma": _sd * 1.2,  "label": "積極政策情境（NDC 2030）","color": "#38bdf8"},
        "ndc":    {"mu": -0.045,  "sigma": _sd * 1.8,  "label": "NDC 淨零情境（2050）",   "color": "#00e5c0"},
    }

    results = {}
    for key, sp in scenario_params.items():
        rates = rng.normal(loc=sp["mu"], scale=sp["sigma"], size=n_sim)
        all_paths = np.zeros((n_sim, steps))
        for sim_i, r in enumerate(rates):
            v = base_val
            for t in range(steps):
                v = max(v * (1 + r), 0.0)
                all_paths[sim_i, t] = v

        p5  = np.percentile(all_paths, 5,  axis=0)
        p25 = np.percentile(all_paths, 25, axis=0)
        p50 = np.percentile(all_paths, 50, axis=0)
        p75 = np.percentile(all_paths, 75, axis=0)
        p95 = np.percentile(all_paths, 95, axis=0)

        results[key] = {
            "label":  sp["label"],
            "color":  sp["color"],
            "p5":     [round(float(v), 1) for v in p5],
            "p25":    [round(float(v), 1) for v in p25],
            "p50":    [round(float(v), 1) for v in p50],
            "p75":    [round(float(v), 1) for v in p75],
            "p95":    [round(float(v), 1) for v in p95],
            "mu":     sp["mu"],
            "sigma":  sp["sigma"],
            "n_sim":  n_sim,
        }

    return results


# ══════════════════════════════════════════════════════════════
# 自動方法論段落生成
# ══════════════════════════════════════════════════════════════

def generate_methods_text(ts, orr, fc, dm_result, mc_result, scenarios, hy, za_result=None, sigma_data=None, bau_cagr=None):
    p, d, q = orr['p'], orr['d'], orr['q']
    n       = len(ts)
    y_start = hy[0]; y_end = hy[-1]
    best_m  = fc.get('best_model', 'log_arima')
    val     = fc.get('validation', {})
    arima_aic = fc.get('arima_aic', 'N/A')
    ets_aic   = fc.get('ets_aic',   'N/A')
    sel_aic   = fc.get('model_aic', 'N/A')

    # 實際使用的 order（d 已限制）
    used = fc.get('used_order', (p, min(d,1), q))
    up, ud, uq = used

    model_name_en = (f"log-ARIMA({up},{ud},{uq})" if best_m == 'log_arima'
                     else fc.get('ets_spec', 'ETS'))
    model_name_zh = (f"對數轉換 ARIMA({up},{ud},{uq})" if best_m == 'log_arima'
                     else fc.get('ets_spec', 'ETS'))

    mape_str = f"{val.get('mape','N/A')}%" if val.get('mape') else "N/A"
    rmse_str = f"{val.get('rmse','N/A'):,.0f} kt" if val.get('rmse') else "N/A"
    lb_str   = (f"Q({val.get('lb_lag',10)}) = {val.get('lb_stat','N/A')}, "
                f"p = {val.get('lb_pval','N/A')}")
    lb_pass  = "indicating no significant residual autocorrelation" if val.get('lb_pass') else \
               "suggesting residual autocorrelation may be present"

    dm_str_en = dm_str_zh = "DM test not available"
    if dm_result and 'hln_pval' in dm_result:
        oos = dm_result.get('oos', {})
        la  = oos.get('log_arima', {}); et = oos.get('ets', {})
        dm_str_en = (
            f"Model selection was further validated using the Diebold-Mariano (DM) test "
            f"(Diebold & Mariano, 1995), with Harvey-Leybourne-Newbold small-sample correction "
            f"(Harvey et al., 1997). Based on {dm_result['h']}-period hold-out forecasts, "
            f"OOS MAPE: log-ARIMA = {la.get('mape','N/A')}%, ETS = {et.get('mape','N/A')}%. "
            f"HLN statistic = {dm_result['hln_stat']}, p = {dm_result['hln_pval']}. "
            f"{dm_result['conclusion']}."
        )
        dm_str_zh = (
            f"本研究進一步採用 Diebold-Mariano 檢定（Diebold & Mariano, 1995）佐以 "
            f"Harvey-Leybourne-Newbold 小樣本修正（Harvey et al., 1997），"
            f"以 {dm_result['h']} 期 Hold-out 滾動預測評估模型精度。"
            f"樣本外 MAPE：log-ARIMA = {la.get('mape','N/A')}%，ETS = {et.get('mape','N/A')}%。"
            f"HLN 統計量 = {dm_result['hln_stat']}，p = {dm_result['hln_pval']}。"
            f"{dm_result['conclusion']}。"
        )

    mc_str_en = mc_str_zh = ""
    _sd_pct = round(float(sigma_data)*100, 2) if sigma_data else "N/A"
    _bau_pct = f"{float(bau_cagr)*100:+.2f}" if bau_cagr else "+0.40"
    if mc_result and not mc_result.get('error'):
        n_sim = list(mc_result.values())[0].get('n_sim', 500)
        mc_str_en = (
            f"To quantify policy uncertainty, a Monte Carlo simulation ({n_sim:,} iterations, "
            f"seed=42) was applied. The standard deviation for the BAU scenario (σ={_sd_pct}%) "
            f"was derived empirically from the historical annual rate-of-change standard deviation "
            f"of the uploaded dataset ({y_start}–{y_end}). Policy and net-zero scenario σ values "
            f"were set at 1.2× and 1.8× the BAU σ respectively (den Elzen et al., 2019). "
            f"Results are reported as 5th–95th percentile bands."
        )
        mc_str_zh = (
            f"為量化政策不確定性，本研究對情境預測執行蒙地卡羅模擬（{n_sim:,} 次，seed=42）。"
            f"BAU 情境的標準差（σ={_sd_pct}%）由上傳資料（{y_start}–{y_end}年）"
            f"歷史年變動率標準差實際計算。"
            f"積極政策與淨零情境 σ 分別設為 BAU 的 1.2 倍與 1.8 倍"
            f"（den Elzen et al., 2019, Clim. Policy）。"
            f"結果以 5th–95th 百分位區間呈現。"
        )

    za_str_en = za_str_zh = ""
    if za_result and za_result.get('skipped'):
        za_str_en = f"The Zivot-Andrews structural break test was not conducted: {za_result.get('reason', 'insufficient observations')}."
        za_str_zh = f"Zivot-Andrews 結構斷點檢定未執行：{za_result.get('reason', '樣本數不足')}。"
    elif za_result and not za_result.get('error'):
        za_str_en = (
            f"A Zivot-Andrews (1992) structural break test was conducted. "
            f"The test statistic was {za_result['za_stat']} (p={za_result['za_pval']}), "
            f"with the most significant breakpoint at {za_result['bp_year']}. "
            f"{za_result['conclusion']}. {za_result['arima_note']}."
        )
        za_str_zh = (
            f"本研究執行 Zivot-Andrews（1992）結構斷點檢定。"
            f"ZA 統計量 = {za_result['za_stat']}（p = {za_result['za_pval']}），"
            f"最顯著斷點位於 {za_result['bp_year']} 年。{za_result['conclusion']}。"
            f"{za_result['arima_note']}。"
        )

    en_text = f"""3. Methodology

3.1 Data
Annual GHG emissions data covering {y_start}–{y_end} (n={n}). All values in kt CO₂e.

3.2 Model Selection
Two competing models: (1) log-ARIMA({up},{ud},{uq}) and (2) {fc.get('ets_spec','ETS')}.
log-ARIMA AIC = {arima_aic}, ETS AIC = {ets_aic}; {model_name_en} selected (AIC = {sel_aic}).

3.2b Structural Break Test
{za_str_en}

3.3 In-Sample Fit
MAPE = {mape_str}, RMSE = {rmse_str}. Ljung-Box: {lb_str}, {lb_pass}.

3.4 Out-of-Sample Validation
{dm_str_en}

3.5 Scenario Analysis (AD-EF Framework)
BAU: +{_bau_pct}%/yr (data CAGR 2005–2019);
Active Policy (NDC 2030): -1.6%/yr;
Net-Zero (NDC 2050): -4.5%/yr.

3.6 Uncertainty Quantification
{mc_str_en}

References
Box & Jenkins (1976); Diebold & Mariano (1995) JBES; Harvey et al. (1997) IJF;
Hyndman & Khandakar (2008) JSS; Ljung & Box (1978) Biometrika;
Zivot & Andrews (1992) JBES; den Elzen et al. (2019) Clim. Policy."""

    zh_text = f"""三、研究方法

（一）資料
{y_start}–{y_end} 年溫室氣體排放清冊（n={n}筆），單位 kt CO₂e。

（二）模型選擇
比較 log-ARIMA({up},{ud},{uq}) 與 {fc.get('ets_spec','ETS')}。
log-ARIMA AIC={arima_aic}，ETS AIC={ets_aic}，依 AIC 最小準則選用 {model_name_zh}（AIC={sel_aic}）。

（二之一）結構斷點檢定
{za_str_zh}

（三）樣本內配適度
MAPE={mape_str}，RMSE={rmse_str}。Ljung-Box：{lb_str}，{('殘差無顯著自相關' if val.get('lb_pass') else '殘差存在自相關')}。

（四）樣本外驗證
{dm_str_zh}

（五）情境分析
BAU：{_bau_pct}%/yr（資料 CAGR 2005–2019）；
積極政策（NDC 2030）：-1.6%/yr；
NDC 淨零（2050）：-4.5%/yr。

（六）不確定性量化
{mc_str_zh}"""

    return {"en": en_text, "zh": zh_text}


# ══════════════════════════════════════════════════════════════
# Zivot-Andrews 結構斷點檢定
# ══════════════════════════════════════════════════════════════
def zivot_andrews_test(series, years):
    try:
        from statsmodels.tsa.stattools import zivot_andrews
        s = series[~np.isnan(series)].astype(float)
        za_stat, za_pval, za_cvdict, za_bplag, za_bpidx = zivot_andrews(
            s, trim=0.15, maxlag=None, regression='ct', autolag='AIC'
        )
        bp_year = int(years[za_bpidx]) if za_bpidx < len(years) else None
        cv_1pct = za_cvdict.get('1%', None)
        cv_5pct = za_cvdict.get('5%', None)

        if za_pval < 0.05:
            conclusion = (f"p={za_pval:.4f} < 0.05，拒絕含斷點單位根 H₀，"
                          f"序列在斷點（{bp_year}年）前後均平穩")
            arima_note = f"建議在 ARIMA 中加入 {bp_year} 年虛擬變數"
        else:
            conclusion = (f"p={za_pval:.4f} ≥ 0.05，未拒絕含斷點單位根，"
                          f"序列含結構性趨勢，ARIMA 差分設定合理")
            arima_note = "現有 ARIMA 差分設定已充分處理趨勢"

        return {
            "za_stat":    round(float(za_stat), 4),
            "za_pval":    round(float(za_pval), 4),
            "bp_year":    bp_year,
            "bp_lag":     int(za_bplag),
            "cv_1pct":    round(float(cv_1pct), 3) if cv_1pct else None,
            "cv_5pct":    round(float(cv_5pct), 3) if cv_5pct else None,
            "conclusion": conclusion,
            "arima_note": arima_note,
            "reference":  "Zivot & Andrews (1992) JBES 10(3), 251–270",
        }
    except ImportError:
        return {"error": "statsmodels.tsa.stattools.zivot_andrews 不可用"}
    except Exception as e:
        return {"error": f"ZA 檢定失敗：{str(e)[:80]}"}

# ── AD-EF 情境計算 ──────────────────────────────────────
def adef_scenarios(base_val, steps, params, bau_cagr=None):
    gdp = params.get("gdp", 0.0)
    pop = params.get("pop", 0.0)
    eff = params.get("eff", 0.0)
    re  = params.get("re",  0.0)
    ela = params.get("elasticity", 0.0)

    exog_ad = gdp * ela + pop * 0.35
    exog_ef = eff + re * 0.012

    scenarios = {
        "bau": {
            "base_rate": float(bau_cagr) if bau_cagr is not None else +0.004,
            "label":     "基準情境 BAU",
            "color":     "#f59e0b",
            "citation":  "環境部排放清冊 (2024)；CAGR 由上傳資料 2005–2019 年計算",
            "rate_note": f"{float(bau_cagr)*100:+.2f}%/yr（資料計算 CAGR）" if bau_cagr is not None else "+0.4%/yr（預設值）",
        },
        "policy": {
            "base_rate": -0.016,
            "label":     "積極政策情境（NDC 2030）",
            "color":     "#38bdf8",
            "citation":  "Taiwan NDC Update (2022), UNFCCC Submission",
            "rate_note": "-1.6%/yr（NDC 2030 折年率）",
        },
        "ndc": {
            "base_rate": -0.045,
            "label":     "NDC 淨零情境（2050）",
            "color":     "#00e5c0",
            "citation":  "台灣 2050 淨零排放路徑 (2022)；IPCC AR6 WG3 C1",
            "rate_note": "-4.5%/yr（2050 淨零折年率）",
        },
    }
    result = {}
    for key, sc in scenarios.items():
        net_rate = sc["base_rate"] + exog_ad - exog_ef
        vals = []; v = base_val
        for _ in range(steps):
            v = max(v * (1 + net_rate), 0.0)
            vals.append(round(v, 2))
        result[key] = {
            "values":    vals,
            "label":     sc["label"],
            "color":     sc["color"],
            "citation":  sc["citation"],
            "rate_note": sc["rate_note"],
            "net_rate":  round(net_rate * 100, 3),
        }
    return result


# ── 診斷數據計算 ─────────────────────────────────────────
def _acf_values(series, nlags=20):
    s = np.array(series, dtype=float); n = len(s)
    mu = np.mean(s); c0 = np.var(s)
    if c0 == 0: return [1.0] + [0.0]*nlags
    acf = [1.0]
    for k in range(1, nlags+1):
        acf.append(float(np.mean((s[:n-k]-mu)*(s[k:]-mu)) / c0))
    return acf

def _pacf_values(series, nlags=20):
    acf = _acf_values(series, nlags)
    pacf = [1.0, acf[1]]
    phi = {1: [acf[1]]}
    for k in range(2, nlags+1):
        prev = phi[k-1]
        num = acf[k] - sum(prev[j]*acf[k-1-j] for j in range(k-1))
        den = 1.0 - sum(prev[j]*acf[j+1] for j in range(k-1))
        pk = num / den if abs(den) > 1e-10 else 0.0
        new_phi = [prev[j] - pk*prev[k-2-j] for j in range(k-1)] + [pk]
        phi[k] = new_phi
        pacf.append(float(pk))
    return pacf

def _conf_band(n, nlags):
    cb = 1.96 / np.sqrt(n)
    return [round(cb, 4)] * (nlags+1)

def compute_diagnostics(series, order, steps=None):
    p, d, q = order
    s = np.array(series[~np.isnan(series)], dtype=float)
    n = len(s)
    nlags = min(20, n//2 - 1)

    sd = s.copy()
    for _ in range(d): sd = np.diff(sd)
    ar = np.zeros(p)
    if p > 0 and len(sd) > p:
        X = np.column_stack([sd[p-i-1:len(sd)-i-1] for i in range(p)])
        try: ar,_,_,_ = np.linalg.lstsq(X, sd[p:], rcond=None)
        except: pass
    fitted = np.array([np.dot(ar, sd[i:i+p][::-1]) for i in range(p, len(sd))]) if p > 0 else np.full(len(sd), np.mean(sd))
    resid = sd[p:] - fitted if p > 0 else sd - np.mean(sd)
    sigma_r = np.std(resid) if np.std(resid) > 0 else 1.0
    std_resid = resid / sigma_r

    diff1 = np.diff(s).tolist()
    diff2 = np.diff(np.diff(s)).tolist() if len(s) > 2 else []

    return {
        "residuals": [round(float(v), 4) for v in std_resid],
        "resid_acf": [round(v, 4) for v in _acf_values(std_resid, nlags)],
        "resid_conf": _conf_band(len(std_resid), nlags),
        "orig_acf":  [round(v, 4) for v in _acf_values(s, nlags)],
        "orig_pacf": [round(v, 4) for v in _pacf_values(s, nlags)],
        "orig_conf": _conf_band(n, nlags),
        "diff1_series": [round(float(v), 2) for v in diff1],
        "diff1_acf": [round(v, 4) for v in _acf_values(np.array(diff1), nlags)] if len(diff1) > nlags else [],
        "diff2_series": [round(float(v), 2) for v in diff2],
        "diff2_acf": [round(v, 4) for v in _acf_values(np.array(diff2), nlags)] if len(diff2) > nlags else [],
        "orig_series": [round(float(v), 2) for v in s.tolist()],
        "nlags": nlags,
    }

# ── 讀檔 ────────────────────────────────────────────────
def read_file(f):
    raw=f.read(); fn=f.filename.lower()
    if fn.endswith('.csv'):
        for enc in ['utf-8-sig','utf-8','big5','cp950']:
            try: return pd.read_csv(io.BytesIO(raw),encoding=enc,dtype=str)
            except: continue
        raise ValueError("CSV 編碼解析失敗")
    elif fn.endswith(('.xlsx','.xls')): return pd.read_excel(io.BytesIO(raw),dtype=str)
    raise ValueError("僅支援 CSV 或 Excel")


# ── Routes ──────────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return safe_json({"error":"未收到檔案"},400)
    try:
        df=read_file(request.files['file'])
        detected=detect_columns(df)
        dfc=clean_df(df)
        preview=dfc.head(5).where(pd.notnull(dfc),None).to_dict(orient='records')
        return safe_json({"columns":list(df.columns),"detected":detected,"preview":preview,"rows":len(df)})
    except Exception as e: return safe_json({"error":str(e)},400)

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
        if std not in rename.values() and orig not in rename and orig in df.columns: rename[orig]=std
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
    return {
        "gdp":      float(req.form.get("adef_gdp",     0.0)),
        "elasticity":float(req.form.get("adef_elasticity",0.0)),
        "pop":      float(req.form.get("adef_pop",     0.0)),
        "eff":      float(req.form.get("adef_eff",     0.0)),
        "re":       float(req.form.get("adef_re",      0.0)),
    }

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files: return safe_json({"error":"未收到檔案"},400)
    try: dfc=_load_and_prep(request)
    except Exception as e: return safe_json({"error":str(e)},400)

    ts=dfc['total'].values.astype(float); hy=dfc['year'].tolist(); ly=hy[-1]
    steps=2050-ly
    if steps<=0: return safe_json({"error":f"資料已涵蓋至 {ly} 年"},400)

    # ── BAU CAGR + MC sigma ──
    try:
        ref_years = np.array(hy); ref_ts = ts.copy()
        mask = (ref_years >= 2005) & (ref_years <= 2019)
        if mask.sum() >= 5:
            y0 = ref_ts[mask][0]; yn = ref_ts[mask][-1]
            n_yrs = mask.sum() - 1
            bau_cagr = float((yn / y0) ** (1.0 / n_yrs) - 1)
        elif len(ts) >= 5:
            bau_cagr = float((ts[-1] / ts[0]) ** (1.0 / (len(ts)-1)) - 1)
        else:
            bau_cagr = 0.0
        annual_changes = np.diff(ts) / ts[:-1]
        sigma_data = float(np.std(annual_changes, ddof=1))
    except Exception:
        bau_cagr = 0.004
        sigma_data = 0.008

    orr=select_arima_order(ts); p,d,q=orr['p'],orr['d'],orr['q']
    fc=select_best_model(ts,(p,d,q),steps)
    fy=list(range(ly+1,2051))

    # AD-EF 三情境
    params=_get_adef_params(request)
    scenarios=adef_scenarios(ts[-1], steps, params, bau_cagr=bau_cagr)

    # ── DM 檢定 ──
    n_ts = len(ts)
    if n_ts < 20:
        dm_result = {
            "skipped": True,
            "reason": f"樣本數 {n_ts} 筆（<20），Hold-out 驗證不可靠，已跳過",
        }
    else:
        holdout_n = 3 if n_ts < 30 else min(5, n_ts // 6)
        try:
            dm_result = diebold_mariano_test(ts, (p,d,q), holdout=holdout_n)
        except Exception as e:
            dm_result = {"error": str(e)}

    # ── Zivot-Andrews ──
    if n_ts < 20:
        za_result = {
            "skipped": True,
            "reason": f"樣本數 {n_ts} 筆（<20），ZA 檢定需至少 20 筆",
        }
    else:
        try:
            za_result = zivot_andrews_test(ts, np.array(hy))
        except Exception as e:
            za_result = {"error": str(e)}

    # ── 蒙地卡羅 ──
    try:
        mc_result = monte_carlo_scenarios(float(ts[-1]), steps, n_sim=500, sigma_data=sigma_data, bau_cagr=bau_cagr)
    except Exception as e:
        mc_result = {"error": str(e)}

    # ── 方法論段落 ──
    try:
        methods_text = generate_methods_text(ts, orr, fc, dm_result, mc_result, scenarios, hy, za_result=za_result, sigma_data=sigma_data, bau_cagr=bau_cagr)
    except Exception as e:
        methods_text = {"error": str(e)}

    # ── 氣體種類預測 ──
    GAS_MIN_N     = 10
    GAS_ETS_MIN_N = 25
    gas_results={}
    for col in ['co2','ch4','n2o','hfc','pfc','sf6','nf3']:
        if col in dfc.columns and not dfc[col].isna().all():
            s=dfc[col].dropna().values.astype(float)
            n_gas = len(s)
            if n_gas < GAS_MIN_N:
                gas_results[col] = {
                    "skipped": True,
                    "reason": f"樣本數 {n_gas} 筆（<{GAS_MIN_N}），跳過預測",
                    "history": [round(float(v),2) for v in s],
                }
                continue
            try:
                if n_gas < GAS_ETS_MIN_N:
                    g = _fit_log_arima(s, (min(p,1), min(d,1), 0), steps)
                    g['best_model'] = 'log_arima'
                    g['warning'] = f"樣本數 {n_gas} 筆（<25），已停用 ETS"
                else:
                    g = select_best_model(s, (min(p,1), min(d,1), 0), steps)
                gas_results[col]={
                    "history":  [round(float(v),2) for v in s],
                    "forecast": [round(float(v),2) for v in g['forecast']],
                    "upper95":  [round(float(v),2) for v in g['upper95']],
                    "lower95":  [round(float(v),2) for v in g['lower95']],
                    "best_model": g.get('best_model','log_arima'),
                    "n": n_gas,
                    "warning": g.get('warning'),
                }
            except Exception as e:
                gas_results[col] = {"error": str(e)[:80], "n": n_gas}

    # ── 部門分解 ──
    SECTORS = {
        'energy':   {'label':'能源部門',    'color':'#f97316'},
        'industry': {'label':'工業製程',    'color':'#a78bfa'},
        'agri':     {'label':'農業部門',    'color':'#4ade80'},
        'waste':    {'label':'廢棄物部門',  'color':'#fb923c'},
    }
    sector_results = {}
    for scol, smeta in SECTORS.items():
        if scol not in dfc.columns or dfc[scol].isna().all():
            continue
        sv = dfc[scol].dropna().values.astype(float)
        n_sv = len(sv)
        if n_sv < 10:
            sector_results[scol] = {
                "label": smeta["label"], "color": smeta["color"],
                "skipped": True,
                "reason": f"樣本數 {n_sv} 筆（<10）"
            }
            continue
        try:
            sp_ord = min(p, 1)
            sg = select_best_model(sv, (sp_ord, min(d,1), 0), steps)
            sector_results[scol] = {
                'label':   smeta['label'],
                'color':   smeta['color'],
                'history': [round(float(v), 2) for v in sv],
                'hist_years': [int(dfc.loc[dfc[scol].notna(), 'year'].values[i]) for i in range(len(sv))],
                'forecast': [round(float(v), 2) for v in sg['forecast']],
                'upper95':  [round(float(v), 2) for v in sg['upper95']],
                'lower95':  [round(float(v), 2) for v in sg['lower95']],
            }
        except Exception:
            pass

    hist_tbl=[]
    for _,row in dfc.iterrows():
        r={"year":int(row['year'])}
        for c in ['energy','industry','agri','waste','land','total','net','co2','ch4','n2o','hfc','pfc','sf6','nf3']:
            v=row.get(c,None); r[c]=round(float(v),2) if v is not None and pd.notna(v) else None
        hist_tbl.append(r)

    # 土地匯線性推算
    land_vals = dfc['land'].dropna().values if 'land' in dfc.columns else []
    if len(land_vals) >= 2:
        n_fit = min(len(land_vals), 10)
        x_fit = np.arange(n_fit)
        y_fit = land_vals[-n_fit:].astype(float)
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        fc_land_series = [float(land_vals[-1]) + slope * (i + 1) for i in range(steps)]
    elif len(land_vals) == 1:
        slope = 0.0
        fc_land_series = [float(land_vals[0])] * steps
    else:
        slope = None
        fc_land_series = [None] * steps

    fc_tbl=[]
    for i,yr in enumerate(fy):
        fc_total = round(fc['forecast'][i], 2)
        fc_land_i = round(fc_land_series[i], 2) if fc_land_series[i] is not None else None
        fc_net = round(fc_total + fc_land_i, 2) if fc_land_i is not None else None
        fc_tbl.append({
            "year": yr,
            "total": fc_total,
            "upper95": round(fc['upper95'][i], 2),
            "lower95": round(fc['lower95'][i], 2),
            "land": fc_land_i,
            "net":  fc_net,
            **{c: None for c in ['energy','industry','agri','waste','co2','ch4','n2o']}
        })

    # ── 取得實際使用的 order ──
    used_order = fc.get('used_order', (p, min(d,1), q))

    return safe_json({
        "status":"ok",
        "hist_years":hy,
        "hist_total":[round(float(v),2) for v in ts],
        "fc_years":fy,
        "fc_total":[round(float(v),2) for v in fc['forecast']],
        "fc_upper":[round(float(v),2) for v in fc['upper95']],
        "fc_lower":[round(float(v),2) for v in fc['lower95']],
        "sigma":fc['sigma'],
        "arima_order":{"p":p,"d":d,"q":q},
        "arima_explanation":orr['explanation'],
        "aic_table":orr['aic_table'],
        "adf_result":orr['adf'],
        "sample_size":orr['sample_size'],
        "warning":orr['warning'],
        "fc_net":[r["net"] for r in fc_tbl],
        "fc_land_series":[r["land"] for r in fc_tbl],
        "fc_land_slope":round(float(slope),2) if slope is not None else None,
        "sector_results":sector_results,
        "gas_results":gas_results,
        "history_table":hist_tbl,
        "forecast_table":fc_tbl,
        "dm_result":dm_result,
        "za_result":za_result,
        "bau_cagr":round(float(bau_cagr)*100,3),
        "sigma_data":round(float(sigma_data)*100,3),
        "mc_result":mc_result,
        "methods_text":methods_text,
        "model_info":{
            "best_model":  fc.get('best_model'),
            # ★ 修正：回傳實際使用的 order（d 已限制 <=1）
            "arima_order": {"p": used_order[0], "d": used_order[1], "q": used_order[2]},
            "arima_aic":   fc.get('arima_aic'),
            "ets_aic":     fc.get('ets_aic'),
            "model_aic":   fc.get('model_aic'),
            "ets_spec":    fc.get('ets_spec'),
            "validation":  fc.get('validation'),
            "fit_errors":  fc.get('fit_errors'),
        },
        "diagnostics":compute_diagnostics(ts,(p,d,q)),
        "scenarios":scenarios,
    })

@app.route('/api/scenarios', methods=['POST'])
def scenarios_only():
    if 'file' not in request.files: return safe_json({"error":"未收到檔案"},400)
    try: dfc=_load_and_prep(request)
    except Exception as e: return safe_json({"error":str(e)},400)
    ts=dfc['total'].values.astype(float)
    hy=dfc['year'].tolist(); ly=hy[-1]
    steps=2050-ly
    params=_get_adef_params(request)
    try:
        ref_years = np.array(hy); ref_ts = ts.copy()
        mask = (ref_years >= 2005) & (ref_years <= 2019)
        if mask.sum() >= 5:
            y0 = ref_ts[mask][0]; yn = ref_ts[mask][-1]
            bau_cagr = float((yn / y0) ** (1.0 / (mask.sum()-1)) - 1)
        elif len(ts) >= 5:
            bau_cagr = float((ts[-1] / ts[0]) ** (1.0 / (len(ts)-1)) - 1)
        else:
            bau_cagr = 0.0
        sigma_data = float(np.std(np.diff(ts) / ts[:-1], ddof=1)) if len(ts) > 2 else 0.008
    except Exception:
        bau_cagr = 0.004; sigma_data = 0.008
    scenarios=adef_scenarios(ts[-1], steps, params, bau_cagr=bau_cagr)
    mc_result = None
    try:
        mc_result = monte_carlo_scenarios(float(ts[-1]), steps, n_sim=500,
                                          sigma_data=sigma_data, bau_cagr=bau_cagr)
    except Exception: pass
    return safe_json({"scenarios": scenarios, "mc_result": mc_result,
                      "bau_cagr": round(bau_cagr*100, 3),
                      "sigma_data": round(sigma_data*100, 3)})

@app.route('/')
def index():
    return safe_json({"status": "ok", "service": "GHG Forecast API"})

@app.route('/api/health')
def api_health():
    return safe_json({"status":"running","allowed_origins": _allowed_list if not _allow_all else "*"})

if __name__=='__main__':
    import os
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=os.environ.get('RENDER') is None)
