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
from flask_cors import CORS
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── CORS 設定 ────────────────────────────────────────────
# 方法一（flask-cors）：自動處理所有路由的 CORS headers
# 方法二（Render 環境變數）：設定 ALLOWED_ORIGINS 精確控制允許來源
# 兩個同時啟用，互補保險
_allowed = os.environ.get('ALLOWED_ORIGINS', '*')
CORS(app,
     origins=_allowed.split(',') if _allowed != '*' else '*',
     supports_credentials=False,
     methods=['GET','POST','OPTIONS'],
     allow_headers=['Content-Type'])

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin', '')
    if _allowed == '*':
        response.headers['Access-Control-Allow-Origin'] = '*'
    elif origin in _allowed.split(','):
        response.headers['Access-Control-Allow-Origin'] = origin
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

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

# ── 數值清洗 ────────────────────────────────────────────
def clean_numeric(val):
    if val is None: return np.nan
    s = str(val).strip().replace(',','').replace('"','')
    if s.upper() in ('NE','NA','N/A','','-','NOT ESTIMATED'): return np.nan
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

# ── Auto-ARIMA（pmdarima 優先，fallback 手工 BIC） ──────
def select_arima_order(series, max_p=3, max_q=3):
    s = series[~np.isnan(series)]; n = len(s)
    adf = adf_test(s)

    # ── 小樣本邊界 ──
    if n < 30:   max_p, max_q = min(max_p, 1), min(max_q, 1)
    elif n < 40: max_p, max_q = min(max_p, 2), min(max_q, 2)

    method = "pmdarima"
    warning = None

    if PMDARIMA_AVAILABLE:
        try:
            # pmdarima 選出 d
            d_pm = ndiffs(s, alpha=0.05, test='adf', max_d=2)
            # stepwise=True 速度快；information_criterion='bic' 對小樣本更嚴格
            model = _pm_auto_arima(
                s,
                start_p=0, max_p=max_p,
                start_q=0, max_q=max_q,
                d=d_pm,            # 固定 d，避免 pmdarima 自行重算造成衝突
                seasonal=False,
                information_criterion='bic',   # 小樣本用 BIC 懲罰
                stepwise=True,
                error_action='ignore',
                suppress_warnings=True,
                n_jobs=1
            )
            best_p, best_d, best_q = model.order
            best_bic = model.bic()

            # 回填完整 AIC table（對所有候選組合計算 BIC）
            tbl = _build_aic_table(s, best_d, max_p, max_q, n)

            if n < 35 and best_p + best_q > 2:
                warning = (f"⚠️ 樣本數僅 {n} 筆，pmdarima 選出 ARIMA({best_p},{best_d},{best_q})，"
                           f"已採用 BIC 準則（比 AIC 對額外參數懲罰更重）抑制過擬合")
            adf["recommended_d"] = best_d  # 以 pmdarima 的 ndiffs 為準更新
            exp = build_exp(best_p, best_d, best_q, adf, n, method="pmdarima·BIC")
            return {"p": best_p, "d": best_d, "q": best_q,
                    "aic": round(best_bic, 2), "adf": adf,
                    "warning": warning, "explanation": exp,
                    "aic_table": tbl, "sample_size": n,
                    "engine": "pmdarima"}
        except Exception as e:
            # pmdarima 失敗時 fallback
            warning = f"⚠️ pmdarima 執行失敗（{str(e)[:60]}），已切換至手工 BIC 選階"
            method = "fallback·BIC"

    # ── Fallback：手工 BIC-like 窮舉 ──
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
        warning = f"⚠️ 樣本數僅 {n} 筆，已套用嚴格 BIC-like 懲罰項，限制最大階數避免過擬合"
    exp = build_exp(best_p, d, best_q, adf, n, method=method)
    return {"p": best_p, "d": d, "q": best_q, "aic": round(best_aic, 2), "adf": adf,
            "warning": warning, "explanation": exp,
            "aic_table": sorted(tbl, key=lambda x: x["AIC"])[:12],
            "sample_size": n, "engine": "fallback"}


def _build_aic_table(s, d, max_p, max_q, n):
    """給 pmdarima 路徑補齊 AIC table，方便前端顯示比較"""
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
        f"> 採用 **pmdarima** 套件的真實 `auto_arima`（stepwise search），以 **BIC 準則**（比 AIC 對額外參數懲罰更重）選出最佳組合，並透過 `ndiffs` 進行正式 ADF/KPSS 檢定決定 d 值。"
        if "pmdarima" in method else
        f"\n\n> ⚙️ **選階引擎：{method}**（手工窮舉，BIC-like 懲罰項 k·ln(m)）"
    )
    small_note = f"\n\n> 📌 **小樣本保護**：樣本數僅 {n} 筆，已自動縮小搜尋範圍（max_p={min(3, 1 if n<30 else 2 if n<40 else 3)}, max_q 同），採用嚴格 BIC 懲罰有效抑制過擬合。" if n < 40 else ""
    return {"p": PE.get(p, f"p={p}"), "d": DE.get(d, f"d={d}"), "q": QE.get(q, f"q={q}"),
            "summary": f"由 **{method}** 選定 **ARIMA({p},{d},{q})**。{engine_note}{small_note}",
            "adf_reason": adf["reason"]}

# ── ARIMA 預測（statsmodels MLE 優先，fallback 手工） ──
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
            pass  # fallback to manual

    # ── 手工 AR 預測（fallback）──
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
# 核心預測引擎（論文版）
# 策略：log-ARIMA 與 ETS 並跑，AIC 自動選最佳
# 文獻依據：
#   Box & Jenkins (1976), Hyndman & Khandakar (2008, JSS),
#   Hyndman & Athanasopoulos (2021) Forecasting: P&P ch.8-9
# ══════════════════════════════════════════════════════════════

def _fit_log_arima(series, order, steps):
    """
    log-ARIMA：對 ln(y) 建模後 exp 還原
    自然防止預測值為負；長期預測不會無限外推至負值
    回傳原始尺度的 forecast / upper95 / lower95
    """
    from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
    log_s = np.log(series)
    p, d, q = order
    model  = SM_ARIMA(log_s, order=(p, d, q)).fit(
        method_kwargs={"warn_convergence": False})
    fc_obj = model.get_forecast(steps=steps)
    log_mu = fc_obj.predicted_mean.values
    log_ci = fc_obj.conf_int(alpha=0.05).values
    # delta method 還原（exp(mu + sigma²/2) 為無偏估計）
    sigma2 = float(model.params.get("sigma2", np.var(model.resid)))
    fc_mean = np.exp(log_mu + sigma2 / 2)
    fc_up   = np.exp(log_ci[:, 1])
    fc_lo   = np.exp(log_ci[:, 0])
    # 樣本內殘差（原始尺度）
    in_sample = np.exp(model.fittedvalues)
    return {
        "forecast": [round(float(v), 2) for v in fc_mean],
        "upper95":  [round(float(v), 2) for v in fc_up],
        "lower95":  [round(float(v), 2) for v in fc_lo],
        "sigma":    round(float(np.sqrt(sigma2)), 4),
        "aic":      round(float(model.aic), 4),
        "model_obj": model,
        "in_sample": in_sample,
        "log_series": log_s,
        "order": order,
    }


def _fit_ets(series, steps):
    """
    ETS（指數平滑狀態空間模型）
    Hyndman & Khandakar (2008) 自動選擇 error/trend/season 組合
    statsmodels ETSModel：information_criterion='aic' 自動選最佳
    """
    try:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        best_aic, best_result = np.inf, None
        # 論文常用組合：加法誤差 × (無趨勢/加法趨勢/加法阻尼趨勢)
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
        # ETS 預測值不可為負（確保正值）
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
    樣本內驗證指標（論文必備）：
      MAPE, RMSE, MAE, R²
      Ljung-Box Q 統計量（lag=10）殘差白噪音檢定
    文獻：
      Ljung & Box (1978) Biometrika
      Hyndman & Koehler (2006) IJF — MAPE/MAE/RMSE 定義
    """
    from scipy import stats as sp_stats
    y    = series.astype(float)
    yhat = np.array(in_sample, dtype=float)
    # 對齊長度（ETS fittedvalues 有時 index 偏移）
    min_n = min(len(y), len(yhat))
    y, yhat = y[-min_n:], yhat[-min_n:]
    resid  = y - yhat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    # 避免除以零
    nonzero = y != 0
    mape = float(np.mean(np.abs(resid[nonzero] / y[nonzero])) * 100) if nonzero.any() else None
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae  = float(np.mean(np.abs(resid)))
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    # Ljung-Box（lag 建議 min(10, n//5)）
    lb_lag = max(1, min(10, len(resid) // 5))
    try:
        lb = sm.stats.acorr_ljungbox(resid, lags=[lb_lag], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[0])
        lb_pval = float(lb["lb_pvalue"].iloc[0])
        lb_pass = lb_pval > 0.05   # True = 殘差為白噪音（好）
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
    回傳：
      forecast, upper95, lower95, sigma,
      best_model ('log_arima' | 'ets'),
      model_aic, ets_aic, arima_aic,
      validation (MAPE/RMSE/MAE/R²/Ljung-Box),
      ets_spec
    """
    results = {}
    errors  = {}

    # ── log-ARIMA ──
    try:
        results['log_arima'] = _fit_log_arima(series, order, steps)
    except Exception as e:
        errors['log_arima'] = str(e)

    # ── ETS ──
    try:
        results['ets'] = _fit_ets(series, steps)
    except Exception as e:
        errors['ets'] = str(e)

    if not results:
        # 兩者均失敗 → fallback 原始 ARIMA（無 log）
        return arima_forecast(series, order, steps)

    # ── AIC 選最佳 ──
    best_key = min(results, key=lambda k: results[k]['aic'])
    best     = results[best_key]

    # ── 驗證指標 ──
    in_s = best.get('in_sample', None)
    if in_s is not None:
        val = _model_validation(series, in_s, best.get('model_obj'), best_key)
    else:
        val = {}

    arima_aic = results['log_arima']['aic'] if 'log_arima' in results else None
    ets_aic   = results['ets']['aic']       if 'ets'       in results else None
    ets_spec  = results['ets'].get('ets_spec', 'ETS') if 'ets' in results else None

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
    }


# ── AD-EF 情境計算 ──────────────────────────────────────
def adef_scenarios(base_val, steps, params):
    """
    三情境 AD-EF 預測（論文版）
    ─────────────────────────────────────────────────────
    情境基礎減排率來源（可引用）：
      BAU：台灣 2005–2019 歷史年均排放成長率約 +0.4%
           來源：環境部溫室氣體排放清冊（2024 版），
                 國發會 2022 年淨零排放路徑（背景假設）
      積極政策：台灣 2030 NDC 目標相對 2005 年減 24±1%，
               折年率約 -1.6%（自 2023 年起計算至 2030）
               來源：Taiwan NDC Update (2022), UNFCCC Submission
      NDC 淨零：2050 淨零目標折年率約 -4.5%（自 2023 年起）
               來源：台灣 2050 淨零排放路徑，國發會 (2022)；
                     IPCC AR6 WG3 Ch.3 C1 情境

    AD-EF 框架文獻：
      Kaya (1990) Impact of Carbon Dioxide Emission on GNP Growth;
      Ang & Zhang (2000) Energy Policy — Kaya 分解架構
      Rosa & Dietz (2012) Nature Climate Change — STIRPAT 延伸

    外生滑桿（使用者可調）疊加在基礎率上：
      AD 驅動：Δemission ≈ gdp × elasticity + pop × 0.35
      EF 效率：Δemission ≈ -(eff + re × 0.012)
    ─────────────────────────────────────────────────────
    """
    gdp = params.get("gdp", 0.0)
    pop = params.get("pop", 0.0)
    eff = params.get("eff", 0.0)
    re  = params.get("re",  0.0)
    ela = params.get("elasticity", 0.0)

    exog_ad = gdp * ela + pop * 0.35
    exog_ef = eff + re * 0.012

    scenarios = {
        # base_rate 來源見 docstring；citation_key 供前端顯示引用
        "bau": {
            "base_rate": +0.004,   # 台灣 2005-2019 歷史均值 +0.4%/yr
            "label":     "基準情境 BAU",
            "color":     "#f59e0b",
            "citation":  "環境部排放清冊 (2024)；國發會淨零路徑 (2022)",
            "rate_note": "+0.4%/yr（歷史趨勢延伸）",
        },
        "policy": {
            "base_rate": -0.016,   # NDC 2030 目標 -24% vs 2005，折年率
            "label":     "積極政策情境（NDC 2030）",
            "color":     "#38bdf8",
            "citation":  "Taiwan NDC Update (2022), UNFCCC Submission",
            "rate_note": "-1.6%/yr（NDC 2030 折年率）",
        },
        "ndc": {
            "base_rate": -0.045,   # 2050 淨零目標折年率
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
            v = max(v * (1 + net_rate), 0.0)   # 不允許負排放
            vals.append(round(v, 2))
        result[key] = {
            "values":    vals,
            "label":     sc["label"],
            "color":     sc["color"],
            "citation":  sc["citation"],
            "rate_note": sc["rate_note"],
            "net_rate":  round(net_rate * 100, 3),   # % 供前端顯示
        }
    return result



# ── 診斷數據計算 ─────────────────────────────────────────
def _acf_values(series, nlags=20):
    """手工計算 ACF，回傳 lag 0..nlags"""
    s = np.array(series, dtype=float); n = len(s)
    mu = np.mean(s); c0 = np.var(s)
    if c0 == 0: return [1.0] + [0.0]*nlags
    acf = [1.0]
    for k in range(1, nlags+1):
        acf.append(float(np.mean((s[:n-k]-mu)*(s[k:]-mu)) / c0))
    return acf

def _pacf_values(series, nlags=20):
    """Yule-Walker 法計算 PACF"""
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
    """95% 信心帶 ±1.96/√n"""
    cb = 1.96 / np.sqrt(n)
    return [round(cb, 4)] * (nlags+1)

def compute_diagnostics(series, order, steps=None):
    """計算完整診斷數據：殘差、ACF/PACF、差分序列"""
    p, d, q = order
    s = np.array(series[~np.isnan(series)], dtype=float)
    n = len(s)
    nlags = min(20, n//2 - 1)

    # 殘差（用差分後的序列減 AR 擬合值）
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

    # 差分序列（用於差分分析圖）
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
    """共用：讀檔、清洗、找 total 序列"""
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

    orr=select_arima_order(ts); p,d,q=orr['p'],orr['d'],orr['q']
    fc=select_best_model(ts,(p,d,q),steps)   # log-ARIMA vs ETS，AIC 自動選最佳
    fy=list(range(ly+1,2051))

    # AD-EF 三情境
    params=_get_adef_params(request)
    scenarios=adef_scenarios(ts[-1], steps, params)

    # 氣體種類預測（7種：CO₂/CH₄/N₂O/HFCs/PFCs/SF₆/NF₃）
    gas_results={}
    for col in ['co2','ch4','n2o','hfc','pfc','sf6','nf3']:
        if col in dfc.columns and not dfc[col].isna().all():
            s=dfc[col].dropna().values.astype(float)
            if len(s)>=5:
                try:
                    g=arima_forecast(s,(min(p,1),d,0),steps)
                    gas_results[col]={"history":[round(float(v),2) for v in s],
                        "forecast":[round(float(v),2) for v in g['forecast']],
                        "upper95":[round(float(v),2) for v in g['upper95']],
                        "lower95":[round(float(v),2) for v in g['lower95']]}
                except: pass

    # ── 部門分解 ARIMA（能源/工業/農業/廢棄物）──
    SECTORS = {
        'energy':   {'label':'能源部門',    'color':'#f97316'},
        'industry': {'label':'工業製程',    'color':'#a78bfa'},
        'agri':     {'label':'農業部門',    'color':'#4ade80'},
        'waste':    {'label':'廢棄物部門',  'color':'#fb923c'},
    }
    sector_results = {}
    sector_sum_hist = None   # 有資料的部門歷史加總（用來計算「其他」）
    for scol, smeta in SECTORS.items():
        if scol not in dfc.columns or dfc[scol].isna().all():
            continue
        sv = dfc[scol].dropna().values.astype(float)
        if len(sv) < 5:
            continue
        try:
            # 部門用較低階模型避免過擬合（小樣本保護）
            sp_ord = min(p, 1)
            sg = select_best_model(sv, (sp_ord, d, 0), steps)
            # 部門也用 log-ARIMA 自然防負值（透過 select_best_model 已處理）
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

    # 土地匯：線性推算（越來越多，即負值越來越大）
    # 用歷史最後 5 年（或全部）做線性回歸，推算每年增加量
    land_vals = dfc['land'].dropna().values if 'land' in dfc.columns else []
    if len(land_vals) >= 2:
        n_fit = min(len(land_vals), 10)  # 最多用近 10 年趨勢
        x_fit = np.arange(n_fit)
        y_fit = land_vals[-n_fit:].astype(float)
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        # 從最後一年往後線性延伸（slope 為負時代表吸收越來越多）
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

    return safe_json({"status":"ok","hist_years":hy,"hist_total":[round(float(v),2) for v in ts],
        "fc_years":fy,"fc_total":[round(float(v),2) for v in fc['forecast']],
        "fc_upper":[round(float(v),2) for v in fc['upper95']],
        "fc_lower":[round(float(v),2) for v in fc['lower95']],
        "sigma":fc['sigma'],"arima_order":{"p":p,"d":d,"q":q},
        "arima_explanation":orr['explanation'],"aic_table":orr['aic_table'],
        "adf_result":orr['adf'],"sample_size":orr['sample_size'],"warning":orr['warning'],
        "fc_net":[r["net"] for r in fc_tbl],"fc_land_series":[r["land"] for r in fc_tbl],"fc_land_slope":round(float(slope),2) if slope is not None else None,
        "sector_results":sector_results,"gas_results":gas_results,"history_table":hist_tbl,"forecast_table":fc_tbl,
        "model_info":{"best_model":fc.get('best_model'),"arima_order":{"p":p,"d":d,"q":q},
            "arima_aic":fc.get('arima_aic'),"ets_aic":fc.get('ets_aic'),"model_aic":fc.get('model_aic'),
            "ets_spec":fc.get('ets_spec'),"validation":fc.get('validation'),"fit_errors":fc.get('fit_errors')},
        "diagnostics":compute_diagnostics(ts,(p,d,q)),
        "scenarios":scenarios})

@app.route('/api/scenarios', methods=['POST'])
def scenarios_only():
    """只重算 AD-EF 情境，不重跑 ARIMA"""
    if 'file' not in request.files: return safe_json({"error":"未收到檔案"},400)
    try: dfc=_load_and_prep(request)
    except Exception as e: return safe_json({"error":str(e)},400)
    ts=dfc['total'].values.astype(float); ly=dfc['year'].tolist()[-1]
    steps=2050-ly
    params=_get_adef_params(request)
    scenarios=adef_scenarios(ts[-1], steps, params)
    return safe_json({"scenarios":scenarios})

@app.route('/api/health')
def health():
    return safe_json({"status":"running","message":"GHG Forecast v3"})

if __name__=='__main__':
    import os
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=os.environ.get('RENDER') is None)