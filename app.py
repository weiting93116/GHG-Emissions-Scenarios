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

# ── AD-EF 情境計算 ──────────────────────────────────────
def adef_scenarios(base_val, steps, params):
    """
    三情境 AD-EF 預測
    base_val: 基準年排放量
    params: {gdp, pop, eff, re, elasticity}
    回傳: {bau, policy, ndc} 各 steps 步預測值
    """
    gdp     = params.get("gdp", 0.025)      # GDP 成長率
    pop     = params.get("pop", 0.003)      # 人口成長率
    eff     = params.get("eff", 0.015)      # 能效改善率
    re      = params.get("re",  0.30)       # 再生能源目標
    ela     = params.get("elasticity", 0.65) # GDP 彈性係數

    scenarios = {
        "bau":    {"ad_mult":1.0, "ef_mult":1.0,  "label":"基準情境 (BAU)",      "color":"#f59e0b"},
        "policy": {"ad_mult":0.9, "ef_mult":1.4,  "label":"積極政策情境",         "color":"#38bdf8"},
        "ndc":    {"ad_mult":0.8, "ef_mult":1.8,  "label":"NDC 目標情境",         "color":"#00e5c0"},
    }
    result = {}
    for key, sc in scenarios.items():
        ad_growth  = (gdp * ela + pop * 0.35) * sc["ad_mult"]
        ef_reduction = (eff + re * 0.012) * sc["ef_mult"]
        net_rate   = ad_growth - ef_reduction
        vals = []
        v = base_val
        for _ in range(steps):
            v = v * (1 + net_rate)
            vals.append(round(v, 2))
        result[key] = {"values": vals, **{k:v2 for k,v2 in sc.items()}}
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
        "gdp":      float(req.form.get("adef_gdp",     0.025)),
        "elasticity":float(req.form.get("adef_elasticity",0.65)),
        "pop":      float(req.form.get("adef_pop",     0.003)),
        "eff":      float(req.form.get("adef_eff",     0.015)),
        "re":       float(req.form.get("adef_re",      0.30)),
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
    fc=arima_forecast(ts,(p,d,q),steps); fy=list(range(ly+1,2051))

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

    hist_tbl=[]
    for _,row in dfc.iterrows():
        r={"year":int(row['year'])}
        for c in ['energy','industry','agri','land','total','net','co2','ch4','n2o','hfc','pfc','sf6','nf3']:
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
            **{c: None for c in ['energy','industry','agri','co2','ch4','n2o']}
        })

    return safe_json({"status":"ok","hist_years":hy,"hist_total":[round(float(v),2) for v in ts],
        "fc_years":fy,"fc_total":[round(float(v),2) for v in fc['forecast']],
        "fc_upper":[round(float(v),2) for v in fc['upper95']],
        "fc_lower":[round(float(v),2) for v in fc['lower95']],
        "sigma":fc['sigma'],"arima_order":{"p":p,"d":d,"q":q},
        "arima_explanation":orr['explanation'],"aic_table":orr['aic_table'],
        "adf_result":orr['adf'],"sample_size":orr['sample_size'],"warning":orr['warning'],
        "fc_net":[r["net"] for r in fc_tbl],"fc_land_series":[r["land"] for r in fc_tbl],"fc_land_slope":round(float(slope),2) if slope is not None else None,
        "gas_results":gas_results,"history_table":hist_tbl,"forecast_table":fc_tbl,
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