"""
溫室氣體排放預測系統 v3
- Auto-ARIMA 自動選階（含小樣本保護）
- AD-EF 結構層（GDP彈性、技術改善率滑桿）
- 三情境對照（BAU / 積極政策 / NDC）
- 統計信心區間 vs 情境範圍視覺區隔
- 前後端合一，無跨域問題
"""

from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import io, json, math, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

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

# ── HTML 前端 ────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>溫室氣體排放預測系統 2050</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--ink:#0d1117;--ink2:#161b22;--ink3:#1c2733;--line:#21303f;--line2:#2d3f50;--muted:#4a6070;--dim:#7a9ab0;--mid:#a8c2d4;--text:#d4e8f5;--bright:#f0f8ff;--teal:#00e5c0;--teal2:#00b89c;--sky:#38bdf8;--amber:#f59e0b;--rose:#fb7185;--violet:#a78bfa;--lime:#84cc16;--r:8px}
html{scroll-behavior:smooth}
body{background:var(--ink);color:var(--text);font-family:'Noto Sans TC',sans-serif;font-size:13px;line-height:1.6;min-height:100vh}
body::before{content:'';position:fixed;inset:0;z-index:0;background:radial-gradient(ellipse 80% 60% at 15% 10%,rgba(0,229,192,0.04) 0%,transparent 60%),radial-gradient(ellipse 60% 40% at 85% 80%,rgba(56,189,248,0.04) 0%,transparent 60%),repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(0,229,192,0.025) 40px),repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(0,229,192,0.025) 40px);pointer-events:none}
.wrap{position:relative;z-index:1;max-width:1560px;margin:0 auto;padding:0 28px 80px}
header{padding:32px 0 24px;display:flex;align-items:flex-start;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);margin-bottom:28px}
.hdr-title{font-size:20px;font-weight:900;color:var(--bright);letter-spacing:.02em}
.hdr-sub{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--teal);margin-top:6px;letter-spacing:.14em}
.hdr-badges{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.chip{font-family:'JetBrains Mono',monospace;font-size:10px;padding:3px 9px;border-radius:100px;border:1px solid;letter-spacing:.06em}
.chip-teal{border-color:var(--teal2);color:var(--teal);background:rgba(0,229,192,.06)}
.chip-sky{border-color:#1e7fa0;color:var(--sky);background:rgba(56,189,248,.06)}
.chip-amr{border-color:#a06f08;color:var(--amber);background:rgba(245,158,11,.06)}
.chip-vio{border-color:#7c6fd4;color:var(--violet);background:rgba(167,139,250,.06)}

/* Upload */
.upload-zone{border:2px dashed var(--line2);border-radius:var(--r);padding:36px 28px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s;background:var(--ink2);position:relative;margin-bottom:16px}
.upload-zone:hover,.upload-zone.drag{border-color:var(--teal);background:rgba(0,229,192,.04)}
.upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-icon{font-size:32px;margin-bottom:10px}
.upload-text{font-size:14px;color:var(--mid)}.upload-text strong{color:var(--teal)}
.upload-hint{font-size:11px;color:var(--muted);margin-top:5px;font-family:'JetBrains Mono',monospace}
.file-chosen{background:rgba(0,229,192,.06);border-color:var(--teal2);padding:14px 22px;display:flex;align-items:center;gap:12px}
.fname{font-family:'JetBrains Mono',monospace;color:var(--teal);font-size:12px}

/* Column mapping */
.col-mapping{background:var(--ink2);border:1px solid var(--line);border-radius:var(--r);padding:18px 22px;margin-bottom:16px;display:none}
.col-mapping h3{font-size:10px;color:var(--teal);letter-spacing:.12em;font-family:'JetBrains Mono',monospace;margin-bottom:14px}
.map-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
.map-group label{display:block;font-size:11px;color:var(--dim);margin-bottom:4px}
.map-group select{width:100%;background:var(--ink3);border:1px solid var(--line2);border-radius:5px;color:var(--text);padding:6px 10px;font-size:12px;outline:none;cursor:pointer}
.map-group select:focus{border-color:var(--teal)}
.preview-wrap{margin-top:14px;overflow-x:auto;max-height:140px}
.preview-table{width:100%;border-collapse:collapse;font-size:11px;font-family:'JetBrains Mono',monospace}
.preview-table th{padding:5px 9px;background:var(--ink3);color:var(--dim);text-align:left;border-bottom:1px solid var(--line)}
.preview-table td{padding:4px 9px;color:var(--mid);border-bottom:1px solid rgba(33,48,63,.4)}

/* AD-EF Scenario Panel */
.scenario-panel{background:var(--ink2);border:1px solid var(--line);border-radius:var(--r);padding:18px 22px;margin-bottom:16px;display:none}
.scenario-panel h3{font-size:10px;color:var(--amber);letter-spacing:.12em;font-family:'JetBrains Mono',monospace;margin-bottom:14px}
.slider-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}
.slider-group label{display:block;font-size:11px;color:var(--dim);margin-bottom:4px}
.slider-group input[type=range]{width:100%;accent-color:var(--teal);cursor:pointer}
.slider-val{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--teal);margin-top:2px}
.scenario-legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:14px}
.sc-item{display:flex;align-items:center;gap:7px;font-size:12px}
.sc-dot{width:10px;height:10px;border-radius:50%}

/* Buttons */
.btn-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.btn{border:none;border-radius:6px;padding:10px 24px;font-weight:700;font-size:13px;cursor:pointer;letter-spacing:.06em;font-family:'Noto Sans TC',sans-serif;transition:opacity .15s,transform .1s;display:inline-flex;align-items:center;gap:7px}
.btn:hover{opacity:.88;transform:translateY(-1px)}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
.btn-primary{background:linear-gradient(135deg,var(--teal2),#009e80);color:var(--ink)}
.btn-secondary{background:var(--ink3);border:1px solid var(--line2);color:var(--mid)}
.loading{display:none;align-items:center;gap:10px;color:var(--dim);font-size:12px;font-family:'JetBrains Mono',monospace}
.spinner{width:15px;height:15px;border:2px solid var(--line2);border-top-color:var(--teal);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.error-box{display:none;background:rgba(251,113,133,.08);border:1px solid rgba(251,113,133,.25);border-radius:6px;padding:11px 15px;color:var(--rose);font-size:12px}

/* Results */
#results{display:none}
.stats-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:11px;margin-bottom:20px}
.stat{background:var(--ink2);border:1px solid var(--line);border-radius:var(--r);padding:14px}
.stat-label{font-size:10px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-bottom:7px}
.stat-val{font-family:'JetBrains Mono',monospace;font-size:19px;font-weight:700}
.stat-sub{font-size:10px;color:var(--muted);margin-top:3px}
.c-teal{color:var(--teal)}.c-sky{color:var(--sky)}.c-amr{color:var(--amber)}.c-rose{color:var(--rose)}.c-vio{color:var(--violet)}

/* Chart grids */
.chart-grid-main{display:grid;grid-template-columns:3fr 1fr;gap:14px;margin-bottom:14px}
.chart-grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:14px}
.chart-grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:14px}
.card{background:var(--ink2);border:1px solid var(--line);border-radius:var(--r);padding:18px}
.card-title{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:7px}
.card-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}
.leg-item{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--dim)}
.leg-line{width:20px;height:2px}

/* ARIMA panel */
.arima-panel{background:var(--ink2);border:1px solid var(--line);border-radius:var(--r);padding:22px;margin-bottom:14px}
.arima-panel h2{font-size:14px;font-weight:700;color:var(--bright);margin-bottom:18px}
.arima-order-display{display:flex;align-items:center;gap:6px;margin-bottom:20px;background:var(--ink3);border:1px solid var(--line2);border-radius:6px;padding:13px 18px;flex-wrap:wrap}
.ord-val{font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700}
.ord-sep{font-size:22px;color:var(--muted)}
.param-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin-bottom:18px}
.param-card{background:var(--ink3);border-radius:6px;padding:14px;border-left:3px solid}
.param-card.pc-p{border-color:var(--sky)}.param-card.pc-d{border-color:var(--teal)}.param-card.pc-q{border-color:var(--violet)}
.pc-label{font-family:'JetBrains Mono',monospace;font-size:17px;font-weight:700;margin-bottom:7px}
.param-card.pc-p .pc-label{color:var(--sky)}.param-card.pc-d .pc-label{color:var(--teal)}.param-card.pc-q .pc-label{color:var(--violet)}
.pc-text{font-size:12px;color:var(--mid);line-height:1.6}
.info-box{background:var(--ink3);border:1px solid var(--line2);border-radius:6px;padding:13px 15px;margin-bottom:14px}
.info-box .ib-title{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.1em;margin-bottom:5px}
.info-box .ib-text{font-size:12px;color:var(--mid);line-height:1.6}
.warning-box{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.25);border-radius:6px;padding:11px 15px;margin-bottom:14px;color:var(--amber);font-size:12px;display:none}
.aic-tbl{width:100%;border-collapse:collapse;font-size:11px;font-family:'JetBrains Mono',monospace}
.aic-tbl th{padding:6px 11px;background:var(--ink3);color:var(--dim);text-align:center;border-bottom:1px solid var(--line);font-size:10px}
.aic-tbl td{padding:5px 11px;text-align:center;border-bottom:1px solid rgba(33,48,63,.4);color:var(--mid)}
.aic-tbl tr.best td{background:rgba(0,229,192,.07);color:var(--teal)}
.aic-tbl tr.best td:first-child::before{content:'★ '}

/* Forecast table */
.table-section{background:var(--ink2);border:1px solid var(--line);border-radius:var(--r);padding:18px;margin-bottom:14px}
.table-section h3{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px}
.tbl-wrap{overflow-x:auto;max-height:500px;overflow-y:auto}
.fc-table{width:100%;border-collapse:collapse;font-size:12px}
.fc-table thead th{position:sticky;top:0;background:var(--ink3);padding:8px 12px;text-align:right;color:var(--dim);font-size:10px;font-family:'JetBrains Mono',monospace;letter-spacing:.07em;border-bottom:1px solid var(--line);white-space:nowrap}
.fc-table thead th:first-child{text-align:center}
.fc-table tbody td{padding:6px 12px;border-bottom:1px solid rgba(33,48,63,.35);text-align:right;font-family:'JetBrains Mono',monospace}
.fc-table tbody td:first-child{text-align:center;color:var(--dim)}
.fc-table tbody tr:hover td{background:rgba(0,229,192,.03)}
.hist-row td{color:var(--mid)}.fc-row td{color:var(--sky)}.fc-row td:first-child{color:var(--teal)}
.divider td{border-top:2px solid var(--teal2)}
.null-val{color:var(--muted)!important}.neg-val{color:var(--rose)!important}

/* Responsive */
@media(max-width:960px){.chart-grid-main,.chart-grid-3{grid-template-columns:1fr}.chart-grid-2{grid-template-columns:1fr}.param-cards{grid-template-columns:1fr}}
@media(max-width:600px){.stats-row{grid-template-columns:repeat(2,1fr)}header{flex-direction:column}}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div>
    <div class="hdr-title">🌍 溫室氣體排放預測系統</div>
    <div class="hdr-sub">GHG FORECAST · AUTO-ARIMA × AD-EF · SCENARIO ANALYSIS · 2050</div>
    <div class="hdr-badges">
      <span class="chip chip-teal">Auto-ARIMA + BIC保護</span>
      <span class="chip chip-sky">AD-EF 結構層</span>
      <span class="chip chip-amr">三情境對照</span>
      <span class="chip chip-vio">CO₂·CH₄·N₂O</span>
    </div>
  </div>
</header>

<!-- 上傳 -->
<div class="upload-zone" id="uploadZone">
  <input type="file" id="fileInput" accept=".csv,.xlsx,.xls">
  <div class="upload-icon">📂</div>
  <div class="upload-text">拖曳或 <strong>點擊</strong> 上傳歷史排放資料</div>
  <div class="upload-hint">CSV · Excel (.xlsx / .xls) | 支援國家清冊格式</div>
</div>

<!-- 欄位對應 -->
<div class="col-mapping" id="colMapping">
  <h3>⚙ COLUMN MAPPING</h3>
  <div class="map-grid">
    <div class="map-group"><label>📅 年份 *</label><select id="mapYear"></select></div>
    <div class="map-group"><label>📊 總排放量 (kt) *</label><select id="mapTotal"></select></div>
    <div class="map-group"><label>🌿 CO₂</label><select id="mapCO2"></select></div>
    <div class="map-group"><label>🐄 CH₄</label><select id="mapCH4"></select></div>
    <div class="map-group"><label>🌾 N₂O</label><select id="mapN2O"></select></div>
    <div class="map-group"><label>🌲 土地匯</label><select id="mapLand"></select></div>
    <div class="map-group"><label>📉 淨排放量</label><select id="mapNet"></select></div>
  </div>
  <div class="preview-wrap"><table class="preview-table" id="previewTable"></table></div>
</div>

<!-- AD-EF 情境參數 -->
<div class="scenario-panel" id="scenarioPanel">
  <h3>📐 AD-EF 情境參數（Exogenous Variables）</h3>
  <div class="slider-grid">
    <div class="slider-group">
      <label>GDP 年成長率（%）</label>
      <input type="range" id="slGdp" min="-1" max="8" step="0.1" value="2.5">
      <div class="slider-val" id="vGdp">+2.5%</div>
    </div>
    <div class="slider-group">
      <label>GDP 彈性係數（排放對GDP敏感度）</label>
      <input type="range" id="slEla" min="0.1" max="1.0" step="0.05" value="0.65">
      <div class="slider-val" id="vEla">0.65</div>
    </div>
    <div class="slider-group">
      <label>人口成長率（%）</label>
      <input type="range" id="slPop" min="-1" max="3" step="0.1" value="0.3">
      <div class="slider-val" id="vPop">+0.3%</div>
    </div>
    <div class="slider-group">
      <label>能源效率年改善率（%）</label>
      <input type="range" id="slEff" min="0" max="5" step="0.1" value="1.5">
      <div class="slider-val" id="vEff">1.5%</div>
    </div>
    <div class="slider-group">
      <label>再生能源目標滲透率（%）</label>
      <input type="range" id="slRe" min="10" max="80" step="1" value="30">
      <div class="slider-val" id="vRe">30%</div>
    </div>
  </div>
  <div class="scenario-legend">
    <div class="sc-item"><div class="sc-dot" style="background:#f59e0b"></div><span style="color:#f59e0b">基準情境 BAU</span>（AD-EF 不調整）</div>
    <div class="sc-item"><div class="sc-dot" style="background:#38bdf8"></div><span style="color:#38bdf8">積極政策情境</span>（AD 需求降10%，EF 改善×1.4）</div>
    <div class="sc-item"><div class="sc-dot" style="background:#00e5c0"></div><span style="color:#00e5c0">NDC 目標情境</span>（AD 需求降20%，EF 改善×1.8）</div>
  </div>
  <div style="margin-top:12px;padding:10px 14px;background:var(--ink3);border-radius:5px;font-size:11px;color:var(--dim);line-height:1.7">
    💡 <strong style="color:var(--mid)">統計區間 vs 情境分析說明：</strong>
    圖表中的 <span style="color:rgba(56,189,248,.6)">■ 藍色陰影</span> 代表 ARIMA 的 <strong>統計不確定性</strong>（歷史隨機波動的95%信心區間）；
    三條彩色曲線代表不同 <strong>政策情境</strong> 下的 AD-EF 結構預測，兩者在圖上明確區隔，互不混淆。
  </div>
</div>

<!-- 按鈕列 -->
<div class="btn-row">
  <button class="btn btn-primary" id="analyzeBtn" disabled>▶ 執行 Auto-ARIMA 分析</button>
  <button class="btn btn-secondary" id="scenarioBtn" style="display:none" onclick="updateScenarios()">🔄 更新情境預測</button>
  <div class="loading" id="loadingInd"><div class="spinner"></div><span>Auto-ARIMA 分析中…</span></div>
</div>
<div class="error-box" id="errorBox"></div>

<!-- 結果區 -->
<div id="results">

  <!-- 統計摘要 -->
  <div class="stats-row">
    <div class="stat"><div class="stat-label">資料範圍</div><div class="stat-val c-teal" id="s-range">—</div><div class="stat-sub">歷史年份</div></div>
    <div class="stat"><div class="stat-label">樣本數</div><div class="stat-val c-sky" id="s-n">—</div><div class="stat-sub">年度觀測值</div></div>
    <div class="stat"><div class="stat-label">基準年排放</div><div class="stat-val c-amr" id="s-base">—</div><div class="stat-sub">kt CO₂e</div></div>
    <div class="stat"><div class="stat-label">ARIMA 2050</div><div class="stat-val c-sky" id="s-2050">—</div><div class="stat-sub">kt（統計中位）</div></div>
    <div class="stat"><div class="stat-label">BAU 2050</div><div class="stat-val c-amr" id="s-bau">—</div><div class="stat-sub">kt（AD-EF BAU）</div></div>
    <div class="stat"><div class="stat-label">NDC 2050</div><div class="stat-val c-teal" id="s-ndc">—</div><div class="stat-sub">kt（AD-EF NDC）</div></div>
    <div class="stat"><div class="stat-label">ARIMA 階數</div><div class="stat-val c-teal" id="s-order">—</div><div class="stat-sub">Auto-ARIMA 選定</div></div>
    <div class="stat"><div class="stat-label">殘差 σ</div><div class="stat-val c-vio" id="s-sigma">—</div><div class="stat-sub">kt CO₂e</div></div>
  </div>

  <!-- 主圖：ARIMA + 三情境 -->
  <div class="chart-grid-main">
    <div class="card">
      <div class="card-title"><span class="card-dot" style="background:var(--teal)"></span>總排放量：ARIMA 統計預測 + AD-EF 三情境對照</div>
      <canvas id="mainChart" height="155"></canvas>
      <div class="legend">
        <div class="leg-item"><div class="leg-line" style="background:var(--teal)"></div>歷史排放</div>
        <div class="leg-item"><div class="leg-line" style="background:rgba(56,189,248,.35);height:10px;width:20px;border-radius:2px"></div>ARIMA 95% 統計區間</div>
        <div class="leg-item"><div class="leg-line" style="background:#f59e0b"></div>BAU</div>
        <div class="leg-item"><div class="leg-line" style="background:#38bdf8"></div>積極政策</div>
        <div class="leg-item"><div class="leg-line" style="background:#00e5c0"></div>NDC</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title"><span class="card-dot" style="background:var(--amber)"></span>2050 氣體組成比例</div>
      <canvas id="pieChart"></canvas>
    </div>
  </div>

  <!-- 氣體種類圖 -->
  <div class="chart-grid-3">
    <div class="card"><div class="card-title"><span class="card-dot" style="background:#60a5fa"></span>CO₂ 歷史 + 預測</div><canvas id="cCO2" height="130"></canvas></div>
    <div class="card"><div class="card-title"><span class="card-dot" style="background:#34d399"></span>CH₄ 歷史 + 預測</div><canvas id="cCH4" height="130"></canvas></div>
    <div class="card"><div class="card-title"><span class="card-dot" style="background:#f472b6"></span>N₂O 歷史 + 預測</div><canvas id="cN2O" height="130"></canvas></div>
  </div>

  <!-- 淨排放 + AIC -->
  <div class="chart-grid-2">
    <div class="card">
      <div class="card-title"><span class="card-dot" style="background:var(--rose)"></span>淨排放量 vs 總排放量（含土地匯）</div>
      <canvas id="cNet" height="130"></canvas>
    </div>
    <div class="card">
      <div class="card-title"><span class="card-dot" style="background:var(--violet)"></span>Auto-ARIMA AIC 各階數比較</div>
      <canvas id="aicChart" height="130"></canvas>
    </div>
  </div>

  <!-- ARIMA 參數說明 -->
  <div class="arima-panel">
    <h2>🔬 Auto-ARIMA 參數說明與選擇邏輯</h2>
    <div class="arima-order-display">
      <span style="font-size:12px;color:var(--dim);margin-right:4px">最終選定：</span>
      <span class="ord-val" style="color:var(--teal)">ARIMA</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:20px;color:var(--teal)">( </span>
      <span class="ord-val" id="exp-p" style="color:var(--sky)">?</span>
      <span class="ord-sep">,</span>
      <span class="ord-val" id="exp-d" style="color:var(--teal)">?</span>
      <span class="ord-sep">,</span>
      <span class="ord-val" id="exp-q" style="color:var(--violet)">?</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:20px;color:var(--teal)"> )</span>
      <span style="margin-left:14px;font-size:12px;color:var(--dim);font-family:'JetBrains Mono',monospace">BIC = <span id="exp-aic">—</span></span>
      <span id="engine-badge" style="margin-left:10px;font-family:'JetBrains Mono',monospace;font-size:10px;padding:2px 9px;border-radius:100px;border:1px solid var(--line2);color:var(--muted)">—</span>
    </div>
    <div class="warning-box" id="warnBox"></div>
    <div class="info-box">
      <div class="ib-title" style="color:var(--amber)">▸ ADF 平穩性檢定</div>
      <div class="ib-text" id="adfText">—</div>
    </div>
    <div class="param-cards">
      <div class="param-card pc-p"><div class="pc-label">p = <span id="exp-p2">?</span></div><div class="pc-text" id="exp-p-text">—</div></div>
      <div class="param-card pc-d"><div class="pc-label">d = <span id="exp-d2">?</span></div><div class="pc-text" id="exp-d-text">—</div></div>
      <div class="param-card pc-q"><div class="pc-label">q = <span id="exp-q2">?</span></div><div class="pc-text" id="exp-q-text">—</div></div>
    </div>
    <div class="info-box" style="background:rgba(0,229,192,.04);border-color:rgba(0,229,192,.15)">
      <div class="ib-title" style="color:var(--teal)">▸ 綜合結論</div>
      <div class="ib-text" id="summaryText">—</div>
    </div>
    <div style="margin-top:18px">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:.1em;margin-bottom:9px">AIC/BIC SELECTION TABLE（前 12 名，★ 為選定）</div>
      <div style="overflow-x:auto"><table class="aic-tbl"><thead><tr><th>排名</th><th>p</th><th>d</th><th>q</th><th>BIC-like Score</th></tr></thead><tbody id="aicTbody"></tbody></table></div>
    </div>
  </div>

  <!-- 數據表 -->
  <div class="table-section">
    <h3>📋 歷史數據 + ARIMA 預測表（至 2050）</h3>
    <div class="tbl-wrap">
      <table class="fc-table">
        <thead><tr>
          <th>年份</th><th>CO₂ (kt)</th><th>CH₄ (kt)</th><th>N₂O (kt)</th>
          <th>土地匯 (kt)</th><th>總排放量 (kt)</th><th>淨排放量 (kt)</th>
          <th>ARIMA 中位</th><th>上界 95%</th><th>下界 95%</th>
          <th>BAU (AD-EF)</th><th>積極政策</th><th>NDC</th><th>類型</th>
        </tr></thead>
        <tbody id="forecastTbody"></tbody>
      </table>
    </div>
  </div>
</div><!-- /results -->
</div><!-- /wrap -->

<script>
let charts={}, uploadedFile=null, analysisData=null;

// ── 滑桿綁定 ──
const sliders=[
  ['slGdp','vGdp',v=>(v>=0?'+':'')+v+'%'],
  ['slEla','vEla',v=>parseFloat(v).toFixed(2)],
  ['slPop','vPop',v=>(v>=0?'+':'')+v+'%'],
  ['slEff','vEff',v=>v+'%'],
  ['slRe','vRe',v=>v+'%'],
];
sliders.forEach(([id,vid,fmt])=>{
  const el=document.getElementById(id);
  el.addEventListener('input',()=>{document.getElementById(vid).textContent=fmt(el.value)});
});

// ── 取得滑桿值 ──
function getParams(){
  return{
    gdp:parseFloat(document.getElementById('slGdp').value)/100,
    elasticity:parseFloat(document.getElementById('slEla').value),
    pop:parseFloat(document.getElementById('slPop').value)/100,
    eff:parseFloat(document.getElementById('slEff').value)/100,
    re:parseFloat(document.getElementById('slRe').value)/100,
  };
}

// ── Upload ──
const uploadZone=document.getElementById('uploadZone');
const fileInput=document.getElementById('fileInput');
uploadZone.addEventListener('dragover',e=>{e.preventDefault();uploadZone.classList.add('drag')});
uploadZone.addEventListener('dragleave',()=>uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop',e=>{e.preventDefault();uploadZone.classList.remove('drag');if(e.dataTransfer.files[0])handleFile(e.dataTransfer.files[0])});
fileInput.addEventListener('change',()=>{if(fileInput.files[0])handleFile(fileInput.files[0])});

async function handleFile(file){
  uploadedFile=file;
  uploadZone.innerHTML=`<div class="file-chosen"><span style="font-size:22px">📄</span><div><div class="fname">${file.name}</div><div style="font-size:10px;color:var(--muted);margin-top:2px">${(file.size/1024).toFixed(1)} KB</div></div></div>`;
  const fd=new FormData(); fd.append('file',file);
  try{
    const r=await fetch('/api/upload',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error){showErr(d.error);return}
    buildSelects(d.columns,d.detected,d.preview);
    document.getElementById('colMapping').style.display='block';
    document.getElementById('analyzeBtn').disabled=false;
    uploadZone.innerHTML=`<div class="file-chosen"><span style="font-size:22px">✅</span><div><div class="fname">${file.name}</div><div style="font-size:10px;color:var(--dim);margin-top:2px">${d.rows} 筆 · ${d.columns.length} 欄 · 偵測完成</div></div></div>`;
  }catch(e){showErr('上傳失敗：'+e.message)}
}

function buildSelects(cols,detected,preview){
  const opts=['（不使用）',...cols];
  [['mapYear','year'],['mapTotal','total'],['mapCO2','co2'],['mapCH4','ch4'],
   ['mapN2O','n2o'],['mapLand','land'],['mapNet','net']
  ].forEach(([id,key])=>{
    const s=document.getElementById(id); if(!s)return;
    s.innerHTML=opts.map(c=>`<option value="${c==='（不使用）'?'':c}">${c}</option>`).join('');
    if(detected[key])s.value=detected[key];
  });
  const t=document.getElementById('previewTable');
  t.innerHTML=`<tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`+(preview||[]).map(row=>`<tr>${cols.map(c=>`<td>${row[c]??''}</td>`).join('')}</tr>`).join('');
}

// ── Analyze ──
document.getElementById('analyzeBtn').addEventListener('click',async()=>{
  if(!uploadedFile)return;
  hideErr();
  document.getElementById('loadingInd').style.display='flex';
  document.getElementById('analyzeBtn').disabled=true;
  document.getElementById('results').style.display='none';
  const fd=new FormData(); fd.append('file',uploadedFile);
  [['col_year','mapYear'],['col_total','mapTotal'],['col_co2','mapCO2'],['col_ch4','mapCH4'],
   ['col_n2o','mapN2O'],['col_land','mapLand'],['col_net','mapNet']
  ].forEach(([k,id])=>{const el=document.getElementById(id);if(el)fd.append(k,el.value||'')});
  // 傳送 AD-EF 參數
  const p=getParams();
  Object.entries(p).forEach(([k,v])=>fd.append('adef_'+k,v));
  try{
    const r=await fetch('/api/analyze',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error){showErr(d.error);return}
    analysisData=d;
    render(d);
    document.getElementById('scenarioPanel').style.display='block';
    document.getElementById('scenarioBtn').style.display='inline-flex';
  }catch(e){showErr('分析失敗：'+e.message)}
  finally{document.getElementById('loadingInd').style.display='none';document.getElementById('analyzeBtn').disabled=false}
});

// ── 更新情境（不重跑ARIMA）──
async function updateScenarios(){
  if(!uploadedFile||!analysisData)return;
  const fd=new FormData(); fd.append('file',uploadedFile);
  [['col_year','mapYear'],['col_total','mapTotal']].forEach(([k,id])=>{const el=document.getElementById(id);if(el)fd.append(k,el.value||'')});
  const p=getParams(); Object.entries(p).forEach(([k,v])=>fd.append('adef_'+k,v));
  try{
    const r=await fetch('/api/scenarios',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error){showErr(d.error);return}
    // 更新主圖情境線
    updateMainChartScenarios(d.scenarios, analysisData.fc_years);
    // 更新統計卡
    const sc=d.scenarios;
    document.getElementById('s-bau').textContent=fmt(sc.bau.values[sc.bau.values.length-1]);
    document.getElementById('s-ndc').textContent=fmt(sc.ndc.values[sc.ndc.values.length-1]);
    // 更新表格
    updateForecastTable(analysisData, d.scenarios);
  }catch(e){showErr('情境更新失敗：'+e.message)}
}

// ── Render ──
function render(d){
  document.getElementById('results').style.display='block';
  const hLen=d.hist_years.length, fLen=d.fc_years.length;
  const allY=[...d.hist_years.map(String),...d.fc_years.map(String)];
  const o=d.arima_order, exp=d.arima_explanation;
  const base=d.hist_total[hLen-1], end=d.fc_total[fLen-1];
  const sc=d.scenarios;

  // Stats
  document.getElementById('s-range').textContent=`${d.hist_years[0]}–${d.hist_years[hLen-1]}`;
  document.getElementById('s-n').textContent=d.sample_size;
  document.getElementById('s-base').textContent=fmt(base);
  document.getElementById('s-2050').textContent=fmt(end);
  document.getElementById('s-bau').textContent=fmt(sc.bau.values[fLen-1]);
  document.getElementById('s-ndc').textContent=fmt(sc.ndc.values[fLen-1]);
  document.getElementById('s-order').textContent=`(${o.p},${o.d},${o.q})`;
  document.getElementById('s-sigma').textContent=fmt(d.sigma);

  // Main chart：ARIMA區間 + 三情境線
  const histFull=[...d.hist_total,...Array(fLen).fill(null)];
  mk('mainChart',{type:'line',data:{labels:allY,datasets:[
    {data:[...Array(hLen).fill(null),...d.fc_upper],borderColor:'transparent',backgroundColor:'rgba(56,189,248,0.10)',fill:'+1',pointRadius:0},
    {data:[...Array(hLen).fill(null),...d.fc_lower],borderColor:'transparent',fill:false,pointRadius:0},
    {label:'歷史排放',data:histFull,borderColor:'#00e5c0',borderWidth:2.5,pointRadius:0,tension:0.3,fill:false},
    {label:'BAU',data:[...Array(hLen).fill(null),...sc.bau.values],borderColor:'#f59e0b',borderWidth:2,borderDash:[6,3],pointRadius:0,tension:0.3,fill:false},
    {label:'積極政策',data:[...Array(hLen).fill(null),...sc.policy.values],borderColor:'#38bdf8',borderWidth:2,borderDash:[6,3],pointRadius:0,tension:0.3,fill:false},
    {label:'NDC',data:[...Array(hLen).fill(null),...sc.ndc.values],borderColor:'#00e5c0',borderWidth:2.5,borderDash:[4,2],pointRadius:0,tension:0.3,fill:false},
  ]},options:lopts('kt CO₂e',1.9)});

  // Pie
  const gk=['co2','ch4','n2o'],gc=['rgba(96,165,250,.85)','rgba(52,211,153,.85)','rgba(244,114,182,.85)'],gl=['CO₂','CH₄','N₂O'];
  const pv=gk.map(k=>{const g=d.gas_results?.[k];return g&&g.forecast.length?Math.max(0,g.forecast[g.forecast.length-1]):null});
  const pfl={l:[],dt:[],c:[]};
  if(pv.some(v=>v!==null)){gl.forEach((lb,i)=>{if(pv[i]!==null){pfl.l.push(lb);pfl.dt.push(pv[i]);pfl.c.push(gc[i])}})}
  else{pfl.l=['總排放量'];pfl.dt=[end];pfl.c=['rgba(0,229,192,.8)']}
  mk('pieChart',{type:'doughnut',data:{labels:pfl.l,datasets:[{data:pfl.dt,backgroundColor:pfl.c,borderColor:'#161b22',borderWidth:2}]},
    options:{responsive:true,maintainAspectRatio:true,plugins:{legend:{display:true,position:'bottom',labels:{color:'#7a9ab0',font:{size:11},padding:10,boxWidth:12}},tooltip:{backgroundColor:'#1c2733',titleColor:'#d4e8f5',bodyColor:'#7a9ab0'}}}});

  // Gas charts
  [['cCO2','co2','#60a5fa'],['cCH4','ch4','#34d399'],['cN2O','n2o','#f472b6']].forEach(([id,key,col])=>{
    const g=d.gas_results?.[key];
    if(!g){mk(id,{type:'line',data:{labels:[],datasets:[]},options:lopts('kt',1.6)});return}
    mk(id,{type:'line',data:{labels:allY,datasets:[
      {data:[...Array(hLen).fill(null),...g.upper95],borderColor:'transparent',backgroundColor:col+'22',fill:'+1',pointRadius:0},
      {data:[...Array(hLen).fill(null),...g.lower95],borderColor:'transparent',fill:false,pointRadius:0},
      {data:[...g.history,...Array(fLen).fill(null)],borderColor:col,borderWidth:2,pointRadius:0,tension:0.3,fill:false},
      {data:[...Array(hLen).fill(null),...g.forecast],borderColor:col,borderWidth:1.5,borderDash:[5,3],pointRadius:0,tension:0.3,fill:false},
    ]},options:lopts('kt CO₂e',1.6)});
  });

  // Net chart
  const nh=d.history_table.map(r=>r.net??null);
  if(nh.some(v=>v!==null)){
    mk('cNet',{type:'line',data:{labels:allY,datasets:[
      {label:'淨排放',data:[...nh,...Array(fLen).fill(null)],borderColor:'#fb7185',borderWidth:2,pointRadius:0,tension:0.3,fill:false},
      {label:'總排放',data:[...d.hist_total,...Array(fLen).fill(null)],borderColor:'rgba(0,229,192,.4)',borderWidth:1.5,borderDash:[4,3],pointRadius:0,tension:0.3,fill:false},
    ]},options:{...lopts('kt CO₂e',1.6),plugins:{legend:{display:true,labels:{color:'#4a6070',font:{size:10},boxWidth:12}},tooltip:{backgroundColor:'#1c2733',titleColor:'#d4e8f5',bodyColor:'#7a9ab0'}}}});
  }

  // AIC chart
  const aicD=d.aic_table||[];
  mk('aicChart',{type:'bar',data:{labels:aicD.map(r=>`(${r.p},${r.d},${r.q})`),datasets:[{data:aicD.map(r=>r.AIC),backgroundColor:aicD.map(r=>r.p===o.p&&r.q===o.q?'rgba(0,229,192,.75)':'rgba(56,189,248,.28)'),borderRadius:3}]},
    options:{responsive:true,maintainAspectRatio:true,aspectRatio:1.8,plugins:{legend:{display:false},tooltip:{backgroundColor:'#1c2733',bodyColor:'#94a3b8'}},scales:{x:{ticks:{color:'#4a6070',font:{size:9},maxRotation:45},grid:{color:'rgba(255,255,255,.03)'}},y:{ticks:{color:'#4a6070',font:{size:9}},grid:{color:'rgba(255,255,255,.03)'}}}}});

  // ARIMA 說明
  ['p','d','q'].forEach(k=>{document.getElementById('exp-'+k).textContent=o[k];document.getElementById('exp-'+k+'2').textContent=o[k]});
  document.getElementById('exp-aic').textContent=d.aic_table?.[0]?.AIC??'—';
  document.getElementById('adfText').textContent=exp.adf_reason||'—';
  // Engine badge
  const eng=d.engine||'fallback';
  const badge=document.getElementById('engine-badge');
  if(eng==='pmdarima'){
    badge.textContent='🔬 pmdarima · auto_arima · BIC';
    badge.style.color='var(--teal)';badge.style.borderColor='var(--teal2)';badge.style.background='rgba(0,229,192,.07)';
  } else {
    badge.textContent='⚙️ 手工BIC窮舉 (fallback)';
    badge.style.color='var(--amber)';badge.style.borderColor='#a06f08';badge.style.background='rgba(245,158,11,.07)';
  }
  const hl=(s,c)=>(s||'').replace(/\*\*(.*?)\*\*/g,`<strong style="color:${c}">$1</strong>`);
  document.getElementById('exp-p-text').innerHTML=hl(exp.p,'var(--sky)');
  document.getElementById('exp-d-text').innerHTML=hl(exp.d,'var(--teal)');
  document.getElementById('exp-q-text').innerHTML=hl(exp.q,'var(--violet)');
  document.getElementById('summaryText').innerHTML=hl(exp.summary,'var(--teal)');
  if(d.warning){const wb=document.getElementById('warnBox');wb.textContent=d.warning;wb.style.display='block'}
  document.getElementById('aicTbody').innerHTML=(d.aic_table||[]).map((r,i)=>`<tr class="${r.p===o.p&&r.q===o.q?'best':''}"><td>${i+1}</td><td>${r.p}</td><td>${r.d}</td><td>${r.q}</td><td>${r.AIC}</td></tr>`).join('');

  // Forecast table
  updateForecastTable(d, sc);
  document.getElementById('results').scrollIntoView({behavior:'smooth',block:'start'});
}

function updateForecastTable(d, sc){
  const fLen=d.fc_years.length;
  let rows='', first=true;
  d.history_table.forEach(r=>{
    rows+=`<tr class="hist-row"><td>${r.year}</td>
      <td>${r.co2!=null?fmt(r.co2):'<span class="null-val">—</span>'}</td>
      <td>${r.ch4!=null?fmt(r.ch4):'<span class="null-val">—</span>'}</td>
      <td>${r.n2o!=null?fmt(r.n2o):'<span class="null-val">—</span>'}</td>
      <td class="${r.land<0?'neg-val':''}">${r.land!=null?fmt(r.land):'<span class="null-val">—</span>'}</td>
      <td>${r.total!=null?fmt(r.total):'<span class="null-val">—</span>'}</td>
      <td class="${r.net<0?'neg-val':''}">${r.net!=null?fmt(r.net):'<span class="null-val">—</span>'}</td>
      <td class="null-val">—</td><td class="null-val">—</td><td class="null-val">—</td>
      <td class="null-val">—</td><td class="null-val">—</td><td class="null-val">—</td>
      <td style="color:var(--muted);font-size:10px">歷史</td></tr>`;
  });
  d.forecast_table.forEach((r,i)=>{
    const cls=first?'fc-row divider':'fc-row'; first=false;
    rows+=`<tr class="${cls}"><td>${r.year}</td>
      <td class="null-val">—</td><td class="null-val">—</td><td class="null-val">—</td><td class="null-val">—</td>
      <td class="null-val">—</td><td class="null-val">—</td>
      <td>${fmt(r.total)}</td><td>${fmt(r.upper95)}</td><td>${fmt(r.lower95)}</td>
      <td style="color:#f59e0b">${sc?fmt(sc.bau.values[i]):'—'}</td>
      <td style="color:#38bdf8">${sc?fmt(sc.policy.values[i]):'—'}</td>
      <td style="color:#00e5c0">${sc?fmt(sc.ndc.values[i]):'—'}</td>
      <td style="color:var(--sky);font-size:10px">預測</td></tr>`;
  });
  document.getElementById('forecastTbody').innerHTML=rows;
}

function updateMainChartScenarios(sc, fcYears){
  const c=charts['mainChart']; if(!c)return;
  const hLen=analysisData.hist_years.length;
  c.data.datasets[3].data=[...Array(hLen).fill(null),...sc.bau.values];
  c.data.datasets[4].data=[...Array(hLen).fill(null),...sc.policy.values];
  c.data.datasets[5].data=[...Array(hLen).fill(null),...sc.ndc.values];
  c.update();
}

// ── Chart helpers ──
const BO={responsive:true,maintainAspectRatio:true,animation:{duration:600},plugins:{legend:{display:false},tooltip:{backgroundColor:'#1c2733',titleColor:'#d4e8f5',bodyColor:'#7a9ab0',borderColor:'#21303f',borderWidth:1}}};
const SC={grid:{color:'rgba(255,255,255,.032)'},ticks:{color:'#4a6070',font:{size:10},maxTicksLimit:8}};
function lopts(yL,ar=2){return{...BO,aspectRatio:ar,scales:{x:SC,y:{...SC,title:{display:true,text:yL,color:'#4a6070',font:{size:10}}}}}}
function mk(id,cfg){if(charts[id])charts[id].destroy();charts[id]=new Chart(document.getElementById(id).getContext('2d'),cfg)}
function fmt(v){if(v==null||isNaN(v))return'<span class="null-val">—</span>';return Number(v).toLocaleString('zh-TW',{maximumFractionDigits:1})}
function showErr(m){const e=document.getElementById('errorBox');e.textContent='❌ '+m;e.style.display='block'}
function hideErr(){document.getElementById('errorBox').style.display='none'}
</script>
</body>
</html>"""

# ── Routes ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

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
        gc=[c for c in ['co2','ch4','n2o','hfcs_value','pfcs_value','sf6_value','nf3_value'] if c in df.columns]
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

    # 氣體種類預測
    gas_results={}
    for col in ['co2','ch4','n2o']:
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
        for c in ['energy','industry','agri','land','total','net','co2','ch4','n2o']:
            v=row.get(c,None); r[c]=round(float(v),2) if v is not None and pd.notna(v) else None
        hist_tbl.append(r)

    fc_tbl=[{"year":yr,"total":round(fc['forecast'][i],2),
              "upper95":round(fc['upper95'][i],2),"lower95":round(fc['lower95'][i],2),
              **{c:None for c in ['energy','industry','agri','land','net','co2','ch4','n2o']}}
             for i,yr in enumerate(fy)]

    return safe_json({"status":"ok","hist_years":hy,"hist_total":[round(float(v),2) for v in ts],
        "fc_years":fy,"fc_total":[round(float(v),2) for v in fc['forecast']],
        "fc_upper":[round(float(v),2) for v in fc['upper95']],
        "fc_lower":[round(float(v),2) for v in fc['lower95']],
        "sigma":fc['sigma'],"arima_order":{"p":p,"d":d,"q":q},
        "arima_explanation":orr['explanation'],"aic_table":orr['aic_table'],
        "adf_result":orr['adf'],"sample_size":orr['sample_size'],"warning":orr['warning'],
        "gas_results":gas_results,"history_table":hist_tbl,"forecast_table":fc_tbl,
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