"""
溫室氣體排放預測系統 - Flask 後端 v2
支援國家溫室氣體排放清冊 CSV 格式
欄位：year, co2_value, ch4_value, n2o_value, hfcs_value, pfcs_value, sf6_value,
      nf3_value, total_ghg_emission_value, co2_absorption_value, net_ghg_emission_value
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────
# 數值清洗：處理千分位逗號、NE、空值
# ──────────────────────────────────────────────

def clean_numeric(val):
    if val is None:
        return np.nan
    s = str(val).strip().replace(',', '').replace('"', '')
    if s.upper() in ('NE', 'NA', 'N/A', '', '-', 'NO', 'NOT ESTIMATED'):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_df(df):
    df = df.copy()
    for col in df.columns:
        if col == 'year':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = df[col].apply(clean_numeric)
    return df


# ──────────────────────────────────────────────
# 欄位偵測
# ──────────────────────────────────────────────

def detect_columns(df):
    mapping = {}
    cols_lower = {c: c.lower().replace(' ', '_') for c in df.columns}

    patterns = {
        "year":    ["year", "年份", "年度", "西元年"],
        "co2":     ["co2_value", "co2", "二氧化碳"],
        "ch4":     ["ch4_value", "ch4", "甲烷"],
        "n2o":     ["n2o_value", "n2o", "氧化亞氮"],
        "hfcs":    ["hfcs_value", "hfcs"],
        "pfcs":    ["pfcs_value", "pfcs"],
        "sf6":     ["sf6_value", "sf6"],
        "nf3":     ["nf3_value", "nf3"],
        "total":   ["total_ghg_emission_value", "total_ghg", "total", "總排放", "合計", "總量"],
        "land":    ["co2_absorption_value", "absorption", "land", "土地匯", "lulucf", "co2_absorption"],
        "net":     ["net_ghg_emission_value", "net_ghg", "net", "淨排放"],
        "energy":  ["energy", "能源"],
        "industry":["industry", "工業製程"],
        "agri":    ["agri", "農業"],
    }

    for key, candidates in patterns.items():
        for orig_col, col_l in cols_lower.items():
            if any(col_l == p or col_l.startswith(p) for p in candidates):
                if key not in mapping:
                    mapping[key] = orig_col
        
    return mapping


# ──────────────────────────────────────────────
# ADF 平穩性檢定
# ──────────────────────────────────────────────

def adf_test(series):
    series = series[~np.isnan(series)]
    vr0 = np.var(series)
    vr1 = np.var(np.diff(series))
    vr2 = np.var(np.diff(np.diff(series))) if len(series) > 3 else vr1
    x = np.arange(len(series))
    slope = np.polyfit(x, series, 1)[0]
    trend_strength = abs(slope) / (np.std(series) + 1e-10)

    if trend_strength > 0.05 or vr1 < vr0 * 0.7:
        if vr2 < vr1 * 0.7:
            d, stationary = 2, False
            reason = f"原序列具明顯趨勢（斜率強度={trend_strength:.3f}），一階差分後仍不平穩（Var比={vr1/vr0:.3f}），建議 d=2"
        else:
            d, stationary = 1, False
            reason = f"原序列具明顯趨勢（斜率強度={trend_strength:.3f}），一階差分後達到平穩（Var比={vr1/vr0:.3f}），建議 d=1"
    else:
        d, stationary = 0, True
        reason = f"原序列已接近平穩（斜率強度={trend_strength:.3f}），無需差分，建議 d=0"

    return {"stationary": stationary, "recommended_d": d, "reason": reason}


# ──────────────────────────────────────────────
# ARIMA 自動選階
# ──────────────────────────────────────────────

def select_arima_order(series, max_p=3, max_q=3):
    series = series[~np.isnan(series)]
    n = len(series)
    adf_result = adf_test(series)
    d = adf_result["recommended_d"]

    s = series.copy().astype(float)
    for _ in range(d):
        s = np.diff(s)

    best_aic = np.inf
    best_p, best_q = 0, 0
    aic_table = []

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                if p > 0 and len(s) > p + 1:
                    X = np.column_stack([s[p-i-1:len(s)-i-1] for i in range(p)] + [np.ones(len(s)-p)])
                    y = s[p:]
                    if X.shape[0] < X.shape[1] + 2:
                        continue
                    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    resid = y - X @ coef
                    sigma2 = np.var(resid)
                    k = p + q + 1
                else:
                    resid = s - np.mean(s)
                    sigma2 = np.var(resid)
                    k = 1

                if sigma2 <= 0:
                    continue
                m = len(s) - p
                if m < 2:
                    continue
                log_lik = -0.5 * m * np.log(2 * np.pi * sigma2) - 0.5 * m
                aic = 2 * k - 2 * log_lik
                aic_table.append({"p": p, "d": d, "q": q, "AIC": round(aic, 2)})
                if aic < best_aic:
                    best_aic = aic
                    best_p, best_q = p, q
            except Exception:
                continue

    warning = None
    if n < 35 and (best_p + best_q) > 2:
        warning = f"⚠️ 樣本數僅 {n} 筆，原始最佳 ARIMA({best_p},{d},{best_q}) 有過擬合風險，已限制 p+q ≤ 2"
        candidates = [r for r in aic_table if r["p"] + r["q"] <= 2]
        if candidates:
            best = min(candidates, key=lambda x: x["AIC"])
            best_p, best_q = best["p"], best["q"]

    explanation = build_param_explanation(best_p, d, best_q, adf_result, n, warning)
    aic_table_sorted = sorted(aic_table, key=lambda x: x["AIC"])[:12]

    return {
        "p": best_p, "d": d, "q": best_q,
        "aic": round(best_aic, 2),
        "adf": adf_result,
        "warning": warning,
        "explanation": explanation,
        "aic_table": aic_table_sorted,
        "sample_size": n
    }


def build_param_explanation(p, d, q, adf, n, warning):
    d_explain = {
        0: "**d=0（不差分）**：原序列已平穩，沒有明顯線性趨勢，直接對原始數據建模。",
        1: f"**d=1（一階差分）**：{adf['reason']}。差分後序列 Yₜ-Yₜ₋₁ 趨於平穩，消除線性趨勢。這是年度排放量最常見的設定。",
        2: f"**d=2（二階差分）**：{adf['reason']}。排放量具加速趨勢，需要兩次差分才能平穩。"
    }
    p_explain = {
        0: "**p=0（無 AR 項）**：當期排放量與過去各期無顯著自相關，殘差近似白雜訊，預測主要依賴趨勢項。",
        1: "**p=1（AR(1)）**：當期排放量受前一年影響，具一階自回歸結構（PACF 在 lag=1 截尾）。",
        2: "**p=2（AR(2)）**：當期排放量受前兩年共同影響（PACF 在 lag=2 截尾），常見於具景氣循環慣性的情境。",
        3: "**p=3（AR(3)）**：三期自回歸，序列具較長記憶性。需注意小樣本下的過擬合風險。"
    }
    q_explain = {
        0: "**q=0（無 MA 項）**：殘差之間無顯著移動平均結構，衝擊效果不持續跨期，殘差獨立。",
        1: "**q=1（MA(1)）**：外部衝擊（如政策衝擊、能源危機）對排放的影響持續一年後消退。",
        2: "**q=2（MA(2)）**：衝擊效果延續兩年，常見於需要 1-2 年才能反映的政策調整情境。",
        3: "**q=3（MA(3)）**：衝擊效果延續三年，適合有重大政策轉折點的歷史序列。"
    }
    small_note = f"\n\n> 📌 **小樣本警告**：本資料集僅有 {n} 個年度觀測值，p+q 已限制在合理範圍內以避免過擬合。" if n < 35 else ""
    return {
        "p": p_explain.get(p, f"p={p}"),
        "d": d_explain.get(d, f"d={d}"),
        "q": q_explain.get(q, f"q={q}"),
        "summary": f"根據 AIC 最小化準則，最終選定 **ARIMA({p},{d},{q})**。{small_note}",
        "adf_reason": adf["reason"]
    }


# ──────────────────────────────────────────────
# ARIMA 預測
# ──────────────────────────────────────────────

def arima_forecast(series, order, steps):
    p, d, q = order
    series = series[~np.isnan(series)]
    orig = series.copy().astype(float)

    s = orig.copy()
    for _ in range(d):
        s = np.diff(s)

    n = len(s)
    mean_s = np.mean(s)
    ar_coefs = np.zeros(p)

    if p > 0 and n > p:
        X = np.column_stack([s[p-i-1:n-i-1] for i in range(p)])
        y = s[p:]
        try:
            ar_coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            ar_coefs = np.zeros(p)

    if p > 0 and n > p:
        fitted = np.array([np.dot(ar_coefs, s[i:i+p][::-1]) for i in range(p, n)])
        resid = s[p:] - fitted
    else:
        resid = s - mean_s
    sigma = np.std(resid)

    s_extended = list(s)
    forecasts_diff = []
    for i in range(steps):
        ar_part = np.dot(ar_coefs, s_extended[-p:][::-1]) if p > 0 and len(s_extended) >= p else mean_s
        forecasts_diff.append(ar_part)
        s_extended.append(ar_part)

    preds = []
    for i, fd in enumerate(forecasts_diff):
        if d == 0:
            preds.append(fd)
        elif d == 1:
            base = orig[-1] if i == 0 else preds[-1]
            preds.append(base + fd)
        elif d == 2:
            if i == 0:
                prev = orig[-1] + (orig[-1] - orig[-2])
            else:
                prev = preds[-1] + (preds[-1] - (orig[-1] if i == 1 else preds[-2]))
            preds.append(prev + fd)

    preds = np.array(preds)
    ci_upper = preds + 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
    ci_lower = preds - 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))

    return {
        "forecast": preds.tolist(),
        "upper95": ci_upper.tolist(),
        "lower95": ci_lower.tolist(),
        "sigma": round(float(sigma), 4)
    }


# ──────────────────────────────────────────────
# 讀檔
# ──────────────────────────────────────────────

def read_file(f):
    filename = f.filename.lower()
    raw = f.read()
    if filename.endswith('.csv'):
        for enc in ['utf-8-sig', 'utf-8', 'big5', 'cp950']:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str)
            except Exception:
                continue
        raise ValueError("CSV 編碼解析失敗")
    elif filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(raw), dtype=str)
    else:
        raise ValueError("僅支援 CSV 或 Excel 格式")


# ──────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "未收到檔案"}), 400
    f = request.files["file"]
    try:
        df = read_file(f)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    detected = detect_columns(df)
    df_clean = clean_df(df)
    preview = df_clean.head(5).where(pd.notnull(df_clean), None).to_dict(orient="records")

    return jsonify({
        "columns": list(df.columns),
        "detected": detected,
        "preview": preview,
        "rows": len(df)
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "未收到檔案"}), 400
    f = request.files["file"]
    try:
        df = read_file(f)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    col_map = {
        "year": request.form.get("col_year", ""),
        "energy": request.form.get("col_energy", ""),
        "industry": request.form.get("col_industry", ""),
        "agri": request.form.get("col_agri", ""),
        "land": request.form.get("col_land", ""),
        "total": request.form.get("col_total", ""),
        "net": request.form.get("col_net", ""),
        "co2": request.form.get("col_co2", ""),
        "ch4": request.form.get("col_ch4", ""),
        "n2o": request.form.get("col_n2o", ""),
    }

    rename = {orig: std for std, orig in col_map.items() if orig and orig in df.columns}

    # ── Fallback：前端沒送欄位對應時，自動偵測 ──
    detected = detect_columns(df)
    for std, orig in detected.items():
        if std not in rename.values() and orig not in rename and orig in df.columns:
            rename[orig] = std

    df = df.rename(columns=rename)

    if "year" not in df.columns:
        df = df.rename(columns={df.columns[0]: "year"})

    # 也嘗試自動把 hfcs/pfcs/sf6/nf3 保留進來
    for c in ["hfcs_value", "pfcs_value", "sf6_value", "nf3_value"]:
        if c in df.columns:
            df = df.rename(columns={c: c})  # keep as-is

    df = clean_df(df)
    df = df.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)
    df["year"] = df["year"].astype(int)

    # 若無 total，從所有氣體加總
    if "total" not in df.columns:
        gas_cols = [c for c in ["co2", "ch4", "n2o", "hfcs_value", "pfcs_value", "sf6_value", "nf3_value"] if c in df.columns]
        if gas_cols:
            df["total"] = df[gas_cols].sum(axis=1, min_count=1)

    if "total" not in df.columns:
        return jsonify({"error": "找不到總排放量欄位，請在欄位對應中選擇「總排放量」"}), 400

    df_clean = df.dropna(subset=["total"]).copy()
    if len(df_clean) < 5:
        return jsonify({"error": f"有效數據不足（{len(df_clean)} 筆），至少需要 5 筆年度數據"}), 400

    total_series = df_clean["total"].values.astype(float)
    hist_years = df_clean["year"].tolist()
    last_year = hist_years[-1]
    steps = 2050 - last_year
    if steps <= 0:
        return jsonify({"error": f"資料已涵蓋至 {last_year} 年，無需預測至 2050"}), 400

    order_result = select_arima_order(total_series)
    p, d, q = order_result["p"], order_result["d"], order_result["q"]
    fc = arima_forecast(total_series, (p, d, q), steps)
    fc_years = list(range(last_year + 1, 2051))

    # 氣體種類個別預測
    gas_results = {}
    for col in ["co2", "ch4", "n2o"]:
        if col in df_clean.columns and not df_clean[col].isna().all():
            s = df_clean[col].dropna().values.astype(float)
            if len(s) >= 5:
                try:
                    gfc = arima_forecast(s, (min(p, 1), d, 0), steps)
                    gas_results[col] = {
                        "history": [round(float(v), 2) for v in s],
                        "forecast": [round(float(v), 2) for v in gfc["forecast"]],
                        "upper95": [round(float(v), 2) for v in gfc["upper95"]],
                        "lower95": [round(float(v), 2) for v in gfc["lower95"]],
                    }
                except Exception:
                    pass

    # 歷史表
    history_table = []
    for _, row in df_clean.iterrows():
        r = {"year": int(row["year"])}
        for col in ["energy", "industry", "agri", "land", "total", "net", "co2", "ch4", "n2o"]:
            v = row.get(col, None)
            r[col] = round(float(v), 2) if v is not None and pd.notna(v) else None
        history_table.append(r)

    # 預測表
    forecast_table = []
    for i, yr in enumerate(fc_years):
        r = {
            "year": yr,
            "total": round(fc["forecast"][i], 2),
            "upper95": round(fc["upper95"][i], 2),
            "lower95": round(fc["lower95"][i], 2),
        }
        for col in ["energy", "industry", "agri", "land", "net", "co2", "ch4", "n2o"]:
            r[col] = None
        forecast_table.append(r)

    return jsonify({
        "status": "ok",
        "hist_years": hist_years,
        "hist_total": [round(float(v), 2) for v in total_series],
        "fc_years": fc_years,
        "fc_total": [round(float(v), 2) for v in fc["forecast"]],
        "fc_upper": [round(float(v), 2) for v in fc["upper95"]],
        "fc_lower": [round(float(v), 2) for v in fc["lower95"]],
        "sigma": fc["sigma"],
        "arima_order": {"p": p, "d": d, "q": q},
        "arima_explanation": order_result["explanation"],
        "aic_table": order_result["aic_table"],
        "adf_result": order_result["adf"],
        "sample_size": order_result["sample_size"],
        "warning": order_result["warning"],
        "gas_results": gas_results,
        "history_table": history_table,
        "forecast_table": forecast_table,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "message": "GHG Forecast API v2 is online"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("RENDER") is None
    print("=" * 50)
    print("🌍 GHG 預測系統後端 v2 啟動")
    print(f"   http://0.0.0.0:{port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=debug)
