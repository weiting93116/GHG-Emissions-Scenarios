"""
溫室氣體排放預測系統 - Flask 後端
支援 CSV / Excel 上傳、ARIMA 自動定階、預測至 2050
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # 允許前端跨域請求

# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────

def detect_columns(df: pd.DataFrame) -> dict:
    """
    自動偵測欄位對應：
    支援中英文欄位名稱，找不到則回傳 None
    """
    mapping = {}
    col_lower = {c: c.lower() for c in df.columns}

    patterns = {
        "year":    ["year", "年份", "年度", "西元年"],
        "energy":  ["energy", "能源", "energy sector", "能源部門"],
        "industry":["industry", "工業", "工業製程", "industrial"],
        "agri":    ["agri", "農業", "agriculture", "農業/廢棄物", "農業廢棄物", "agri/waste"],
        "land":    ["land", "土地", "lulucf", "土地匯", "forest", "林業"],
        "total":   ["total", "總排放", "總量", "total emission", "合計"],
    }

    for key, candidates in patterns.items():
        for col, col_l in col_lower.items():
            if any(p in col_l for p in candidates):
                mapping[key] = col
                break

    return mapping


def adf_test(series: np.ndarray) -> dict:
    """
    手動實作 Augmented Dickey-Fuller 檢定簡化版
    若序列差分後趨於平穩 → d 建議值
    回傳: { 'stationary': bool, 'recommended_d': int, 'reason': str }
    """
    def variance_ratio(s):
        if len(s) < 4:
            return 1.0
        diff1 = np.diff(s)
        return np.var(diff1) / (np.var(s) + 1e-10)

    vr0 = np.var(series)
    vr1 = np.var(np.diff(series))
    vr2 = np.var(np.diff(np.diff(series))) if len(series) > 3 else vr1

    # 趨勢強度：線性回歸斜率 / 標準差
    x = np.arange(len(series))
    slope = np.polyfit(x, series, 1)[0]
    trend_strength = abs(slope) / (np.std(series) + 1e-10)

    if trend_strength > 0.05 or vr1 < vr0 * 0.7:
        # 原序列非平穩，一階差分改善顯著
        if vr2 < vr1 * 0.7:
            d = 2
            reason = f"原序列具明顯趨勢（斜率強度={trend_strength:.3f}），一階差分後仍不平穩（Var比={vr1/vr0:.3f}），建議 d=2"
        else:
            d = 1
            reason = f"原序列具明顯趨勢（斜率強度={trend_strength:.3f}），一階差分後達到平穩（Var比={vr1/vr0:.3f}），建議 d=1"
        stationary = False
    else:
        d = 0
        reason = f"原序列已接近平穩（斜率強度={trend_strength:.3f}），無需差分，建議 d=0"
        stationary = True

    return {"stationary": stationary, "recommended_d": d, "reason": reason}


def select_arima_order(series: np.ndarray, max_p: int = 3, max_q: int = 3) -> dict:
    """
    使用 AIC 準則選擇最佳 (p, d, q)
    - d 由 ADF-like 方法決定
    - p, q 窮舉 AIC 最小組合
    回傳完整說明
    """
    n = len(series)
    adf_result = adf_test(series)
    d = adf_result["recommended_d"]

    # 差分後序列
    s = series.copy().astype(float)
    for _ in range(d):
        s = np.diff(s)

    best_aic = np.inf
    best_p, best_q = 0, 0
    aic_table = []

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                # ARIMA(0,d,0): 純隨機漫步
                resid = s - np.mean(s)
                sigma2 = np.var(resid)
                k = 1  # 只有截距
            else:
                # 簡化 AR(p) 用 OLS 估計
                try:
                    if p > 0 and len(s) > p + 1:
                        X = np.column_stack([s[p-i-1:len(s)-i-1] for i in range(p)] + [np.ones(len(s)-p)])
                        y = s[p:]
                        if X.shape[0] < X.shape[1] + 2:
                            continue
                        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        pred = X @ coef
                        resid = y - pred
                        sigma2 = np.var(resid)
                        k = p + q + 1
                    else:
                        resid = s - np.mean(s)
                        sigma2 = np.var(resid)
                        k = 1
                except Exception:
                    continue

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

    # 限制小樣本過擬合：若樣本 < 30 且 p+q > 2，降階
    warning = None
    if n < 35 and (best_p + best_q) > 2:
        warning = f"⚠️ 樣本數僅 {n} 筆，原始最佳 ARIMA({best_p},{d},{best_q}) 有過擬合風險，已限制 p+q ≤ 2"
        candidates = [r for r in aic_table if r["p"] + r["q"] <= 2]
        if candidates:
            best = min(candidates, key=lambda x: x["AIC"])
            best_p, best_q = best["p"], best["q"]

    # 建立說明文字
    param_explanation = build_param_explanation(best_p, d, best_q, adf_result, n, warning)

    # 排序 AIC 表
    aic_table_sorted = sorted(aic_table, key=lambda x: x["AIC"])[:12]

    return {
        "p": best_p,
        "d": d,
        "q": best_q,
        "aic": round(best_aic, 2),
        "adf": adf_result,
        "warning": warning,
        "explanation": param_explanation,
        "aic_table": aic_table_sorted,
        "sample_size": n
    }


def build_param_explanation(p: int, d: int, q: int, adf: dict, n: int, warning) -> dict:
    """產生 ARIMA(p,d,q) 各參數的中文說明"""

    d_explain = {
        0: f"**d=0（不差分）**：原序列已平穩，沒有明顯線性趨勢，均值與變異數穩定。直接對原始數據建模。",
        1: f"**d=1（一階差分）**：{adf['reason']}。差分後序列 Yₜ-Yₜ₋₁ 趨於平穩，消除線性趨勢。這是年度排放量最常見的設定。",
        2: f"**d=2（二階差分）**：{adf['reason']}。排放量具加速上升或下降趨勢，需要兩次差分才能平穩。"
    }

    p_explain = {
        0: "**p=0（無 AR 項）**：當期排放量與過去各期無顯著自相關（ACF 無明顯截尾），殘差近似白雜訊，預測主要依賴趨勢項。",
        1: "**p=1（AR(1)）**：當期排放量受前一年影響，具一階自回歸結構（PACF 在 lag=1 截尾）。常見於具慣性的政策驅動排放序列。",
        2: "**p=2（AR(2)）**：當期排放量受前兩年共同影響，呈現兩期自回歸（PACF 在 lag=2 截尾），常見於排放具景氣循環慣性的情境。",
        3: "**p=3（AR(3)）**：三期自回歸，序列具較長記憶性。需注意小樣本下的過擬合風險。"
    }

    q_explain = {
        0: "**q=0（無 MA 項）**：殘差之間無顯著移動平均結構（ACF 無截尾），衝擊效果不持續跨期，殘差獨立。",
        1: "**q=1（MA(1)）**：外部衝擊（如政策衝擊、能源危機）對排放的影響持續一年後消退（ACF 在 lag=1 截尾）。",
        2: "**q=2（MA(2)）**：衝擊效果延續兩年，常見於需要 1-2 年才能反映的政策調整情境。",
        3: "**q=3（MA(3)）**：衝擊效果延續三年，適合有重大政策轉折點的歷史序列。"
    }

    small_sample_note = ""
    if n < 35:
        small_sample_note = f"\n\n> 📌 **小樣本警告**：本資料集僅有 {n} 個年度觀測值。統計理論建議 ARIMA 模型參數數量不超過樣本數的 10%（即最多 {max(1, n//10)} 個自由參數）。p+q 已限制在合理範圍內以避免過擬合。"

    return {
        "p": p_explain.get(p, f"p={p}：高階自回歸項"),
        "d": d_explain.get(d, f"d={d}：多階差分"),
        "q": q_explain.get(q, f"q={q}：高階移動平均項"),
        "summary": f"根據 AIC 最小化準則，最終選定 **ARIMA({p},{d},{q})**。{small_sample_note}",
        "adf_reason": adf["reason"]
    }


def arima_forecast(series: np.ndarray, order: tuple, steps: int) -> dict:
    """
    手動實作 ARIMA(p,d,q) 預測
    回傳: 預測值、95% 信心區間上下界、殘差標準差
    """
    p, d, q = order
    orig = series.copy().astype(float)

    # 差分
    s = orig.copy()
    for _ in range(d):
        s = np.diff(s)

    n = len(s)
    mean_s = np.mean(s)

    # AR 係數估計（OLS）
    ar_coefs = np.zeros(p)
    if p > 0 and n > p:
        X = np.column_stack([s[p-i-1:n-i-1] for i in range(p)])
        y = s[p:]
        try:
            ar_coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            ar_coefs = np.zeros(p)

    # 殘差 & 標準差
    if p > 0 and n > p:
        fitted = np.array([np.dot(ar_coefs, s[i:i+p][::-1]) for i in range(p, n)])
        resid = s[p:] - fitted
    else:
        resid = s - mean_s
    sigma = np.std(resid)

    # 向前預測（差分序列）
    s_extended = list(s)
    forecasts_diff = []
    for i in range(steps):
        if p > 0 and len(s_extended) >= p:
            ar_part = np.dot(ar_coefs, s_extended[-p:][::-1])
        else:
            ar_part = mean_s
        forecasts_diff.append(ar_part)
        s_extended.append(ar_part)

    # 逆差分還原
    last_vals = list(orig[-d:]) if d > 0 else []
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

    # 信心區間（隨時間擴張）
    ci_upper = preds + 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
    ci_lower = preds - 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))

    return {
        "forecast": preds.tolist(),
        "upper95": ci_upper.tolist(),
        "lower95": ci_lower.tolist(),
        "sigma": round(float(sigma), 4)
    }


# ──────────────────────────────────────────────
# API 路由
# ──────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    接收前端上傳的 CSV 或 Excel 檔案
    回傳：解析後的欄位清單，供前端讓用戶確認欄位映射
    """
    if "file" not in request.files:
        return jsonify({"error": "未收到檔案"}), 400

    f = request.files["file"]
    filename = f.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(f.read()), encoding="utf-8-sig")
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(f.read()))
        else:
            return jsonify({"error": "僅支援 CSV 或 Excel 格式"}), 400
    except Exception as e:
        return jsonify({"error": f"檔案解析失敗：{str(e)}"}), 400

    # 自動偵測欄位
    col_mapping = detect_columns(df)

    return jsonify({
        "columns": list(df.columns),
        "detected": col_mapping,
        "preview": df.head(5).fillna("").to_dict(orient="records"),
        "rows": len(df)
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    接收：
      - file: 上傳的 CSV/Excel
      - col_year, col_energy, col_industry, col_agri, col_land, col_total: 欄位映射
    回傳：ARIMA 分析結果、預測至 2050 數據、參數說明
    """
    if "file" not in request.files:
        return jsonify({"error": "未收到檔案"}), 400

    f = request.files["file"]
    filename = f.filename.lower()
    col_year     = request.form.get("col_year", "")
    col_energy   = request.form.get("col_energy", "")
    col_industry = request.form.get("col_industry", "")
    col_agri     = request.form.get("col_agri", "")
    col_land     = request.form.get("col_land", "")
    col_total    = request.form.get("col_total", "")

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(f.read()), encoding="utf-8-sig")
        else:
            df = pd.read_excel(io.BytesIO(f.read()))
    except Exception as e:
        return jsonify({"error": f"檔案解析失敗：{str(e)}"}), 400

    # 建立統一欄位名
    rename = {}
    cols_needed = {
        "year": col_year, "energy": col_energy,
        "industry": col_industry, "agri": col_agri,
        "land": col_land, "total": col_total
    }
    for std, orig in cols_needed.items():
        if orig and orig in df.columns:
            rename[orig] = std

    df = df.rename(columns=rename)

    # 確保有年份欄
    if "year" not in df.columns:
        # 嘗試第一欄作為年份
        df = df.rename(columns={df.columns[0]: "year"})

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)
    df["year"] = df["year"].astype(int)

    # 數值化各部門
    for col in ["energy", "industry", "agri", "land", "total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 若無 total 欄，計算加總（land 為負匯，需處理 NaN）
    sector_cols = [c for c in ["energy", "industry", "agri"] if c in df.columns]
    if "total" not in df.columns and sector_cols:
        df["total"] = df[sector_cols].sum(axis=1, min_count=1)
        if "land" in df.columns:
            df["total"] = df["total"] + df["land"].fillna(0)

    if "total" not in df.columns:
        return jsonify({"error": "無法計算總排放量，請確認欄位設定"}), 400

    # 準備序列（去除 NaN）
    df_clean = df.dropna(subset=["total"]).copy()
    total_series = df_clean["total"].values
    hist_years = df_clean["year"].tolist()

    if len(total_series) < 5:
        return jsonify({"error": f"有效數據不足（{len(total_series)} 筆），至少需要 5 筆"}), 400

    last_year = hist_years[-1]
    forecast_end = 2050
    steps = forecast_end - last_year

    if steps <= 0:
        return jsonify({"error": f"資料已涵蓋至 {last_year} 年，無需預測至 2050"}), 400

    # ARIMA 自動選階
    order_result = select_arima_order(total_series, max_p=3, max_q=3)
    p, d, q = order_result["p"], order_result["d"], order_result["q"]

    # 執行預測
    fc = arima_forecast(total_series, (p, d, q), steps)
    fc_years = list(range(last_year + 1, forecast_end + 1))

    # 各部門預測（比例外推 + ARIMA 各自計算）
    sector_results = {}
    for col in ["energy", "industry", "agri", "land"]:
        if col in df_clean.columns and not df_clean[col].isna().all():
            sec_series = df_clean[col].fillna(0).values
            try:
                sec_fc = arima_forecast(sec_series, (min(p, 1), d, 0), steps)
                sector_results[col] = {
                    "history": [round(float(v), 2) for v in sec_series],
                    "forecast": [round(float(v), 2) for v in sec_fc["forecast"]],
                    "upper95": [round(float(v), 2) for v in sec_fc["upper95"]],
                    "lower95": [round(float(v), 2) for v in sec_fc["lower95"]],
                }
            except Exception:
                sector_results[col] = {"history": sec_series.tolist(), "forecast": [], "upper95": [], "lower95": []}

    # 歷史數據整理
    history_table = []
    for _, row in df_clean.iterrows():
        record = {"year": int(row["year"])}
        for col in ["energy", "industry", "agri", "land", "total"]:
            if col in row and not pd.isna(row[col]):
                record[col] = round(float(row[col]), 2)
            else:
                record[col] = None
        history_table.append(record)

    # 預測數據表
    forecast_table = []
    for i, yr in enumerate(fc_years):
        record = {
            "year": yr,
            "total": round(fc["forecast"][i], 2),
            "upper95": round(fc["upper95"][i], 2),
            "lower95": round(fc["lower95"][i], 2),
        }
        for col in ["energy", "industry", "agri", "land"]:
            if col in sector_results and i < len(sector_results[col]["forecast"]):
                record[col] = round(sector_results[col]["forecast"][i], 2)
            else:
                record[col] = None
        forecast_table.append(record)

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
        "sector_results": sector_results,
        "history_table": history_table,
        "forecast_table": forecast_table,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "message": "GHG Forecast API is online"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("RENDER") is None  # 本機才開 debug
    print("=" * 50)
    print("🌍 溫室氣體預測系統後端啟動中...")
    print(f"   API 位址：http://0.0.0.0:{port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=debug)
