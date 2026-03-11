"""
溫室氣體排放預測系統 - 前後端合一版
Flask 直接 serve HTML，不需要跨域，直接部署到 Render
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")

import json, math

app = Flask(__name__)

def nan_to_none(obj):
    """遞迴把所有 float NaN / Inf 換成 None，避免 JSON 序列化失敗"""
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

def safe_json(data, status=200):
    resp = app.response_class(
        response=json.dumps(nan_to_none(data), ensure_ascii=False),
        status=status,
        mimetype='application/json'
    )
    return resp

# ─── 前端 HTML（內嵌）─────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
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
header{padding:36px 0 28px;display:flex;align-items:flex-start;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);margin-bottom:32px}
.hdr-title{font-size:20px;font-weight:900;color:var(--bright);letter-spacing:.02em;line-height:1.2}
.hdr-sub{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--teal);margin-top:6px;letter-spacing:.14em}
.hdr-badges{display:flex;gap:6px;flex-wrap:wrap;margin-top:4px}
.chip{font-family:'JetBrains Mono',monospace;font-size:10px;padding:3px 9px;border-radius:100px;border:1px solid;letter-spacing:.06em}
.chip-teal{border-color:var(--teal2);color:var(--teal);background:rgba(0,229,192,.06)}
.chip-sky{border-color:#1e7fa0;color:var(--sky);background:rgba(56,189,248,.06)}
.chip-amr{border-color:#a06f08;color:var(--amber);background:rgba(245,158,11,.06)}
.upload-zone{border:2px dashed var(--line2);border-radius:var(--r);padding:40px 28px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s;background:var(--ink2);position:relative;margin-bottom:20px}
.upload-zone:hover,.upload-zone.drag{border-color:var(--teal);background:rgba(0,229,192,.04)}
.upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-icon{font-size:36px;margin-bottom:12px}
.upload-text{font-size:14px;color:var(--mid)}
.upload-text strong{color:var(--teal)}
.upload-hint{font-size:11px;color:var(--muted);margin-top:6px;font-family:'JetBrains Mono',monospace}
.file-chosen{background:rgba(0,229,192,.06);border-color:var(--teal2);padding:16px 24px;display:flex;align-items:center;gap:12px}
.file-chosen .fname{font-family:'JetBrains Mono',monospace;color:var(--teal);font-size:12px}
.col-mapping{background:var(--ink2);border:1px solid var(--line);border-radius:var(--r);padding:20px 24px;margin-bottom:20px;display:none}
.col-mapping h3{font-size:11px;color:var(--teal);letter-spacing:.12em;font-family:'JetBrains Mono',monospace;margin-bottom:16px}
.map-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.map-group label{display:block;font-size:11px;color:var(--dim);margin-bottom:5px}
.map-group select{width:100%;background:var(--ink3);border:1px solid var(--line2);border-radius:5px;color:var(--text);padding:7px 10px;font-size:12px;font-family:'Noto Sans TC',sans-serif;outline:none;cursor:pointer}
.map-group select:focus{border-color:var(--teal)}
.map-preview{margin-top:16px;overflow-x:auto}
.preview-table{width:100%;border-collapse:collapse;font-size:11px;font-family:'JetBrains Mono',monospace}
.preview-table th{padding:6px 10px;background:var(--ink3);color:var(--dim);text-align:left;border-bottom:1px solid var(--line)}
.preview-table td{padding:5px 10px;color:var(--mid);border-bottom:1px solid rgba(33,48,63,.5)}
.btn-analyze{background:linear-gradient(135deg,var(--teal2),#009e80);color:var(--ink);border:none;border-radius:6px;padding:12px 32px;font-weight:700;font-size:13px;cursor:pointer;letter-spacing:.06em;font-family:'Noto Sans TC',sans-serif;transition:opacity .15s,transform .1s;display:inline-flex;align-items:center;gap:8px}
.btn-analyze:hover{opacity:.88;transform:translateY(-1px)}
.btn-analyze:disabled{opacity:.4;cursor:not-allowed;transform:none}
.loading{display:none;align-items:center;gap:10px;color:var(--dim);font-size:12px;font-family:'JetBrains Mono',monospace}
.spinner{width:16px;height:16px;border:2px solid var(--line2);border-top-color:var(--teal);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.error-box{display:none;background:rgba(251,113,133,.08);border:1px solid rgba(251,113,133,.25);border-radius:6px;padding:12px 16px;color:var(--rose);font-size:12px;margin-top:12px}
#results{display:none}
.stats-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-bottom:24px}
.stat{background:var(--ink2);border:1px solid var(--line);border-radius:var(--r);padding:16px}
.stat-label{font-size:10px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px}
.stat-val{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700}
.stat-sub{font-size:10px;color:var(--muted);margin-top:3px}
.c-teal{color:var(--teal)}.c-sky{color:var(--sky)}.c-amr{color:var(--amber)}.c-rose{color:var(--rose)}.c-vio{color:var(--violet)}
.chart-grid-main{display:grid;grid-template-columns:3fr 1fr;gap:16px;margin-bottom:16px}
.chart-grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:16px}
.chart-grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-bottom:16px}
.card{background:var(--ink2);border:1px solid var(--line);border-radius:var(--r);padding:20px}
.card-title{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.card-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.arima-panel{background:var(--ink2);border:1px solid var(--line);border-radius:var(--r);padding:24px;margin-bottom:16px}
.arima-panel h2{font-size:14px;font-weight:700;color:var(--bright);margin-bottom:20px}
.arima-order-display{display:flex;align-items:center;gap:6px;margin-bottom:24px;background:var(--ink3);border:1px solid var(--line2);border-radius:6px;padding:14px 20px;flex-wrap:wrap}
.ord-val{font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700}
.ord-sep{font-size:24px;color:var(--muted)}
.param-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}
.param-card{background:var(--ink3);border-radius:6px;padding:16px;border-left:3px solid}
.param-card.pc-p{border-color:var(--sky)}.param-card.pc-d{border-color:var(--teal)}.param-card.pc-q{border-color:var(--violet)}
.param-card .pc-label{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;margin-bottom:8px}
.param-card.pc-p .pc-label{color:var(--sky)}.param-card.pc-d .pc-label{color:var(--teal)}.param-card.pc-q .pc-label{color:var(--violet)}
.param-card .pc-text{font-size:12px;color:var(--mid);line-height:1.65}
.adf-box{background:var(--ink3);border:1px solid var(--line2);border-radius:6px;padding:14px 16px;margin-bottom:16px}
.adf-box .adf-title{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--amber);letter-spacing:.1em;margin-bottom:6px}
.adf-box .adf-text{font-size:12px;color:var(--mid);line-height:1.6}
.warning-box{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.25);border-radius:6px;padding:12px 16px;margin-bottom:16px;color:var(--amber);font-size:12px;display:none}
.aic-table-wrap{overflow-x:auto}
.aic-tbl{width:100%;border-collapse:collapse;font-size:11px;font-family:'JetBrains Mono',monospace}
.aic-tbl th{padding:7px 12px;background:var(--ink3);color:var(--dim);text-align:center;border-bottom:1px solid var(--line);font-size:10px}
.aic-tbl td{padding:6px 12px;text-align:center;border-bottom:1px solid rgba(33,48,63,.5);color:var(--mid)}
.aic-tbl tr.best td{background:rgba(0,229,192,.07);color:var(--teal)}
.aic-tbl tr.best td:first-child::before{content:'★ '}
.table-section{background:var(--ink2);border:1px solid var(--line);border-radius:var(--r);padding:20px;margin-bottom:16px}
.table-section h3{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:16px}
.tbl-wrap{overflow-x:auto;max-height:520px;overflow-y:auto}
.fc-table{width:100%;border-collapse:collapse;font-size:12px}
.fc-table thead th{position:sticky;top:0;background:var(--ink3);padding:9px 14px;text-align:right;color:var(--dim);font-size:10px;font-family:'JetBrains Mono',monospace;letter-spacing:.08em;border-bottom:1px solid var(--line);white-space:nowrap}
.fc-table thead th:first-child{text-align:center}
.fc-table tbody td{padding:7px 14px;border-bottom:1px solid rgba(33,48,63,.4);text-align:right;font-family:'JetBrains Mono',monospace}
.fc-table tbody td:first-child{text-align:center;color:var(--dim)}
.fc-table tbody tr:hover td{background:rgba(0,229,192,.03)}
.fc-table tbody tr.hist-row td{color:var(--mid)}
.fc-table tbody tr.fc-row td{color:var(--sky)}
.fc-table tbody tr.fc-row td:first-child{color:var(--teal)}
.fc-table tbody tr.divider td{border-top:2px solid var(--teal2)}
.null-val{color:var(--muted)!important}.neg-val{color:var(--rose)!important}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px}
.leg-item{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--dim)}
.leg-line{width:22px;height:2px}
@media(max-width:960px){.chart-grid-main,.chart-grid-3{grid-template-columns:1fr}.chart-grid-2{grid-template-columns:1fr}.param-cards{grid-template-columns:1fr}}
@media(max-width:600px){.stats-row{grid-template-columns:repeat(2,1fr)}header{flex-direction:column}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <div class="hdr-title">🌍 溫室氣體排放預測系統</div>
    <div class="hdr-sub">GHG EMISSION FORECAST · ARIMA AUTO-ORDER · TO 2050</div>
    <div class="hdr-badges">
      <span class="chip chip-teal">ARIMA Auto-Order</span>
      <span class="chip chip-sky">95% CI</span>
      <span class="chip chip-amr">CO₂ · CH₄ · N₂O</span>
    </div>
  </div>
</header>

<div style="margin-bottom:24px">
  <div class="upload-zone" id="uploadZone">
    <input type="file" id="fileInput" accept=".csv,.xlsx,.xls">
    <div class="upload-icon">📂</div>
    <div class="upload-text">拖曳或 <strong>點擊</strong> 上傳歷史排放資料</div>
    <div class="upload-hint">支援 CSV · Excel (.xlsx / .xls)</div>
  </div>

  <div class="col-mapping" id="colMapping">
    <h3>⚙ COLUMN MAPPING — 欄位對應設定</h3>
    <div class="map-grid">
      <div class="map-group"><label>📅 年份欄位 *</label><select id="mapYear"></select></div>
      <div class="map-group"><label>📊 總排放量 (kt CO₂e) *</label><select id="mapTotal"></select></div>
      <div class="map-group"><label>🌿 CO₂ 排放量</label><select id="mapCO2"></select></div>
      <div class="map-group"><label>🐄 CH₄ 排放量</label><select id="mapCH4"></select></div>
      <div class="map-group"><label>🌾 N₂O 排放量</label><select id="mapN2O"></select></div>
      <div class="map-group"><label>🌲 土地匯 / CO₂ 吸收量</label><select id="mapLand"></select></div>
      <div class="map-group"><label>📉 淨排放量（含土地匯）</label><select id="mapNet"></select></div>
      <div class="map-group"><label>⚡ 能源部門（選填）</label><select id="mapEnergy"></select></div>
      <div class="map-group"><label>🏭 工業製程（選填）</label><select id="mapIndustry"></select></div>
      <div class="map-group"><label>🌾 農業/廢棄物（選填）</label><select id="mapAgri"></select></div>
    </div>
    <div class="map-preview">
      <div style="font-size:10px;color:var(--muted);margin-bottom:6px;font-family:'JetBrains Mono',monospace">DATA PREVIEW (first 5 rows)</div>
      <div class="tbl-wrap" style="max-height:160px"><table class="preview-table" id="previewTable"></table></div>
    </div>
  </div>

  <div style="display:flex;align-items:center;gap:16px;margin-top:16px">
    <button class="btn-analyze" id="analyzeBtn" disabled>▶ 執行 ARIMA 分析 → 預測至 2050</button>
    <div class="loading" id="loadingInd"><div class="spinner"></div><span>分析中，請稍候…</span></div>
  </div>
  <div class="error-box" id="errorBox"></div>
</div>

<div id="results">
  <div class="stats-row">
    <div class="stat"><div class="stat-label">資料範圍</div><div class="stat-val c-teal" id="s-range">—</div><div class="stat-sub">歷史年份</div></div>
    <div class="stat"><div class="stat-label">樣本數</div><div class="stat-val c-sky" id="s-n">—</div><div class="stat-sub">年度觀測值</div></div>
    <div class="stat"><div class="stat-label">基準年排放</div><div class="stat-val c-amr" id="s-base">—</div><div class="stat-sub">kt CO₂e</div></div>
    <div class="stat"><div class="stat-label">2050 預測</div><div class="stat-val c-sky" id="s-2050">—</div><div class="stat-sub">kt CO₂e（中位）</div></div>
    <div class="stat"><div class="stat-label">變化率</div><div class="stat-val" id="s-chg">—</div><div class="stat-sub">相對基準年</div></div>
    <div class="stat"><div class="stat-label">ARIMA 階數</div><div class="stat-val c-teal" id="s-order">—</div><div class="stat-sub">最佳 AIC</div></div>
    <div class="stat"><div class="stat-label">殘差 σ</div><div class="stat-val c-vio" id="s-sigma">—</div><div class="stat-sub">kt CO₂e</div></div>
  </div>

  <div class="chart-grid-main">
    <div class="card">
      <div class="card-title"><span class="card-dot" style="background:var(--teal)"></span>總排放量歷史 + ARIMA 預測（至 2050）</div>
      <canvas id="mainChart" height="160"></canvas>
      <div class="legend">
        <div class="leg-item"><div class="leg-line" style="background:var(--teal)"></div>歷史排放</div>
        <div class="leg-item"><div class="leg-line" style="background:var(--sky)"></div>ARIMA 預測</div>
        <div class="leg-item"><div class="leg-line" style="background:rgba(56,189,248,.2);height:10px;width:22px;border-radius:2px"></div>95% 信心區間</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title"><span class="card-dot" style="background:var(--amber)"></span>2050 年氣體組成</div>
      <canvas id="pieChart"></canvas>
    </div>
  </div>

  <div class="chart-grid-3">
    <div class="card"><div class="card-title"><span class="card-dot" style="background:#60a5fa"></span>CO₂ 排放趨勢</div><canvas id="cCO2" height="130"></canvas></div>
    <div class="card"><div class="card-title"><span class="card-dot" style="background:#34d399"></span>CH₄ 排放趨勢</div><canvas id="cCH4" height="130"></canvas></div>
    <div class="card"><div class="card-title"><span class="card-dot" style="background:#f472b6"></span>N₂O 排放趨勢</div><canvas id="cN2O" height="130"></canvas></div>
  </div>
  <div class="chart-grid-2">
    <div class="card"><div class="card-title"><span class="card-dot" style="background:var(--rose)"></span>淨排放量（含土地匯）</div><canvas id="cNet" height="130"></canvas></div>
    <div class="card"><div class="card-title"><span class="card-dot" style="background:var(--violet)"></span>AIC 各階數比較</div><canvas id="aicChart" height="130"></canvas></div>
  </div>

  <div class="arima-panel">
    <h2>🔬 ARIMA 參數設定說明</h2>
    <div class="arima-order-display">
      <span style="font-size:12px;color:var(--dim);margin-right:4px">最終模型：</span>
      <span class="ord-val" style="color:var(--teal)">ARIMA</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:20px;color:var(--teal)">( </span>
      <span class="ord-val" id="exp-p" style="color:var(--sky)">?</span>
      <span class="ord-sep">,</span>
      <span class="ord-val" id="exp-d" style="color:var(--teal)">?</span>
      <span class="ord-sep">,</span>
      <span class="ord-val" id="exp-q" style="color:var(--violet)">?</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:20px;color:var(--teal)"> )</span>
      <span style="margin-left:16px;font-size:12px;color:var(--dim);font-family:'JetBrains Mono',monospace">AIC = <span id="exp-aic">—</span></span>
    </div>
    <div class="warning-box" id="warnBox"></div>
    <div class="adf-box">
      <div class="adf-title">▸ ADF 平穩性檢定結果</div>
      <div class="adf-text" id="adfText">—</div>
    </div>
    <div class="param-cards">
      <div class="param-card pc-p"><div class="pc-label">p = <span id="exp-p2">?</span></div><div class="pc-text" id="exp-p-text">—</div></div>
      <div class="param-card pc-d"><div class="pc-label">d = <span id="exp-d2">?</span></div><div class="pc-text" id="exp-d-text">—</div></div>
      <div class="param-card pc-q"><div class="pc-label">q = <span id="exp-q2">?</span></div><div class="pc-text" id="exp-q-text">—</div></div>
    </div>
    <div class="adf-box" style="background:rgba(0,229,192,.04);border-color:rgba(0,229,192,.15)">
      <div class="adf-title" style="color:var(--teal)">▸ 綜合結論</div>
      <div class="adf-text" id="summaryText">—</div>
    </div>
    <div style="margin-top:20px">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:.1em;margin-bottom:10px">AIC SELECTION TABLE（前 12 名）</div>
      <div class="aic-table-wrap">
        <table class="aic-tbl">
          <thead><tr><th>排名</th><th>p</th><th>d</th><th>q</th><th>AIC</th></tr></thead>
          <tbody id="aicTbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="table-section">
    <h3>📋 歷史 + 預測數據表（至 2050 年）</h3>
    <div class="tbl-wrap">
      <table class="fc-table">
        <thead><tr>
          <th>年份</th><th>CO₂ (kt)</th><th>CH₄ (kt)</th><th>N₂O (kt)</th>
          <th>土地匯 (kt)</th><th>總排放量 (kt)</th><th>淨排放量 (kt)</th>
          <th>預測上界 95%</th><th>預測下界 95%</th><th>類型</th>
        </tr></thead>
        <tbody id="forecastTbody"></tbody>
      </table>
    </div>
  </div>
</div>
</div>

<script>
const API = '';  // 空字串 = 同一網域，自動對應
let charts = {}, uploadedFile = null;

// ── Upload ──
const uploadZone = document.getElementById('uploadZone');
const fileInput  = document.getElementById('fileInput');
uploadZone.addEventListener('dragover', e=>{e.preventDefault();uploadZone.classList.add('drag')});
uploadZone.addEventListener('dragleave',()=>uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop',e=>{e.preventDefault();uploadZone.classList.remove('drag');if(e.dataTransfer.files[0])handleFile(e.dataTransfer.files[0])});
fileInput.addEventListener('change',()=>{if(fileInput.files[0])handleFile(fileInput.files[0])});

async function handleFile(file) {
  uploadedFile = file;
  uploadZone.innerHTML = `<div class="file-chosen"><span style="font-size:24px">📄</span><div><div class="fname">${file.name}</div><div style="font-size:10px;color:var(--muted);margin-top:2px">${(file.size/1024).toFixed(1)} KB · 上傳中…</div></div></div>`;
  const fd = new FormData(); fd.append('file', file);
  try {
    const r = await fetch('/api/upload',{method:'POST',body:fd});
    const d = await r.json();
    if(d.error){showErr(d.error);return}
    buildSelects(d.columns, d.detected, d.preview);
    document.getElementById('colMapping').style.display='block';
    document.getElementById('analyzeBtn').disabled=false;
    uploadZone.innerHTML=`<div class="file-chosen"><span style="font-size:24px">✅</span><div><div class="fname">${file.name}</div><div style="font-size:10px;color:var(--dim);margin-top:2px">${d.rows} 筆 · ${d.columns.length} 欄 · 偵測完成</div></div></div>`;
  } catch(e){showErr('上傳失敗：'+e.message)}
}

function buildSelects(cols, detected, preview){
  const opts=['（不使用）',...cols];
  [['mapYear','year'],['mapTotal','total'],['mapCO2','co2'],['mapCH4','ch4'],['mapN2O','n2o'],
   ['mapLand','land'],['mapNet','net'],['mapEnergy','energy'],['mapIndustry','industry'],['mapAgri','agri']
  ].forEach(([id,key])=>{
    const s=document.getElementById(id); if(!s)return;
    s.innerHTML=opts.map(c=>`<option value="${c==='（不使用）'?'':c}">${c}</option>`).join('');
    if(detected[key]) s.value=detected[key];
  });
  const t=document.getElementById('previewTable');
  t.innerHTML=`<tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`+(preview||[]).map(row=>`<tr>${cols.map(c=>`<td>${row[c]??''}</td>`).join('')}</tr>`).join('');
}

// ── Analyze ──
document.getElementById('analyzeBtn').addEventListener('click', async()=>{
  if(!uploadedFile)return;
  hideErr();
  document.getElementById('loadingInd').style.display='flex';
  document.getElementById('analyzeBtn').disabled=true;
  document.getElementById('results').style.display='none';
  const fd=new FormData(); fd.append('file',uploadedFile);
  [['col_year','mapYear'],['col_total','mapTotal'],['col_co2','mapCO2'],['col_ch4','mapCH4'],
   ['col_n2o','mapN2O'],['col_land','mapLand'],['col_net','mapNet'],
   ['col_energy','mapEnergy'],['col_industry','mapIndustry'],['col_agri','mapAgri']
  ].forEach(([k,id])=>{const el=document.getElementById(id);if(el)fd.append(k,el.value||'')});
  try{
    const r=await fetch('/api/analyze',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error){showErr(d.error);return}
    render(d);
  }catch(e){showErr('分析失敗：'+e.message)}
  finally{document.getElementById('loadingInd').style.display='none';document.getElementById('analyzeBtn').disabled=false}
});

// ── Render ──
function render(d){
  document.getElementById('results').style.display='block';
  const hLen=d.hist_years.length, fLen=d.fc_years.length;
  const allY=[...d.hist_years.map(String),...d.fc_years.map(String)];
  const o=d.arima_order, exp=d.arima_explanation;
  const base=d.hist_total[hLen-1], end=d.fc_total[fLen-1];
  const chg=((end-base)/base*100).toFixed(1);

  document.getElementById('s-range').textContent=`${d.hist_years[0]}–${d.hist_years[hLen-1]}`;
  document.getElementById('s-n').textContent=d.sample_size;
  document.getElementById('s-base').textContent=fmt(base);
  document.getElementById('s-2050').textContent=fmt(end);
  const ce=document.getElementById('s-chg'); ce.textContent=(chg>0?'+':'')+chg+'%'; ce.className='stat-val '+(chg<0?'c-teal':'c-rose');
  document.getElementById('s-order').textContent=`(${o.p},${o.d},${o.q})`;
  document.getElementById('s-sigma').textContent=fmt(d.sigma);

  // Main chart
  mk('mainChart',{type:'line',data:{labels:allY,datasets:[
    {data:[...Array(hLen).fill(null),...d.fc_upper],borderColor:'transparent',backgroundColor:'rgba(56,189,248,0.12)',fill:'+1',pointRadius:0},
    {data:[...Array(hLen).fill(null),...d.fc_lower],borderColor:'transparent',fill:false,pointRadius:0},
    {data:[...d.hist_total,...Array(fLen).fill(null)],borderColor:'#00e5c0',borderWidth:2.5,pointRadius:0,tension:0.3},
    {data:[...Array(hLen).fill(null),...d.fc_total],borderColor:'#38bdf8',borderWidth:2,borderDash:[7,4],pointRadius:0,tension:0.3},
  ]},options:lopts('kt CO₂e',1.9)});

  // Pie
  const gk=['co2','ch4','n2o'],gc=['rgba(96,165,250,.85)','rgba(52,211,153,.85)','rgba(244,114,182,.85)'],gl=['CO₂','CH₄','N₂O'];
  const pv=gk.map(k=>{const g=d.gas_results?.[k];return g&&g.forecast.length?Math.max(0,g.forecast[g.forecast.length-1]):null});
  const pfl={l:[],d:[],c:[]};
  if(pv.some(v=>v!==null)){gl.forEach((lb,i)=>{if(pv[i]!==null){pfl.l.push(lb);pfl.d.push(pv[i]);pfl.c.push(gc[i])}})}
  else{pfl.l=['總排放量'];pfl.d=[end];pfl.c=['rgba(0,229,192,.8)']}
  mk('pieChart',{type:'doughnut',data:{labels:pfl.l,datasets:[{data:pfl.d,backgroundColor:pfl.c,borderColor:'#161b22',borderWidth:2}]},
    options:{responsive:true,maintainAspectRatio:true,plugins:{legend:{display:true,position:'bottom',labels:{color:'#7a9ab0',font:{size:11},padding:10,boxWidth:12}},tooltip:{backgroundColor:'#1c2733',titleColor:'#d4e8f5',bodyColor:'#7a9ab0'}}}});

  // Gas charts
  [['cCO2','co2','#60a5fa'],['cCH4','ch4','#34d399'],['cN2O','n2o','#f472b6']].forEach(([id,key,col])=>{
    const g=d.gas_results?.[key];
    if(!g){mk(id,{type:'line',data:{labels:[],datasets:[]},options:lopts('kt',1.6)});return}
    mk(id,{type:'line',data:{labels:allY,datasets:[
      {data:[...Array(hLen).fill(null),...g.upper95],borderColor:'transparent',backgroundColor:col+'28',fill:'+1',pointRadius:0},
      {data:[...Array(hLen).fill(null),...g.lower95],borderColor:'transparent',fill:false,pointRadius:0},
      {data:[...g.history,...Array(fLen).fill(null)],borderColor:col,borderWidth:2,pointRadius:0,tension:0.3},
      {data:[...Array(hLen).fill(null),...g.forecast],borderColor:col,borderWidth:1.5,borderDash:[5,3],pointRadius:0,tension:0.3},
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
  mk('aicChart',{type:'bar',data:{labels:aicD.map(r=>`(${r.p},${r.d},${r.q})`),datasets:[{data:aicD.map(r=>r.AIC),backgroundColor:aicD.map(r=>(r.p===o.p&&r.q===o.q?'rgba(0,229,192,.7)':'rgba(56,189,248,.3)')),borderRadius:3}]},
    options:{responsive:true,maintainAspectRatio:true,aspectRatio:1.8,plugins:{legend:{display:false},tooltip:{backgroundColor:'#1c2733',bodyColor:'#94a3b8'}},scales:{x:{ticks:{color:'#4a6070',font:{size:9},maxRotation:45},grid:{color:'rgba(255,255,255,.03)'}},y:{ticks:{color:'#4a6070',font:{size:9}},grid:{color:'rgba(255,255,255,.03)'}}}}});

  // ARIMA explanation
  ['p','d','q'].forEach(k=>{document.getElementById('exp-'+k).textContent=o[k];document.getElementById('exp-'+k+'2').textContent=o[k]});
  document.getElementById('exp-aic').textContent=d.aic_table?.[0]?.AIC??'—';
  document.getElementById('adfText').textContent=exp.adf_reason||'—';
  const hl=s=>(s||'').replace(/\*\*(.*?)\*\*/g,'<strong style="color:var(--teal)">$1</strong>');
  document.getElementById('exp-p-text').innerHTML=(exp.p||'').replace(/\*\*(.*?)\*\*/g,'<strong style="color:var(--sky)">$1</strong>');
  document.getElementById('exp-d-text').innerHTML=hl(exp.d);
  document.getElementById('exp-q-text').innerHTML=(exp.q||'').replace(/\*\*(.*?)\*\*/g,'<strong style="color:var(--violet)">$1</strong>');
  document.getElementById('summaryText').innerHTML=hl(exp.summary);
  if(d.warning){const wb=document.getElementById('warnBox');wb.textContent=d.warning;wb.style.display='block'}

  // AIC table
  document.getElementById('aicTbody').innerHTML=(d.aic_table||[]).map((r,i)=>`<tr class="${r.p===o.p&&r.q===o.q?'best':''}"><td>${i+1}</td><td>${r.p}</td><td>${r.d}</td><td>${r.q}</td><td>${r.AIC}</td></tr>`).join('');

  // Forecast table
  let rows='', first=true;
  d.history_table.forEach(r=>{
    rows+=`<tr class="hist-row"><td>${r.year}</td><td>${r.co2!=null?fmt(r.co2):'<span class="null-val">—</span>'}</td><td>${r.ch4!=null?fmt(r.ch4):'<span class="null-val">—</span>'}</td><td>${r.n2o!=null?fmt(r.n2o):'<span class="null-val">—</span>'}</td><td class="${r.land<0?'neg-val':''}">${r.land!=null?fmt(r.land):'<span class="null-val">—</span>'}</td><td>${r.total!=null?fmt(r.total):'<span class="null-val">—</span>'}</td><td class="${r.net<0?'neg-val':''}">${r.net!=null?fmt(r.net):'<span class="null-val">—</span>'}</td><td class="null-val">—</td><td class="null-val">—</td><td style="color:var(--muted);font-size:10px">歷史</td></tr>`;
  });
  d.forecast_table.forEach(r=>{
    const cls=first?'fc-row divider':'fc-row'; first=false;
    rows+=`<tr class="${cls}"><td>${r.year}</td><td class="null-val">—</td><td class="null-val">—</td><td class="null-val">—</td><td class="null-val">—</td><td>${fmt(r.total)}</td><td class="null-val">—</td><td>${fmt(r.upper95)}</td><td>${fmt(r.lower95)}</td><td style="color:var(--sky);font-size:10px">預測</td></tr>`;
  });
  document.getElementById('forecastTbody').innerHTML=rows;
  document.getElementById('results').scrollIntoView({behavior:'smooth',block:'start'});
}

// ── Helpers ──
const BO={responsive:true,maintainAspectRatio:true,animation:{duration:700},plugins:{legend:{display:false},tooltip:{backgroundColor:'#1c2733',titleColor:'#d4e8f5',bodyColor:'#7a9ab0',borderColor:'#21303f',borderWidth:1}}};
const SC={grid:{color:'rgba(255,255,255,.035)'},ticks:{color:'#4a6070',font:{size:10},maxTicksLimit:8}};
function lopts(yL,ar=2){return{...BO,aspectRatio:ar,scales:{x:SC,y:{...SC,title:{display:true,text:yL,color:'#4a6070',font:{size:10}}}}}}
function mk(id,cfg){if(charts[id])charts[id].destroy();charts[id]=new Chart(document.getElementById(id).getContext('2d'),cfg)}
function fmt(v){if(v==null||isNaN(v))return '<span class="null-val">—</span>';return Number(v).toLocaleString('zh-TW',{maximumFractionDigits:1})}
function showErr(m){const e=document.getElementById('errorBox');e.textContent='❌ '+m;e.style.display='block'}
function hideErr(){document.getElementById('errorBox').style.display='none'}
</script>
</body>
</html>"""


# ─── Python 後端函式 ────────────────────────────────────

def clean_numeric(val):
    if val is None: return np.nan
    s = str(val).strip().replace(',', '').replace('"', '')
    if s.upper() in ('NE','NA','N/A','','-','NO','NOT ESTIMATED'): return np.nan
    try: return float(s)
    except: return np.nan

def clean_df(df):
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') if col=='year' else df[col].apply(clean_numeric)
    return df

def detect_columns(df):
    mapping = {}
    cl = {c: c.lower().replace(' ','_') for c in df.columns}
    patterns = {
        "year":["year","年份","年度"],"co2":["co2_value","co2","二氧化碳"],
        "ch4":["ch4_value","ch4","甲烷"],"n2o":["n2o_value","n2o","氧化亞氮"],
        "total":["total_ghg_emission_value","total_ghg","total","總排放","合計"],
        "land":["co2_absorption_value","absorption","land","土地匯","lulucf"],
        "net":["net_ghg_emission_value","net_ghg","net","淨排放"],
        "energy":["energy","能源"],"industry":["industry","工業"],"agri":["agri","農業"],
    }
    for key, cands in patterns.items():
        for oc, ol in cl.items():
            if any(ol==p or ol.startswith(p) for p in cands):
                if key not in mapping: mapping[key]=oc
    return mapping

def adf_test(series):
    s = series[~np.isnan(series)]
    vr0,vr1 = np.var(s), np.var(np.diff(s))
    vr2 = np.var(np.diff(np.diff(s))) if len(s)>3 else vr1
    slope = np.polyfit(np.arange(len(s)), s, 1)[0]
    ts = abs(slope)/(np.std(s)+1e-10)
    if ts>0.05 or vr1<vr0*0.7:
        if vr2<vr1*0.7: d,stat,reason=2,False,f"原序列具明顯趨勢（斜率強度={ts:.3f}），一階差分後仍不平穩（Var比={vr1/vr0:.3f}），建議 d=2"
        else: d,stat,reason=1,False,f"原序列具明顯趨勢（斜率強度={ts:.3f}），一階差分後達到平穩（Var比={vr1/vr0:.3f}），建議 d=1"
    else: d,stat,reason=0,True,f"原序列已接近平穩（斜率強度={ts:.3f}），無需差分，建議 d=0"
    return {"stationary":stat,"recommended_d":d,"reason":reason}

def select_arima_order(series):
    s = series[~np.isnan(series)]; n=len(s)
    adf=adf_test(s); d=adf["recommended_d"]
    sd=s.copy().astype(float)
    for _ in range(d): sd=np.diff(sd)
    best_aic,best_p,best_q=np.inf,0,0; tbl=[]
    for p in range(4):
        for q in range(4):
            try:
                if p>0 and len(sd)>p+1:
                    X=np.column_stack([sd[p-i-1:len(sd)-i-1] for i in range(p)]+[np.ones(len(sd)-p)])
                    y=sd[p:]
                    if X.shape[0]<X.shape[1]+2: continue
                    coef,_,_,_=np.linalg.lstsq(X,y,rcond=None)
                    resid=y-X@coef; sig2=np.var(resid); k=p+q+1
                else: resid=sd-np.mean(sd); sig2=np.var(resid); k=1
                if sig2<=0: continue
                m=len(sd)-p
                if m<2: continue
                ll=-0.5*m*np.log(2*np.pi*sig2)-0.5*m; aic=2*k-2*ll
                tbl.append({"p":p,"d":d,"q":q,"AIC":round(aic,2)})
                if aic<best_aic: best_aic=aic; best_p,best_q=p,q
            except: continue
    warning=None
    if n<35 and best_p+best_q>2:
        warning=f"⚠️ 樣本數僅 {n} 筆，已限制 p+q ≤ 2 避免過擬合"
        cands=[r for r in tbl if r["p"]+r["q"]<=2]
        if cands: bst=min(cands,key=lambda x:x["AIC"]); best_p,best_q=bst["p"],bst["q"]
    exp=build_exp(best_p,d,best_q,adf,n)
    return {"p":best_p,"d":d,"q":best_q,"aic":round(best_aic,2),"adf":adf,"warning":warning,"explanation":exp,"aic_table":sorted(tbl,key=lambda x:x["AIC"])[:12],"sample_size":n}

def build_exp(p,d,q,adf,n):
    DE={0:"**d=0（不差分）**：原序列已平穩，無需差分，直接建模。",
        1:f"**d=1（一階差分）**：{adf['reason']}。差分後趨於平穩，此為年度排放最常見設定。",
        2:f"**d=2（二階差分）**：{adf['reason']}。需兩次差分才能平穩。"}
    PE={0:"**p=0（無 AR 項）**：與過去各期無顯著自相關，殘差近似白雜訊。",
        1:"**p=1（AR(1)）**：當期受前一年影響，PACF 在 lag=1 截尾，具一年慣性。",
        2:"**p=2（AR(2)）**：當期受前兩年影響，PACF 在 lag=2 截尾，具景氣循環慣性。",
        3:"**p=3（AR(3)）**：三期自回歸，序列具較長記憶性，小樣本需注意過擬合。"}
    QE={0:"**q=0（無 MA 項）**：殘差無移動平均結構，衝擊效果不跨期，各期獨立。",
        1:"**q=1（MA(1)）**：衝擊（如政策、危機）影響持續一年後消退，ACF 在 lag=1 截尾。",
        2:"**q=2（MA(2)）**：衝擊效果延續兩年，適合政策需 1-2 年反映的情境。",
        3:"**q=3（MA(3)）**：衝擊效果延續三年，適合有重大政策轉折點的序列。"}
    note=f"\n\n> 📌 **小樣本警告**：僅 {n} 個年度觀測值，p+q 已限制避免過擬合。" if n<35 else ""
    return {"p":PE.get(p,f"p={p}"),"d":DE.get(d,f"d={d}"),"q":QE.get(q,f"q={q}"),
            "summary":f"根據 AIC 最小化準則，最終選定 **ARIMA({p},{d},{q})**。{note}","adf_reason":adf["reason"]}

def arima_forecast(series, order, steps):
    p,d,q=order; s=series[~np.isnan(series)]; orig=s.copy().astype(float)
    sd=orig.copy()
    for _ in range(d): sd=np.diff(sd)
    n=len(sd); ms=np.mean(sd); ar=np.zeros(p)
    if p>0 and n>p:
        X=np.column_stack([sd[p-i-1:n-i-1] for i in range(p)]); y=sd[p:]
        try: ar,_,_,_=np.linalg.lstsq(X,y,rcond=None)
        except: ar=np.zeros(p)
    resid=(sd[p:]-np.array([np.dot(ar,sd[i:i+p][::-1]) for i in range(p,n)])) if p>0 and n>p else sd-ms
    sigma=np.std(resid); ext=list(sd); fd=[]
    for _ in range(steps):
        ap=np.dot(ar,ext[-p:][::-1]) if p>0 and len(ext)>=p else ms
        fd.append(ap); ext.append(ap)
    preds=[]
    for i,f in enumerate(fd):
        if d==0: preds.append(f)
        elif d==1: preds.append((orig[-1] if i==0 else preds[-1])+f)
        elif d==2:
            prev=orig[-1]+(orig[-1]-orig[-2]) if i==0 else preds[-1]+(preds[-1]-(orig[-1] if i==1 else preds[-2]))
            preds.append(prev+f)
    preds=np.array(preds); sq=np.sqrt(np.arange(1,steps+1))
    return {"forecast":preds.tolist(),"upper95":(preds+1.96*sigma*sq).tolist(),"lower95":(preds-1.96*sigma*sq).tolist(),"sigma":round(float(sigma),4)}

def read_file(f):
    raw=f.read(); fn=f.filename.lower()
    if fn.endswith('.csv'):
        for enc in ['utf-8-sig','utf-8','big5','cp950']:
            try: return pd.read_csv(io.BytesIO(raw),encoding=enc,dtype=str)
            except: continue
        raise ValueError("CSV 編碼解析失敗")
    elif fn.endswith(('.xlsx','.xls')): return pd.read_excel(io.BytesIO(raw),dtype=str)
    raise ValueError("僅支援 CSV 或 Excel")


# ─── Routes ────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return safe_json({"error":"未收到檔案"}, 400)
    try:
        df=read_file(request.files['file'])
        detected=detect_columns(df)
        dfc=clean_df(df)
        preview=dfc.head(5).where(pd.notnull(dfc),None).to_dict(orient='records')
        return safe_json({"columns":list(df.columns),"detected":detected,"preview":preview,"rows":len(df)})
    except Exception as e: return safe_json({"error":str(e)}, 400)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files: return safe_json({"error":"未收到檔案"}, 400)
    try: df=read_file(request.files['file'])
    except Exception as e: return safe_json({"error":str(e)}, 400)

    cm={"year":request.form.get("col_year",""),"energy":request.form.get("col_energy",""),
        "industry":request.form.get("col_industry",""),"agri":request.form.get("col_agri",""),
        "land":request.form.get("col_land",""),"total":request.form.get("col_total",""),
        "net":request.form.get("col_net",""),"co2":request.form.get("col_co2",""),
        "ch4":request.form.get("col_ch4",""),"n2o":request.form.get("col_n2o","")}
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
    if 'total' not in df.columns: return safe_json({"error":"找不到總排放量欄位，請確認欄位對應"}, 400)
    dfc=df.dropna(subset=['total']).copy()
    if len(dfc)<5: return safe_json({"error":f"有效數據不足（{len(dfc)} 筆），至少需要 5 筆"}, 400)

    ts=dfc['total'].values.astype(float); hy=dfc['year'].tolist(); ly=hy[-1]
    steps=2050-ly
    if steps<=0: return safe_json({"error":f"資料已涵蓋至 {ly} 年"}, 400)

    orr=select_arima_order(ts); p,d,q=orr['p'],orr['d'],orr['q']
    fc=arima_forecast(ts,(p,d,q),steps); fy=list(range(ly+1,2051))

    gas_results={}
    for col in ['co2','ch4','n2o']:
        if col in dfc.columns and not dfc[col].isna().all():
            s=dfc[col].dropna().values.astype(float)
            if len(s)>=5:
                try:
                    g=arima_forecast(s,(min(p,1),d,0),steps)
                    gas_results[col]={"history":[round(float(v),2) for v in s],"forecast":[round(float(v),2) for v in g['forecast']],"upper95":[round(float(v),2) for v in g['upper95']],"lower95":[round(float(v),2) for v in g['lower95']]}
                except: pass

    hist_tbl=[]
    for _,row in dfc.iterrows():
        r={"year":int(row['year'])}
        for c in ['energy','industry','agri','land','total','net','co2','ch4','n2o']:
            v=row.get(c,None); r[c]=round(float(v),2) if v is not None and pd.notna(v) else None
        hist_tbl.append(r)

    fc_tbl=[{"year":yr,"total":round(fc['forecast'][i],2),"upper95":round(fc['upper95'][i],2),"lower95":round(fc['lower95'][i],2),**{c:None for c in ['energy','industry','agri','land','net','co2','ch4','n2o']}} for i,yr in enumerate(fy)]

    return safe_json({"status":"ok","hist_years":hy,"hist_total":[round(float(v),2) for v in ts],
        "fc_years":fy,"fc_total":[round(float(v),2) for v in fc['forecast']],
        "fc_upper":[round(float(v),2) for v in fc['upper95']],"fc_lower":[round(float(v),2) for v in fc['lower95']],
        "sigma":fc['sigma'],"arima_order":{"p":p,"d":d,"q":q},"arima_explanation":orr['explanation'],
        "aic_table":orr['aic_table'],"adf_result":orr['adf'],"sample_size":orr['sample_size'],
        "warning":orr['warning'],"gas_results":gas_results,"history_table":hist_tbl,"forecast_table":fc_tbl})

@app.route('/api/health')
def health():
    return safe_json({"status":"running","message":"GHG Forecast API is online"})

if __name__=='__main__':
    import os
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=os.environ.get('RENDER') is None)