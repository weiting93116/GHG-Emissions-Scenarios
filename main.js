// ── Datalabels Plugin 全域設定 ──────────────────────────
Chart.register(ChartDataLabels);
Chart.defaults.plugins.datalabels.display = false;

// ── API Base URL ─────────────────────────────────────────
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:5000'
  : 'https://ghg-emissions-scenarios.onrender.com';

// Render 冷啟動喚醒
(async()=>{
  try{
    await fetch(`${API_BASE}/api/health`,{method:'GET',signal:AbortSignal.timeout(8000)});
  }catch(e){}
})();

let charts={}, uploadedFile=null, analysisData=null;

// ════════════════════════════════════════
// ── Custom Crosshair Tooltip ──────────
// ════════════════════════════════════════
const ttEl   = ()=>document.getElementById('chartTooltip');
const ttYear = ()=>document.getElementById('ttYear');
const ttRows = ()=>document.getElementById('ttRows');

function buildTooltipPlugin(chartId, seriesMeta){
  return {
    id:'crosshairTooltip_'+chartId,
    afterEvent(chart, args){
      const {event}=args;
      if(event.type==='mouseleave'){ttEl().style.display='none'; return;}
      const pts=chart.getElementsAtEventForMode(event.native,'index',{intersect:false},false);
      if(!pts.length){ttEl().style.display='none'; return;}
      const idx=pts[0].index;
      const label=chart.data.labels[idx];
      ttYear().textContent=label+' 年';
      let html='';
      seriesMeta.forEach(m=>{
        const ds=chart.data.datasets[m.datasetIdx];
        if(!ds)return;
        const v=ds.data[idx];
        if(v==null||isNaN(v))return;
        const col=m.color||ds.borderColor||'#aaa';
        html+=`<div class="tt-row">
          <span class="tt-label"><span class="tt-dot" style="background:${col}"></span>${m.label}</span>
          <span class="tt-val">${Number(v).toLocaleString('zh-TW',{maximumFractionDigits:1})}<span class="tt-unit">${m.unit||'kt'}</span></span>
        </div>`;
      });
      ttRows().innerHTML=html||'<div style="color:var(--muted);font-size:11px">無資料</div>';
      const tt=ttEl();
      tt.style.display='block';
      const mx=event.native.clientX, my=event.native.clientY;
      const tw=tt.offsetWidth||200, th=tt.offsetHeight||120;
      const vw=window.innerWidth, vh=window.innerHeight;
      tt.style.left=(mx+tw+16>vw ? mx-tw-10 : mx+14)+'px';
      tt.style.top =(my+th+10>vh ? my-th-10 : my+4)+'px';
    }
  };
}

document.addEventListener('mouseleave',()=>{ttEl().style.display='none'},{capture:true});

// ════════════════════════════════════════
// ── Lightbox ──────────────────────────
// ════════════════════════════════════════
function expandChart(chartId, title){
  const c=charts[chartId]; if(!c)return;
  const lb=document.getElementById('chartLightbox');
  const lbContent=document.getElementById('lbContent');
  lbContent.innerHTML=`<div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--teal);margin-bottom:12px;letter-spacing:.1em">${title.toUpperCase()}</div>
    <canvas id="lbCanvas" style="width:100%;max-height:70vh"></canvas>`;
  lb.classList.add('open');
  setTimeout(()=>{
    const orig=c.config;
    const cfg=JSON.parse(JSON.stringify(orig));
    if(cfg.options?.scales){
      ['x','y'].forEach(ax=>{if(cfg.options.scales[ax]?.ticks)cfg.options.scales[ax].ticks.font={size:13}});
    }
    cfg.options=cfg.options||{};
    cfg.options.animation={duration:400};
    cfg.options.aspectRatio=2.2;
    if(charts['_lb'])charts['_lb'].destroy();
    charts['_lb']=new Chart(document.getElementById('lbCanvas').getContext('2d'), cfg);
  },50);
}
function closeLightbox(){
  document.getElementById('chartLightbox').classList.remove('open');
  if(charts['_lb']){charts['_lb'].destroy();delete charts['_lb'];}
}
document.getElementById('chartLightbox').addEventListener('click',function(e){
  if(e.target===this)closeLightbox();
});

// ════════════════════════════════════════
// ── Chart PNG export ──────────────────
// ════════════════════════════════════════
function exportChartPng(chartId, filename){
  const c=charts[chartId]; if(!c)return;
  const skipTypes = new Set(['doughnut','bar']);
  const isSkip = c.config.type && skipTypes.has(c.config.type);
  if(!isSkip){
    c.data.datasets.forEach((ds,di)=>{
      const isMainLine = ds.borderWidth >= 1.5 && ds.borderDash === undefined && ds.label && !ds.label.includes('界') && !ds.label.includes('CI');
      if(!isMainLine) return;
      const data = ds.data || [];
      ds._exportLabels = ds.datalabels || {};
      ds.datalabels = {
        display: (ctx) => {
          const v = data[ctx.dataIndex];
          if(v == null) return false;
          const lbl = c.data.labels?.[ctx.dataIndex];
          const yr = parseInt(lbl);
          return (!isNaN(yr) && yr % 5 === 0) || ctx.dataIndex === data.length - 1;
        },
        formatter: (v) => {
          if(v == null) return '';
          const n = Math.abs(v);
          if(n >= 100000) return (v/1000).toFixed(0)+'M';
          if(n >= 1000)   return (v/1000).toFixed(1)+'k';
          return Number(v).toFixed(1);
        },
        color: '#e2e8f0',
        font: { size: 9, weight: 'bold', family: 'Arial' },
        anchor: 'end', align: 'top', offset: 2,
        backgroundColor: 'rgba(13,17,23,0.65)', borderRadius: 3,
        padding: { top:2, bottom:2, left:4, right:4 }, clip: true,
      };
    });
    c.update('none');
  }
  const origCanvas = c.canvas;
  const tmp = document.createElement('canvas');
  const scale = 2;
  tmp.width  = origCanvas.width  * scale;
  tmp.height = origCanvas.height * scale;
  const ctx = tmp.getContext('2d');
  ctx.scale(scale, scale);
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, origCanvas.width, origCanvas.height);
  ctx.drawImage(origCanvas, 0, 0);
  const a = document.createElement('a');
  a.href = tmp.toDataURL('image/png');
  a.download = (filename || chartId) + '.png';
  a.click();
  if(!isSkip){
    c.data.datasets.forEach(ds=>{
      if(ds._exportLabels !== undefined){
        ds.datalabels = ds._exportLabels;
        delete ds._exportLabels;
      }
    });
    c.update('none');
  }
}
async function doExportCharts(){
  const ids=['mainChart','cCO2','cCH4','cN2O','cNet','aicChart','cMC','diffSeriesChart','diffAcfChart','acfChart','residChart'];
  const names=['總排放量預測','CO2預測','CH4預測','N2O預測','淨排放','AIC比較','MC模擬','差分分析','差分ACF','ACF_PACF','殘差診斷'];
  for(let i=0;i<ids.length;i++){
    if(charts[ids[i]]){
      exportChartPng(ids[i],names[i]);
      await new Promise(r=>setTimeout(r,120));
    }
  }
}

// ════════════════════════════════════════
// ── Export Panel ──────────────────────
// ════════════════════════════════════════
const EXPORT_COLS=[
  {id:'year',    label:'年份',          group:'base',   always:true},
  {id:'total',   label:'總排放量 (kt)', group:'hist'},
  {id:'co2',     label:'CO₂ (kt)',      group:'hist'},
  {id:'ch4',     label:'CH₄ (kt)',      group:'hist'},
  {id:'n2o',     label:'N₂O (kt)',      group:'hist'},
  {id:'land',    label:'土地匯 (kt)',   group:'hist'},
  {id:'net',     label:'淨排放量 (kt)', group:'hist'},
  {id:'fc_mid',  label:'ARIMA 中位預測',group:'fc'},
  {id:'fc_up',   label:'預測上界 95%', group:'fc'},
  {id:'fc_lo',   label:'預測下界 95%', group:'fc'},
  {id:'bau',     label:'BAU 情境',      group:'fc'},
  {id:'policy',  label:'積極政策情境',  group:'fc'},
  {id:'ndc',     label:'NDC 情境',      group:'fc'},
];

function toggleExport(){
  const p=document.getElementById('exportPanel');
  const btn=document.getElementById('exportToggleBtn');
  if(p.style.display==='block'){
    p.style.display='none'; btn.textContent='📤 匯出資料';
  } else {
    p.style.display='block'; btn.textContent='✕ 關閉匯出';
    buildExportPanel();
  }
}

function buildExportPanel(){
  if(!analysisData)return;
  const d=analysisData;
  const allYears=[...d.hist_years,...d.fc_years];
  document.getElementById('expYearFrom').value=allYears[0];
  document.getElementById('expYearTo').value=allYears[allYears.length-1];
  const list=document.getElementById('cbList');
  list.innerHTML=EXPORT_COLS.map(c=>`
    <div class="cb-item">
      <input type="checkbox" id="cb_${c.id}" ${c.always||c.group==='hist'||c.group==='fc'?'checked':''} ${c.always?'disabled':''}>
      <label for="cb_${c.id}">
        <span style="font-size:9px;padding:1px 5px;border-radius:3px;margin-right:5px;background:${c.group==='hist'?'rgba(0,229,192,.12)':c.group==='fc'?'rgba(56,189,248,.12)':'rgba(100,100,100,.15)'}; color:${c.group==='hist'?'var(--teal)':c.group==='fc'?'var(--sky)':'var(--muted)'}">
          ${c.group==='hist'?'歷史':c.group==='fc'?'預測':'—'}
        </span>
        ${c.label}
      </label>
    </div>
  `).join('');
  updateExportPreview();
  document.getElementById('expYearFrom').oninput=updateExportPreview;
  document.getElementById('expYearTo').oninput=updateExportPreview;
  list.querySelectorAll('input[type=checkbox]').forEach(el=>el.addEventListener('change',updateExportPreview));
}

function selAll(v){
  document.querySelectorAll('#cbList input[type=checkbox]:not(:disabled)').forEach(el=>el.checked=v);
  updateExportPreview();
}
function selPreset(preset){
  EXPORT_COLS.forEach(c=>{
    const el=document.getElementById('cb_'+c.id);
    if(!el||el.disabled)return;
    el.checked = (preset==='hist' ? c.group==='hist' : c.group==='fc') || c.always;
  });
  updateExportPreview();
}

function getExportRows(){
  if(!analysisData)return[];
  const d=analysisData;
  const yFrom=parseInt(document.getElementById('expYearFrom').value)||0;
  const yTo  =parseInt(document.getElementById('expYearTo').value)||9999;
  const selCols=EXPORT_COLS.filter(c=>{const el=document.getElementById('cb_'+c.id);return el&&el.checked;});
  const sc=d.scenarios||{};
  const rows=[];
  d.history_table.forEach(r=>{
    if(r.year<yFrom||r.year>yTo)return;
    const row={year:r.year,total:r.total,co2:r.co2,ch4:r.ch4,n2o:r.n2o,land:r.land,net:r.net,
               fc_mid:null,fc_up:null,fc_lo:null,bau:null,policy:null,ndc:null,_type:'歷史'};
    rows.push(row);
  });
  d.forecast_table.forEach((r,i)=>{
    if(r.year<yFrom||r.year>yTo)return;
    const row={year:r.year,total:null,co2:null,ch4:null,n2o:null,land:null,net:null,
               fc_mid:r.total,fc_up:r.upper95,fc_lo:r.lower95,
               bau:sc.bau?.values[i]??null,policy:sc.policy?.values[i]??null,ndc:sc.ndc?.values[i]??null,
               _type:'預測'};
    rows.push(row);
  });
  return rows.map(r=>{
    const out={};
    selCols.forEach(c=>{out[c.label]=r[c.id]??'';});
    out['類型']=r._type;
    return out;
  });
}

function updateExportPreview(){
  const rows=getExportRows();
  const el=document.getElementById('expPreview');
  if(!rows.length){el.textContent='無符合條件的資料';return;}
  const cols=Object.keys(rows[0]).length;
  el.innerHTML=`<span style="color:var(--teal)">${rows.length}</span> 行 × <span style="color:var(--sky)">${cols}</span> 欄`;
}

function doExportCSV(){
  const rows=getExportRows();
  if(!rows.length){showErr('無可匯出資料');return;}
  const yFrom=document.getElementById('expYearFrom').value;
  const yTo=document.getElementById('expYearTo').value;
  const headers=Object.keys(rows[0]);
  const csvRows=[headers.join(',')];
  rows.forEach(r=>csvRows.push(headers.map(h=>{
    const v=r[h];
    if(v===null||v===undefined||v==='')return '';
    if(typeof v==='string')return `"${v}"`;
    return v;
  }).join(',')));
  const bom='\uFEFF';
  const blob=new Blob([bom+csvRows.join('\r\n')],{type:'text/csv;charset=utf-8'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download=`GHG預測_${yFrom}-${yTo}.csv`; a.click();
}

function doExportClip(){
  const rows=getExportRows();
  if(!rows.length){showErr('無可匯出資料');return;}
  const headers=Object.keys(rows[0]);
  const tsv=[headers.join('\t'),...rows.map(r=>headers.map(h=>r[h]??'').join('\t'))].join('\n');
  navigator.clipboard.writeText(tsv).then(()=>{
    const btn=event.currentTarget;
    const orig=btn.textContent; btn.textContent='✅ 已複製！';
    setTimeout(()=>btn.textContent=orig,2000);
  }).catch(()=>showErr('複製失敗'));
}

// ════════════════════════════════════════
// ── Sliders ───────────────────────────
// ════════════════════════════════════════
const sliders=[
  ['slGdp','vGdp',v=>(v>=0?'+':'')+parseFloat(v).toFixed(1)+'%'],
  ['slEla','vEla',v=>parseFloat(v).toFixed(2)],
  ['slPop','vPop',v=>(v>=0?'+':'')+parseFloat(v).toFixed(1)+'%'],
  ['slEff','vEff',v=>parseFloat(v).toFixed(1)+'%'],
  ['slRe','vRe',v=>v+'%'],
];
sliders.forEach(([id,vid,fmt])=>{
  const el=document.getElementById(id);
  el.addEventListener('input',()=>{document.getElementById(vid).textContent=fmt(el.value)});
  document.getElementById(vid).textContent=fmt(el.value);
});

function getParams(){
  return{
    gdp:parseFloat(document.getElementById('slGdp').value)/100,
    elasticity:parseFloat(document.getElementById('slEla').value),
    pop:parseFloat(document.getElementById('slPop').value)/100,
    eff:parseFloat(document.getElementById('slEff').value)/100,
    re:parseFloat(document.getElementById('slRe').value)/100,
  };
}

// ════════════════════════════════════════
// ── Upload ────────────────────────────
// ════════════════════════════════════════
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
    const r=await fetch(`${API_BASE}/api/upload`,{method:'POST',body:fd});
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

// ════════════════════════════════════════
// ── Analyze ───────────────────────────
// ════════════════════════════════════════
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
  const p=getParams();
  Object.entries(p).forEach(([k,v])=>fd.append('adef_'+k,v));
  try{
    const r=await fetch(`${API_BASE}/api/analyze`,{method:'POST',body:fd});
    const d=await r.json();
    if(d.error){showErr(d.error);return}
    analysisData=d;
    render(d);
    document.getElementById('scenarioPanel').style.display='block';
    document.getElementById('scenarioBtn').style.display='inline-flex';
  }catch(e){showErr('分析失敗：'+e.message)}
  finally{document.getElementById('loadingInd').style.display='none';document.getElementById('analyzeBtn').disabled=false}
});

async function updateScenarios(){
  if(!uploadedFile||!analysisData)return;
  const fd=new FormData(); fd.append('file',uploadedFile);
  [['col_year','mapYear'],['col_total','mapTotal']].forEach(([k,id])=>{const el=document.getElementById(id);if(el)fd.append(k,el.value||'')});
  const p=getParams(); Object.entries(p).forEach(([k,v])=>fd.append('adef_'+k,v));
  try{
    const r=await fetch(`${API_BASE}/api/scenarios`,{method:'POST',body:fd});
    const d=await r.json();
    if(d.error){showErr(d.error);return}
    updateMainChartScenarios(d.scenarios, analysisData.fc_years);
    const sc=d.scenarios;
    document.getElementById('s-bau').textContent=fmtN(sc.bau.values[sc.bau.values.length-1]);
    document.getElementById('s-ndc').textContent=fmtN(sc.ndc.values[sc.ndc.values.length-1]);
    analysisData.scenarios=sc;
    updateForecastTable(analysisData, d.scenarios);
    renderFcTable(analysisData, analysisData.scenarios);
    renderValidation(analysisData);
    if(d.mc_result) renderMC(d.mc_result, analysisData.fc_years, analysisData.hist_years.length);
    if(document.getElementById('exportPanel').style.display==='block') buildExportPanel();
  }catch(e){showErr('情境更新失敗：'+e.message)}
}

// ════════════════════════════════════════
// ── Render ────────────────────────────
// ════════════════════════════════════════
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
  document.getElementById('s-base').textContent=fmtN(base);
  document.getElementById('s-2050').textContent=fmtN(end);
  document.getElementById('s-bau').textContent=fmtN(sc.bau.values[fLen-1]);
  document.getElementById('s-ndc').textContent=fmtN(sc.ndc.values[fLen-1]);
  document.getElementById('s-order').textContent=`(${o.p},${o.d},${o.q})`;
  document.getElementById('s-sigma').textContent=fmtN(d.sigma);

  // ── Main chart ──
  const histFull=[...d.hist_total,...Array(fLen).fill(null)];
  const mainMeta=[
    {label:'歷史排放',color:'#00e5c0',datasetIdx:2},
    {label:'BAU',color:'#f59e0b',datasetIdx:3},
    {label:'積極政策',color:'#38bdf8',datasetIdx:4},
    {label:'NDC',color:'#00e5c0',datasetIdx:5},
  ];
  mk('mainChart',{type:'line',data:{labels:allY,datasets:[
    {data:[...Array(hLen).fill(null),...d.fc_upper],borderColor:'transparent',backgroundColor:'rgba(56,189,248,0.10)',fill:'+1',pointRadius:0,label:'95%上界'},
    {data:[...Array(hLen).fill(null),...d.fc_lower],borderColor:'transparent',fill:false,pointRadius:0,label:'95%下界'},
    {label:'歷史排放',data:histFull,borderColor:'#00e5c0',borderWidth:2.5,pointRadius:0,tension:0.3,fill:false},
    {label:'BAU',data:[...Array(hLen).fill(null),...sc.bau.values],borderColor:'#f59e0b',borderWidth:2,borderDash:[6,3],pointRadius:0,tension:0.3,fill:false},
    {label:'積極政策',data:[...Array(hLen).fill(null),...sc.policy.values],borderColor:'#38bdf8',borderWidth:2,borderDash:[6,3],pointRadius:0,tension:0.3,fill:false},
    {label:'NDC',data:[...Array(hLen).fill(null),...sc.ndc.values],borderColor:'#00e5c0',borderWidth:2.5,borderDash:[4,2],pointRadius:0,tension:0.3,fill:false},
  ]},options:{
    ...lopts('kt CO₂e',1.9),
    plugins:{...lopts('kt CO₂e',1.9).plugins, legend:{display:false}}
  },plugins:[buildTooltipPlugin('mainChart',mainMeta)]});

  // ── Pie ──
  const GAS_KEYS =['co2','ch4','n2o','hfc','pfc','sf6','nf3'];
  const GAS_LABELS=['CO₂','CH₄','N₂O','HFCs','PFCs','SF₆','NF₃'];
  const GAS_COLORS=['rgba(96,165,250,.85)','rgba(52,211,153,.85)','rgba(244,114,182,.85)',
                    'rgba(251,191,36,.85)','rgba(167,139,250,.85)','rgba(251,113,133,.85)','rgba(20,184,166,.85)'];
  const pfl={l:[],dt:[],c:[]};
  const lastHist = d.history_table[d.history_table.length-1] || {};
  GAS_KEYS.forEach((k,i)=>{
    let v = (lastHist[k] != null && lastHist[k] > 0) ? lastHist[k] : null;
    if(v===null){
      const g=d.gas_results?.[k];
      v = g&&g.forecast&&g.forecast.length>0 ? Math.max(0,g.forecast[0]) : null;
    }
    if(v!==null&&v>0){pfl.l.push(GAS_LABELS[i]);pfl.dt.push(v);pfl.c.push(GAS_COLORS[i]);}
  });
  const pieYear = lastHist.year || d.hist_years[d.hist_years.length-1];
  const pieTitle = document.querySelector('#pieChart')?.closest('.card')?.querySelector('.card-title');
  if(pieTitle) pieTitle.innerHTML = `<span class="card-dot" style="background:var(--amber)"></span>${pieYear} 年氣體組成比例`;
  if(pfl.dt.length===0){pfl.l=['總排放量'];pfl.dt=[end];pfl.c=['rgba(0,229,192,.8)'];}
  mk('pieChart',{type:'doughnut',data:{labels:pfl.l,datasets:[{data:pfl.dt,backgroundColor:pfl.c,borderColor:'#161b22',borderWidth:2}]},
    options:{responsive:true,maintainAspectRatio:true,plugins:{
      legend:{display:true,position:'bottom',labels:{color:'#7a9ab0',font:{size:11},padding:10,boxWidth:12}},
      tooltip:{backgroundColor:'#1c2733',titleColor:'#d4e8f5',bodyColor:'#7a9ab0',
        callbacks:{label:(ctx)=>`${ctx.label}: ${Number(ctx.parsed).toLocaleString('zh-TW',{maximumFractionDigits:1})} kt (${(ctx.parsed/pfl.dt.reduce((a,b)=>a+b,0)*100).toFixed(1)}%)`}}
    }}});

  // ── Sector charts ──
  const sr = d.sector_results || {};
  const sectorKeys = Object.keys(sr);
  if(sectorKeys.length > 0){
    document.getElementById('sectorCard').style.display = '';
    document.getElementById('sectorIndGrid').style.display = '';
    const stackDatasets = sectorKeys.map(k => {
      const s = sr[k];
      const histPad  = [...s.history, ...Array(fLen).fill(null)];
      const fcPad    = [...Array(hLen).fill(null), ...s.forecast];
      return [
        {label: s.label+'（歷史）', data: histPad, borderColor: s.color, backgroundColor: s.color+'33', borderWidth:2, pointRadius:0, tension:0.3, fill:true},
        {label: s.label+'（預測）', data: fcPad,   borderColor: s.color, backgroundColor: s.color+'22', borderWidth:1.5, borderDash:[5,3], pointRadius:0, tension:0.3, fill:true},
      ];
    }).flat();
    mk('cSectorStack', {type:'line', data:{labels:allY, datasets:stackDatasets},
      options:{...lopts('kt CO₂e', 2.0), plugins:{...lopts('kt CO₂e',2.0).plugins,
        legend:{display:true, labels:{color:'#4a6070', font:{size:10}, boxWidth:12, filter: item => item.text.includes('歷史')}},
      }}});
    const SECTOR_CANVAS = {energy:'cEnergy', industry:'cIndustry', agri:'cAgri', waste:'cWaste'};
    sectorKeys.forEach(k => {
      const cid = SECTOR_CANVAS[k]; if(!cid) return;
      const s = sr[k];
      mk(cid, {type:'line', data:{labels:allY, datasets:[
        {data:[...Array(hLen).fill(null),...s.upper95], borderColor:'transparent', backgroundColor:s.color+'18', fill:'+1', pointRadius:0, label:'上界'},
        {data:[...Array(hLen).fill(null),...s.lower95], borderColor:'transparent', fill:false, pointRadius:0, label:'下界'},
        {label:s.label+'（歷史）', data:[...s.history,...Array(fLen).fill(null)], borderColor:s.color, borderWidth:2, pointRadius:0, tension:0.3, fill:false},
        {label:s.label+'（預測）', data:[...Array(hLen).fill(null),...s.forecast], borderColor:s.color, borderWidth:1.5, borderDash:[5,3], pointRadius:0, tension:0.3, fill:false},
      ]}, options:{...lopts('kt CO₂e',1.6), plugins:{...lopts('kt CO₂e',1.6).plugins, legend:{display:false}}}});
    });
  }

  // ── Gas charts ──
  [['cCO2','co2','#60a5fa','CO₂'],['cCH4','ch4','#34d399','CH₄'],['cN2O','n2o','#f472b6','N₂O']].forEach(([id,key,col,gasLabel])=>{
    const g=d.gas_results?.[key];
    if(!g||g.skipped||g.error){mk(id,{type:'line',data:{labels:[],datasets:[]},options:lopts('kt',1.6)});return}
    mk(id,{type:'line',data:{labels:allY,datasets:[
      {data:[...Array(hLen).fill(null),...g.upper95],borderColor:'transparent',backgroundColor:col+'22',fill:'+1',pointRadius:0,label:'上界'},
      {data:[...Array(hLen).fill(null),...g.lower95],borderColor:'transparent',fill:false,pointRadius:0,label:'下界'},
      {data:[...g.history,...Array(fLen).fill(null)],borderColor:col,borderWidth:2,pointRadius:0,tension:0.3,fill:false,label:`${gasLabel} 歷史`},
      {data:[...Array(hLen).fill(null),...g.forecast],borderColor:col,borderWidth:1.5,borderDash:[5,3],pointRadius:0,tension:0.3,fill:false,label:`${gasLabel} 預測`},
    ]},options:{...lopts('kt CO₂e',1.6),plugins:{...lopts('kt CO₂e',1.6).plugins, legend:{display:false}}}});
  });

  // ── Net chart ──
  const nh=d.history_table.map(r=>r.net??null);
  const fcNet = (d.fc_net||[]);
  if(nh.some(v=>v!==null)||fcNet.some(v=>v!==null)){
    const histLand = d.history_table.map(r=>r.land??null);
    const datasets = [
      {label:'淨排放（歷史）', data:[...nh,...Array(fLen).fill(null)], borderColor:'#fb7185',borderWidth:2,pointRadius:0,tension:0.3,fill:false},
      {label:'淨排放（預測）', data:[...Array(hLen).fill(null),...fcNet], borderColor:'#fb7185',borderWidth:1.5,borderDash:[5,3],pointRadius:0,tension:0.3,fill:false},
      {label:'總排放（歷史）', data:[...d.hist_total,...Array(fLen).fill(null)], borderColor:'rgba(0,229,192,.4)',borderWidth:1.5,borderDash:[4,3],pointRadius:0,tension:0.3,fill:false},
      {label:'總排放（預測）', data:[...Array(hLen).fill(null),...d.fc_total], borderColor:'rgba(0,229,192,.3)',borderWidth:1.5,borderDash:[3,4],pointRadius:0,tension:0.3,fill:false},
    ];
    if(histLand.some(v=>v!==null)){
      datasets.push({label:'土地匯', data:[...histLand,...(d.fc_land_series||[])], borderColor:'rgba(132,204,22,.55)',borderWidth:1.5,borderDash:[2,3],pointRadius:0,tension:0.2,fill:false});
    }
    mk('cNet',{type:'line',data:{labels:allY,datasets},options:{...lopts('kt CO₂e',1.8),plugins:{...lopts('kt CO₂e',1.8).plugins,legend:{display:true,labels:{color:'#4a6070',font:{size:10},boxWidth:12,filter:(item)=>item.text.includes('歷史')||item.text.includes('預測')}}}}});
  }

  // ── AIC chart ──
  const aicD=d.aic_table||[];
  mk('aicChart',{type:'bar',data:{labels:aicD.map(r=>`(${r.p},${r.d},${r.q})`),datasets:[{
    data:aicD.map(r=>r.AIC),
    backgroundColor:aicD.map(r=>r.p===o.p&&r.q===o.q?'rgba(0,229,192,.75)':'rgba(56,189,248,.28)'),
    borderRadius:3,label:'BIC'}]},
    options:{responsive:true,maintainAspectRatio:true,aspectRatio:1.8,
      plugins:{legend:{display:false},tooltip:{backgroundColor:'#1c2733',bodyColor:'#94a3b8',
        callbacks:{label:(ctx)=>`BIC: ${ctx.parsed.y.toLocaleString('zh-TW',{maximumFractionDigits:2})}`}}},
      scales:{x:{ticks:{color:'#4a6070',font:{size:9},maxRotation:45},grid:{color:'rgba(255,255,255,.03)'}},
              y:{ticks:{color:'#4a6070',font:{size:9}},grid:{color:'rgba(255,255,255,.03)'}}}
    }});

  // ARIMA 說明面板
  ['p','d','q'].forEach(k=>{document.getElementById('exp-'+k).textContent=o[k];document.getElementById('exp-'+k+'2').textContent=o[k]});
  document.getElementById('exp-aic').textContent=d.aic_table?.[0]?.AIC??'—';
  // ADF+KPSS 雙重檢定結果
  const dtests=d.d_tests||{}; const adf_o=dtests.adf_orig||{}; const kpss_o=dtests.kpss_orig||{};
  const adf_d1=dtests.adf_diff1||{}; const kpss_d1=dtests.kpss_diff1||{};
  document.getElementById('adfText').innerHTML=
    `${d.d_reason||exp.adf_reason||'—'}<br>` +
    `<span style="font-size:10px;color:#475569;font-family:'JetBrains Mono',monospace">` +
    `原序列 ADF p=${adf_o.p??'—'}，KPSS p=${kpss_o.p??'—'}` +
    (adf_d1.p!=null ? `　一階差分 ADF p=${adf_d1.p}，KPSS p=${kpss_d1.p??'—'}` : '') +
    `</span>`;
  const eng=d.engine||'fallback';
  const badge=document.getElementById('engine-badge');
  if(eng==='pmdarima'){
    badge.textContent='🔬 pmdarima · BIC + ADF/KPSS';
    badge.style.color='var(--teal)';badge.style.borderColor='var(--teal2)';badge.style.background='rgba(0,229,192,.07)';
  } else {
    badge.textContent='⚙️ 手工BIC窮舉 (fallback)';
    badge.style.color='var(--amber)';badge.style.borderColor='#a06f08';badge.style.background='rgba(245,158,11,.07)';
  }
  // 選用模型 badge
  const bmbadge=document.getElementById('best-model-badge');
  if(bmbadge && d.model_info){
    const bm=d.model_info.best_model||'';
    const bmLabel={'log_arima':`log-ARIMA(${o.p},${o.d},${o.q})`,'ets':d.model_info.ets_spec||'ETS','holt':d.model_info.holt_spec||'Holt(damped)'}[bm]||bm;
    const bmColor={'log_arima':'#38bdf8','ets':'#a78bfa','holt':'#4ade80'}[bm]||'#e2e8f0';
    bmbadge.textContent=`✓ 選用：${bmLabel}`;
    bmbadge.style.color=bmColor; bmbadge.style.borderColor=bmColor.replace('ff','33');
    bmbadge.style.background=`${bmColor}11`; bmbadge.style.display='';
  }
  const hl=(s,c)=>(s||'').replace(/\*\*(.*?)\*\*/g,`<strong style="color:${c}">$1</strong>`);
  document.getElementById('exp-p-text').innerHTML=hl(exp.p,'var(--sky)');
  document.getElementById('exp-d-text').innerHTML=hl(exp.d,'var(--teal)');
  document.getElementById('exp-q-text').innerHTML=hl(exp.q,'var(--violet)');
  document.getElementById('summaryText').innerHTML=hl(exp.summary,'var(--teal)');
  if(d.warning){const wb=document.getElementById('warnBox');wb.textContent=d.warning;wb.style.display='block'}
  document.getElementById('aicTbody').innerHTML=(d.aic_table||[]).map((r,i)=>`<tr class="${r.p===o.p&&r.q===o.q?'best':''}"><td>${i+1}</td><td>${r.p}</td><td>${r.d}</td><td>${r.q}</td><td>${r.AIC}</td></tr>`).join('');
  // Holt 參數卡（只有選用 Holt 時才顯示）
  const holtCard=document.getElementById('holtParamCard');
  const holtText=document.getElementById('holtParamText');
  if(holtCard && d.model_info?.best_model==='holt' && d.model_info?.holt_params){
    const hp=d.model_info.holt_params;
    holtCard.style.display='';
    if(holtText) holtText.innerHTML=
      `阻尼係數 φ = <strong style="color:#4ade80">${hp.phi??'—'}</strong><br>` +
      `<span style="font-size:10px;color:#475569">φ &lt; 1 使趨勢隨時間衰減，防止長期外推線性發散。α=${hp.alpha??'—'}，β=${hp.beta??'—'}</span><br>` +
      `<span style="font-size:10px;color:#475569">Gardner &amp; McKenzie (1985) Mgmt Sci</span>`;
  } else if(holtCard){
    holtCard.style.display='none';
  }

  // ── 差分分析圖 ──
  const diag = d.diagnostics || {};
  const nlags = diag.nlags || 15;
  const lagLabels = Array.from({length:nlags+1},(_,i)=>i);
  if(diag.orig_series){
    const nOrig = diag.orig_series.length;
    const maxLen = Math.max(nOrig, (diag.diff1_series||[]).length, (diag.diff2_series||[]).length);
    const diffLabels = Array.from({length:maxLen},(_,i)=>d.hist_years[0]+i);
    const padNull = (arr, offset, len) => Array(offset).fill(null).concat(arr).concat(Array(Math.max(0,len-offset-arr.length)).fill(null));
    mk('diffSeriesChart',{type:'line',data:{
      labels: diffLabels,
      datasets:[
        {label:'原序列',   data:padNull(diag.orig_series,0,maxLen), borderColor:'#f87171', borderWidth:2, pointRadius:0,tension:0.3,fill:false},
        {label:'一階差分', data:padNull(diag.diff1_series||[],1,maxLen), borderColor:'#fbbf24', borderWidth:2, pointRadius:0,tension:0.3,fill:false},
        {label:'二階差分', data:padNull(diag.diff2_series||[],2,maxLen), borderColor:'#818cf8', borderWidth:2, pointRadius:0,tension:0.3,fill:false},
      ]
    },options:{...BO,aspectRatio:2.4,
      plugins:{...BO.plugins,legend:{display:true,labels:{color:'#c7d2fe',font:{size:10},boxWidth:12}}},
      scales:{x:{...SC,ticks:{maxTicksLimit:10,color:'#94a3b8'}},y:{...SC,title:{display:true,text:'kt CO₂e',color:'#94a3b8',font:{size:9}},ticks:{color:'#94a3b8'}}}}});
    const acfSets=[];
    if(diag.orig_acf)  acfSets.push({label:'原序列 ACF', data:diag.orig_acf,  backgroundColor:'rgba(248,113,113,.65)', borderColor:'transparent'});
    if(diag.diff1_acf&&diag.diff1_acf.length) acfSets.push({label:'一階差分 ACF',data:diag.diff1_acf,backgroundColor:'rgba(251,191,36,.65)', borderColor:'transparent'});
    if(diag.diff2_acf&&diag.diff2_acf.length) acfSets.push({label:'二階差分 ACF',data:diag.diff2_acf,backgroundColor:'rgba(129,140,248,.65)', borderColor:'transparent'});
    mk('diffAcfChart',{type:'bar',data:{labels:lagLabels,datasets:acfSets},
      options:{...BO,aspectRatio:1.6,plugins:{...BO.plugins,legend:{display:true,labels:{color:'#c7d2fe',font:{size:9},boxWidth:10}}},scales:{x:{...SC,ticks:{color:'#94a3b8'}},y:{...SC,min:-1,max:1,ticks:{color:'#94a3b8'}}}}});
  }
  if(diag.orig_acf && diag.orig_pacf){
    mk('acfChart',{type:'bar',data:{labels:lagLabels,datasets:[
      {label:'ACF', data:diag.orig_acf,  backgroundColor:diag.orig_acf.map((_,i)=>i===0?'transparent':'rgba(56,189,248,.65)'), borderColor:'transparent'},
      {label:'PACF',data:diag.orig_pacf, backgroundColor:diag.orig_pacf.map((_,i)=>i===0?'transparent':'rgba(167,139,250,.65)'), borderColor:'transparent'},
      {label:'+95%CI',data:diag.orig_conf,type:'line',borderColor:'rgba(255,255,255,.3)',borderDash:[4,3],pointRadius:0,fill:false},
      {label:'-95%CI',data:(diag.orig_conf||[]).map(v=>-v),type:'line',borderColor:'rgba(255,255,255,.3)',borderDash:[4,3],pointRadius:0,fill:'-1',backgroundColor:'rgba(255,255,255,.04)'},
    ]},options:{...BO,aspectRatio:1.8,plugins:{...BO.plugins,legend:{display:true,labels:{color:'#4a6070',font:{size:10},boxWidth:12}}},scales:{x:{...SC},y:{...SC,min:-1,max:1,title:{display:true,text:'相關係數',color:'#4a6070',font:{size:9}}}}}});
  }
  if(diag.residuals){
    const nR=diag.residuals.length;
    mk('residChart',{type:'bar',data:{
      labels:Array.from({length:nR},(_,i)=>i+1),
      datasets:[
        {label:'標準化殘差',data:diag.residuals,backgroundColor:diag.residuals.map(v=>Math.abs(v)>2?'rgba(251,113,133,.8)':'rgba(56,189,248,.45)'),borderColor:'transparent'},
        {label:'+2σ',data:Array(nR).fill(2), type:'line',borderColor:'rgba(251,113,133,.5)',borderDash:[5,3],pointRadius:0,fill:false},
        {label:'-2σ',data:Array(nR).fill(-2),type:'line',borderColor:'rgba(251,113,133,.5)',borderDash:[5,3],pointRadius:0,fill:'-1',backgroundColor:'rgba(251,113,133,.04)'},
      ]
    },options:{...BO,aspectRatio:1.8,
      plugins:{...BO.plugins,legend:{display:true,labels:{color:'#4a6070',font:{size:10},boxWidth:12}}},
      scales:{x:{...SC,ticks:{maxTicksLimit:14}},y:{...SC,title:{display:true,text:'σ',color:'#4a6070',font:{size:9}}}}}});
  }

  updateForecastTable(d, sc);
  renderValidation(d);
  renderFcTable(d, sc);

  // ★ 修正：這裡補上呼叫四個之前遺漏的 render 函式
  renderOosDm(d.dm_result);
  renderMC(d.mc_result, d.fc_years, d.hist_years.length);
  renderZA(d.za_result, d.bau_cagr, d.sigma_data);
  renderMethodsText(d.methods_text);

  document.getElementById('results').scrollIntoView({behavior:'smooth',block:'start'});
}


// ════════════════════════════════════════════════════════
// ── OOS驗證 + DM檢定 ────────────────────────────────────
// ════════════════════════════════════════════════════════
function renderOosDm(dm) {
  const el = document.getElementById('oosBody'); if(!el) return;
  if(!dm){ el.innerHTML='<span style="color:var(--muted)">未執行</span>'; return; }
  if(dm.skipped){ el.innerHTML=`<div style="color:#fbbf24;font-size:11px;padding:8px;background:#1a1400;border-radius:4px">⚠️ ${dm.reason}</div>`; return; }
  if(dm.error){ el.innerHTML=`<div style="color:#f87171;font-size:11px">❌ ${dm.error}</div>`; return; }

  // v4：dm.oos 含三模型，dm.arima_vs_ets / arima_vs_holt / ets_vs_holt
  const oos = dm.oos||{};
  const la=oos.log_arima||{}; const et=oos.ets||{}; const ho=oos.holt||{};
  const ave=dm.arima_vs_ets||{}; const avh=dm.arima_vs_holt||{}; const evh=dm.ets_vs_holt||{};
  const oos_rmse=analysisData?.model_info?.oos_rmse||{};
  const sel_meth=analysisData?.model_info?.selection_method||'OOS RMSE';
  const best_m=analysisData?.model_info?.best_model||'';
  const bestColor={'log_arima':'#38bdf8','ets':'#a78bfa','holt':'#4ade80'}[best_m]||'#e2e8f0';
  const bestLabel={'log_arima':`log-ARIMA(${analysisData?.model_info?.arima_order?.p},${analysisData?.model_info?.arima_order?.d},${analysisData?.model_info?.arima_order?.q})`,'ets':analysisData?.model_info?.ets_spec||'ETS','holt':analysisData?.model_info?.holt_spec||'Holt(damped=True)'}[best_m]||best_m;
  const dmRow=(obj,label)=>{
    if(!obj||obj.error) return `<div style="font-size:10px;color:#475569">${label}：資料不足</div>`;
    const pc=obj.hln_pval<0.05?'#4ade80':'#fbbf24';
    return `<div style="padding:3px 0;font-size:10.5px"><strong style="color:#94a3b8">${label}</strong>：HLN p=<strong style="color:${pc}">${obj.hln_pval??'—'}</strong>，${obj.conclusion||'—'}</div>`;
  };
  el.innerHTML=`
    <div style="margin-bottom:6px;font-size:10px;color:#64748b">選模標準：${sel_meth}</div>
    <table style="width:100%;border-collapse:collapse;font-size:11px;margin-bottom:8px">
      <tr style="border-bottom:1px solid #1e293b">
        <th style="text-align:left;color:#64748b;padding:3px 6px;font-weight:400">OOS 指標</th>
        <th style="text-align:right;color:#38bdf8;padding:3px 6px">log-ARIMA</th>
        <th style="text-align:right;color:#a78bfa;padding:3px 6px">ETS</th>
        <th style="text-align:right;color:#4ade80;padding:3px 6px">Holt</th>
      </tr>
      <tr><td style="padding:3px 6px;color:#94a3b8">MAPE (%)</td>
        <td style="text-align:right;${best_m==='log_arima'?'color:#38bdf8;font-weight:700':''}">${la.mape??'—'}</td>
        <td style="text-align:right;${best_m==='ets'?'color:#a78bfa;font-weight:700':''}">${et.mape??'—'}</td>
        <td style="text-align:right;${best_m==='holt'?'color:#4ade80;font-weight:700':''}">${ho.mape??'—'}</td></tr>
      <tr style="background:#0a0e17"><td style="padding:3px 6px;color:#94a3b8">RMSE (kt)</td>
        <td style="text-align:right">${la.rmse!=null?Number(la.rmse).toLocaleString():'—'}</td>
        <td style="text-align:right">${et.rmse!=null?Number(et.rmse).toLocaleString():'—'}</td>
        <td style="text-align:right">${ho.rmse!=null?Number(ho.rmse).toLocaleString():'—'}</td></tr>
      <tr><td style="padding:3px 6px;color:#94a3b8">OOS RMSE（選模）</td>
        <td style="text-align:right;${best_m==='log_arima'?'color:#38bdf8;font-weight:700':''}">${oos_rmse.log_arima??'—'}</td>
        <td style="text-align:right;${best_m==='ets'?'color:#a78bfa;font-weight:700':''}">${oos_rmse.ets??'—'}</td>
        <td style="text-align:right;${best_m==='holt'?'color:#4ade80;font-weight:700':''}">${oos_rmse.holt??'—'}</td></tr>
    </table>
    <div style="background:#0a0e17;border-radius:4px;padding:7px 8px;font-size:10.5px;margin-bottom:6px">
      <div style="color:#e2e8f0;margin-bottom:4px;font-weight:600">✓ 選用：<span style="color:${bestColor}">${bestLabel}</span>（OOS RMSE 最小）</div>
      ${dmRow(ave,'ARIMA vs ETS')}
      ${dmRow(avh,'ARIMA vs Holt')}
      ${dmRow(evh,'ETS vs Holt')}
    </div>
    <div style="font-size:10px;color:#475569">${oos.holdout_n||'?'} 期 rolling origin · DM檢定：Diebold &amp; Mariano (1995) · Harvey et al. (1997)</div>`;
}


// ════════════════════════════════════════════════════════
// ── 蒙地卡羅情境模擬圖（v4：歷史 bootstrap）
// ════════════════════════════════════════════════════════
function renderMC(mc, fy, hLen) {
  if(!mc || !fy) return;
  if(mc.error){
    const el=document.getElementById('mcCard');
    if(el) el.querySelector('canvas')?.insertAdjacentHTML('beforebegin',
      `<div style="color:#f87171;font-size:11px;margin-bottom:8px">MC 失敗：${mc.error}</div>`);
    return;
  }
  // v4：mc 是單一 bootstrap 結果（p5/p25/p50/p75/p95）
  // 不再有 mc.bau / mc.policy / mc.ndc 三情境分開
  const fLen=fy.length;
  const allLabels=[...(analysisData?.hist_years||[]).map(String), ...fy.map(String)];
  const pad=v=>[...Array(hLen).fill(null),...v];
  const datasets=[];

  // Bootstrap 不確定帶（p5~p95）
  if(mc.p95 && mc.p5){
    datasets.push(
      {label:'Bootstrap p5–p95', data:pad(mc.p95), borderColor:'transparent',
       backgroundColor:'rgba(56,189,248,0.12)', fill:'+1', pointRadius:0},
      {label:'', data:pad(mc.p5), borderColor:'transparent', fill:false, pointRadius:0},
    );
  }
  // Bootstrap p25~p75
  if(mc.p75 && mc.p25){
    datasets.push(
      {label:'Bootstrap p25–p75', data:pad(mc.p75), borderColor:'transparent',
       backgroundColor:'rgba(56,189,248,0.18)', fill:'+1', pointRadius:0},
      {label:'', data:pad(mc.p25), borderColor:'transparent', fill:false, pointRadius:0},
    );
  }
  // Bootstrap 中位數
  if(mc.p50){
    datasets.push(
      {label:`Bootstrap 中位數（n=${mc.n_sim||500}次）`,
       data:pad(mc.p50), borderColor:'#38bdf8',
       borderWidth:1.5, borderDash:[4,3], pointRadius:0, tension:0.3, fill:false}
    );
  }
  // 歷史排放
  if(analysisData?.hist_total){
    datasets.push({
      label:'歷史排放', data:[...analysisData.hist_total,...Array(fLen).fill(null)],
      borderColor:'rgba(255,255,255,.6)', borderWidth:2, pointRadius:0, tension:0.3, fill:false
    });
  }
  // ARIMA 預測主線
  if(analysisData?.fc_total){
    datasets.push({
      label:`${analysisData?.model_info?.best_model||'ARIMA'} 預測`,
      data:[...Array(hLen).fill(null),...analysisData.fc_total],
      borderColor:'#00e5c0', borderWidth:1.5, borderDash:[6,3], pointRadius:0, tension:0.3, fill:false
    });
  }

  mk('cMC',{type:'line',data:{labels:allLabels,datasets},
    options:{...lopts('kt CO₂e',1.8),
      plugins:{...lopts('kt CO₂e',1.8).plugins,
        legend:{display:true,labels:{color:'#4a6070',font:{size:9},boxWidth:10,
          filter:item=>item.text&&item.text.length>0}}
      }
    }
  });

  // 在圖表下方顯示 bootstrap 說明
  const card=document.getElementById('mcCard');
  if(card){
    const existing=card.querySelector('.mc-caption');
    if(existing) existing.remove();
    const cap=document.createElement('div');
    cap.className='mc-caption';
    cap.style.cssText='margin-top:6px;font-size:10px;color:#475569;line-height:1.6';
    cap.innerHTML=`歷史年變動率 bootstrap（${mc.n_hist_rates||'?'} 個觀測值，有放回，${mc.n_sim||500} 次）<br>無分布假設 · Efron &amp; Tibshirani (1993)`;
    card.appendChild(cap);
  }
}


// ════════════════════════════════════════════════════════
// ── Zivot-Andrews 斷點檢定 ──────────────────────────────
// ════════════════════════════════════════════════════════
function renderZA(za, bau_cagr, sigma_data) {
  const el = document.getElementById('zaBody'); if(!el) return;
  if(!za){
    el.innerHTML='<span style="color:var(--muted)">未執行</span>'; return;
  }
  if(za.skipped){
    el.innerHTML=`<div style="color:#fbbf24;font-size:11px;padding:8px;background:#1a1400;border-radius:4px">⚠️ ${za.reason}</div>`; return;
  }
  if(za.error){
    el.innerHTML=`<div style="color:#f87171;font-size:11px">❌ ${za.error}</div>`; return;
  }
  const pCol = za.za_pval < 0.05 ? '#4ade80' : '#fbbf24';
  el.innerHTML = `
    <table style="width:100%;border-collapse:collapse;font-size:11px">
      <tr><td style="padding:3px 6px;color:#94a3b8">ZA 統計量</td>
          <td style="text-align:right;font-weight:600;color:#e2e8f0">${za.za_stat}</td></tr>
      <tr style="background:#0a0e17">
          <td style="padding:3px 6px;color:#94a3b8">p 值</td>
          <td style="text-align:right;color:${pCol};font-weight:600">${za.za_pval}</td></tr>
      <tr><td style="padding:3px 6px;color:#94a3b8">斷點年份</td>
          <td style="text-align:right;color:#f472b6;font-weight:700">${za.bp_year??'—'}</td></tr>
      <tr style="background:#0a0e17">
          <td style="padding:3px 6px;color:#94a3b8">臨界值 5%</td>
          <td style="text-align:right">${za.cv_5pct??'—'}</td></tr>
    </table>
    <div style="margin-top:7px;padding:7px 8px;background:#0a0e17;border-left:3px solid ${pCol};border-radius:0 4px 4px 0;font-size:11px">
      ${za.conclusion}<br>
      <span style="color:#475569">${za.arima_note}</span>
    </div>
    <div style="margin-top:6px;font-size:10px;color:#475569">
      BAU CAGR：<strong style="color:#f59e0b">${bau_cagr!=null?(bau_cagr>=0?'+':'')+Number(bau_cagr).toFixed(3)+'%':'N/A'}</strong>　
      MC σ：<strong style="color:#818cf8">${sigma_data!=null?Number(sigma_data).toFixed(3)+'%':'N/A'}</strong><br>
      Zivot &amp; Andrews (1992) JBES 10(3)
    </div>`;
}

// ════════════════════════════════════════════════════════
// ── 方法論段落 ───────────────────────────────────────────
// ════════════════════════════════════════════════════════
let _methodsData = {zh:'', en:''}; let _methodsLang = 'zh';

function renderMethodsText(methods) {
  if(!methods || methods.error) return;
  _methodsData = methods;
  const el = document.getElementById('methodsText');
  const card = document.getElementById('methodsCard');
  if(el) el.textContent = methods.zh || '';
  if(card) card.style.display='';
}

function toggleMethodsLang() {
  _methodsLang = _methodsLang === 'zh' ? 'en' : 'zh';
  const el = document.getElementById('methodsText');
  if(el) el.textContent = _methodsData[_methodsLang] || '';
  const btn = document.getElementById('methodsLangBtn');
  if(btn) btn.textContent = _methodsLang === 'zh' ? '切換英文' : '切換中文';
}

function copyMethods() {
  const txt = document.getElementById('methodsText')?.textContent;
  if(!txt) return;
  navigator.clipboard.writeText(txt).then(()=>{
    const btn = document.querySelector('[onclick="copyMethods()"]');
    if(!btn) return;
    const orig = btn.textContent; btn.textContent='✓ 已複製';
    setTimeout(()=>btn.textContent=orig, 1800);
  });
}

// ════════════════════════════════════════════════════════
// ── 模型驗證 + 情境引用卡片 ─────────────────────────────
// ════════════════════════════════════════════════════════
function renderValidation(d) {
  const mi=d.model_info||{}; const val=mi.validation||{}; const sc=d.scenarios||{};

  // 決定三個模型的標籤
  const arima_label=`log-ARIMA(${mi.arima_order?.p},${mi.arima_order?.d},${mi.arima_order?.q})`;
  const ets_label  =mi.ets_spec  ||'ETS';
  const holt_label =mi.holt_spec ||'Holt(damped=True)';
  const best_m     =mi.best_model||'';
  const bestCol={'log_arima':'#38bdf8','ets':'#a78bfa','holt':'#4ade80'}[best_m]||'#e2e8f0';

  const lbColor =val.lb_pass===true?'#4ade80':val.lb_pass===false?'#f87171':'#94a3b8';
  const lbText  =val.lb_pass===true?'✓ 通過（殘差為白噪音）':val.lb_pass===false?'✗ 未通過（存在自相關）':'—';
  const mapeCol =val.mape!=null?(val.mape<5?'#4ade80':val.mape<10?'#fbbf24':'#f87171'):'#94a3b8';

  // ADF + KPSS 雙重檢定結果
  const d_tests =d.d_tests||{}; const d_reason=d.d_reason||'';
  const adf_o   =d_tests.adf_orig ||{}; const kpss_o=d_tests.kpss_orig||{};
  const adf_d1  =d_tests.adf_diff1||{}; const kpss_d1=d_tests.kpss_diff1||{};
  const oos_rmse=mi.oos_rmse||{}; const sel_meth=mi.selection_method||'OOS RMSE';

  // Holt 參數
  const hp=mi.holt_params||{};

  document.getElementById('modelValidBody').innerHTML=`
    <div style="margin-bottom:8px;padding:6px 8px;background:rgba(56,189,248,.06);border-radius:4px;font-size:10.5px">
      <strong style="color:#e2e8f0">選模標準：</strong><span style="color:#64748b">${sel_meth}</span><br>
      OOS RMSE → log-ARIMA: <strong>${oos_rmse.log_arima??'—'}</strong>，
      ETS: <strong>${oos_rmse.ets??'—'}</strong>，
      Holt: <strong>${oos_rmse.holt??'—'}</strong>
      → 選用 <strong style="color:${bestCol}">${
        {log_arima:arima_label,ets:ets_label,holt:holt_label}[best_m]||best_m
      }</strong>
    </div>
    <table style="width:100%;border-collapse:collapse;margin-bottom:8px">
      <tr style="border-bottom:1px solid #1e293b">
        <th style="text-align:left;color:#64748b;padding:4px 8px;font-weight:400">指標</th>
        <th style="text-align:right;color:#38bdf8;padding:4px 6px;font-size:10px">${arima_label}</th>
        <th style="text-align:right;color:#a78bfa;padding:4px 6px;font-size:10px">${ets_label}</th>
        <th style="text-align:right;color:#4ade80;padding:4px 6px;font-size:10px">${holt_label}</th>
        <th style="text-align:right;color:#e2e8f0;padding:4px 8px;font-weight:600;font-size:10px">✓ 選用</th>
      </tr>
      <tr>
        <td style="padding:4px 8px;color:#94a3b8">AIC（僅參考）</td>
        <td style="text-align:right;padding:4px 6px;font-size:11px">${mi.arima_aic!=null?Number(mi.arima_aic).toFixed(1):'—'}</td>
        <td style="text-align:right;padding:4px 6px;font-size:11px">${mi.ets_aic!=null?Number(mi.ets_aic).toFixed(1):'—'}</td>
        <td style="text-align:right;padding:4px 6px;font-size:11px">${mi.holt_aic!=null?Number(mi.holt_aic).toFixed(1):'—'}</td>
        <td style="text-align:right;padding:4px 8px;color:#64748b;font-size:10px">不可直接比較</td>
      </tr>
      <tr style="background:#0a0e17">
        <td style="padding:4px 8px;color:#94a3b8">樣本內 MAPE (%)</td>
        <td colspan="3" style="text-align:right;padding:4px 6px;color:#64748b;font-size:10px">選用模型回測</td>
        <td style="text-align:right;padding:4px 8px;color:${mapeCol};font-weight:600">${val.mape!=null?Number(val.mape).toFixed(2)+'%':'—'}</td>
      </tr>
      <tr>
        <td style="padding:4px 8px;color:#94a3b8">RMSE (kt)</td>
        <td colspan="3"></td>
        <td style="text-align:right;padding:4px 8px">${val.rmse!=null?Number(val.rmse).toLocaleString():'—'}</td>
      </tr>
      <tr style="background:#0a0e17">
        <td style="padding:4px 8px;color:#94a3b8">R²</td>
        <td colspan="3"></td>
        <td style="text-align:right;padding:4px 8px">${val.r2!=null?Number(val.r2).toFixed(4):'—'}</td>
      </tr>
      <tr>
        <td style="padding:4px 8px;color:#94a3b8">Ljung-Box Q(${val.lb_lag||10})</td>
        <td colspan="3" style="text-align:right;padding:4px 6px">${val.lb_stat!=null?Number(val.lb_stat).toFixed(3):'—'}</td>
        <td style="text-align:right;padding:4px 8px;color:${lbColor};font-weight:600">${lbText}</td>
      </tr>
      ${hp.phi!=null?`<tr style="background:#0a0e17"><td style="padding:4px 8px;color:#94a3b8">Holt φ（阻尼係數）</td>
        <td colspan="3"></td>
        <td style="text-align:right;padding:4px 8px;color:#4ade80">${hp.phi} <span style="color:#475569;font-size:10px">（趨勢衰減率）</span></td></tr>`:''}
    </table>
    <div style="background:#0a0e17;border-radius:4px;padding:8px;font-size:10.5px;margin-bottom:8px">
      <div style="color:#64748b;margin-bottom:4px;font-weight:600">▸ ADF + KPSS 雙重檢定（差分階數 d）</div>
      <div style="color:#94a3b8">
        原序列：ADF p=${adf_o.p??'—'}，KPSS p=${kpss_o.p??'—'}
        ${adf_d1.p!=null?`<br>一階差分：ADF p=${adf_d1.p}，KPSS p=${kpss_d1.p??'—'}`:''}
      </div>
      <div style="color:#64748b;margin-top:4px;font-size:10px">${d_reason}</div>
      <div style="color:#475569;margin-top:3px;font-size:10px">Kwiatkowski et al. (1992) JoE；Hyndman &amp; Athanasopoulos (2021)</div>
    </div>
    <div style="font-size:10.5px;color:#475569;line-height:1.7">
      ✦ AIC 不可跨空間比較（Hyndman &amp; Koehler, 2006）— 以 OOS RMSE 為主要選模準則<br>
      ✦ Holt damped trend：φ&lt;1 使趨勢衰減，防止長期外推發散（Gardner &amp; McKenzie, 1985）<br>
      ✦ MAPE &lt; 5% 優秀；5–10% 良好（Hyndman &amp; Koehler, 2006）
    </div>`;

  const citRows=Object.entries(sc).map(([k,s])=>`
    <tr style="${k!=='bau'?'border-top:1px solid #1e293b':''}">
      <td style="padding:5px 8px;color:${s.color};font-weight:600">${s.label}</td>
      <td style="padding:5px 8px;text-align:right;color:#e2e8f0">${s.rate_note||'—'}</td>
    </tr>
    <tr style="background:#0a0e17">
      <td colspan="2" style="padding:2px 8px 6px;color:#475569;font-size:10.5px">${s.citation||'—'}</td>
    </tr>`).join('');

  document.getElementById('scenarioCitBody').innerHTML=`
    <table style="width:100%;border-collapse:collapse">${citRows}</table>
    <div style="margin-top:8px;font-size:10.5px;color:#475569;line-height:1.6">
      ✦ 折年率從最後資料年動態計算至目標年（非固定起算點）<br>
      ✦ MC：歷史年變動率 bootstrap，無分布假設（Efron &amp; Tibshirani, 1993）<br>
      ✦ Kaya (1990)；Ang &amp; Zhang (2000) Energy Policy
    </div>`;

  document.getElementById('validationSection').style.display='';
}


// ════════════════════════════════════════════════════════
// ── 完整數據表 ───────────────────────────────────────────
// ════════════════════════════════════════════════════════
let _fcShowHist = false;
function toggleHistRows() {
  _fcShowHist = !_fcShowHist;
  document.querySelectorAll('.fc-hist-row').forEach(r => {
    r.style.display = _fcShowHist ? '' : 'none';
  });
  document.getElementById('fcToggleHist').textContent = _fcShowHist ? '隱藏歷史' : '顯示歷史';
}

function renderFcTable(d, sc) {
  if(!d || !d.fc_years) return;
  const fy  = d.fc_years  || [];
  const hy  = d.hist_years|| [];
  const KEY = new Set([2030,2040,2050]);
  const MID = new Set([2025,2035,2045]);
  const fmt = v => v!=null && !isNaN(v) ? Number(Math.round(v)).toLocaleString('zh-TW') : '—';
  const fmtPct = (a,b) => (a!=null&&b!=null&&b!==0) ? ((a-b)/b*100).toFixed(1)+'%' : '—';

  const GAS = ['co2','ch4','n2o','hfc','pfc','sf6','nf3'];
  const GAS_LABEL = ['CO₂','CH₄','N₂O','HFCs','PFCs','SF₆','NF₃'];
  const GAS_COLOR = ['#60a5fa','#34d399','#f472b6','#fbbf24','#a78bfa','#fb7185','#14b8a6'];
  const hasGas = GAS.map(k => {
    const g = d.gas_results?.[k];
    return g && !g.skipped && !g.error && g.forecast?.length > 0;
  });
  const anyGas = hasGas.some(Boolean);

  const modelBadge = document.getElementById('fcTableModelBadge');
  if(modelBadge) {
    const m = d.model_info?.best_model || 'log_arima';
    const aic = d.model_info?.model_aic;
    modelBadge.textContent = `${m==='log_arima'?'log-ARIMA':'ETS'}  AIC=${aic!=null?Number(aic).toFixed(1):'—'}`;
  }

  const sumEl = document.getElementById('fcTableSummary');
  if(sumEl && d.hist_total?.length) {
    const base = d.hist_total[d.hist_total.length-1];
    const fc2030 = fy.indexOf(2030) >= 0 ? d.fc_total[fy.indexOf(2030)] : null;
    const fc2050 = fy.indexOf(2050) >= 0 ? d.fc_total[fy.indexOf(2050)] : null;
    const mape = d.model_info?.validation?.mape;
    const items = [
      {l:'基準年排放', v: fmt(base)+' kt', c:'#e2e8f0'},
      {l:'2030 預測',  v: fc2030!=null ? fmt(fc2030)+' kt' : '—', c:'#38bdf8'},
      {l:'vs 基準年',  v: fmtPct(fc2030,base), c: fc2030<base?'#4ade80':'#f87171'},
      {l:'2050 預測',  v: fc2050!=null ? fmt(fc2050)+' kt' : '—', c:'#a78bfa'},
      {l:'vs 基準年',  v: fmtPct(fc2050,base), c: fc2050<base?'#4ade80':'#f87171'},
      {l:'樣本內MAPE', v: mape!=null ? mape+'%' : '—', c:'#fbbf24'},
      {l:'BAU CAGR',   v: d.bau_cagr!=null ? (d.bau_cagr>=0?'+':'')+Number(d.bau_cagr).toFixed(3)+'%' : '—', c:'#f59e0b'},
    ];
    sumEl.innerHTML = items.map(x=>`
      <div style="background:#0a0e17;border:1px solid #1e293b;border-radius:6px;padding:6px 12px;min-width:100px">
        <div style="font-size:10px;color:#475569;margin-bottom:2px">${x.l}</div>
        <div style="font-size:13px;font-family:'JetBrains Mono',monospace;font-weight:600;color:${x.c}">${x.v}</div>
      </div>`).join('');
  }

  const gasThs = anyGas ? GAS.map((k,i) => hasGas[i]
    ? `<th style="color:${GAS_COLOR[i]};padding:6px 8px;white-space:nowrap">${GAS_LABEL[i]} (kt)</th>`
    : '').join('') : '';

  document.getElementById('fcTableHead').innerHTML = `<tr>
    <th style="text-align:center;padding:6px 10px;min-width:52px">年份</th>
    <th style="text-align:right;padding:6px 10px;color:#94a3b8">實際排放</th>
    <th style="text-align:right;padding:6px 10px;color:#60a5fa">CO₂ (kt)</th>
    <th style="text-align:right;padding:6px 10px;color:#34d399">CH₄ (kt)</th>
    <th style="text-align:right;padding:6px 10px;color:#f472b6">N₂O (kt)</th>
    <th style="text-align:right;padding:6px 10px;color:#84cc16">土地匯 (kt)</th>
    <th style="text-align:right;padding:6px 10px;color:#fb7185">淨排放 (kt)</th>
    <th style="text-align:right;padding:6px 10px;color:#38bdf8">ARIMA預測</th>
    <th style="text-align:right;padding:6px 10px;color:#475569">95%上界</th>
    <th style="text-align:right;padding:6px 10px;color:#475569">95%下界</th>
    <th style="text-align:right;padding:6px 10px;color:#64748b">CI寬度</th>
    <th style="text-align:right;padding:6px 10px;color:#f59e0b">BAU</th>
    <th style="text-align:right;padding:6px 10px;color:#38bdf8">積極政策</th>
    <th style="text-align:right;padding:6px 10px;color:#00e5c0">NDC淨零</th>
    <th style="text-align:right;padding:6px 10px;color:#94a3b8">年變動率</th>
    ${gasThs}
  </tr>`;

  const histRows = hy.map((yr,i) => {
    const isKey = KEY.has(yr);
    const hrow = d.history_table?.[i] || {};
    const v    = d.hist_total?.[i];
    const prev = i>0 ? d.hist_total[i-1] : null;
    const chg  = (v!=null&&prev!=null&&prev!==0) ? ((v-prev)/prev*100).toFixed(2)+'%' : '—';
    const chgCol = (v!=null&&prev!=null) ? (v<prev?'#4ade80':'#f87171') : '#475569';
    const landVal = hrow.land;
    const netVal  = hrow.net;
    // 氣體欄：有 gas_results 預測才顯示欄位；歷史值從 history_table 取
    const gasVals = anyGas ? GAS.map((k,gi) => {
      if(!hasGas[gi]) return '';
      const gv = hrow[k];
      return `<td style="text-align:right;padding:4px 8px;color:${GAS_COLOR[gi]};opacity:.85">${fmt(gv)}</td>`;
    }).join('') : '';
    const bg = isKey ? 'background:rgba(167,139,250,.06)' : (i%2===0?'':'background:rgba(255,255,255,.015)');
    const fw = isKey ? 'font-weight:600' : '';
    return `<tr class="fc-hist-row" style="${bg};${fw};display:${_fcShowHist?'':'none'}">
      <td style="text-align:center;padding:4px 10px;color:${isKey?'#a78bfa':'#64748b'}">${yr}${isKey?'★':''}</td>
      <td style="text-align:right;padding:4px 10px;color:#e2e8f0">${fmt(v)}</td>
      <td style="text-align:right;padding:4px 10px;color:#60a5fa">${fmt(hrow.co2)}</td>
      <td style="text-align:right;padding:4px 10px;color:#34d399">${fmt(hrow.ch4)}</td>
      <td style="text-align:right;padding:4px 10px;color:#f472b6">${fmt(hrow.n2o)}</td>
      <td style="text-align:right;padding:4px 10px;color:${landVal!=null&&landVal<0?'#fb7185':'#84cc16'}">${fmt(landVal)}</td>
      <td style="text-align:right;padding:4px 10px;color:#fb7185">${fmt(netVal)}</td>
      <td style="text-align:right;padding:4px 10px;color:#38bdf8">—</td>
      <td style="text-align:right;padding:4px 10px;color:#475569">—</td>
      <td style="text-align:right;padding:4px 10px;color:#475569">—</td>
      <td style="text-align:right;padding:4px 10px;color:#475569">—</td>
      <td style="text-align:right;padding:4px 10px;color:#f59e0b">—</td>
      <td style="text-align:right;padding:4px 10px;color:#38bdf8">—</td>
      <td style="text-align:right;padding:4px 10px;color:#00e5c0">—</td>
      <td style="text-align:right;padding:4px 10px;color:${chgCol}">${chg}</td>
      ${gasVals}
    </tr>`;
  }).join('');

  const fcRows = fy.map((yr,i) => {
    const isKey = KEY.has(yr); const isMid = MID.has(yr);
    const fc = d.fc_total?.[i]; const up = d.fc_upper?.[i]; const lo = d.fc_lower?.[i];
    const ci = (up!=null&&lo!=null) ? fmt(up-lo) : '—';
    const bau = sc?.bau?.values?.[i]; const pol = sc?.policy?.values?.[i]; const ndc = sc?.ndc?.values?.[i];
    const prev = i>0 ? d.fc_total[i-1] : d.hist_total?.[d.hist_total.length-1];
    const chg = (fc!=null&&prev!=null&&prev!==0) ? ((fc-prev)/prev*100).toFixed(2)+'%' : '—';
    const chgCol = (fc!=null&&prev!=null) ? (fc<prev?'#4ade80':'#f87171') : '#475569';
    const gasVals = anyGas ? GAS.map((k,gi) => {
      if(!hasGas[gi]) return '';
      const gv = d.gas_results?.[k]?.forecast?.[i];
      return `<td style="text-align:right;padding:4px 8px;color:${GAS_COLOR[gi]}">${fmt(gv)}</td>`;
    }).join('') : '';
    const bg  = isKey ? 'background:rgba(167,139,250,.12)' : isMid ? 'background:rgba(56,189,248,.04)' : (i%2===0?'':'background:rgba(255,255,255,.015)');
    const fw  = isKey ? 'font-weight:700' : '';
    const star = isKey ? '★' : isMid ? '·' : '';
    return `<tr class="fc-fc-row" style="${bg};${fw}">
      <td style="text-align:center;padding:4px 10px;color:${isKey?'#a78bfa':'#475569'}">${yr}${star}</td>
      <td style="color:#1e293b;text-align:right;padding:4px 10px">—</td>
      <td style="color:#1e293b;text-align:right;padding:4px 10px">—</td>
      <td style="color:#1e293b;text-align:right;padding:4px 10px">—</td>
      <td style="color:#1e293b;text-align:right;padding:4px 10px">—</td>
      <td style="color:#1e293b;text-align:right;padding:4px 10px">—</td>
      <td style="color:#1e293b;text-align:right;padding:4px 10px">—</td>
      <td style="text-align:right;padding:4px 10px;color:#38bdf8;font-weight:${isKey?700:400}">${fmt(fc)}</td>
      <td style="text-align:right;padding:4px 10px;color:#64748b">${fmt(up)}</td>
      <td style="text-align:right;padding:4px 10px;color:#64748b">${fmt(lo)}</td>
      <td style="text-align:right;padding:4px 10px;color:#475569">${ci}</td>
      <td style="text-align:right;padding:4px 10px;color:#f59e0b">${fmt(bau)}</td>
      <td style="text-align:right;padding:4px 10px;color:#38bdf8">${fmt(pol)}</td>
      <td style="text-align:right;padding:4px 10px;color:#00e5c0">${fmt(ndc)}</td>
      <td style="text-align:right;padding:4px 10px;color:${chgCol}">${chg}</td>
      ${gasVals}
    </tr>`;
  }).join('');

  document.getElementById('fcTableBody').innerHTML = histRows + fcRows;
  document.getElementById('fcTableCard').style.display = '';
}

function exportFcTableCsv() {
  if(!analysisData) return;
  const d  = analysisData; const sc = d.scenarios;
  const GAS = ['co2','ch4','n2o','hfc','pfc','sf6','nf3'];
  const hasGas = GAS.map(k => { const g = d.gas_results?.[k]; return g && !g.skipped && !g.error && g.forecast?.length > 0; });
  const gasHdr = GAS.filter((_,i)=>hasGas[i]).map(k=>k.toUpperCase()+'(kt)');
  const hdr = ['年份','資料類型','實際排放(kt)','ARIMA預測(kt)','95%上界','95%下界','CI寬度','年變動率%','BAU(kt)','積極政策(kt)','NDC淨零(kt)', ...gasHdr];
  const rows = [hdr];
  (d.hist_years||[]).forEach((yr,i) => {
    const v = d.hist_total?.[i] ?? ''; const prev = i>0 ? d.hist_total[i-1] : null;
    const chg = (v!==''&&prev!=null&&prev!==0) ? ((v-prev)/prev*100).toFixed(3) : '';
    const gasVals = GAS.filter((_,gi)=>hasGas[gi]).map(k => d.history_table?.[i]?.[k] ?? '');
    rows.push([yr,'歷史實測',v,'','','','',chg,'','','', ...gasVals]);
  });
  (d.fc_years||[]).forEach((yr,i) => {
    const fc = d.fc_total?.[i] ?? ''; const up = d.fc_upper?.[i] ?? ''; const lo = d.fc_lower?.[i] ?? '';
    const ci = (up!==''&&lo!=='') ? Math.round(up-lo) : '';
    const prev = i>0 ? d.fc_total[i-1] : d.hist_total?.[d.hist_total.length-1];
    const chg = (fc!==''&&prev!=null&&prev!==0) ? ((fc-prev)/prev*100).toFixed(3) : '';
    const gasVals = GAS.filter((_,gi)=>hasGas[gi]).map(k => d.gas_results?.[k]?.forecast?.[i] ?? '');
    rows.push([yr,'ARIMA預測','',fc,up,lo,ci,chg,sc?.bau?.values?.[i]??'',sc?.policy?.values?.[i]??'',sc?.ndc?.values?.[i]??'', ...gasVals]);
  });
  const csv = rows.map(r=>r.map(v=>String(v).includes(',')?`"${v}"`:v).join(',')).join('\n');
  const a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,\uFEFF' + encodeURIComponent(csv);
  a.download = `GHG_forecast_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
}

function updateForecastTable(d, sc){
  const fLen=d.fc_years.length;
  let rows='', first=true;
  d.history_table.forEach(r=>{
    rows+=`<tr class="hist-row"><td>${r.year}</td>
      <td>${r.co2!=null?fmtN(r.co2):'<span class="null-val">—</span>'}</td>
      <td>${r.ch4!=null?fmtN(r.ch4):'<span class="null-val">—</span>'}</td>
      <td>${r.n2o!=null?fmtN(r.n2o):'<span class="null-val">—</span>'}</td>
      <td class="${r.land!=null&&r.land<0?'neg-val':''}">${r.land!=null?fmtN(r.land):'<span class="null-val">—</span>'}</td>
      <td>${r.total!=null?fmtN(r.total):'<span class="null-val">—</span>'}</td>
      <td class="${r.net<0?'neg-val':''}">${r.net!=null?fmtN(r.net):'<span class="null-val">—</span>'}</td>
      <td class="null-val">—</td><td class="null-val">—</td><td class="null-val">—</td>
      <td class="null-val">—</td><td class="null-val">—</td><td class="null-val">—</td>
      <td style="color:var(--muted);font-size:10px">歷史</td></tr>`;
  });
  d.forecast_table.forEach((r,i)=>{
    const cls=first?'fc-row divider':'fc-row'; first=false;
    rows+=`<tr class="${cls}"><td>${r.year}</td>
      <td class="null-val">—</td><td class="null-val">—</td><td class="null-val">—</td><td class="null-val">—</td>
      <td class="null-val">—</td><td class="null-val">—</td>
      <td>${fmtN(r.total)}</td><td>${fmtN(r.upper95)}</td><td>${fmtN(r.lower95)}</td>
      <td style="color:#f59e0b">${sc?fmtN(sc.bau.values[i]):'—'}</td>
      <td style="color:#38bdf8">${sc?fmtN(sc.policy.values[i]):'—'}</td>
      <td style="color:#00e5c0">${sc?fmtN(sc.ndc.values[i]):'—'}</td>
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

// ── Chart helpers ──────────────────────────────────────
const BO={responsive:true,maintainAspectRatio:true,animation:{duration:600},plugins:{legend:{display:false},tooltip:{enabled:false}}};
const SC={grid:{color:'rgba(255,255,255,.032)'},ticks:{color:'#4a6070',font:{size:10},maxTicksLimit:8}};
function lopts(yL,ar=2){return{...BO,aspectRatio:ar,scales:{x:SC,y:{...SC,title:{display:true,text:yL,color:'#4a6070',font:{size:10}}}}}}
function mk(id,cfg){if(charts[id])charts[id].destroy();charts[id]=new Chart(document.getElementById(id).getContext('2d'),cfg)}
function fmtN(v){if(v==null||v===undefined||isNaN(Number(v)))return'<span class="null-val">—</span>';return Number(v).toLocaleString('zh-TW',{maximumFractionDigits:1})}
function fmt(v){if(v==null||v===''||isNaN(Number(v)))return'<span class="null-val">—</span>';return Number(v).toLocaleString('zh-TW',{maximumFractionDigits:1})}
function showErr(m){const e=document.getElementById('errorBox');e.textContent='❌ '+m;e.style.display='block'}
function hideErr(){document.getElementById('errorBox').style.display='none'}
