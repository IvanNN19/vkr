# -*- coding: utf-8 -*-
"""
Генерирует output/cluster_analysis.html — отдельная страница с детальным
анализом кластеров: профили признаков, автоматические описания, география.
"""

import json, os
import pandas as pd
import numpy as np

BASE    = os.path.dirname(__file__) + '/..'
IN_CSV  = f'{BASE}/data/processed/clusters_with_vpr.csv'
OUT_HTML = f'{BASE}/output/cluster_analysis.html'

CLUSTER_LABELS = {
    0: ('Северные ресурсные регионы (ХМАО/ЯНАО/Сахалин)', '#00BCD4'),
    1: ('Юг и Северный Кавказ',                           '#FF9800'),
    2: ('Москва',                                          '#F44336'),
    3: ('Восточная Сибирь и ДФО (удалённые)',              '#795548'),
    4: ('Крупные центры ПФО/УФО/СФО',                     '#4CAF50'),
    5: ('Малые города ЦФО и СЗФО',                        '#2196F3'),
    6: ('Сельские районы ПФО и СФО',                      '#9E9E9E'),
    7: ('Санкт-Петербург и Подмосковье',                   '#9C27B0'),
    8: ('Сибирь и Дальний Восток',                        '#607D8B'),
}

FEATURE_COLS = [
    'dist_to_region_center_km',
    'dist_to_moscow_km',
    'federal_district_code',
    'gps_density_5km',
    'grp_per_capita_2021',
    'population_2021',
    'urban_pct',
]

FEAT_NAMES = {
    'dist_to_region_center_km': 'Удалённость от центра региона',
    'dist_to_moscow_km':        'Удалённость от Москвы',
    'federal_district_code':    'Федеральный округ (запад→восток)',
    'gps_density_5km':          'Транспортная активность (GPS)',
    'grp_per_capita_2021':      'ВРП на душу населения',
    'population_2021':          'Численность населения региона',
    'urban_pct':                'Уровень урбанизации',
}

FEAT_UNITS = {
    'dist_to_region_center_km': 'км',
    'dist_to_moscow_km':        'км',
    'federal_district_code':    '',
    'gps_density_5km':          'треков/ячейка',
    'grp_per_capita_2021':      'тыс. руб.',
    'population_2021':          'тыс. чел.',
    'urban_pct':                '%',
}

FEAT_EXPLAIN = {
    'dist_to_region_center_km': {
        '+': 'сильно удалены от столицы своего региона — глубокая периферия',
        '-': 'находятся близко к столице региона — административные центры',
    },
    'dist_to_moscow_km': {
        '+': 'расположены далеко от Москвы — восточные и отдалённые территории',
        '-': 'расположены близко к Москве — центральная Россия',
    },
    'federal_district_code': {
        '+': 'относятся к восточным федеральным округам (СФО, ДФО)',
        '-': 'относятся к западным федеральным округам (ЦФО, СЗФО, ЮФО)',
    },
    'gps_density_5km': {
        '+': 'высокая транспортная активность — плотная дорожная сеть',
        '-': 'низкая транспортная активность — слабая транспортная доступность',
    },
    'grp_per_capita_2021': {
        '+': 'богатые регионы с высоким ВРП — ресурсные или столичные субъекты',
        '-': 'экономически слабые регионы с низким ВРП',
    },
    'population_2021': {
        '+': 'крупные по населению регионы',
        '-': 'малонаселённые регионы',
    },
    'urban_pct': {
        '+': 'высокая урбанизация — в основном городские школы',
        '-': 'низкая урбанизация — значительная доля сельских школ',
    },
}

FD_NAMES = {
    'ЦФО': 'Центральный', 'СЗФО': 'Северо-Западный',
    'ЮФО': 'Южный',       'СКФО': 'Северо-Кавказский',
    'ПФО': 'Приволжский', 'УФО': 'Уральский',
    'СФО': 'Сибирский',   'ДФО': 'Дальневосточный',
}

# ─────────────────────────────────────────────────────────────────────────────
print('Загружаю данные...')
df = pd.read_csv(IN_CSV)

feat_data     = df[FEATURE_COLS].astype(float)
global_mean   = feat_data.mean()
global_std    = feat_data.std().replace(0, 1)
global_median = feat_data.median()

# ── Собираем данные по каждому кластеру ─────────────────────────────────────
clusters_data = []
for cid in sorted(df['cluster'].unique()):
    sub   = df[df['cluster'] == cid]
    label, color = CLUSTER_LABELS[cid]
    n     = len(sub)

    # z-score профиль
    c_mean = sub[FEATURE_COLS].mean()
    profile = {}
    for f in FEATURE_COLS:
        z    = float((c_mean[f] - global_mean[f]) / global_std[f])
        val  = float(c_mean[f])
        gval = float(global_mean[f])
        profile[f] = {'z': round(z, 2), 'val': round(val, 1), 'global': round(gval, 1)}

    # Топ округа
    fd_dist = sub['federal_district'].value_counts()
    fd_pct  = (fd_dist / n * 100).round(1).to_dict()

    # Топ регионов
    top_regions = sub['region'].value_counts().head(5).to_dict()

    # ВПР
    vpr = {
        'score': round(float(sub['vpr_score_region'].mean()), 1) if 'vpr_score_region' in sub else None,
        'ru':    round(float(sub['vpr_ru_region'].mean()),    1) if 'vpr_ru_region'    in sub else None,
        'ma':    round(float(sub['vpr_ma_region'].mean()),    1) if 'vpr_ma_region'    in sub else None,
        'grade': round(float(sub['vpr_grade_region'].mean()), 2) if 'vpr_grade_region' in sub else None,
    }

    # Автоматическое описание
    sorted_feats = sorted(FEATURE_COLS, key=lambda f: abs(profile[f]['z']), reverse=True)
    auto_desc = []
    for f in sorted_feats[:4]:
        z = profile[f]['z']
        if abs(z) < 0.4:
            continue
        direction = '+' if z > 0 else '-'
        strength = 'Значительно' if abs(z) > 1.5 else ('Заметно' if abs(z) > 0.8 else 'Немного')
        explain  = FEAT_EXPLAIN[f][direction]
        auto_desc.append(f'{strength}: школы {explain} ({profile[f]["val"]:.0f} {FEAT_UNITS[f]})')

    clusters_data.append({
        'id':       int(cid),
        'label':    label,
        'color':    color,
        'n':        n,
        'profile':  profile,
        'fd_pct':   fd_pct,
        'top_regions': {k: int(v) for k, v in list(top_regions.items())[:5]},
        'auto_desc': auto_desc,
        'vpr':      vpr,
    })

# ── Данные для сравнительной таблицы (z-score heatmap) ──────────────────────
comparison = {}
for f in FEATURE_COLS:
    comparison[f] = {}
    for cd in clusters_data:
        comparison[f][cd['id']] = cd['profile'][f]['z']

JS_DATA = json.dumps({
    'clusters':    clusters_data,
    'features':    FEATURE_COLS,
    'feat_names':  FEAT_NAMES,
    'feat_units':  FEAT_UNITS,
    'comparison':  comparison,
    'global_mean': {f: round(float(global_mean[f]), 1) for f in FEATURE_COLS},
    'total':       int(len(df)),
}, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Анализ кластеров школ России</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; color: #222; }}

  /* HEADER */
  .page-header {{
    background: #1a237e;
    color: #fff;
    padding: 18px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,.25);
  }}
  .page-header h1 {{ font-size: 20px; font-weight: 600; }}
  .page-header p  {{ font-size: 13px; opacity: .8; margin-top: 3px; }}
  .header-badge {{
    background: rgba(255,255,255,.15);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 13px;
    margin-left: auto;
  }}
  .back-btn {{
    text-decoration: none;
    background: rgba(255,255,255,.2);
    color: #fff;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 13px;
    border: 1px solid rgba(255,255,255,.3);
  }}
  .back-btn:hover {{ background: rgba(255,255,255,.3); }}

  /* TABS */
  .tabs-wrap {{
    background: #fff;
    border-bottom: 2px solid #e0e0e0;
    padding: 0 32px;
    display: flex;
    gap: 0;
    overflow-x: auto;
  }}
  .tab-btn {{
    padding: 13px 18px;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: #555;
    border-bottom: 3px solid transparent;
    white-space: nowrap;
    transition: color .15s, border-color .15s;
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: -2px;
  }}
  .tab-btn .dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .tab-btn:hover   {{ color: #222; }}
  .tab-btn.active  {{ color: #222; font-weight: 700; }}

  /* CONTENT */
  .content {{ max-width: 1200px; margin: 0 auto; padding: 28px 32px; }}
  .tab-pane {{ display: none; }}
  .tab-pane.active {{ display: block; }}

  /* CLUSTER CARD */
  .cluster-hero {{
    background: #fff;
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    border-left: 6px solid var(--c);
  }}
  .cluster-hero-num {{
    font-size: 36px;
    font-weight: 900;
    color: var(--c);
    line-height: 1;
    min-width: 44px;
  }}
  .cluster-hero-title {{ font-size: 20px; font-weight: 700; }}
  .cluster-hero-sub {{ font-size: 13px; color: #666; margin-top: 4px; }}

  /* TWO-COL */
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  @media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

  .card {{
    background: #fff;
    border-radius: 10px;
    padding: 20px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
  }}
  .card h3 {{ font-size: 14px; font-weight: 700; color: #444; margin-bottom: 14px; text-transform: uppercase; letter-spacing: .5px; }}

  /* DESCRIPTION LIST */
  .desc-list {{ list-style: none; }}
  .desc-list li {{
    padding: 7px 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 13px;
    line-height: 1.5;
    display: flex;
    gap: 10px;
    align-items: flex-start;
  }}
  .desc-list li:last-child {{ border-bottom: none; }}
  .desc-arrow {{ font-size: 16px; line-height: 1.4; flex-shrink: 0; }}

  /* FD BARS */
  .fd-row {{
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    gap: 10px;
    font-size: 13px;
  }}
  .fd-name {{ width: 140px; flex-shrink: 0; color: #555; }}
  .fd-bar-wrap {{ flex: 1; height: 16px; background: #f0f0f0; border-radius: 4px; overflow: hidden; }}
  .fd-bar-fill {{ height: 100%; border-radius: 4px; transition: width .3s; }}
  .fd-pct {{ width: 40px; text-align: right; font-weight: 600; color: #333; }}

  /* FEATURE TABLE */
  .feat-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .feat-table th {{
    text-align: left;
    padding: 8px 10px;
    background: #f5f6fa;
    font-size: 11px;
    text-transform: uppercase;
    color: #888;
    letter-spacing: .4px;
  }}
  .feat-table td {{ padding: 9px 10px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }}
  .feat-table tr:last-child td {{ border-bottom: none; }}
  .z-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 700;
    min-width: 52px;
    text-align: center;
  }}
  .compare-bar {{
    display: inline-block;
    height: 10px;
    border-radius: 3px;
    vertical-align: middle;
    margin-left: 4px;
  }}

  /* COMPARISON TAB */
  .heatmap-wrap {{ overflow-x: auto; }}
  .heatmap-table {{ border-collapse: collapse; font-size: 12px; min-width: 700px; width: 100%; }}
  .heatmap-table th {{
    padding: 10px 8px;
    text-align: center;
    font-size: 11px;
    background: #f5f6fa;
    font-weight: 700;
    white-space: nowrap;
  }}
  .heatmap-table th.feat-col {{ text-align: left; min-width: 200px; }}
  .heatmap-table td {{
    padding: 9px 10px;
    text-align: center;
    font-weight: 700;
    font-size: 13px;
    border-bottom: 1px solid #eee;
    border-right: 1px solid #eee;
  }}
  .heatmap-table td.feat-name {{ text-align: left; font-weight: 400; font-size: 13px; color: #333; }}

  .chart-container {{ position: relative; height: 260px; }}
  .full-chart {{ position: relative; height: 300px; }}

  /* TOP REGIONS */
  .region-chip {{
    display: inline-block;
    background: #f0f0f0;
    border-radius: 16px;
    padding: 4px 11px;
    font-size: 12px;
    margin: 3px 3px 3px 0;
    color: #444;
  }}
  .region-chip b {{ color: #222; }}

  /* INFO BOX */
  .info-box {{
    background: #e8f4fd;
    border-left: 4px solid #2196F3;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    color: #1a237e;
    margin-bottom: 20px;
    line-height: 1.6;
  }}
</style>
</head>
<body>

<div class="page-header">
  <div>
    <h1>📊 Анализ кластеров школ России</h1>
    <p>Кластерный анализ {len(df):,} школ по 7 экономическим и территориальным признакам &nbsp;·&nbsp; K-means, k=9</p>
  </div>
  <div class="header-badge">Силуэт: 0.35</div>
  <a href="map_clusters.html" class="back-btn">← Карта</a>
</div>

<!-- TABS -->
<div class="tabs-wrap" id="tabs-wrap">
  <!-- генерируются JS -->
</div>

<div class="content" id="tab-content">
  <!-- генерируются JS -->
</div>

<script>
const DATA = {JS_DATA};

// ── helpers ────────────────────────────────────────────────────────────────
function fmtVal(f, v) {{
  const u = DATA.feat_units[f];
  if (f === 'gps_density_5km') {{
    if (v >= 1e6) return (v/1e6).toFixed(2) + 'M ' + u;
    if (v >= 1e3) return Math.round(v/1e3) + 'K ' + u;
    return Math.round(v) + ' ' + u;
  }}
  return (Math.round(v * 10)/10).toLocaleString('ru') + (u ? ' ' + u : '');
}}

function zColor(z, clusterColor) {{
  const abs = Math.min(Math.abs(z), 3);
  const alpha = 0.15 + abs/3*0.75;
  if (z > 0) return clusterColor;
  return '#9e9e9e';
}}

function zBg(z) {{
  const abs = Math.min(Math.abs(z), 2.5);
  const t = abs / 2.5;
  if (z >= 0) {{
    const r = Math.round(66  + t*(33-66));
    const g = Math.round(165 + t*(150-165));
    const b = Math.round(245 + t*(243-245));
    return `rgba(${{r}},${{g}},${{b}},0.15)`;
  }} else {{
    return `rgba(180,180,180,${{0.1 + t*0.2}})`;
  }}
}}

// ── build tabs ─────────────────────────────────────────────────────────────
const tabsWrap   = document.getElementById('tabs-wrap');
const tabContent = document.getElementById('tab-content');

function makeTab(id, label, color, active) {{
  const btn = document.createElement('button');
  btn.className = 'tab-btn' + (active ? ' active' : '');
  btn.dataset.tab = id;
  btn.style.borderBottomColor = active ? color : 'transparent';
  btn.innerHTML = `<span class="dot" style="background:${{color}}"></span>${{label}}`;
  btn.onclick = () => switchTab(id, color);
  return btn;
}}

function switchTab(id, color) {{
  document.querySelectorAll('.tab-btn').forEach(b => {{
    b.classList.remove('active');
    b.style.borderBottomColor = 'transparent';
  }});
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  const btn = document.querySelector(`[data-tab="${{id}}"]`);
  if (btn) {{
    btn.classList.add('active');
    btn.style.borderBottomColor = color || '#1a237e';
  }}
  const pane = document.getElementById('pane-' + id);
  if (pane) {{
    pane.classList.add('active');
    renderCharts(id);
  }}
}}

// ── render cluster pane ────────────────────────────────────────────────────
const chartInstances = {{}};

function renderCharts(id) {{
  if (id === 'compare') {{ renderCompareChart(); return; }}
  const cid = parseInt(id);
  const cd  = DATA.clusters.find(c => c.id === cid);
  if (!cd) return;

  // bar chart
  const canvasId = 'chart-' + cid;
  const canvas   = document.getElementById(canvasId);
  if (!canvas) return;
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

  const labels = DATA.features.map(f => DATA.feat_names[f]);
  const zVals  = DATA.features.map(f => cd.profile[f].z);
  const bgColors = zVals.map(z => z >= 0 ? cd.color : '#bdbdbd');
  const borderColors = bgColors;

  chartInstances[canvasId] = new Chart(canvas, {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'Отклонение от среднего (σ)',
        data: zVals,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 4,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const f = DATA.features[ctx.dataIndex];
              const p = cd.profile[f];
              const sign = p.z >= 0 ? '+' : '';
              return `${{sign}}${{p.z.toFixed(2)}}σ  |  ${{fmtVal(f, p.val)}}  (ср. ${{fmtVal(f, p.global)}})`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          min: -3, max: 3,
          grid: {{ color: ctx => ctx.tick.value === 0 ? '#999' : '#eee' }},
          ticks: {{ callback: v => (v>0?'+':'')+v+'σ', font: {{ size: 11 }} }},
          title: {{ display: true, text: 'Стандартных отклонений от среднего (σ)', font: {{ size: 11 }} }}
        }},
        y: {{ ticks: {{ font: {{ size: 12 }} }} }}
      }}
    }}
  }});
}}

function renderCompareChart() {{
  const cid = 'compare-chart';
  const canvas = document.getElementById(cid);
  if (!canvas) return;
  if (chartInstances[cid]) chartInstances[cid].destroy();

  const datasets = DATA.clusters.map(cd => ({{
    label: (cd.id+1) + '. ' + cd.label,
    data: DATA.features.map(f => cd.profile[f].z),
    backgroundColor: cd.color + 'bb',
    borderColor: cd.color,
    borderWidth: 2,
    pointRadius: 4,
    fill: false,
  }}));

  chartInstances[cid] = new Chart(canvas, {{
    type: 'radar',
    data: {{ labels: DATA.features.map(f => DATA.feat_names[f]), datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'right', labels: {{ font: {{ size: 11 }} }} }} }},
      scales: {{
        r: {{
          min: -2.5, max: 2.5,
          ticks: {{ stepSize: 1, callback: v => (v>0?'+':'')+v+'σ', font: {{ size: 10 }} }},
          pointLabels: {{ font: {{ size: 11 }} }}
        }}
      }}
    }}
  }});
}}

// ── build cluster pane HTML ────────────────────────────────────────────────
function buildClusterPane(cd) {{
  const pane = document.createElement('div');
  pane.className = 'tab-pane';
  pane.id = 'pane-' + cd.id;

  // fd bars
  const maxFdPct = Math.max(...Object.values(cd.fd_pct));
  const fdHtml = Object.entries(cd.fd_pct)
    .sort((a,b) => b[1]-a[1])
    .map(([fd, pct]) => `
      <div class="fd-row">
        <div class="fd-name">${{fd}}</div>
        <div class="fd-bar-wrap">
          <div class="fd-bar-fill" style="width:${{pct/maxFdPct*100}}%;background:${{cd.color}}"></div>
        </div>
        <div class="fd-pct">${{pct}}%</div>
      </div>`).join('');

  // feature table
  const featTableRows = DATA.features.map(f => {{
    const p = cd.profile[f];
    const z = p.z;
    const sign = z >= 0 ? '+' : '';
    const bg   = z >= 0 ? cd.color : '#9e9e9e';
    const barW = Math.min(Math.abs(z)/2.5*60, 60);
    const barDir = z >= 0 ? '' : 'margin-left:auto;margin-right:0;';
    return `<tr>
      <td>${{DATA.feat_names[f]}}</td>
      <td><b>${{fmtVal(f, p.val)}}</b></td>
      <td style="color:#888">${{fmtVal(f, p.global)}}</td>
      <td>
        <span class="z-badge" style="background:${{bg}}22;color:${{bg}}">${{sign}}${{z.toFixed(2)}}σ</span>
        <span class="compare-bar" style="width:${{barW}}px;background:${{bg}};opacity:.6;${{barDir}}"></span>
      </td>
    </tr>`;
  }}).join('');

  // auto desc
  const descHtml = cd.auto_desc.length
    ? cd.auto_desc.map(d => `<li><span class="desc-arrow">→</span>${{d}}</li>`).join('')
    : '<li><span class="desc-arrow">—</span>Признаки близки к среднему по всем школам</li>';

  // top regions
  const regHtml = Object.entries(cd.top_regions)
    .map(([r, n]) => `<span class="region-chip">${{r}} <b>${{n}}</b></span>`).join('');

  // VPR block
  const vpr = cd.vpr;
  const allScores = DATA.clusters.map(c => c.vpr.score).filter(v => v != null);
  const minScore = Math.min(...allScores), maxScore = Math.max(...allScores);
  const scoreRange = maxScore - minScore || 1;
  const vprBarW = Math.round((vpr.score - minScore) / scoreRange * 100);
  const vprRuW  = Math.round((vpr.ru  - minScore) / scoreRange * 100);
  const vprMaW  = Math.round((vpr.ma  - minScore) / scoreRange * 100);
  const vprHtml = vpr.score != null ? `
    <div class="card" style="margin-top:20px">
      <h3>Результаты ВПР (внешняя валидация)</h3>
      <p style="font-size:12px;color:#888;margin-bottom:14px">
        Средний балл по регионам кластера · ВПР 4–6 классы, 2019–2020 · источник: ФИОКО
      </p>
      <div class="fd-row">
        <div class="fd-name" style="font-weight:600">Общий балл</div>
        <div class="fd-bar-wrap"><div class="fd-bar-fill" style="width:${{vprBarW}}%;background:${{cd.color}}"></div></div>
        <div class="fd-pct">${{vpr.score}}%</div>
      </div>
      <div class="fd-row">
        <div class="fd-name">Русский язык</div>
        <div class="fd-bar-wrap"><div class="fd-bar-fill" style="width:${{vprRuW}}%;background:#5C6BC0"></div></div>
        <div class="fd-pct">${{vpr.ru}}%</div>
      </div>
      <div class="fd-row">
        <div class="fd-name">Математика</div>
        <div class="fd-bar-wrap"><div class="fd-bar-fill" style="width:${{vprMaW}}%;background:#26A69A"></div></div>
        <div class="fd-pct">${{vpr.ma}}%</div>
      </div>
      <div style="margin-top:12px;font-size:12px;color:#555">
        Средняя отметка: <b>${{vpr.grade}}</b> &nbsp;·&nbsp;
        Шкала: ${{minScore}}% (мин. по кластерам) — ${{maxScore}}% (макс.)
      </div>
    </div>` : '';

  pane.innerHTML = `
    <div class="cluster-hero" style="--c:${{cd.color}}">
      <div class="cluster-hero-num">${{cd.id+1}}</div>
      <div>
        <div class="cluster-hero-title">${{cd.label}}</div>
        <div class="cluster-hero-sub">${{cd.n.toLocaleString('ru')}} школ &nbsp;·&nbsp; ${{(cd.n/DATA.total*100).toFixed(1)}}% от выборки</div>
      </div>
    </div>

    <div class="two-col">
      <div>
        <div class="card" style="margin-bottom:20px">
          <h3>Что отличает этот кластер</h3>
          <ul class="desc-list">${{descHtml}}</ul>
        </div>
        <div class="card" style="margin-bottom:20px">
          <h3>Распределение по округам</h3>
          ${{fdHtml}}
        </div>
        <div class="card">
          <h3>Топ регионов</h3>
          <div style="margin-top:4px">${{regHtml}}</div>
        </div>
        ${{vprHtml}}
      </div>
      <div class="card">
        <h3>Профиль признаков (отклонение от среднего)</h3>
        <div class="chart-container">
          <canvas id="chart-${{cd.id}}"></canvas>
        </div>
        <p style="font-size:11px;color:#aaa;margin-top:10px;text-align:center">
          Положительные значения (цвет) — выше среднего, отрицательные (серые) — ниже среднего
        </p>
      </div>
    </div>

    <div class="card">
      <h3>Детальные значения признаков</h3>
      <table class="feat-table">
        <thead>
          <tr>
            <th>Признак</th>
            <th>Среднее в кластере</th>
            <th>Среднее по всем школам</th>
            <th>Отклонение</th>
          </tr>
        </thead>
        <tbody>${{featTableRows}}</tbody>
      </table>
    </div>
  `;
  return pane;
}}

// ── build compare pane ─────────────────────────────────────────────────────
function buildComparePane() {{
  const pane = document.createElement('div');
  pane.className = 'tab-pane';
  pane.id = 'pane-compare';

  // heatmap table
  const headerCells = DATA.clusters.map(cd =>
    `<th><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{cd.color}};margin-right:4px"></span>${{cd.id+1}}. ${{cd.label.split(' ').slice(0,3).join(' ')}}</th>`
  ).join('');

  const tableRows = DATA.features.map(f => {{
    const cells = DATA.clusters.map(cd => {{
      const z = cd.profile[f].z;
      const sign = z >= 0 ? '+' : '';
      const bg = z >= 0
        ? `rgba(33,150,243,${{Math.min(Math.abs(z)/2.5, 1)*0.7}})`
        : `rgba(158,158,158,${{Math.min(Math.abs(z)/2.5, 1)*0.5}})`;
      const textColor = Math.abs(z) > 1.2 ? '#fff' : '#333';
      return `<td style="background:${{bg}};color:${{textColor}}">${{sign}}${{z.toFixed(1)}}σ</td>`;
    }}).join('');
    return `<tr>
      <td class="feat-name">${{DATA.feat_names[f]}}</td>
      ${{cells}}
    </tr>`;
  }}).join('');

  pane.innerHTML = `
    <div class="info-box">
      Таблица показывает, насколько каждый признак отличается от среднего по всем школам (в стандартных отклонениях σ).
      <b>Синий цвет</b> — значительно выше среднего, <b>серый</b> — ниже среднего.
      Чем насыщеннее цвет, тем сильнее отклонение.
    </div>

    <div class="two-col" style="margin-bottom:20px">
      <div class="card" style="grid-column: 1 / -1;">
        <h3>Радарная диаграмма — профили всех кластеров</h3>
        <div class="full-chart">
          <canvas id="compare-chart"></canvas>
        </div>
      </div>
    </div>

    <div class="card">
      <h3>Тепловая карта признаков по кластерам</h3>
      <div class="heatmap-wrap">
        <table class="heatmap-table">
          <thead>
            <tr>
              <th class="feat-col">Признак</th>
              ${{headerCells}}
            </tr>
          </thead>
          <tbody>${{tableRows}}</tbody>
        </table>
      </div>
    </div>
  `;
  return pane;
}}

// ── init ───────────────────────────────────────────────────────────────────
DATA.clusters.forEach((cd, i) => {{
  tabsWrap.appendChild(makeTab(cd.id, `${{cd.id+1}}. ${{cd.label}}`, cd.color, i === 0));
  tabContent.appendChild(buildClusterPane(cd));
}});

// compare tab
const compareBtn = document.createElement('button');
compareBtn.className = 'tab-btn';
compareBtn.dataset.tab = 'compare';
compareBtn.style.borderBottomColor = 'transparent';
compareBtn.innerHTML = '⚖️ Сравнение всех';
compareBtn.onclick = () => switchTab('compare', '#1a237e');
tabsWrap.appendChild(compareBtn);
tabContent.appendChild(buildComparePane());

// активируем первую вкладку
switchTab(0, DATA.clusters[0].color);
</script>
</body>
</html>"""

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f'Сохранено: {OUT_HTML}')
