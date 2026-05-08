# -*- coding: utf-8 -*-
"""
Интерактивная карта кластеров школ России на Folium.
Кнопка «Анализ кластеров» открывает cluster_analysis.html в новой вкладке.
"""

import os
import folium
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

BASE    = os.path.dirname(__file__) + '/..'
IN_CSV  = f'{BASE}/data/processed/clusters.csv'
OUT_DIR = f'{BASE}/output'

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

print('Загружаю данные...')
df = pd.read_csv(IN_CSV)
print(f'  {len(df)} школ')

# ── Карта 1: точки по кластерам ──────────────────────────────────────────────
print('Строю карту точек...')
m = folium.Map(location=[61, 90], zoom_start=4, tiles='CartoDB positron')

# Легенда
legend_html = (
    '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
    'background:white;padding:12px;border-radius:8px;'
    'border:1px solid #ccc;font-size:12px;line-height:1.8">'
    '<b>Кластеры школ</b><br>'
)
for c, (label, color) in CLUSTER_LABELS.items():
    cnt = (df['cluster'] == c).sum()
    legend_html += f'<span style="color:{color};font-size:16px">●</span> {c+1}. {label} ({cnt})<br>'
legend_html += '</div>'
m.get_root().html.add_child(folium.Element(legend_html))

# Кнопка → отдельная страница анализа
btn_html = """
<a href="cluster_analysis.html" target="_blank"
   style="position:fixed;top:80px;right:10px;z-index:1500;
          background:#1a237e;color:#fff;text-decoration:none;
          border-radius:7px;padding:8px 15px;font-size:13px;font-weight:bold;
          box-shadow:0 2px 8px rgba(0,0,0,.28);font-family:Arial,sans-serif;
          display:flex;align-items:center;gap:7px;">
  📊 Анализ кластеров
</a>
"""
m.get_root().html.add_child(folium.Element(btn_html))

# Слои по кластерам
for cluster_id, (label, color) in CLUSTER_LABELS.items():
    group  = folium.FeatureGroup(name=f'Кластер {cluster_id+1}: {label}')
    subset = df[df['cluster'] == cluster_id]
    mc     = MarkerCluster(options={'maxClusterRadius': 30})

    for _, row in subset.iterrows():
        popup_text = (
            f"<b>{row.get('name', '—')}</b><br>"
            f"Регион: {row.get('region', '—')}<br>"
            f"Округ: {row.get('federal_district', '—')}<br>"
            f"До центра региона: {row.get('dist_to_region_center_km', 0):.0f} км<br>"
            f"До Москвы: {row.get('dist_to_moscow_km', 0):.0f} км<br>"
            f"GPS-активность: {int(row.get('gps_density_5km', 0)):,}<br>"
            f"ВРП на душу: {row.get('grp_per_capita_2021', 0):.0f} тыс. руб.<br>"
            f"Урбанизация: {row.get('urban_pct', 0):.0f}%<br>"
            f"<b>Кластер {cluster_id+1}: {label}</b>"
        )
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=280),
        ).add_to(mc)

    mc.add_to(group)
    group.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

out1 = f'{OUT_DIR}/map_clusters.html'
m.save(out1)
print(f'  Карта точек: {out1}')

# ── Карта 2: тепловая карта GPS-плотности ────────────────────────────────────
print('Строю тепловую карту GPS-плотности...')
m2 = folium.Map(location=[61, 90], zoom_start=4, tiles='CartoDB dark_matter')

heat_data = df[['lat', 'lon', 'gps_density_5km']].dropna()
heat_data = heat_data[heat_data['gps_density_5km'] > 0]
max_val   = heat_data['gps_density_5km'].quantile(0.95)
heat_data = heat_data.copy()
heat_data['weight'] = heat_data['gps_density_5km'].clip(upper=max_val) / max_val

HeatMap(
    heat_data[['lat', 'lon', 'weight']].values.tolist(),
    radius=12, blur=8, min_opacity=0.3,
).add_to(m2)

folium.map.Marker(
    [71, 20],
    icon=folium.DivIcon(
        html='<div style="color:white;font-size:14px;font-weight:bold">'
             'Тепловая карта транспортной активности вокруг школ</div>'
    ),
).add_to(m2)

out2 = f'{OUT_DIR}/map_heatmap.html'
m2.save(out2)
print(f'  Тепловая карта: {out2}')

print('\n=== Готово ===')
print(f'  {out1}')
print(f'  {out2}')
