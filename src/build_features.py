"""
Расчёт признаков для каждой школы:
  - dist_to_region_center_km — расстояние до столицы региона (км)
  - dist_to_moscow_km        — расстояние до Москвы (км)
  - federal_district_code    — федеральный округ (1-8)
  - gps_density_5km          — плотность GPS-точек в радиусе 10 км
  - grp_per_capita_2021      — ВРП региона на душу населения (тыс. руб.)
  - population_2021          — население региона (тыс. чел.)
  - urban_pct                — доля городского населения в регионе (%)
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from math import radians, cos, sin, asin, sqrt

BASE = os.path.dirname(__file__) + '/..'
SCHOOLS_CSV = f'{BASE}/data/processed/schools_with_regions.csv'
GPS_DB      = f'{BASE}/data/raw/gps/2022-12-18_1686603663.sqlite'
ROSSTAT_CSV = f'{BASE}/data/raw/rosstat/rosstat_regions.csv'
OUT_CSV     = f'{BASE}/data/processed/features.csv'

# ── Центры регионов (региональные столицы) ──────────────────────────────────
# lat, lon столиц субъектов РФ (GADM NAME_1 → координаты)
REGION_CENTERS = {
    'Adygey':           (44.609, 40.101),
    'Altay':            (51.957, 82.961),
    'Gorno-Altay':            (50.707, 86.857),
    'Amur':             (50.350, 127.534),
    'Arkhangel\'sk':    (64.543, 40.537),
    'Astrakhan\'':      (46.347, 48.033),
    'Bashkortostan':    (54.735, 55.958),
    'Belgorod':         (50.597, 36.588),
    'Bryansk':          (53.243, 34.364),
    'Buryat':           (51.834, 107.584),
    'Chelyabinsk':      (55.154, 61.430),
    'Chechnya':         (43.317, 45.699),
    'Chukot':           (64.734, 177.514),
    'Chuvash':          (56.139, 47.252),
    'Dagestan':         (42.980, 47.504),
    'Ingush':           (43.166, 44.813),
    'Irkutsk':          (52.290, 104.296),
    'Ivanovo':          (57.000, 40.973),
    'Kabardin-Balkar':  (43.485, 43.607),
    'Kaliningrad':      (54.710, 20.510),
    'Kalmyk':           (46.308, 44.257),
    'Kaluga':           (54.507, 36.252),
    'Kamchatka':        (53.024, 158.653),
    'Karachay-Cherkess':(43.789, 41.738),
    'Karelia':          (61.787, 34.347),
    'Kemerovo':         (55.354, 86.086),
    'Khabarovsk':       (48.480, 135.082),
    'Khakass':          (53.721, 91.442),
    'Khanty-Mansiy':    (61.004, 69.019),
    'Kirov':            (58.597, 49.654),
    'Komi':             (61.668, 50.836),
    'Kostroma':         (57.767, 40.927),
    'Krasnodar':        (45.035, 38.975),
    'Krasnoyarsk':      (56.010, 92.852),
    'Kurgan':           (55.441, 65.341),
    'Kursk':            (51.730, 36.193),
    'Leningrad':        (59.902, 30.316),
    'Lipetsk':          (52.608, 39.600),
    'Magadan':          (59.568, 150.808),
    'Mariy-El':         (56.638, 47.897),
    'Mordovia':         (54.184, 45.184),
    'Moskva':           (55.751, 37.618),
    'MoscowCity':       (55.751, 37.618),
    'Murmansk':         (68.970, 33.075),
    'Nenets':           (67.638, 53.006),
    'Nizhegorod':       (56.328, 44.002),
    'NorthOssetia':     (43.048, 44.677),
    'Novgorod':         (58.522, 31.269),
    'Novosibirsk':      (54.989, 82.905),
    'Omsk':             (54.989, 73.368),
    'Orel':             (52.966, 36.070),
    'Orenburg':         (51.768, 55.097),
    'Penza':            (53.195, 45.018),
    'Perm\'':           (58.010, 56.229),
    "Primor'ye":        (43.116, 131.882),
    'Pskov':            (57.819, 28.332),
    'Rostov':           (47.222, 39.718),
    'Ryazan\'':         (54.629, 39.741),
    'Sakha':            (62.028, 129.733),
    'Sakhalin':         (46.959, 142.738),
    'Samara':           (53.195, 50.150),
    'Saratov':          (51.533, 46.034),
    'Smolensk':         (54.782, 32.045),
    'CityofSt.Petersburg': (59.939, 30.316),
    'Stavropol\'':      (45.042, 41.969),
    'Sverdlovsk':       (56.838, 60.597),
    'Tambov':           (52.721, 41.452),
    'Tatarstan':        (55.796, 49.106),
    'Tomsk':            (56.502, 84.992),
    'Tula':             (54.193, 37.617),
    'Tuva':             (51.719, 94.437),
    'Tver\'':           (56.859, 35.912),
    'Tyumen\'':         (57.153, 68.994),
    'Udmurt':           (56.852, 53.205),
    'Ul\'yanovsk':      (54.317, 48.402),
    'Vladimir':         (56.129, 40.406),
    'Volgograd':        (48.707, 44.517),
    'Vologda':          (59.220, 39.891),
    'Voronezh':         (51.672, 39.184),
    'Yamal-Nenets':     (66.530, 66.613),
    'Yaroslavl\'':      (57.626, 39.893),
    'Yevrey':           (48.480, 132.930),
    'Zabaykal\'ye':     (52.034, 113.500),
}

# Федеральные округа
FEDERAL_DISTRICTS = {
    'ЦФО': ['Belgorod','Bryansk','Vladimir','Voronezh','Ivanovo','Kaluga','Kostroma',
             'Kursk','Lipetsk','Moskva','MoscowCity','Orel','Ryazan\'','Smolensk',
             'Tambov','Tver\'','Tula','Yaroslavl\''],
    'СЗФО': ['Karelia','Komi',"Arkhangel'sk",'Nenets','Vologda','Kaliningrad',
              'Leningrad','Murmansk','Novgorod','Pskov','CityofSt.Petersburg'],
    'ЮФО':  ['Adygey','Kalmyk','Krasnodar','Astrakhan\'','Volgograd','Rostov',
              'Crimea','Sevastopol\''],
    'СКФО': ['Dagestan','Ingush','Kabardin-Balkar','Karachay-Cherkess',
              'NorthOssetia','Chechnya',"Stavropol'"],
    'ПФО':  ['Bashkortostan','Mariy-El','Mordovia','Tatarstan','Udmurt','Chuvash',
              'Kirov','Nizhegorod','Orenburg','Penza','Perm\'','Samara',
              'Saratov','Ul\'yanovsk'],
    'УФО':  ['Kurgan','Sverdlovsk','Tyumen\'','Khanty-Mansiy','Yamal-Nenets','Chelyabinsk'],
    'СФО':  ['Gorno-Altay','Altay','Buryat','Tuva','Khakass','Krasnoyarsk',
              'Irkutsk','Kemerovo','Novosibirsk','Omsk','Tomsk',"Zabaykal'ye"],
    'ДФО':  ['Sakha','Kamchatka',"Primor'ye",'Khabarovsk','Amur','Magadan',
              'Sakhalin','Yevrey','Chukot'],
}
REGION_TO_FD = {r: fd for fd, regions in FEDERAL_DISTRICTS.items() for r in regions}


def haversine(lat1, lon1, lat2, lon2):
    """Расстояние между двумя точками в км."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))


# ── 1. Загрузка школ ─────────────────────────────────────────────────────────
print('Загружаю школы...')
df = pd.read_csv(SCHOOLS_CSV)
df = df.dropna(subset=['lat', 'lon', 'region'])
print(f'  {len(df)} школ с регионом')

# ── 2. Расстояние до центра региона ──────────────────────────────────────────
print('Считаю расстояние до центра региона...')
def dist_to_center(row):
    center = REGION_CENTERS.get(row['region'])
    if center is None:
        return np.nan
    return haversine(row['lat'], row['lon'], center[0], center[1])

df['dist_to_region_center_km'] = df.apply(dist_to_center, axis=1)

# ── 3. Расстояние до Москвы ───────────────────────────────────────────────────
MOSCOW = (55.7558, 37.6176)
print('Считаю расстояние до Москвы...')
df['dist_to_moscow_km'] = df.apply(
    lambda r: haversine(r['lat'], r['lon'], MOSCOW[0], MOSCOW[1]), axis=1
)

# ── 4. Федеральный округ + числовой код ──────────────────────────────────────
FD_CODE = {'ЦФО':1,'СЗФО':2,'ЮФО':3,'СКФО':4,'ПФО':5,'УФО':6,'СФО':7,'ДФО':8}
df['federal_district'] = df['region'].map(REGION_TO_FD).fillna('Неизвестно')
df['federal_district_code'] = df['federal_district'].map(FD_CODE).fillna(0).astype(int)

# ── 4. GPS-плотность через сетку (быстро) ────────────────────────────────────
# Вместо 31К запросов — один запрос агрегирует всё по сетке 0.1° (~10 км),
# потом для каждой школы берём значение ближайшей ячейки.
GRID = 0.1  # градусов (~10 км)

print('Загружаю GPS-плотность по сетке (один запрос)...')
conn = sqlite3.connect(GPS_DB)
grid_df = pd.read_sql_query(f"""
    SELECT
        ROUND(x / {GRID}) * {GRID} AS grid_lon,
        ROUND(y / {GRID}) * {GRID} AS grid_lat,
        COUNT(*) AS gps_count
    FROM trafficpoints
    WHERE x BETWEEN 19 AND 192 AND y BETWEEN 41 AND 82
    GROUP BY grid_lon, grid_lat
""", conn)
conn.close()
print(f'  Сетка: {len(grid_df)} ячеек')

# Для каждой школы — ключ ближайшей ячейки
df['grid_lon'] = (df['lon'] / GRID).round() * GRID
df['grid_lat'] = (df['lat'] / GRID).round() * GRID
df = df.merge(grid_df, on=['grid_lon', 'grid_lat'], how='left')
df['gps_density_5km'] = df['gps_count'].fillna(0).astype(int)
df = df.drop(columns=['grid_lon', 'grid_lat', 'gps_count'])
print(f'  GPS-плотность рассчитана, покрытие: {(df["gps_density_5km"]>0).sum()}/{len(df)}')

# ── 5. Сохранение ─────────────────────────────────────────────────────────────
# ── 6. Добавить данные Росстат ────────────────────────────────────────────────
print('Добавляю данные Росстат...')
rosstat = pd.read_csv(ROSSTAT_CSV)
df = df.merge(rosstat[['region_gadm','grp_per_capita_2021','population_2021','urban_pct']],
              left_on='region', right_on='region_gadm', how='left').drop(columns='region_gadm')
print(f'  Покрытие ВРП: {df["grp_per_capita_2021"].notna().sum()} / {len(df)}')

feature_cols = ['osm_id', 'name', 'lat', 'lon', 'region', 'federal_district',
                'dist_to_region_center_km', 'dist_to_moscow_km',
                'federal_district_code', 'gps_density_5km',
                'grp_per_capita_2021', 'population_2021', 'urban_pct']
result = df[feature_cols]
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
result.to_csv(OUT_CSV, index=False, encoding='utf-8')

print(f'\n=== Готово ===')
print(f'Школ: {len(result)}')
print(f'Признаков: {len(feature_cols)-4}')
print(result[['region','dist_to_region_center_km','gps_density_5km']].describe())
print(f'\nСохранено: {OUT_CSV}')
