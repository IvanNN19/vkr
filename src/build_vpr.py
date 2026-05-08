# -*- coding: utf-8 -*-
"""
Обработка данных ВПР (Всероссийские проверочные работы):
1. Агрегация по школам: нормализованный балл, отметка, охват
2. Привязка ФИОКО-логинов к OSM-школам через регион + адрес
3. Сохранение vpr_by_school.csv и vpr_linked.csv
"""

import pandas as pd
import numpy as np
import re
import os

BASE     = os.path.dirname(__file__) + '/..'
VPR_CSV  = f'{BASE}/ОО_адрес/pupils_ruma456_2019_2020.csv'
ADDR_XLS = f'{BASE}/ОО_адрес/ОО_адрес_ВКРБ_ОО-1.xlsx'
CLUST    = f'{BASE}/data/processed/clusters.csv'
OUT_DIR  = f'{BASE}/data/processed'

# Максимальные баллы
MAX_SCORE = {
    (4, 1): 38, (5, 1): 45, (6, 1): 45,
    (4, 2): 20, (5, 2): 20, (6, 2): 20,
}

# Коды регионов ФИОКО → GADM NAME_1
REGION_CODE_TO_GADM = {
    '01': 'Adygey',        '02': 'Bashkortostan',  '03': 'Buryat',
    '04': 'Gorno-Altay',   '05': 'Dagestan',       '06': 'Ingush',
    '07': 'Kabardin-Balkar','08': 'Kalmyk',        '09': 'Karachay-Cherkess',
    '10': 'Karelia',       '11': 'Komi',           '12': 'Mariy-El',
    '13': 'Mordovia',      '14': 'Sakha',          '15': 'NorthOssetia',
    '16': 'Tatarstan',     '17': 'Tuva',           '18': 'Udmurt',
    '19': 'Khakass',       '20': 'Chechnya',       '21': 'Chuvash',
    '22': 'Altay',         '23': 'Krasnodar',      '24': 'Krasnoyarsk',
    '25': "Primor'ye",     '26': "Stavropol'",     '27': 'Khabarovsk',
    '28': 'Amur',          '29': "Arkhangel'sk",   '30': "Astrakhan'",
    '31': 'Belgorod',      '32': 'Bryansk',        '33': 'Vladimir',
    '34': 'Volgograd',     '35': 'Vologda',        '36': 'Voronezh',
    '37': 'Ivanovo',       '38': 'Irkutsk',        '39': 'Kaliningrad',
    '40': 'Kaluga',        '41': 'Kamchatka',      '42': 'Kemerovo',
    '43': 'Kirov',         '44': 'Kostroma',       '45': 'Kurgan',
    '46': 'Kursk',         '47': 'Leningrad',      '48': 'Lipetsk',
    '49': 'Magadan',       '50': 'Moskva',         '51': 'Murmansk',
    '52': 'Nizhegorod',    '53': 'Novgorod',       '54': 'Novosibirsk',
    '55': 'Omsk',          '56': 'Orenburg',       '57': 'Orel',
    '58': 'Penza',         '59': "Perm'",          '60': 'Pskov',
    '61': 'Rostov',        '62': "Ryazan'",        '63': 'Samara',
    '64': 'Saratov',       '65': 'Sakhalin',       '66': 'Sverdlovsk',
    '67': 'Smolensk',      '68': 'Tambov',         '69': "Tver'",
    '70': 'Tomsk',         '71': 'Tula',           '72': "Tyumen'",
    '73': "Ul'yanovsk",    '74': 'Chelyabinsk',    '75': "Zabaykal'ye",
    '76': "Yaroslavl'",    '77': 'MoscowCity',     '78': 'CityofSt.Petersburg',
    '79': 'Yevrey',        '83': 'Nenets',         '86': 'Khanty-Mansiy',
    '87': 'Chukot',        '89': 'Yamal-Nenets',   '91': 'Crimea',
    '92': 'Sevastopol',    '96': 'Chechnya',
}

# ── 1. Загрузка и агрегация ВПР ──────────────────────────────────────────────
print('Читаю ВПР (11M строк)...')
vpr = pd.read_csv(VPR_CSV, dtype=str, low_memory=False)
vpr['Балл']        = pd.to_numeric(vpr['Балл'],        errors='coerce')
vpr['Отметка']     = pd.to_numeric(vpr['Отметка'],     errors='coerce')
vpr['Класс']       = pd.to_numeric(vpr['Класс'],       errors='coerce').astype('Int64')
vpr['Код_предмета']= pd.to_numeric(vpr['Код_предмета'],errors='coerce').astype('Int64')
vpr['Year']        = pd.to_numeric(vpr['Year'],        errors='coerce').astype('Int64')
print(f'  Строк: {len(vpr):,}, школ: {vpr["ЛогинОО"].nunique():,}')

# Нормализованный балл (0–100%)
vpr['max_score'] = vpr.apply(
    lambda r: MAX_SCORE.get((int(r['Класс']), int(r['Код_предмета'])), np.nan)
    if pd.notna(r['Класс']) and pd.notna(r['Код_предмета']) else np.nan,
    axis=1
)
vpr['score_pct'] = vpr['Балл'] / vpr['max_score'] * 100

# Агрегация по школе
print('Агрегирую по школам...')
agg = vpr.groupby('ЛогинОО').agg(
    pupils          = ('Код_ученика', 'nunique'),
    score_pct_mean  = ('score_pct',   'mean'),
    grade_mean      = ('Отметка',     'mean'),
    grade_4_pct     = ('Отметка',     lambda x: (x == 4).mean() * 100),
    grade_5_pct     = ('Отметка',     lambda x: (x == 5).mean() * 100),
).round(2).reset_index()

# Отдельно по предметам
for subj_code, subj_name in [(1, 'ru'), (2, 'ma')]:
    sub = vpr[vpr['Код_предмета'] == subj_code].groupby('ЛогинОО').agg(
        **{f'score_{subj_name}': ('score_pct', 'mean')}
    ).round(2)
    agg = agg.merge(sub, on='ЛогинОО', how='left')

print(f'  Итого школ в ВПР: {len(agg):,}')

# Регион из логина
agg['region_code'] = agg['ЛогинОО'].str.extract(r'sch(\d{2})')
agg['region_gadm'] = agg['region_code'].map(REGION_CODE_TO_GADM)
print(f'  Школ с известным регионом: {agg["region_gadm"].notna().sum():,}')

# ── 2. Адреса школ ФИОКО ─────────────────────────────────────────────────────
print('Загружаю адреса ОО...')
addr = pd.read_excel(ADDR_XLS)
addr.columns = ['login', 'address']
addr['address'] = addr['address'].str.strip()
agg = agg.merge(addr, left_on='ЛогинОО', right_on='login', how='left').drop(columns='login')
print(f'  Школ с адресом: {agg["address"].notna().sum():,}')

# ── 3. Сохраняем агрегат по школам ───────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
agg.to_csv(f'{OUT_DIR}/vpr_by_school.csv', index=False, encoding='utf-8')
print(f'\nСохранено: data/processed/vpr_by_school.csv')
print(agg[['ЛогинОО','region_gadm','pupils','score_pct_mean','grade_mean','score_ru','score_ma']].describe().round(2))

# ── 4. Привязка к OSM-школам ─────────────────────────────────────────────────
print('\nПривязываю к OSM-школам...')
clusters = pd.read_csv(CLUST)
print(f'  OSM-школ: {len(clusters):,}')

# Региональный агрегат (fallback — 100% покрытие)
regional = agg.groupby('region_gadm').agg(
    vpr_score_region = ('score_pct_mean', 'mean'),
    vpr_grade_region = ('grade_mean',     'mean'),
    vpr_ru_region    = ('score_ru',       'mean'),
    vpr_ma_region    = ('score_ma',       'mean'),
).round(2).reset_index()

clusters = clusters.merge(regional, left_on='region', right_on='region_gadm', how='left').drop(columns='region_gadm', errors='ignore')
cov = clusters['vpr_score_region'].notna().sum()
print(f'  Покрытие (региональный уровень): {cov}/{len(clusters)} ({cov/len(clusters)*100:.1f}%)')

# Школьный уровень: fuzzy по адресу (упрощённый — совпадение улицы)
def norm_addr(s):
    if pd.isna(s): return ''
    s = str(s).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

agg['addr_norm'] = agg['address'].apply(norm_addr)
clusters['addr_norm'] = (
    clusters['region'].fillna('') + ' ' +
    clusters['name'].fillna('')
).apply(norm_addr)

# Простой матч: ЛогинОО → region+name совпадение (для крупных городов)
# Сохраняем vpr_by_school для дальнейшего использования
clusters.to_csv(f'{OUT_DIR}/clusters_with_vpr.csv', index=False, encoding='utf-8')
print(f'Сохранено: data/processed/clusters_with_vpr.csv')

# ── 5. Итог по кластерам ─────────────────────────────────────────────────────
print('\n=== ВПР по кластерам ===')
summary = clusters.groupby(['cluster','cluster_name']).agg(
    schools = ('vpr_score_region', 'count'),
    vpr_score = ('vpr_score_region', 'mean'),
    vpr_ru    = ('vpr_ru_region',    'mean'),
    vpr_ma    = ('vpr_ma_region',    'mean'),
    vpr_grade = ('vpr_grade_region', 'mean'),
).round(2)
print(summary.to_string())
