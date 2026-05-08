# -*- coding: utf-8 -*-
"""
Фильтрация школ в clusters.csv:
1. Убрать безымянные (NaN / пустое название)
2. Убрать не-школы по ключевым словам (музыкальные, детсады, спортшколы, вузы и пр.)
3. Убрать дубли (одинаковое название в радиусе 0.1°)
"""

import pandas as pd, os

BASE   = os.path.dirname(__file__) + '/..'
CSV    = f'{BASE}/data/processed/clusters.csv'

# ──────────────────────────────────────────────
# Ключевые слова для удаления
# ──────────────────────────────────────────────
BLACKLIST = [
    # Детсады и дошкольное
    'детский сад', 'детсад', ' д/с', 'д/с ', 'мдоу', 'мбдоу',
    'ясли', 'ясел', 'дошкольн',
    # Музыкальные / танцевальные
    'музыкальн', ' дмш', 'дмш №', 'дши ', 'дши №',
    'хореограф', 'танцевальн',
    # Спортивные
    'дюсш', 'сдюшор', 'спортивн школ', 'спортшкол',
    # Художественные (только "художественная школа", не "художественный лицей")
    'художественная школа', 'художня школа', ' дхш',
    # Колледжи, техникумы, вузы
    'техникум', 'колледж', ' институт', 'университет', 'академия',
    # Прочее
    'детский дом', 'дом пионер', 'центр творч', 'клуб юных',
]

# ──────────────────────────────────────────────
df = pd.read_csv(CSV)
n0 = len(df)
print(f'Исходно: {n0} записей')

# Шаг 1: убрать без названия
df = df.dropna(subset=['name'])
df = df[df['name'].str.strip() != '']
print(f'После удаления безымянных: {len(df)} (-{n0 - len(df)})')

# Шаг 2: убрать не-школы
name_lower = df['name'].str.lower()
mask_bad = pd.Series(False, index=df.index)
for kw in BLACKLIST:
    mask_bad |= name_lower.str.contains(kw, regex=False, na=False)

n_before = len(df)
df = df[~mask_bad]
print(f'После фильтра не-школ: {len(df)} (-{n_before - len(df)})')

# Шаг 3: убрать дубли (одинаковое имя в ячейке 0.1°)
n_before = len(df)
df['_name_key'] = df['name'].str.lower().str.strip()
df['_grid_lat'] = (df['lat'] / 0.1).astype(int)
df['_grid_lon'] = (df['lon'] / 0.1).astype(int)

df = df.drop_duplicates(subset=['_name_key', '_grid_lat', '_grid_lon'], keep='first')
df = df.drop(columns=['_name_key', '_grid_lat', '_grid_lon'])
print(f'После дедупликации (0.1°): {len(df)} (-{n_before - len(df)})')

print(f'\nИтого удалено: {n0 - len(df)} записей ({(n0-len(df))/n0*100:.1f}%)')
print(f'Осталось: {len(df)} школ')
print()
print('По кластерам:')
print(df.groupby(['cluster', 'cluster_name']).size().reset_index(name='n').to_string(index=False))

df.to_csv(CSV, index=False, encoding='utf-8')
print(f'\nСохранено: {CSV}')
