"""
Кластеризация школ по экономическим и территориальным признакам.
Методы: K-means (основной) + иерархическая (для проверки).
Подбор числа кластеров: метод локтя + силуэтный коэффициент.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os, warnings
warnings.filterwarnings('ignore')

BASE    = os.path.dirname(__file__) + '/..'
IN_CSV  = f'{BASE}/data/processed/features.csv'
OUT_CSV = f'{BASE}/data/processed/clusters.csv'
OUT_DIR = f'{BASE}/output'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Загрузка и очистка ─────────────────────────────────────────────────────
print('Загружаю данные...')
df = pd.read_csv(IN_CSV)

FEATURE_COLS = [
    'dist_to_region_center_km',   # территориальный: удалённость от центра региона
    'dist_to_moscow_km',          # территориальный: удалённость от Москвы
    'federal_district_code',      # территориальный: федеральный округ (1-8)
    'gps_density_5km',            # транспортный: активность трафика
    'grp_per_capita_2021',        # экономический: ВРП на душу
    'population_2021',            # экономический: население региона
    'urban_pct',                  # экономический: урбанизация
]

df_clean = df.dropna(subset=FEATURE_COLS).copy()
print(f'  Школ после удаления пропусков: {len(df_clean)} / {len(df)}')

X = df_clean[FEATURE_COLS].values

# ── 2. Нормализация ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f'  Признаков: {len(FEATURE_COLS)}')

# ── 3. Подбор числа кластеров ─────────────────────────────────────────────────
print('\nПодбираю число кластеров (2–12)...')
inertias   = []
silhouettes = []
K_RANGE = range(2, 13)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
    silhouettes.append(sil)
    print(f'  k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.3f}')

# График метода локтя + силуэт
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(K_RANGE), inertias, 'bo-', linewidth=2)
ax1.set_xlabel('Число кластеров k')
ax1.set_ylabel('Инерция (SSE)')
ax1.set_title('Метод локтя')
ax1.grid(True, alpha=0.3)

ax2.plot(list(K_RANGE), silhouettes, 'rs-', linewidth=2)
ax2.set_xlabel('Число кластеров k')
ax2.set_ylabel('Силуэтный коэффициент')
ax2.set_title('Силуэтный анализ')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\n  График сохранён: output/elbow_silhouette.png')

# ── 4. Финальная кластеризация (k=9) ─────────────────────────────────────────
BEST_K = 9
# Названия уточняются после анализа профилей
CLUSTER_NAMES = {i: f'Кластер {i+1}' for i in range(BEST_K)}
print(f'\nКластеризация K-means с k={BEST_K}...')
km_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=20)
df_clean['cluster'] = km_final.fit_predict(X_scaled)

# ── 5. Описание кластеров ─────────────────────────────────────────────────────
print('\n=== Профили кластеров ===')
cluster_profile = df_clean.groupby('cluster')[FEATURE_COLS].mean().round(1)
cluster_profile['count'] = df_clean.groupby('cluster').size()
print(cluster_profile.to_string())

# ── 6. PCA для визуализации ───────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.tab10(np.linspace(0, 1, BEST_K))
for i in range(BEST_K):
    mask = df_clean['cluster'] == i
    cnt = mask.sum()
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[colors[i]], alpha=0.4, s=8, label=f'Кластер {i+1} (n={cnt})')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title(f'PCA-проекция кластеров (k={BEST_K})')
ax.legend(loc='best', fontsize=8, markerscale=3)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/pca_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  PCA-график сохранён: output/pca_clusters.png')

# ── 7. Сохранение результатов ─────────────────────────────────────────────────
df_clean['cluster_name'] = df_clean['cluster'].map(CLUSTER_NAMES)
result = df_clean[['osm_id', 'name', 'lat', 'lon', 'region', 'federal_district', 'cluster', 'cluster_name'] + FEATURE_COLS]
result.to_csv(OUT_CSV, index=False, encoding='utf-8')

print(f'\n=== Готово ===')
print(f'Результат: {OUT_CSV}')
print(f'Распределение по кластерам:')
print(df_clean['cluster'].value_counts().sort_index().to_string())
