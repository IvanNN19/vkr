import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import zipfile, os

BASE     = os.path.dirname(__file__) + '/..'
OSM_CSV  = f'{BASE}/data/raw/schools/schools_osm.csv'
DFO_CSV  = f'{BASE}/data/raw/schools/schools_dfo_extra.csv'
GADM_ZIP = f'{BASE}/data/raw/osm/gadm41_RUS_1.zip'
OUT_CSV  = f'{BASE}/data/processed/schools_with_regions.csv'

# Точный маппинг GADM NAME_1 → федеральный округ
REGION_TO_FD = {
    # ЦФО
    'Belgorod':'ЦФО','Bryansk':'ЦФО','Vladimir':'ЦФО','Voronezh':'ЦФО',
    'Ivanovo':'ЦФО','Kaluga':'ЦФО','Kostroma':'ЦФО','Kursk':'ЦФО',
    'Lipetsk':'ЦФО','Moskva':'ЦФО','MoscowCity':'ЦФО','Orel':'ЦФО',
    "Ryazan'":'ЦФО','Smolensk':'ЦФО','Tambov':'ЦФО',"Tver'":'ЦФО',
    'Tula':'ЦФО',"Yaroslavl'":'ЦФО',
    # СЗФО
    'Karelia':'СЗФО','Komi':'СЗФО',"Arkhangel'sk":'СЗФО','Nenets':'СЗФО',
    'Vologda':'СЗФО','Kaliningrad':'СЗФО','Leningrad':'СЗФО',
    'Murmansk':'СЗФО','Novgorod':'СЗФО','Pskov':'СЗФО',
    'CityofSt.Petersburg':'СЗФО',
    # ЮФО
    'Adygey':'ЮФО','Kalmyk':'ЮФО','Krasnodar':'ЮФО',"Astrakhan'":'ЮФО',
    'Volgograd':'ЮФО','Rostov':'ЮФО',
    # СКФО
    'Dagestan':'СКФО','Ingush':'СКФО','Kabardin-Balkar':'СКФО',
    'Karachay-Cherkess':'СКФО','NorthOssetia':'СКФО','Chechnya':'СКФО',
    "Stavropol'":'СКФО',
    # ПФО
    'Bashkortostan':'ПФО','Mariy-El':'ПФО','Mordovia':'ПФО',
    'Tatarstan':'ПФО','Udmurt':'ПФО','Chuvash':'ПФО','Kirov':'ПФО',
    'Nizhegorod':'ПФО','Orenburg':'ПФО','Penza':'ПФО',"Perm'":'ПФО',
    'Samara':'ПФО','Saratov':'ПФО',"Ul'yanovsk":'ПФО',
    # УФО
    'Kurgan':'УФО','Sverdlovsk':'УФО',"Tyumen'":'УФО',
    'Khanty-Mansiy':'УФО','Yamal-Nenets':'УФО','Chelyabinsk':'УФО',
    # СФО
    'Gorno-Altay':'СФО','Altay':'СФО','Buryat':'СФО','Tuva':'СФО',
    'Khakass':'СФО','Krasnoyarsk':'СФО','Irkutsk':'СФО','Kemerovo':'СФО',
    'Novosibirsk':'СФО','Omsk':'СФО','Tomsk':'СФО',"Zabaykal'ye":'СФО',
    # ДФО
    "Primor'ye":'ДФО','Khabarovsk':'ДФО','Amur':'ДФО','Magadan':'ДФО',
    'Sakhalin':'ДФО','Yevrey':'ДФО','Chukot':'ДФО','Sakha':'ДФО',
    'Kamchatka':'ДФО',
}

# 1. Объединить основной OSM + доп. ДФО
print('Загружаю школы...')
df = pd.read_csv(OSM_CSV)
if os.path.exists(DFO_CSV):
    dfo = pd.read_csv(DFO_CSV)
    df = pd.concat([df, dfo], ignore_index=True).drop_duplicates(subset='osm_id')
    print(f'  После добавления ДФО: {len(df)} школ')
else:
    print(f'  {len(df)} школ (ДФО-файл не найден)')

# 2. Загрузить границы регионов
print('Загружаю границы регионов GADM...')
with zipfile.ZipFile(GADM_ZIP) as z:
    fname = [f for f in z.namelist() if f.endswith('.json')][0]
    with z.open(fname) as f:
        regions = gpd.read_file(f).to_crs('EPSG:4326')
print(f'  {len(regions)} регионов')

# 3. GeoDataFrame школ
print('Создаю геоданные...')
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])],
    crs='EPSG:4326'
)

# 4. Spatial join — сначала within, потом nearest для непопавших
print('Spatial join (within)...')
joined = gpd.sjoin(gdf, regions[['NAME_1','geometry']], how='left', predicate='within')
joined = joined.rename(columns={'NAME_1': 'region'}).drop(columns=['index_right'], errors='ignore')

no_region = joined['region'].isna().sum()
print(f'  Без региона после within: {no_region}')

if no_region > 0:
    print('  Spatial join (nearest) для оставшихся...')
    regions_proj = regions.to_crs('EPSG:3857')
    missing_idx = joined[joined['region'].isna()].index
    missing = gdf.loc[missing_idx].copy().to_crs('EPSG:3857')
    nearest = gpd.sjoin_nearest(missing, regions_proj[['NAME_1','geometry']], how='left')
    nearest = nearest.rename(columns={'NAME_1': 'region_nearest'})
    joined.loc[missing_idx, 'region'] = nearest['region_nearest'].values
    print(f'  Без региона после nearest: {joined["region"].isna().sum()}')

# 5. Добавить федеральный округ
joined['federal_district'] = joined['region'].map(REGION_TO_FD).fillna('Неизвестно')

# Отчёт
unknown = joined[joined['federal_district'] == 'Неизвестно']['region'].value_counts()
if len(unknown):
    print(f'  Регионы без округа: {unknown.to_dict()}')

# 6. Сохранить
joined = joined.drop(columns=['geometry'], errors='ignore')
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
joined.to_csv(OUT_CSV, index=False, encoding='utf-8')

print(f'\n=== Готово ===')
print(f'Школ: {len(joined)}')
print(f'С регионом: {joined["region"].notna().sum()}')
print(f'\nПо федеральным округам:')
print(joined.groupby('federal_district').size().sort_values(ascending=False))
print(f'\nСохранено: {OUT_CSV}')
