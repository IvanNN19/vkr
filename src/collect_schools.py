import requests
import pandas as pd
import time
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'schools', 'schools_osm.csv')
OVERPASS_URL = 'https://overpass.kumi.systems/api/interpreter'

FEDERAL_DISTRICTS = {
    'ЦФО':  (50.0, 31.0, 57.5, 41.5),
    'СЗФО': (54.0, 26.5, 70.0, 62.0),
    'ЮФО':  (43.5, 36.5, 50.0, 47.0),
    'СКФО': (43.0, 40.5, 47.0, 49.5),
    'ПФО':  (51.0, 44.0, 62.0, 59.0),
    'УФО':  (54.0, 58.0, 72.0, 69.0),
    'СФО':  (49.0, 68.0, 74.0, 99.0),
    'ДФО-запад': (42.0, 99.0, 77.0, 141.0),
    'ДФО-восток':(60.0, 141.0, 77.0, 192.0),
}

def fetch_schools(name, bbox, timeout=90, retries=3):
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""
    [out:json][timeout:{timeout}];
    (
      node["amenity"="school"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["amenity"="school"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """
    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={'data': query}, timeout=timeout+15)
            resp.raise_for_status()
            return resp.json().get('elements', [])
        except Exception as e:
            print(f'  [{name}] Попытка {attempt+1}/{retries}: {e}')
            time.sleep(15)
    return []

def parse_element(el):
    tags = el.get('tags', {})
    if el['type'] == 'way':
        lat = el.get('center', {}).get('lat')
        lon = el.get('center', {}).get('lon')
    else:
        lat = el.get('lat')
        lon = el.get('lon')
    return {
        'osm_id':           el.get('id'),
        'osm_type':         el.get('type'),
        'lat':              lat,
        'lon':              lon,
        'name':             tags.get('name', ''),
        'addr_region':      tags.get('addr:region', ''),
        'addr_city':        tags.get('addr:city', ''),
        'addr_street':      tags.get('addr:street', ''),
        'addr_housenumber': tags.get('addr:housenumber', ''),
        'operator':         tags.get('operator', ''),
        'capacity':         tags.get('capacity', ''),
    }

all_records = []
for district, bbox in FEDERAL_DISTRICTS.items():
    print(f'Скачиваю {district}...', end=' ', flush=True)
    elements = fetch_schools(district, bbox)
    records = [parse_element(el) for el in elements]
    records = [r for r in records if r['lat'] is not None]
    all_records.extend(records)
    print(f'{len(records)} школ')
    time.sleep(8)

df = pd.DataFrame(all_records)
df = df.drop_duplicates(subset='osm_id')
df = df.dropna(subset=['lat', 'lon'])
df = df[(df['lat'].between(41, 82)) & (df['lon'].between(19, 192))]

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

print(f'\n=== Готово ===')
print(f'Всего школ: {len(df)}')
print(f'Сохранено: {OUTPUT_PATH}')
