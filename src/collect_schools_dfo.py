"""Доскачиваем пропущенные регионы ДФО отдельными запросами."""
import requests, pandas as pd, time, os

OUTPUT = os.path.dirname(__file__) + '/../data/raw/schools/schools_dfo_extra.csv'
OVERPASS_URL = 'https://overpass.kumi.systems/api/interpreter'

REGIONS = {
    'Primor_ye':  (42.0, 130.0, 48.5, 136.5),   # Приморский край
    'Sakhalin':   (46.0, 141.0, 55.0, 145.5),   # Сахалинская обл
    'Magadan':    (57.0, 147.0, 64.0, 157.0),   # Магаданская обл
    'Kamchatka':  (50.5, 155.0, 61.5, 165.0),   # Камчатский край
    'Chukotka':   (60.0, 160.0, 77.0, 192.0),   # Чукотский АО
}

def fetch(name, bbox, timeout=120):
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""
    [out:json][timeout:{timeout}];
    (
      node["amenity"="school"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["amenity"="school"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """
    for attempt in range(4):
        try:
            r = requests.post(OVERPASS_URL, data={'data': query}, timeout=timeout+20)
            r.raise_for_status()
            return r.json().get('elements', [])
        except Exception as e:
            print(f'  [{name}] попытка {attempt+1}/4: {e}')
            time.sleep(20)
    return []

def parse(el):
    tags = el.get('tags', {})
    if el['type'] == 'way':
        lat = el.get('center', {}).get('lat')
        lon = el.get('center', {}).get('lon')
    else:
        lat, lon = el.get('lat'), el.get('lon')
    return {
        'osm_id': el.get('id'), 'osm_type': el.get('type'),
        'lat': lat, 'lon': lon,
        'name': tags.get('name',''),
        'addr_region': tags.get('addr:region',''),
        'addr_city':   tags.get('addr:city',''),
        'addr_street': tags.get('addr:street',''),
        'addr_housenumber': tags.get('addr:housenumber',''),
        'operator': tags.get('operator',''),
        'capacity': tags.get('capacity',''),
    }

all_records = []
for region, bbox in REGIONS.items():
    print(f'Скачиваю {region}...', end=' ', flush=True)
    els = fetch(region, bbox)
    recs = [parse(e) for e in els if e.get('lat') or e.get('center')]
    all_records.extend(recs)
    print(f'{len(recs)} школ')
    time.sleep(10)

df = pd.DataFrame(all_records).drop_duplicates(subset='osm_id').dropna(subset=['lat','lon'])
df.to_csv(OUTPUT, index=False, encoding='utf-8')
print(f'\nГотово: {len(df)} школ → {OUTPUT}')
