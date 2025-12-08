import requests
import pandas as pd
import re
from PIL import Image
from io import BytesIO
import mercantile

def get_best_prefire_layer(fire_date="2025-01-07"):
    """
    Finds the best available Esri Wayback imagery layer before the fire date.
    """
    wayback_config_url = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
    
    print("Fetching Wayback Catalog...")
    try:
        r = requests.get(wayback_config_url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise Exception(f"Failed to connect to Esri: {e}")
    
    versions = []
    # Regex to find "Wayback 2024-01-01" in the title
    date_pattern = re.compile(r"Wayback (\d{4}-\d{2}-\d{2})")
    
    for key, val in data.items():
        # 1. Try explicit keys first (legacy support)
        date_str = val.get('releaseDatme') or val.get('releaseDate')
        
        # 2. Fallback: Extract from Title
        if not date_str and 'itemTitle' in val:
            match = date_pattern.search(val['itemTitle'])
            if match:
                date_str = match.group(1)
        
        if date_str:
            versions.append({
                'id': key,
                'date': pd.to_datetime(date_str),
                'url_template': val['itemURL']
            })
            
    if not versions:
        raise Exception("Could not extract dates. Pattern match failed.")
        
    df_v = pd.DataFrame(versions)
    
    # Filter: Strictly BEFORE the fire
    fire_ts = pd.to_datetime(fire_date)
    pre_fire = df_v[df_v['date'] < fire_ts].sort_values('date', ascending=False)
    
    if pre_fire.empty:
        raise Exception(f"No maps found before {fire_date}.")
        
    best_layer = pre_fire.iloc[0]
    print(f"✅ Selected Vintage: {best_layer['date'].date()} (ID: {best_layer['id']})")
    
    # Construct the Direct Tile URL Template
    base_url = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile"
    final_url_template = f"{base_url}/{best_layer['id']}/{{z}}/{{y}}/{{x}}"
    
    return final_url_template

def download_stitched_home(lat, lon, layer_url, zoom=20, crop_size=450):
    """
    Downloads tiles, stitches them, and crops to the home location.
    """
    # 1. Fetch 3x3 Grid (Source Data)
    def fetch_grid(target_zoom):
        center_tile = mercantile.tile(lon, lat, target_zoom)
        stitched = Image.new('RGB', (256*3, 256*3))
        tiles_found = 0
        
        for i, dx in enumerate([-1, 0, 1]):
            for j, dy in enumerate([-1, 0, 1]):
                tile = mercantile.Tile(center_tile.x + dx, center_tile.y + dy, target_zoom)
                url = layer_url.format(z=target_zoom, y=tile.y, x=tile.x)
                
                try:
                    r = requests.get(url, timeout=3)
                    if r.status_code == 200:
                        tile_img = Image.open(BytesIO(r.content))
                        stitched.paste(tile_img, (i*256, j*256))
                        tiles_found += 1
                except:
                    pass
        return stitched, tiles_found

    # Try Zoom 20 first (Max Detail). Fallback to 19.
    img, count = fetch_grid(zoom)
    if count < 5 and zoom > 18:
        img, count = fetch_grid(zoom - 1)

    # 2. The "Sweet Spot" Crop
    # Center of 768x768 canvas is (384, 384)
    half = crop_size // 2
    left = 384 - half
    top = 384 - half
    right = 384 + half
    bottom = 384 + half
    
    tight_crop = img.crop((left, top, right, bottom))
    
    # Resize to 640x640 (Standard input for Neural Nets)
    final_img = tight_crop.resize((640, 640), Image.Resampling.LANCZOS)
    
    return final_img

