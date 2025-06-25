# Improved script to avoid Earth Engine timeouts by processing in subregions
import ee
from openai import OpenAI
import os
import logging

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def authenticate_earth_engine(project_id=None):
    proj = project_id or os.getenv("EARTHENGINE_PROJECT")
    if not proj: raise RuntimeError("Set EARTHENGINE_PROJECT or pass project_id.")
    ee.Authenticate()
    ee.Initialize(project=proj)
    logging.info(f"Earth Engine initialized with project: {proj}")


def authenticate_openai():
    key = os.getenv("OPENAI_API_KEY")
    if not key: raise EnvironmentError("Missing OPENAI_API_KEY.")
    OpenAI.api_key = key
    logging.info("OpenAI authenticated.")


def split_aoi(aoi, tile_size_deg=2.0):
    """
    Split a polygon AOI into a grid of smaller tiles (in degrees) to reduce per-request size.
    Returns list of ee.Geometry.Polygon tiles.
    """
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    min_lon, max_lon, min_lat, max_lat = min(lons), max(lons), min(lats), max(lats)
    tiles = []
    lon = min_lon
    while lon < max_lon:
        next_lon = min(lon + tile_size_deg, max_lon)
        lat = min_lat
        while lat < max_lat:
            next_lat = min(lat + tile_size_deg, max_lat)
            tile = ee.Geometry.Polygon([[lon, lat], [next_lon, lat], [next_lon, next_lat], [lon, next_lat]])
            tiles.append(tile.intersection(aoi, 1e-8))
            lat = next_lat
        lon = next_lon
    logging.info(f"Split AOI into {len(tiles)} tiles of ~{tile_size_deg}°")
    return tiles


def detect_in_tile(tile, collection, dem, ndvi_thresh, elev_thresh, min_area):
    """
    Run detection logic on a single tile and return features as Python dicts.
    """
    comp = collection.median().clip(tile)
    ndvi = comp.normalizedDifference(['B8','B4']).rename('NDVI')
    dem_clip = dem.select('elevation').clip(tile)

    mask = ndvi.lt(ndvi_thresh).And(dem_clip.gt(elev_thresh)).selfMask()
    vectors = mask.reduceToVectors(geometry=tile, scale=10, maxPixels=1e13)
    def add_stats(feat):
        stats = ndvi.addBands(dem_clip).reduceRegion(ee.Reducer.mean(), feat.geometry(), 10)
        area = feat.geometry().area(1)
        return feat.set({'mean_ndvi': stats.get('NDVI'), 'mean_elev': stats.get('elevation'), 'area_m2': area})
    stats_fc = vectors.map(add_stats)
    filtered = stats_fc.filter(ee.Filter.gt('area_m2', min_area))
    # Retrieve only geometry and properties
    info = filtered.getInfo()
    results = []
    for f in info.get('features', []):
        results.append({'geometry': f['geometry'], 'properties': f['properties']})
    return results


def main():
    authenticate_earth_engine()
    authenticate_openai()

    # Define large Amazon AOI
    amazon = ee.Geometry.Polygon([[-64,-10],[-54,-10],[-54,0],[-64,0]])
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate('2024-01-01','2024-12-31').filterBounds(amazon).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',10))
    dem = ee.Image('USGS/SRTMGL1_003')
    ndvi_thresh, elev_thresh, min_area = 0.3, 200, 10000

    # Split AOI into smaller tiles
    tiles = split_aoi(amazon, tile_size_deg=2.0)
    all_sites = []
    for idx, tile in enumerate(tiles, 1):
        logging.info(f"Processing tile {idx}/{len(tiles)}...")
        try:
            sites = detect_in_tile(tile, collection, dem, ndvi_thresh, elev_thresh, min_area)
            logging.info(f"Tile {idx}: found {len(sites)} sites")
            all_sites.extend(sites)
        except Exception as e:
            logging.warning(f"Tile {idx} failed: {e}")
    logging.info(f"Total candidate sites: {len(all_sites)}")

    # Analyze top N sites
    for i, site in enumerate(all_sites[:5], 1):
        logging.info(f"Analyzing site {i} with ChatGPT...")
        summary = OpenAI.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role':'user','content':f"Site properties: {site['properties']}"}]).choices[0].message.content
        print(f"\n--- Site {i} ---\n{summary}\n")

if __name__ == '__main__':
    main()
