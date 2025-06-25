import ee
from openai import OpenAI
import os
import logging
import sys
import csv
import json

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)

client = None

def authenticate_earth_engine(project_id=None):
    """Authenticate and initialize the Earth Engine API."""
    proj = project_id or os.getenv("EARTHENGINE_PROJECT")
    if not proj:
        raise RuntimeError("Set EARTHENGINE_PROJECT or pass project_id.")
    ee.Authenticate()
    ee.Initialize(project=proj)
    logging.info(f"Earth Engine initialized with project: {proj}")

def authenticate_OpenAI():
    """Authenticate the OpenAI client."""
    global client
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("Missing OPENAI_API_KEY.")
    client = OpenAI(api_key=key)
    print(client.models.list())  # Test the connection
    logging.info("OpenAI authenticated.")

def split_aoi(aoi, tile_size_deg=0.5):
    """
    Split a polygon AOI into a grid of smaller tiles (in degrees).
    Uses zero tolerance for intersection to avoid reprojection errors.
    Returns a list of ee.Geometry.Polygon tiles.
    """
    coords = aoi.bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    tiles = []
    lon = min_lon
    while lon < max_lon:
        next_lon = min(lon + tile_size_deg, max_lon)
        lat = min_lat
        while lat < max_lat:
            next_lat = min(lat + tile_size_deg, max_lat)
            tile = ee.Geometry.Polygon([
                [lon, lat],
                [next_lon, lat],
                [next_lon, next_lat],
                [lon, next_lat]
            ])
            # zero maxError avoids tiny reprojection tolerance errors
            tiles.append(tile.intersection(aoi, 0))
            lat = next_lat
        lon = next_lon

    logging.info(f"Split AOI into {len(tiles)} tiles of ~{tile_size_deg}°")
    return tiles

def detect_in_tile(tile, collection, dem, ndvi_thresh, elev_thresh, min_area,
                   vec_scale=100, tile_scale=4):
    """
    Process one tile: compute NDVI, mask, vectorize with tileScale, then
    compute per-feature stats and filter by area.
    Returns a list of features as dicts.
    """
    comp = collection.median().clip(tile)
    ndvi = comp.normalizedDifference(['B8', 'B4']).rename('NDVI')
    dem_clip = dem.select('elevation').clip(tile)

    mask = ndvi.lt(ndvi_thresh).And(dem_clip.gt(elev_thresh)).selfMask()

    vectors = mask.reduceToVectors(
        geometry=tile,
        scale=vec_scale,
        maxPixels=1e13,
        tileScale=tile_scale,
        bestEffort=True,
        geometryType='polygon',
        eightConnected=False
    )

    raw_count = vectors.size().getInfo()
    logging.info(f"  raw vectors in tile: {raw_count}")

    def add_stats(feat):
        stats = ndvi.addBands(dem_clip).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feat.geometry(),
            scale=vec_scale,
            maxPixels=1e13,
            tileScale=tile_scale
        )
        return feat.set({
            'mean_ndvi': stats.get('NDVI'),
            'mean_elev': stats.get('elevation'),
            'area_m2': feat.geometry().area(1)
        })

    stats_fc = vectors.map(add_stats)
    filtered = stats_fc.filter(ee.Filter.gt('area_m2', min_area))

    info = filtered.getInfo()
    results = []
    for f in info.get('features', []):
        results.append({
            'geometry': f['geometry'],
            'properties': f['properties']
        })
    return results

def export_sites_to_csv(sites, path='candidate_sites.csv'):
    """
    Writes a CSV with one row per region. Columns:  
      - geometry: the GeoJSON geometry as a string  
      - <each property> as its own column
    """
    if not sites:
        print("No sites to export.")
        return

    # Determine all property names (in case they vary)
    props = sorted({k for s in sites for k in s['properties'].keys()})

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['geometry'] + props)
        # Rows
        for s in sites:
            geom = s['geometry']  # a dict
            geom_str = json.dumps(geom)
            row = [geom_str] + [s['properties'].get(p) for p in props]
            writer.writerow(row)

    print(f"Wrote {len(sites)} sites to {path}")

def select_possible_site(aoi):
    authenticate_earth_engine()
    authenticate_OpenAI()

    # User parameters
    tile_size_deg = 0.5
    ndvi_thresh   = 0.3      # NDVI below this marks candidate
    elev_thresh   = 200      # elevation above this marks candidate
    min_area      = 10_000   # in m²
    vec_scale     = 1000      # meters per pixel for vectorization
    tile_scale    = 4        # internal aggregation factor

    # Define your Amazon AOI
    amazon = ee.Geometry.Polygon(aoi)

    # Build Sentinel-2 composite
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate('2024-01-01', '2024-12-31')
        .filterBounds(amazon)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    )
    dem = ee.Image('USGS/SRTMGL1_003')

    tiles = split_aoi(amazon, tile_size_deg)
    all_sites = []

    for idx, tile in enumerate(tiles, start=1):
        logging.info(f"Processing tile {idx}/{len(tiles)}...")
        try:
            sites = detect_in_tile(
                tile, collection, dem,
                ndvi_thresh, elev_thresh, min_area,
                vec_scale, tile_scale
            )
            logging.info(f"Tile {idx}: found {len(sites)} sites")
            all_sites.extend(sites)
        except Exception as e:
            logging.warning(f"Tile {idx} failed: {e}")

    logging.info(f"Total candidate sites: {len(all_sites)}")

    export_sites_to_csv(all_sites, "candidate_sites.csv")

    return all_sites
