#!/usr/bin/env python3
"""
subregions_evaluation.py

Cleaned end-to-end pipeline with top-N selection:
1. Authenticate Earth Engine
2. Load candidate sites
3. Subdivide all sites into 100m×100m cells
4. Build combined NDVI+DEM image
5. Bulk compute NDVI/elevation for all subcells
6. Load results into pandas and compute anomaly scores
7. Select top 300 most anomalous subregions
8. Export
9. Detailed logging
"""
import os
import json
import pandas as pd
import numpy as np
import ee
import logging
from shapely.geometry import shape, mapping, box
from possibleSiteSelection import authenticate_earth_engine

# Configuration
INPUT_CSV = os.path.expanduser("~/Desktop/OpenAiToZ/candidate_sites.csv")
CELL_SIZE = 100.0  # meters
OUTPUT_CSV = os.path.expanduser("~/Desktop/OpenAiToZ/subregions_evaluated.csv")
TOP_K = 300

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Subdivide geometry into grid cells
def subdivide_geometry(geom, size):
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(minx, maxx, size)
    ys = np.arange(miny, maxy, size)
    cells = []
    for x in xs:
        for y in ys:
            cell = box(x, y, x + size, y + size)
            inter = geom.intersection(cell)
            if not inter.is_empty:
                cells.append(inter)
    return cells

# Build NDVI+DEM composite
def build_combined_image():
    ndvi = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate('2024-01-01', '2024-12-31')
            .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
            .median())
    dem = ee.Image('USGS/SRTMGL1_003').select('elevation')
    return ndvi.addBands(dem)

def export_subregions():
    # 1. Authenticate Earth Engine
    logging.info("Authenticating Earth Engine...")
    authenticate_earth_engine()

    # 2. Load candidates
    logging.info(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    df['geom'] = df['geometry'].apply(lambda v: shape(json.loads(v)))
    logging.info(f"Loaded {len(df)} sites.")

    # 3. Subdivide all sites
    logging.info("Subdividing sites into cells...")
    ee_feats = []
    for idx, row in df.iterrows():
        subs = subdivide_geometry(row['geom'], CELL_SIZE)
        logging.info(f"Site {idx+1}: {len(subs)} subcells")
        for i, sub in enumerate(subs):
            props = row.drop(['geometry', 'geom']).to_dict()
            props.update({'site_index': idx, 'subcell_id': i})
            ee_feats.append(ee.Feature(ee.Geometry(mapping(sub)), props))
    logging.info(f"Total subcells: {len(ee_feats)}")

    # 4. Build composite image
    combined = build_combined_image()

    # 5. Bulk reduceRegions
    logging.info("Running reduceRegions on all subcells...")
    fc = ee.FeatureCollection(ee_feats)
    stats_fc = combined.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=CELL_SIZE,
        tileScale=16
    )

    # 6. Retrieve stats
    logging.info("Retrieving stats to DataFrame...")
    features = stats_fc.getInfo().get('features', [])
    if not features:
        logging.error("No features returned from reduceRegions. Exiting.")
        return
    records = []
    for f in features:
        p = f['properties']
        rec = {k: v for k, v in p.items() if k not in ['NDVI', 'elevation']}
        rec['NDVI'] = p.get('NDVI') or p.get('NDVI_mean')
        rec['elevation'] = p.get('elevation') or p.get('elevation_mean')
        rec['geometry'] = json.dumps(f['geometry'])
        records.append(rec)
    stats_df = pd.DataFrame(records)
    logging.info(f"Stats DataFrame shape: {stats_df.shape}")

    # 7. Compute anomaly score and select top K
    # Z-score normalize
    stats_df['ndvi_z'] = (stats_df['NDVI'] - stats_df['NDVI'].mean()) / stats_df['NDVI'].std()
    stats_df['elev_z'] = (stats_df['elevation'] - stats_df['elevation'].mean()) / stats_df['elevation'].std()
    # Composite score: lower NDVI and higher elevation are more anomalous
    stats_df['score'] = -stats_df['ndvi_z'] + stats_df['elev_z']
    # Select top K
    top_df = stats_df.nlargest(TOP_K, 'score')
    logging.info(f"Selected top {len(top_df)} subregions by anomaly score.")

    # 8. Export
    top_df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Exported top {len(top_df)} records to {OUTPUT_CSV}.")

    return top_df

if __name__ == '__main__':
    export_subregions()