#!/usr/bin/env python3
"""
process_subregions.py

This script reads subregions from a CSV, computes metric-based scores, and selects top K sites.
Enhancements:
- Projects geometries to EPSG:3857 for accurate area/perimeter
- Calculates compactness and composite anomaly score using NDVI, elevation, and shape metrics
- Uses logging instead of prints
- Exports top K candidates
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

# -------------------------------------------
# Configuration
# -------------------------------------------
INPUT_CSV = os.path.expanduser('~/Desktop/OpenAiToZ/subregions_evaluated.csv')
OUTPUT_CSV = os.path.expanduser('~/Desktop/OpenAiToZ/top25_sites.csv')
TOP_K = 25
# Metric weights
WEIGHTS = {
    'mean_ndvi': -1.0,   # lower NDVI more anomalous
    'mean_elev':  1.0,   # higher elevation more likely
    'compactness': 1.0   # more compact shapes preferred
}

# -------------------------------------------
# Logging setup
# -------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logging = logging.getlogging(__name__)

# -------------------------------------------
# Main processing function
# -------------------------------------------
def process_sites():
    # 1. Load data
    logging.info(f"Loading subregions from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Loaded {len(df)} subregions.")

    # 2. Parse geometries
    logging.info("Parsing GeoJSON geometries...")
    df['geometry_obj'] = df['geometry'].apply(lambda g: shape(json.loads(g)))

    # 3. Project to metric CRS
    logging.info("Projecting geometries to EPSG:3857...")
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    df['geom_proj'] = df['geometry_obj'].apply(lambda geom: transform(transformer.transform, geom))

    # 4. Compute shape metrics
    logging.info("Computing area and perimeter...")
    df['area_m2'] = df['geom_proj'].apply(lambda g: g.area)
    df['perimeter'] = df['geom_proj'].apply(lambda g: g.length)

    logging.info("Computing compactness (perimeter / sqrt(area))...")
    df['compactness'] = df['perimeter'] / np.sqrt(df['area_m2'])

    # 5. Normalize metrics and compute composite score
    logging.info("Normalizing metrics and computing composite score...")
    for metric, weight in WEIGHTS.items():
        if metric not in df.columns:
            logging.error(f"Metric '{metric}' not found in DataFrame.")
            return
        zcol = f"{metric}_z"
        df[zcol] = (df[metric] - df[metric].mean()) / df[metric].std(ddof=0)
    df['score'] = sum(df[f"{m}_z"] * w for m, w in WEIGHTS.items())
    logging.info("Composite scores computed.")

    # 6. Select top K candidates
    logging.info(f"Selecting top {TOP_K} sites by score...")
    top_df = df.nlargest(TOP_K, 'score').copy()
    logging.info(f"Selected {len(top_df)} sites.")

    # 7. Export results
    cols_to_drop = ['geometry_obj', 'geom_proj']
    top_df = top_df.drop(columns=cols_to_drop, errors='ignore')
    logging.info(f"Exporting top {len(top_df)} sites to {OUTPUT_CSV}...")
    top_df.to_csv(OUTPUT_CSV, index=False)
    logging.info("Process complete.")

    return top_df

if __name__ == '__main__':
    process_sites()