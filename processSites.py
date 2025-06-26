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
logger = logging.getLogger(__name__)

# -------------------------------------------
# Main processing function
# -------------------------------------------
def process_sites():
    # 1. Load data
    logger.info(f"Loading subregions from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} subregions.")

    # 2. Parse geometries
    logger.info("Parsing GeoJSON geometries...")
    df['geometry_obj'] = df['geometry'].apply(lambda g: shape(json.loads(g)))

    # 3. Project to metric CRS
    logger.info("Projecting geometries to EPSG:3857...")
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    df['geom_proj'] = df['geometry_obj'].apply(lambda geom: transform(transformer.transform, geom))

    # 4. Compute shape metrics
    logger.info("Computing area and perimeter...")
    df['area_m2'] = df['geom_proj'].apply(lambda g: g.area)
    df['perimeter'] = df['geom_proj'].apply(lambda g: g.length)

    logger.info("Computing compactness (perimeter / sqrt(area))...")
    df['compactness'] = df['perimeter'] / np.sqrt(df['area_m2'])

    # 5. Normalize metrics and compute composite score
    logger.info("Normalizing metrics and computing composite score...")
    for metric, weight in WEIGHTS.items():
        if metric not in df.columns:
            logger.error(f"Metric '{metric}' not found in DataFrame.")
            return
        zcol = f"{metric}_z"
        df[zcol] = (df[metric] - df[metric].mean()) / df[metric].std(ddof=0)
    df['score'] = sum(df[f"{m}_z"] * w for m, w in WEIGHTS.items())
    logger.info("Composite scores computed.")

    # 6. Select top K candidates
    logger.info(f"Selecting top {TOP_K} sites by score...")
    top_df = df.nlargest(TOP_K, 'score').copy()
    logger.info(f"Selected {len(top_df)} sites.")

    # 7. Export results
    cols_to_drop = ['geometry_obj', 'geom_proj']
    top_df = top_df.drop(columns=cols_to_drop, errors='ignore')
    logger.info(f"Exporting top {len(top_df)} sites to {OUTPUT_CSV}...")
    top_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Process complete.")

    return top_df

if __name__ == '__main__':
    process_sites()