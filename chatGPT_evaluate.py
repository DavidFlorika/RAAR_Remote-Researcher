import logging
import pandas as pd
import os
from openai import OpenAI
import time

def authenticate_OpenAI():
    """Authenticate the OpenAI client."""
    global client
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("Missing OPENAI_API_KEY.")
    client = OpenAI(api_key=key)
    print(client.models.list())  # Test the connection
    logging.info("OpenAI authenticated.")

def evaluate_sites_with_chatgpt(sites_df, model_name, advice_csv):
    global client

    if sites_df.empty:
        logging.warning('No sites to evaluate.')
        return sites_df

    # ensure authenticated
    if client is None:
        authenticate_OpenAI()

    logging.info(f"Evaluating {len(sites_df)} sites with {model_name}...")
    advice_list = []
    for idx, row in sites_df.iterrows():
        prompt = f"""
Site {idx+1}:
  - Mean NDVI: {row['mean_ndvi']:.3f}
  - Mean Elevation: {row['mean_elev']:.1f} m
  - Compactness: {row['compactness']:.3f}

You are an expert in archaeology and remote sensing. Based on the above metrics for a site within the Amazon region, evaluate its potential as an archaeological site.

Please:
  1. Provide your reasoning based on elevation, NDVI, and compactness.
  2. Rate the site's archaeological potential on a scale of 1 to 10 (1 = very unlikely, 10 = highly likely).
  3. Give a brief summary of key considerations.
"""
        # retry loop for rate limits (no import of missing submodule)
        while True:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                advice_text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                err_str = str(e).lower()
                if 'rate limit' in err_str or '429' in err_str:
                    backoff_sec = 5
                    logging.warning(f"Rate limit hit. Retrying in {backoff_sec}s...")
                    time.sleep(backoff_sec)
                    continue
                else:
                    logging.error(f"Failed ChatGPT for site {idx+1}: {e}")
                    advice_text = '<error>'
                    break
        advice_list.append(advice_text)
        logging.info(f"Site {idx+1} advice received.")
        print(f"Site {idx+1} advice: {advice_text}")

    sites_df['advice'] = advice_list
    sites_df.to_csv(advice_csv, index=False)
    logging.info(f"Exported advice to {advice_csv}.")
    return sites_df

if __name__ == "__main__":

    client = None
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    MODEL_NAME = "gpt-4o-mini"
    ADVICE_CSV = "site_advice.csv"
    TOP_CSV = "top25_sites.csv"
    df = pd.read_csv(TOP_CSV)
    
    evaluate_sites_with_chatgpt(df, model_name=MODEL_NAME, advice_csv=ADVICE_CSV)
    logging.info("Site evaluation completed.")