from possibleSiteSelection import select_possible_site
from subregions import export_subregions
from processSites import process_sites
from chatGPT_evaluate import evaluate_sites_with_chatgpt
import logging

def main():

    MODEL_NAME = "gpt-4o-mini"
    ADVICE_CSV = "site_advice.csv"

    aoi = [
        [-64, -10],
        [-54, -10],
        [-54,   0],
        [-64,   0]
    ]

    possible_sites = select_possible_site(aoi)

    # Print the results
    print("Possible sites selected:")
    for site in possible_sites:
        print(site)
    
    subregions_df = export_subregions()
    logging.info(f"Subregions created with {len(subregions_df)} entries.")

    final_sites = process_sites()
    logging.info(f"Processed sites with {len(final_sites)} entries.")

    logging.info(f"Final sites exported: {len(final_sites)} sites.")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    evaluate_sites_with_chatgpt(final_sites, model_name=MODEL_NAME, advice_csv=ADVICE_CSV)
    logging.info("Site evaluation completed.")

if __name__ == '__main__':
    main()
    logging.info("Pipeline completed successfully.")
