from possibleSiteSelection import select_possible_site
from subregions import export_subregions
from processSites import process_sites
from openai import OpenAI
import os
import logging

def authenticate_OpenAI():
    """Authenticate the OpenAI client."""
    global client
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("Missing OPENAI_API_KEY.")
    client = OpenAI(api_key=key)
    print(client.models.list())  # Test the connection
    logging.info("OpenAI authenticated.")


def main():

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

if __name__ == '__main__':
    main()
    logging.info("Pipeline completed successfully.")
