from possibleSiteSelection import select_possible_site

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

if __name__ == '__main__':
    main()
