# Archeological Sites Search

---

## 1. Overview

**This project was originally intented for the Kaggle [OpenAi A-to-Z Challenge](https://www.kaggle.com/competitions/openai-to-z-challenge) Competition.**

This project uses GEE(Google Earth Engine) and ChatGPT to search over the Amazonia area in finding possible archeological sites. We used *gpt-4o-mini* in getting acheological feedback for computed data.

GEE Data set was used to compute NDVI(Index for plant richness) and DEM(Elevation) to possibly select candidate sites.

*Note:* Archeological Sites Search is a premature attempt to use ChatGPT combined with GEE to process geological data and investigate possible usage of AI in aarcheological studies.


**Reference**:

Peripato, V., Levis, C., Moreira, G., Gamerman, D., Steege, H., Pitman, N., et al. (2023). More than 10,000 pre-Columbian earthworks are still hidden throughout Amazonia. Science. https://www.science.org/doi/10.1126/science.ade2541

Google Earth Engine. https://earthengine.google.com.

### 1.1 Competition Description

**Competition Description:**
_"We challenge you to bring legends to life by finding previously unknown archaeological site(s), using available open-source data. Findings should be reasonably bound by the Amazon biome in Northern South America. Focus on Brazil, with allowed extension into the outskirts of Bolivia, Columbia, Ecuador, Guyana, Peru, Suriname, Venezuela, and French Guiana."_

### 1.2 Requirements

#### 1.2.1 Setting Up the Keys

**How to add keys?**

- Go to [https://github.com/DavidFlorika/AmazonArcheologicalSiteSearch/blob/main/environment_keys_tutorial.pdf](https://github.com/DavidFlorika/AmazonArcheologicalSiteSearch/blob/main/environment_keys_tutorial.pdf) for a documented tutorial for both Windows and Mac

**How to get ChatGPT API?**

- Go to [https://www.merge.dev/blog/chatgpt-api-key](https://www.merge.dev/blog/chatgpt-api-key)

**How to set up Google Earth Engine project?**

- Go to [https://github.com/DavidFlorika/AmazonArcheologicalSiteSearch/blob/main/gee_project_id_tutorial.pdf](https://github.com/DavidFlorika/AmazonArcheologicalSiteSearch/blob/main/gee_project_id_tutorial.pdf)

#### 1.2.2 Downloading Requirements

Download all requirements by pasting the following code into your terminal:
`pip install -r requirements.txt`

## 2. Workflow

**Area of Interest (AOI)**

Our area of interest lies in north-western Brazil and covers roughly 1 233 204 km². It is defined by the following bounding polygon (longitude, latitude):
`
aoi = [
        [-64, -10],
        [-54, -10],
        [-54,   0],
        [-64,   0]
    ]
`

This project is structured to follow a process that ensures reproducibility, scalability, and interpretability. Tiling: The AOI is divided into 0.5 * 0.5 degree tiles for smaller and easier computations.

Site Detection: Using NVDI and DEM, we find low forest cover and elevation anomalies regions that suggest human modification.

CSV Export: Final candidates are exported to a CSV file and further process using ChatGPT to find the most possible 25 regions.

Although the pipeline is still under refinement, initial runs have yielded hundreds of candidate regions exhibiting unusual patterns of vegetation and elevation. These include: Ridges possibly used as roads or walls

Circular or rectangular clearings that suggests of residential enclosures

Terracing or elevated mounds of agriculture fields

The resulting CSV output gives the spatial coordinates, mean NDVI/elevation values, and area for each polygon. All results are stored into the results folder. Final result of 25 highly possible sites with ChatGPT analysis: site_advice.csv.

**Workflow Conclusion**

The Amazon holds a hidden historical record of complex human civilizations. Many of these cultures changed the landscapes, leaving behind physical traces that are detectable by improving technology.

This project demonstrates a scalable, reproducable workflow for detecting potential archaeological features across a vast rainforest. By using NDVI, DEM, spatial analysis, and AI-powered interpretation, we can find new discoveries in Amazonian history and guide future explorations.

## 3. Conclusion

This project provides a introductory insight into the usage of AI (ChatGPT) and automated data processing in archeological research and other data science applications. The implementation of automated data processing could potentially be improved by including more accurate researches and increasing the depth of datasets and number of dimensions when analysing problems or cases using pure data analysis.
The results of it demonstrates values of artificial intelligence in remote sensing researches. Nontheless, if human archeological experts were involved in the evaluation and identification phase with the help from ChatGPT, more precise and refined results could perhaps be generated, and more valuable implications in real-life archeological studies may be applied. 

## 4. Contributors

- DavidFlorika
- George Xue
