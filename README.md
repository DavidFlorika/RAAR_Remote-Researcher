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
'pip install -r requirements.txt'


