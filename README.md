# Data Mining Project Overview

## Project Goal
This project analyzes Airbnb data across two connected perspectives:
- listing-level market behavior and trends
- review-level sentiment and text semantics

Together, the notebooks provide an end-to-end view of how listing characteristics, geography, and textual feedback can be mined to produce insights, comparisons across years, and predictive modeling outputs.

## Notebooks and Their Roles

### 1) Listings Analysis and Recommendation
Path: `Airbnb_Listings_Analysis_And_Recommendation/Data_Mining.ipynb`

Focus:
- Build unified listing datasets for 2019 and 2023
- Perform exploratory and comparative analysis (prices, neighborhoods, room types, reviews, availability)
- Create visual outputs to support interpretation
- Build a simple content-based recommender using TF-IDF and cosine similarity

### 2) Review Sentiment Modeling (2019)
Path: `Airbnb_Review_Sentiment_Modeling/Revs_19.ipynb`

Focus:
- Clean and filter review text
- Label sentiment using lexicon-assisted filtering plus transformer inference
- Train and evaluate models (SVM, Random Forest, KNN) with TF-IDF and Word2Vec features
- Export predictions and evaluation outputs

### 3) Review Sentiment Modeling (2023)
Path: `Airbnb_Review_Sentiment_Modeling/Revs_23.ipynb`

Focus:
- Apply the same pipeline as 2019 for comparability
- Reproduce feature extraction, modeling, cross-validation, and semantic neighborhood analysis for 2023 data

## How Everything Connects
- The listings notebook explains supply-side structure and market dynamics.
- The review notebooks explain customer-side sentiment and language signals.
- Combined, they form a unified data-mining workflow for descriptive analytics, text mining, and predictive modeling across time.

## Execution Context
The notebooks are written for Google Colab with Google Drive-based paths and produce intermediate CSV/TSV/PKL artifacts plus figures used in analysis and modeling.
