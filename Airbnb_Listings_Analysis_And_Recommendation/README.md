# Airbnb Listings Analysis and Recommendation

## Notebook Covered
- `Data_Mining.ipynb`

## Goal
This notebook builds a full Airbnb listings workflow in two major parts:
1. Build cleaned, unified yearly datasets (2019 and 2023) from multiple month folders.
2. Perform exploratory analysis and visual comparisons across years.
3. Build a simple content-based recommender using TF-IDF + cosine similarity on listing text.

In short, it answers business-style questions about listing distribution, price behavior, review patterns, and neighborhood dynamics, then adds a recommendation component for similar listings.

## High-Level Structure

### Part A: Data Preparation and Comparative Analysis (Question 1.x)
- Mounts Google Drive and reads raw `listings.csv` files from:
  - 2019: February, March, April
  - 2023: March, June, September
- Selects relevant columns and handles mixed CSV/TSV edge cases.
- Samples data and merges into:
  - `train_2019.csv`
  - `train_2023.csv`

Then it executes multiple analysis blocks (`Question 1.1` through `Question 1.14`) that generate plots and save figures such as:
- most common categorical values
- average price trends by month
- top neighborhoods by reviews/listing counts
- neighborhood distributions
- room type prevalence
- mean price comparisons
- geo map visualization with Folium
- word clouds for text fields
- review score and availability relationships
- host listing concentration

`Question 1.15` is a written interpretation section (Greek) summarizing differences between 2019 and 2023.

### Part B: Text Similarity and Recommendations (Question 2.x)
- Builds a combined text field from `name` + `description`.
- Computes TF-IDF with uni-grams and bi-grams.
- Computes listing-to-listing cosine similarity.
- Recommends top-N similar listings for a selected listing.
- Extracts top frequent bigrams from cleaned text.

This part provides a baseline content-based recommendation approach using textual listing metadata.

## How the Goal Is Achieved
- **Data integration:** monthly raw files are cleaned and merged to yearly training files.
- **Feature normalization:** key columns are parsed and sanitized before analysis.
- **Exploratory analytics:** each `Question 1.x` cell computes a focused metric and visual output.
- **Comparative framing:** the same analyses are run for both years to identify changes over time.
- **Recommendation logic:** TF-IDF vectors represent listing text; cosine similarity identifies close neighbors.

## Inputs and Outputs

### Expected Inputs
- Google Drive mounted at `/content/drive`.
- Raw Airbnb listing files in the month/year folder layout used by the notebook.

### Main Produced Files
- `train_2019.csv`, `train_2023.csv`
- Multiple figure files such as `question_1_*.png`
- Interactive map output (rendered in notebook)
- Console recommendation outputs for top-N similar listings

## Dependencies
Core libraries used:
- `pandas`, `numpy`, `matplotlib`
- `folium`, `wordcloud`
- `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
- `nltk` bigram utilities

The notebook is written primarily for Google Colab execution.

## Practical Notes
- Several paths are hard-coded for Colab Drive; adjust if running locally.
- Sampling (`sample(n=6000)`) is used in data assembly and may affect reproducibility unless random seeds are fixed.
- Some logic branches depend on fixed substring positions in file paths (year detection), which is brittle if paths change.
- The recommendation section is interactive and expects user input.

## Suggested Refactor Direction
If you plan to continue this project, consider extracting the notebook into:
- one module for data ingestion/cleaning,
- one module for EDA plotting,
- one module for recommendation,
- and a small config file for paths and parameters.
