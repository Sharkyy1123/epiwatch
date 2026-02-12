# 📊 Dataset Setup Guide

## Downloading the COVID-19 Dataset

1. **Visit the Kaggle Dataset Page:**
   - Go to: https://www.kaggle.com/datasets/josephassaker/covid19-global-dataset

2. **Download the Dataset:**
   - Click the "Download" button
   - You may need to sign in to Kaggle (free account)
   - Extract the ZIP file if needed

3. **Place the CSV File:**
   - Copy the CSV file to the `epiwatch-ai` folder
   - Rename it to one of these (if needed):
     - `covid19-global-dataset.csv` (preferred)
     - `covid19_global_dataset.csv`
     - `covid19-global.csv`
     - `covid_19_data.csv`
     - `data.csv`
     - `covid19.csv`

4. **Verify the File:**
   - The file should be in the same folder as `app.py`
   - The app will automatically detect and load it

## Expected Dataset Format

The dataset should have columns like:
- **Date** or **ObservationDate**: Date column
- **Country/Region** or **Country**: Country name
- **Confirmed** or **Cases**: Number of confirmed cases
- **Deaths** (optional): Number of deaths
- **Recovered** (optional): Number of recovered cases

## Note

If the dataset file is not found, the app will automatically use sample data for demonstration purposes.
