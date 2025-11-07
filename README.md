# Multi-CSV Mental Health Dashboard (Shiny)

This Shiny app dynamically loads all CSV files from the project folder and lets you:

- Preview datasets
- Run quick EDA (histograms, correlation heatmap, categorical vs target)
- Detect binary targets (factor columns with 2 levels)
- Train a model (GLM or Random Forest) when a binary target exists
- View confusion matrix, variable importance, and ROC
- Make predictions using trained model and download results as CSV

Files:
- `app.R` — the Shiny application

Dependencies (R packages):
- shiny, tidyverse, caret, randomForest, plotly, DT, pROC
- recipes (for tidy preprocessing pipelines)
- rlang (helper for tidy programming)

Optional / Useful for extended EDA:
- GGally (pair plots), skimr (data summaries)

How to run locally:

1. Open R or RStudio and set working directory to this project folder, for example:

   In R (or RStudio):
   ```r
   setwd("c:/Users/Joseph/Desktop/projects/mental_health_dashboard_R")
   shiny::runApp("app.R")
   ```

   Or from PowerShell (if R is on PATH):
   ```powershell
   cd C:\Users\Joseph\Desktop\projects\mental_health_dashboard_R;
   R -e "shiny::runApp('app.R', launch.browser=TRUE)"
   ```

2. The app will list CSV files found in the folder. Select one, optionally select a binary target and train a model.

- The app auto-detects factor columns with exactly 2 levels as potential targets.
- For large CSVs training might take time — consider sampling or increasing available memory.
- If you want a specific column to be considered a target, convert it to a factor with two levels beforehand.

Notes & tips:
- The app auto-detects factor columns with exactly 2 levels as potential targets. It also demonstrates common preprocessing steps using the `recipes` package: median imputation, centering/scaling, dummy encoding and PCA.
- Use the "Train and compare both GLM and RF" option to learn how different algorithms perform on the same dataset.
- The app includes extra plots (boxplots and density by target) to help you explore distributions and feature-target relationships.
- For large CSVs training might take time — consider sampling or increasing available memory.
- If you want a specific column to be considered a target, convert it to a factor with two levels beforehand.
- The app auto-detects factor columns with exactly 2 levels as potential targets.
- For large CSVs training might take time — consider sampling or increasing available memory.
- If you want a specific column to be considered a target, convert it to a factor with two levels beforehand.

If you want extra features (multi-target models, automated preprocessing pipelines, or model export), tell me what to add next.
