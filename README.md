# 🧠 Multi-CSV Mental Health Dashboard (Shiny)

An **educational machine learning dashboard** for exploring mental health datasets and learning about binary classification, preprocessing, and model tuning.

## ✨ Features

### 📊 Data Exploration
- **Dynamic CSV Loading**: Automatically discovers all CSV files in project folder
- **Interactive EDA**: 10+ visualization types (histogram, density, boxplot, scatter, pair plots, etc.)
- **Data Preview**: View datasets with summary statistics and missing value analysis
- **Binary Target Detection**: Automatically identifies columns suitable for classification

### 🔧 Preprocessing Pipeline
- **Visual Preview**: See before/after data transformation
- **Median Imputation**: Handle missing numeric values
- **Scaling & Normalization**: Standardize features
- **One-Hot Encoding**: Convert categorical variables
- **PCA**: Dimensionality reduction with configurable components

### 🎯 Educational Model Tuning
- **Train/Test Split Control**: Experiment with different split ratios (50%-95%)
- **Cross-Validation Folds**: Configurable CV (2-10 folds)
- **Random Seed Control**: Ensure reproducibility or try different splits
- **Algorithm Selection**: GLM (Logistic Regression) or Random Forest
- **Random Forest Hyperparameters**:
  - Number of trees (10-2000)
  - Auto or manual mtry tuning

### 📈 Model Evaluation
- **Comprehensive Metrics**: Accuracy, Sensitivity, Specificity, ROC AUC
- **Confusion Matrix**: Visual performance breakdown
- **Feature Importance**: See which variables matter most
- **ROC Curves**: Interactive visualization
- **Educational Insights**: Learn what each metric means and how to interpret results
- **🆕 Overfitting Detection**: Automatic comparison of train vs test accuracy with explanations
- **🆕 Class Imbalance Warnings**: Alerts when dataset has imbalanced classes with guidance on metric interpretation

### 🎓 Enhanced Learning Features (NEW!)
- **Smart Parameter Tooltips**: Detailed explanations of WHY each parameter matters, not just what it does
  - Train/Test Split: Understand tradeoffs between training data size and test reliability
  - Cross-Validation: Learn how CV reduces variance and when to use different fold counts
  - Random Forest Parameters: Deep explanations of tree count, mtry, and their impacts
  - Preprocessing Steps**: When to use each technique and why order matters

- **Proactive Educational Warnings**:
  - Overfitting alerts when train accuracy >> test accuracy
  - Class imbalance detection with impact explanations
  - Best practice recommendations

- **Code-Visible Learning**: All R code remains visible so students can see HOW metrics are calculated
- **Context-Aware Guidance**: Warnings and tips reference specific line numbers in the R code

### 🔮 Predictions & Export
- **Interactive Prediction**: Enter values and get instant predictions
- **Download Models**: Save trained models as RDS files
- **Export Predictions**: Download predictions as CSV
- **Export Visualizations**: Save plots as PNG

## 📦 Installation

### Prerequisites
- R version 4.0 or higher
- RStudio (recommended)

### Required Packages
```r
install.packages(c(
  "shiny",       # Web framework
  "tidyverse",   # Data manipulation
  "caret",       # Machine learning
  "randomForest",# RF algorithm
  "plotly",      # Interactive plots
  "DT",          # Data tables
  "pROC",        # ROC curves
  "recipes",     # Preprocessing
  "rlang",       # Tidy programming
  "bslib"        # Bootstrap theming
))
```

### Optional Packages
```r
install.packages(c(
  "GGally",      # Enhanced pair plots
  "skimr"        # Data summaries
))
```

## 🚀 Quick Start

### Method 1: RStudio
1. Open `app.R` in RStudio
2. Click **"Run App"** button (top-right of editor)
3. Dashboard opens in viewer pane or browser

### Method 2: R Console
```r
setwd("path/to/Mental_Health_Dashboard_In_R")
shiny::runApp("app.R")
```

### Method 3: Command Line
```bash
cd /path/to/Mental_Health_Dashboard_In_R
R -e "shiny::runApp('app.R', launch.browser=TRUE)"
```

## 📚 How to Use

### Step 1: Select Dataset
- Dashboard automatically detects all CSV files in project folder
- Choose from dropdown in sidebar
- Preview data in "Data Preview" tab

### Step 2: Explore Data (EDA Tab)
- Select visualization type (10+ options)
- Customize appearance (colors, bins, transparency)
- View correlation heatmaps and missing value maps

### Step 3: Configure Preprocessing (Optional)
- Choose preprocessing steps (imputation, scaling, encoding, PCA)
- Click **"Preview Preprocessing"** to see before/after comparison
- Experiment with different combinations

### Step 4: Tune Model Parameters
- **Train/Test Split**: Adjust ratio (default 80/20)
- **CV Folds**: Set cross-validation folds (default 5)
- **Random Seed**: Control reproducibility
- **Algorithm**: Choose GLM or Random Forest
- **RF Hyperparameters**: Trees and mtry tuning

### Step 5: Train Model
- Click **"Train Model"** button
- Monitor progress bar
- View results in "Model Output" tab

### Step 6: Interpret Results
- Review training configuration
- Analyze confusion matrix and metrics
- Examine feature importance and ROC curve
- Read educational insights for guidance

### Step 7: Make Predictions (Optional)
- Scroll to prediction panel (appears after training)
- Enter feature values
- Click **"Predict"** for instant results
- Download predictions as CSV

## 🎓 Educational Use Cases

### For Students
1. **Understanding Train/Test Splits**: Change ratio and observe test accuracy changes
2. **Cross-Validation Impact**: Increase CV folds to see more stable estimates
3. **Preprocessing Effects**: Toggle steps to see which improve performance
4. **Algorithm Comparison**: Compare GLM vs Random Forest on same data
5. **Hyperparameter Tuning**: Experiment with RF trees and mtry values

### For Instructors
- Demonstrate overfitting with high train/low test split
- Show importance of feature scaling for certain algorithms
- Illustrate bias-variance tradeoff with CV folds
- Teach model evaluation metrics interactively
- Create reproducible examples with seed control

## 📊 Included Datasets

| Dataset | Rows | Features | Use Case |
|---------|------|----------|----------|
| Mental Health Dataset.csv | 292K | 17 | Large-scale binary classification |
| survey.csv | 1,260 | 27 | Tech worker mental health survey |
| Age-standardized suicide rates.csv | 550 | 5+ | Time-series country analysis |
| Crude suicide rates.csv | 550 | 5+ | Age-group suicide rates |
| Facilities.csv | 113 | 7 | Mental health infrastructure |
| Human Resources.csv | 108 | 6 | Healthcare worker distribution |

## 🐛 Troubleshooting

### "No binary target detected"
- Ensure column has exactly 2 unique values
- Convert to factor: `df$target <- as.factor(df$target)`

### Model training fails
- Check for sufficient data (minimum 10 rows)
- Verify no constant columns (zero variance)
- Try enabling "Impute median" for missing values

### Preprocessing errors
- Ensure at least 2 numeric columns for PCA
- Check that categorical variables have reasonable cardinality
- Preview preprocessing before training

### Out of memory
- Enable sampling option
- Reduce sample size for large datasets
- Close other R sessions

## 📝 Notes & Best Practices

- **Binary targets**: App auto-detects factor columns with 2 levels or numeric 0/1 columns
- **Large datasets**: Use sampling for faster experimentation (can toggle on/off)
- **Reproducibility**: Enable custom seed and document your configuration
- **Comparison mode**: "Train both GLM and RF" for side-by-side algorithm comparison
- **Feature engineering**: Preprocess data outside app for advanced transformations

## 🔄 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and improvements.

**Current Version**: 2.0 (Educational Enhancement & Bug Fixes)

## 🤝 Contributing

Suggestions for improvement:
- Additional algorithms (SVM, XGBoost, neural networks)
- Automated hyperparameter tuning (grid search)
- Multi-class classification support
- Time-series analysis features
- Automated testing framework

## 📄 License

This project is open-source and available for educational purposes.
