# Quick Start Guide - Mental Health Dashboard

## How to Launch the App

### Option 1: Terminal
```bash
cd /home/maxwell/Desktop/Mental_Health_Dashboard_In_R
R -e "shiny::runApp('app.R', launch.browser=TRUE)"
```

### Option 2: RStudio
- Open `app.R`
- Click "Run App" button (top-right)

---

## First-Time Workflow (Recommended)

### **Step 1: Select a Dataset**
- **Beginner-friendly**: Start with `survey.csv` (1,259 rows, 27 features)
- Dashboard automatically detects all CSV files in the folder
- Click the dropdown in the sidebar to choose

### **Step 2: Explore Your Data**
Navigate to the **"Data Preview"** tab:
- See the first 50 rows
- Check summary statistics
- Identify missing values

Then go to **"EDA"** tab:
- Try "Histogram" to see distributions
- Try "Correlation Heatmap" to see feature relationships
- Try "Missing Map" to visualize missing data patterns

### **Step 3: Choose a Target Variable**
Back in the sidebar:
- Select a **binary target** (e.g., `treatment` for survey.csv)
- 🆕 **Watch for class imbalance warnings!** The app will alert you if classes are unbalanced

### **Step 4: Configure Preprocessing (Optional)**
Try these combinations:
- **None**: See baseline performance
- **Just "Center & scale"**: Good for GLM
- **"Center & scale" + "PCA"**: Dimensionality reduction (🆕 Read the enhanced tooltips!)

Click **"👁️ Preview Preprocessing"** to see before/after data transformation

### **Step 5: Tune Model Parameters**
Read the 🆕 **enhanced tooltips** to understand each parameter:

**Train/Test Split**:
- Default: 80/20 (standard)
- Try: 70/30 (more conservative test)
- 🆕 Tooltip explains the tradeoff!

**Cross-Validation Folds**:
- Default: 5 folds
- Try: 10 folds for more reliable estimates
- 🆕 Tooltip explains how CV reduces variance

**Algorithm**:
- Start with **GLM** (fast, interpretable)
- Then try **Random Forest** (usually better, but slower)

For Random Forest:
- **Trees**: 100-500 is typical (🆕 tooltip explains why)
- **mtry**: Keep auto-tune enabled (🆕 tooltip explains what mtry means)

### **Step 6: Train Your First Model**
Click **"🚀 Train Model"**

Watch for:
- Progress bar
- 🆕 **Overfitting warnings** (if CV accuracy >> test accuracy)
- Success notification

### **Step 7: Interpret Results**
Navigate to **"Model Output"** tab:

**Training Configuration**:
- See all your settings

**🆕 Educational Insights**:
- 🆕 **Overfitting analysis** (if detected)
- Metric explanations
- Experiment suggestions

**Confusion Matrix**:
- See True Positives, False Positives, etc.

**Feature Importance**:
- Which variables matter most?

**ROC Curve**:
- Visual performance assessment

---

## Learning Experiments to Try

### **Experiment 1: Understanding Overfitting**
1. Use survey.csv with target=`treatment`
2. Set train/test split to **95/5** (very high)
3. Train Random Forest with **500 trees**
4. 🆕 Watch for the overfitting warning!
5. Now try **80/20 split** - see the difference?

### **Experiment 2: Class Imbalance Impact**
1. Choose a dataset/target with imbalance (🆕 warning will appear)
2. Train a model
3. Compare **Accuracy** vs **Sensitivity** vs **Specificity**
4. Notice how accuracy can be misleading?

### **Experiment 3: Preprocessing Impact**
1. Train GLM with **no preprocessing** → note accuracy
2. Train GLM with **"Center & scale"** → compare
3. Train GLM with **"Center & scale" + "PCA"** → compare again
4. Which worked best? Why? (🆕 tooltips explain!)

### **Experiment 4: Algorithm Comparison**
1. Enable **"Train and compare both GLM and RF"** checkbox
2. Train both models simultaneously
3. Compare performance and training time
4. When is RF worth the extra time?

### **Experiment 5: Hyperparameter Tuning**
1. Train RF with **100 trees** → note accuracy
2. Train RF with **500 trees** → compare
3. See diminishing returns? (🆕 tooltip predicted this!)

---

## Advanced Features

### **Reproducibility**
- Enable "Use custom random seed"
- Set seed = 123
- Train multiple times → same results every time!

### **Large Dataset Sampling**
For `Mental Health Dataset.csv` (292K rows):
- Enable "Use sampling for training"
- Set sample size (e.g., 5000 rows)
- Faster experimentation!

### **Make Predictions**
After training:
- Scroll down in sidebar to **"Prediction"** section
- Enter feature values
- Click "Predict"
- Get instant classification!

### **Download Results**
- **Download model**: Save trained model as RDS
- **Download predictions**: Export predictions as CSV
- **Download plots**: Save visualizations as PNG

---

## 🆕 New Educational Features

### **Smart Warnings**
- 🆕 **Overfitting alerts**: Automatic detection when train >> test
- 🆕 **Class imbalance warnings**: Before you even train!
- Both explain WHY it matters and WHAT to do

### **Enhanced Tooltips**
- 🆕 Every parameter now has detailed "WHY" explanations
- 🆕 Hover over any control to learn best practices
- 🆕 Understand tradeoffs before experimenting

### **Code Learning**
- 🆕 Warnings reference specific R code lines
- Open `app.R` and search for the line numbers
- See exactly HOW we calculate metrics!

---

## Common Issues & Solutions

**Q: App crashes when training**
A: ✅ Fixed! The app now handles missing values properly.

**Q: Should I always use all preprocessing steps?**
A: No! Read the tooltips - scaling is essential for PCA but not for Random Forest.

**Q: What's a good accuracy?**
A: Depends! With class imbalance, even 70% accuracy might be bad if it's just predicting the majority class.

**Q: Why is Random Forest slower?**
A: It builds hundreds of trees. Try reducing trees to 100 for faster experimentation.

---

## Suggested Learning Path

### **Session 1: Basics (30 min)**
1. Load survey.csv
2. Explore in EDA tab
3. Train first GLM (no preprocessing)
4. Understand the metrics

### **Session 2: Preprocessing (30 min)**
1. Compare GLM with/without scaling
2. Try PCA with different component counts
3. Preview preprocessing to see transformations

### **Session 3: Model Comparison (30 min)**
1. Compare GLM vs Random Forest
2. Tune RF hyperparameters
3. Understand the tradeoffs

### **Session 4: Advanced Topics (30 min)**
1. Handle class imbalance
2. Detect and fix overfitting
3. Cross-validation experiments

---

## Educational Use Cases

### **For Students**:
- Learn ML concepts interactively
- See immediate feedback from experiments
- Understand "why" through enhanced tooltips
- Connect concepts to R code

### **For Instructors**:
- Demonstrate overfitting live (95/5 split)
- Show class imbalance impact
- Compare algorithms side-by-side
- Assign experiments as homework

### **For Self-Learners**:
- Experiment safely (can't break anything!)
- Rich tooltips = self-guided learning
- Try "what if?" scenarios easily

---

## Your First 5-Minute Session

```
1. Run app: shiny::runApp("app.R")
2. Select "survey.csv"
3. Go to "Data Preview" tab → explore
4. Choose target = "treatment"
5. Read the class imbalance warning (if any)
6. Click "Train Model" (use defaults)
7. Go to "Model Output" tab
8. Read the Educational Insights
9. Look at Confusion Matrix
10. Celebrate your first ML model! 🎉
```

---

## Included Datasets

| Dataset | Rows | Features | Best For |
|---------|------|----------|----------|
| **survey.csv** | 1,260 | 27 | Beginner practice |
| **Mental Health Dataset.csv** | 292K | 17 | Large-scale training, sampling |
| **Age-standardized suicide rates.csv** | 550 | 6 | Time-series analysis |
| **Crude suicide rates.csv** | 550 | 10 | Age-group analysis |
| **Facilities.csv** | 113 | 7 | EDA practice |
| **Human Resources.csv** | 108 | 6 | EDA practice |

---

## Tips for Success

1. **Start simple**: Use defaults first, then experiment
2. **Read tooltips**: They explain WHY, not just WHAT
3. **Compare models**: Use the comparison mode to learn
4. **Check warnings**: They teach best practices proactively
5. **Explore the code**: Warnings reference line numbers - go look!
6. **Experiment fearlessly**: You can't break anything

---

## Getting Help

- **Tooltips**: Hover over any parameter for explanations
- **Educational Insights**: Read the panel in Model Output tab
- **Warnings**: Pay attention to alerts - they teach important concepts
- **Documentation**: See `README.md` for full feature list
- **Code**: See `EDUCATIONAL_ENHANCEMENTS.md` for implementation details

---

**Ready to start? Run the app and explore!** 🚀
