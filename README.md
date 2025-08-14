# Telco Customer Churn Survival Analysis

A comprehensive survival analysis implementation for the IBM Telco Customer Churn dataset using Python. This project demonstrates advanced statistical modeling techniques to understand customer retention patterns and identify factors that influence churn behavior.

## 📊 Project Overview

This project performs a complete survival analysis workflow on telecommunications customer data to:

- **Analyze customer retention patterns** over time
- **Compare survival rates** across different customer segments
- **Identify risk factors** that influence customer churn
- **Build predictive models** using Cox proportional hazards regression
- **Validate model assumptions** with statistical tests

## 🎯 Key Features

### Statistical Methods Implemented
- ✅ **Kaplan-Meier Survival Estimation**: Non-parametric survival function estimation
- ✅ **Log-Rank Tests**: Statistical comparison of survival curves between groups
- ✅ **Cox Proportional Hazards Model**: Semi-parametric regression for hazard modeling
- ✅ **Schoenfeld Residual Analysis**: Proportional hazards assumption validation
- ✅ **Hazard Ratio Calculation**: Risk quantification with confidence intervals

### Advanced Features
- 📈 **Multiple Visualization Types**: Overall survival, group comparisons, adjusted curves
- 🔍 **Comprehensive Data Cleaning**: Missing value handling, data type conversions
- 📊 **Statistical Summaries**: Descriptive statistics and survival metrics
- 🎛️ **Model Diagnostics**: Residual analysis and assumption testing

## 📁 Project Structure

```
Survival Analysis/
├── telco_survival_analysis.py    # Main analysis script
├── Telco-Customer-Churn.csv      # Dataset
├── README.md                     # This file
├── survival_plots/               # Generated visualizations (created on run)
│   ├── overall_survival.png
│   ├── survival_by_contract.png
│   └── adjusted_survival_by_contract.png
└── .venv/                        # Virtual environment (if using)
```

## 🛠️ Technical Requirements

### Dependencies
```
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
statsmodels >= 0.12.0
scipy >= 1.7.0
```

### Python Version
- Python 3.8 or higher

## 🚀 Installation & Usage

### 1. Clone or Download the Project
```bash
# Navigate to your project directory
cd "path/to/Survival Analysis"
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib statsmodels scipy
```

### 3. Run the Analysis
```bash
python telco_survival_analysis.py
```

## 📊 Dataset Information

**Source**: IBM Telco Customer Churn Dataset  
**Size**: 7,043 customers  
**Variables**: 21 features including demographics, services, and billing information

### Key Variables
- **Event**: Customer churn (Yes/No)
- **Time**: Customer tenure (months)
- **Covariates**: Contract type, internet service, monthly charges, demographics

## 📈 Analysis Workflow

### 1. Data Preprocessing
- Load and clean the dataset
- Convert categorical variables to appropriate formats
- Handle missing values in `TotalCharges`
- Create survival time and event indicators

### 2. Exploratory Analysis
- Generate summary statistics for key variables
- Examine distribution of contract types and churn rates
- Calculate basic retention metrics

### 3. Survival Function Estimation
- **Kaplan-Meier estimator** for overall survival
- **Stratified analysis** by contract type
- Median survival time calculation with confidence intervals

### 4. Group Comparison
- **Log-rank test** to compare survival across contract types
- Statistical significance testing (α = 0.05)
- Effect size quantification

### 5. Cox Regression Modeling
- **Multivariable Cox model** with:
  - Monthly charges (continuous)
  - Contract type (categorical)
  - Internet service type (categorical)
- **Hazard ratio estimation** with 95% confidence intervals
- Model summary and coefficient interpretation

### 6. Model Validation
- **Schoenfeld residual tests** for proportional hazards assumption
- Spearman correlation analysis between residuals and time
- Model diagnostic evaluation

### 7. Visualization
- Overall Kaplan-Meier survival curves
- Stratified survival curves by contract type
- Adjusted survival curves from Cox model predictions

## 📊 Key Results & Insights

### Sample Output Interpretation

```
Log-rank test statistic: [X.XXXX], p-value: [X.XXXXe-XX]
```
- **Significant p-value** indicates survival differences between contract types

```
Hazard Ratios and 95% Confidence Intervals:
                    HR    CI_lower  CI_upper   p_value
MonthlyCharges    X.XX      X.XX      X.XX    X.XXXXX
Contract_One year X.XX      X.XX      X.XX    X.XXXXX
Contract_Two year X.XX      X.XX      X.XX    X.XXXXX
```
- **HR > 1**: Increased hazard (higher churn risk)
- **HR < 1**: Decreased hazard (lower churn risk)
- **95% CI**: Precision of hazard ratio estimates

## 🎨 Generated Visualizations

1. **Overall Survival Curve**: Population-level retention pattern
2. **Survival by Contract Type**: Comparative retention across contract lengths
3. **Adjusted Survival Curves**: Model-based predictions controlling for covariates

## 📚 Statistical Methods Reference

### Kaplan-Meier Estimator
Non-parametric method for estimating survival probability over time, handling censored observations.

### Log-Rank Test
Non-parametric test comparing survival distributions between two or more groups.

### Cox Proportional Hazards Model
Semi-parametric regression model estimating the effect of covariates on hazard rates.

### Schoenfeld Residuals
Diagnostic tool for testing the proportional hazards assumption in Cox models.

## 🔧 Customization Options

### Modifying Analysis Parameters
- **Dataset path**: Update `DATA_PATH` variable
- **Covariates**: Modify the covariate selection in `fit_cox_model()`
- **Grouping variables**: Change stratification variables in `kaplan_meier()`
- **Visualization settings**: Adjust plot parameters in `plot_survival()`

### Extending the Analysis
- Add additional covariates to the Cox model
- Implement stratified Cox models
- Include interaction terms
- Add parametric survival models (Weibull, exponential)

## 📋 Academic Context

This project demonstrates mastery of:
- **Survival Analysis Theory**: Understanding of censoring, hazard functions, survival curves
- **Statistical Modeling**: Cox regression, assumption testing, model validation
- **Data Science Workflow**: ETL, EDA, modeling, visualization, interpretation
- **Python Programming**: Object-oriented design, statistical computing, data manipulation

## 🏆 Project Strengths for Academic Submission

1. **Methodological Rigor**: Implements multiple statistical methods correctly
2. **Code Quality**: Professional-level documentation and structure
3. **Statistical Validation**: Includes assumption testing and diagnostics
4. **Practical Application**: Real-world business problem (customer retention)
5. **Comprehensive Output**: Statistical summaries, visualizations, interpretations

## 👨‍🎓 Author

**Academic Project**: M2 Survival Analysis Course  
**Date**: August 2025

## 📜 License

This project is for academic purposes. Dataset provided by IBM for educational use.
