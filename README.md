# 📈 Sales Forecasting & Demand Analysis

A Machine Learning-based project that analyzes historical sales data, identifies sales patterns and trends, and predicts future sales using time-based features and a Random Forest regression model.

## 📌 Project Overview

Sales forecasting helps businesses estimate future demand using historical sales information. This project uses Python and Machine Learning to analyze historical daily sales data and generate predictions for future sales.

The project follows a complete workflow:

**Data Collection → Data Cleaning → EDA → Feature Engineering → Time-Based Train/Test Split → Model Training → Prediction → Evaluation → Visualization**

A **Random Forest Regressor** is used to predict sales values based on historical sales patterns and time-related features.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze historical sales data.
- Clean and preprocess sales records.
- Perform Exploratory Data Analysis (EDA).
- Identify sales trends and patterns.
- Create time-based and historical sales features.
- Train a Machine Learning regression model.
- Evaluate prediction performance.
- Compare actual and predicted sales.
- Generate visual reports for analysis.

---

## ✨ Features

- Sales data cleaning
- Missing-value handling
- Duplicate-value checking
- Date and time preprocessing
- Exploratory Data Analysis
- Sales trend visualization
- Time-based feature engineering
- Lag features
- Rolling average feature
- Time-based train-test split
- Random Forest regression
- Model evaluation
- Actual vs Predicted visualization
- Forecast result generation
- Simple Streamlit dashboard

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### Machine Learning
- Scikit-learn

### Visualization
- Matplotlib

### Web Application
- Streamlit

### Development Tools
- VS Code
- Git
- GitHub

---

## 📊 Dataset

The dataset contains historical daily sales information.

### Main Columns

| Column | Description |
|---|---|
| Date | Date of the sales record |
| Sales | Number/value of sales for that date |

The dataset is used to identify historical sales patterns and generate future sales predictions.

---

## 🧹 Data Cleaning

The following preprocessing steps are performed:

- Load the dataset using Pandas.
- Convert the date column into datetime format.
- Sort records chronologically.
- Check for missing values.
- Remove duplicate records.
- Remove incomplete records where required.
- Prepare the data for feature engineering.

---

## 🔍 Exploratory Data Analysis

EDA is performed to understand the sales dataset before Machine Learning.

The analysis includes:

- Dataset structure
- Statistical summary
- Missing-value analysis
- Duplicate-value analysis
- Sales distribution
- Daily sales trends
- Historical sales patterns

A sales trend chart is generated to visualize how sales change over time.

---

## ⚙️ Feature Engineering

Time-based and historical features are created to help the model understand sales patterns.

### Time-Based Features

- Day
- Day of week
- Month
- Week of year

### Historical Features

- 1-day lag sales
- 7-day lag sales
- 7-day rolling average

These features provide information about recent and historical sales behavior.

---

## 🤖 Machine Learning Model

### Algorithm Used

**Random Forest Regressor**

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make predictions.

It is suitable for this project because it can capture non-linear relationships between historical sales patterns and future sales values.

---

## 🔄 Forecasting Workflow

```text
Historical Sales Data
          ↓
      Data Cleaning
          ↓
         EDA
          ↓
  Date Feature Extraction
          ↓
     Lag Features
          ↓
   Rolling Mean Feature
          ↓
  Time-Based Train/Test Split
          ↓
    Random Forest Model
          ↓
       Prediction
          ↓
      Evaluation
          ↓
Actual vs Predicted Analysis
