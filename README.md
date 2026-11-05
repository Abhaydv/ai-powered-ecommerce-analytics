# 🛒 AI-Powered E-Commerce Analytics

An end-to-end E-Commerce Analytics and Machine Learning project that transforms raw transactional data into actionable business insights through Python, SQL, ETL pipelines, Machine Learning, and interactive Streamlit dashboards.

## 📌 Project Overview

This project focuses on analyzing e-commerce business performance, understanding customer purchasing behavior, evaluating product performance, and identifying potential repeat customers using Machine Learning.

It combines data engineering, data analytics, business intelligence, and predictive analytics into a unified analytics solution.

## 🎯 Project Objectives

* Analyze overall e-commerce sales and revenue performance.
* Understand customer purchasing patterns and behavior.
* Identify top-performing products and categories.
* Evaluate seller and delivery performance.
* Build ETL pipelines for data cleaning and transformation.
* Develop Machine Learning models for repeat purchase prediction.
* Generate interactive dashboards and AI-driven business insights.

## 🚀 Key Features

### 1. Sales Analytics

* Total revenue and order analysis.
* Monthly revenue trends.
* Monthly order trends.
* State-wise sales performance.
* Average Order Value (AOV).

### 2. Customer Analytics

* Customer segmentation.
* Repeat vs. one-time customer analysis.
* Customer revenue distribution.
* Purchase frequency analysis.
* Customer-level business insights.

### 3. Product Analytics

* Product performance analysis.
* Category-level revenue analysis.
* Best-performing products.
* Product sales insights.

### 4. AI Insights

* Repeat purchase prediction.
* Customer probability scoring.
* Prediction-based customer segmentation.
* Model evaluation metrics.
* Business recommendations.

## 🧰 Technologies Used

| Category            | Technologies                       |
| ------------------- | ---------------------------------- |
| Programming         | Python                             |
| Data Analysis       | Pandas, NumPy                      |
| Data Visualization  | Plotly, Matplotlib, Seaborn        |
| Database & Querying | SQL                                |
| Machine Learning    | Scikit-learn                       |
| Models              | Logistic Regression, Random Forest |
| Dashboard           | Streamlit                          |
| Development         | Jupyter Notebook, VS Code          |
| Version Control     | Git, GitHub                        |

## 🏗️ Project Architecture

```text
Raw E-Commerce Data
        |
        v
Data Profiling
        |
        v
Data Cleaning & Transformation
        |
        v
Master Dataset Creation
        |
        v
Exploratory Data Analysis
        |
        v
SQL Business Analysis
        |
        v
Machine Learning
        |
        v
Repeat Purchase Prediction
        |
        v
Streamlit Analytics Dashboard
        |
        v
Business Insights
```

## 📂 Repository Structure

```text
ai-powered-ecommerce-analytics/
│
├── ai/
│   └── 06_time_based_model_training.py
│
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── 1_Sales_Analytics.py
│       ├── 2_Customer_Analytics.py
│       ├── 3_Product_Analytics.py
│       └── 4_AI_Insights.py
│
├── etl/
│   ├── 01_profile_data.py
│   ├── 02_clean_data.py
│   └── 03_create_master_dataset.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── model_metadata.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 01_eda_github_version.ipynb
│
├── sql/
│   ├── 01_revenue_analysis.sql
│   ├── 02_customer_analysis.sql
│   ├── 03_product_analysis.sql
│   ├── 04_seller_analysis.sql
│   ├── 05_delivery_analysis.sql
│   └── 06_repeat_customer_analysis.sql
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Abhaydv/ai-powered-ecommerce-analytics.git
```

### 2. Navigate to the project directory

```bash
cd ai-powered-ecommerce-analytics
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```powershell
venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

## 📊 Machine Learning

The project explores classification models to predict repeat customer purchasing behavior.

Models include:

* Logistic Regression
* Random Forest
* Time-based model training

Model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC

**Note:** The repeat-purchase dataset is highly imbalanced. Therefore, accuracy alone is not a reliable indicator of model performance. Precision, recall, F1-score, and PR-AUC are also considered.

## 📈 Dashboard Modules

The Streamlit dashboard contains five sections:

1. Main Dashboard
2. Sales Analytics
3. Customer Analytics
4. Product Analytics
5. AI Insights

## 📁 Dataset Information

The processed datasets are stored locally under:

```text
data/
├── raw/
└── processed/
```

Large datasets are excluded from GitHub to keep the repository lightweight.

To run the complete dashboard, the required processed datasets must be available locally:

* `master_ecommerce_dataset.csv`
* `repeat_purchase_test_predictions_v2.csv`

These files are not included in the GitHub repository.

## 🔮 Future Enhancements

* Cloud deployment of the analytics dashboard.
* Automated ETL workflows.
* Real-time data integration.
* Improved repeat-purchase prediction models.
* Advanced customer segmentation.
* Automated business reporting.
* Integration with cloud data platforms.

## 👨‍💻 Author

**Abhay Yadav**

B.Tech – Computer Science Engineering

Interested in Data Analytics, Data Engineering, Machine Learning, and Business Intelligence.

GitHub: [Abhaydv](https://github.com/Abhaydv)

## ⭐ Support

If you find this project interesting, consider giving the repository a star!
