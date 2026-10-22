# Dataset Info
Buy Now Pay Later (BNPL) Default Risk  
https://www.kaggle.com/datasets/itzzomkar/buy-now-pay-later-bnpl-default-risk  
License: CC0: Public Domain

The "Buy Now, Pay Later" (BNPL) industry has experienced explosive growth in recent years. Services like Klarna, Afterpay, and Affirm allow consumers to split e-commerce purchases into smaller, interest-free installments. While convenient, this model has raised concerns about hidden debt accumulation and financial instability, especially among younger demographics. This dataset provides a comprehensive, synthetic snapshot of 10,000 online shoppers. It is designed to help data scientists analyze the demographic, employment, and behavioral factors that indicate whether a consumer will default on their BNPL loans.

# Requirements
* Python 3.13.2  
* Jupyter  
* Required dependencies for environment listed in `requirements.txt`.  
* Run `dataset_importer.ipynb` to import the BNPL dataset.

# Approach
The goal of this project is to perform multi-class classification using `Default_Risk` (Low, Medium, High) as the target variable to identify the most impactful features within the dataset.

_This project was done by a team of 4 data scientist undergraduates._

> [!CAUTION]
> Due to the synthetic nature of features and target variable, the results of this project should not be used as a reflection on real world behaviors or observations, but rather a practice in technique for learning. The CSV file also notes that "The target variable is 'Default_Risk' (Low, Medium, High), which has been mathematically modeled to correlate realistically with the user's debt-to-income ratio, credit score, and existing loan volume."

<ins>Two approaches were performed</ins> with an initial attempt `Initial_Approach.ipynb`, and another for a further, more accurate in-depth analysis `Final_Approach.ipynb`.

The **<ins>final approach</ins>** was able to best closely identify the _nonlinear relationships_ with the target variable classes and _correlation patterns_ for the features that the creator of the dataset most likely used for the dataset through an `XGBoost` model.  
* The model resulted in a `Macro-F1 score` of `0.876668`.  
* The top 5 identified important features are associated with `Employment_Status`, `Income_USD`, `Total_BNPL_Debt_USD`, & `Late_Payment_History`.

## Pre-processing & Baseline
Both approaches perform **EDA** to identify inherent data issues and encode categorical data with **One-Hot-Encoding** before performing a **train-test split**. The initial approach imputes missing values with _mean_ values and uses _logistic regression_ as a baseline model, while the final approach imputes with _median_ values and uses the `most_frequent` strategy with `scikit-learn`’s `DummyClassifier` rule.

The final approach also performs further EDA by analyzing association metrics between features through grouped boxplots, ANOVA tests, and chi-squared tests.

## Modeling & Optimization
The initial approach tests **feature reduction**, **logistic scaling** for skewed features, and handling of feature imbalance through **random over/undersampling** and **SMOTE** techniques.  
Results for the initial approach show a strong performance in the baseline model and SMOTE models, leading to hyperparameterization tuning for each respective model.  
* **Logistic regression** was optimized by testing different `C` values, `solver`, and `class_weight`.  
* **SMOTE** was optimized by testing different `C` values, `k-neighbors`, and `solver`. 

The final approach tests using **logistic regression**, **random forest**, and **XGBoost** models with bare-bones, default parameters to view initial metrics before performing further optimization with **5-fold CV** through **<ins>Macro-F1 score</ins>**.

> [!NOTE]
> Due to the synthetic nature of the dataset, sets of different features were categorized and tested with each model separately to identify and analyze the weights of each category through differences in resulting model metrics.

* **Logistic regression** was optimized by testing different `C` values and `class_weight`.  
* **Random forest** was optimized by testing different `n_estimators`, `max_depth`, `min_samples_leaf`, and `class_weight`.  
* **XGBoost** was optimized by testing different `n_estimators`, `max_depth`, `learning_rate`, and `subsample`.
