# Projects for testing and learning ML models
*by Patcharanat P.*

## Table of contents
0. [**Machine Learning Algorithms Research**](./ML_research.md)
1. Edible or Poisonous Mushroom Classification
2. Time Series Forecasting using Regression Model for Temperature Prediction

...


### 1. Edible or Poisonous Mushroom Classification
---
*folder: mushroom*

**Overview**

Since a dataset is downloaded from Kaggle, the dataset is already cleaned and ready to apply to the models. The purpose of this project is only to learn the basics of machine learning models and data-preprocessing approaches. The work would further expand feature engineering, such as feature selection later.

dataset: [Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification) *By UCI Machine Learning* from Kaggle

Models used in the project
- DecisionTreeClassifier
- RandomForestClassifier
- ExtraTreesClassifier
- Support Vector Classification (SVC)
- XGBoost Classifier (XGBClassifier)

Aspects learned in the project
- Compared accuracy score between Label encoding and One hot encoding approach
- Hyperparameter tuning
- Using GridSearchCV and RandomizedSearchCV
- Metrics used to evaluated in classification tasks including precision, recall, f-score (f1-score), confusion matrix
- Overfitting checking (accuracy score on a training set and on a test set)

-- see more detail: *mushroom/[mushroom_notebook.ipynb](https://github.com/Patcharanat/ML-Learning/blob/master/mushroom/mushroom_notebook.ipynb)*

### 2. Time Series Forecasting using Regression Model for Temperature Prediction
---
*folder: regression*

**Overview**

The main purpose of this project is to learn different approach of developing ML model in a regression task which is temperature prediction. The dataset was collected roughly, so there's processes of data cleaning included. Attributes definition of the dataset weren't given, so domain knowledge and comprehensive for data meaning and pattern in this project will be not emphasized. The most represented in this project are **Multivariate Imputer** and **Regression Result Analysis**.

dataset: [Weather Conditions in World War Two](https://www.kaggle.com/datasets/smid80/weatherww2?select=Summary+of+Weather.csv) *By United States National Oceanic and Atmospheric Administration* from Kaggle

Model used in the project
- RandomForestRegressor with Random Search

Aspects learned in the project
- Filling missing value with a Mean or Median value
- Multivariate feature Imputation (IterativeImputer)
- (MSE and MAE explaination)
- A Baseline model & A Benchmark model
- Normalization (MinMaxScaler, RobustScaler, StandardScaler exclude Normalizer)
- Random Forest Tuning
- Using time series cross validation
- Dimensional Reduction with PCA and t-SNE (In progress...)

-- see more detail: *regression/[regression_test.ipynb](https://github.com/Patcharanat/ML-Learning/blob/master/regression/regression_test.ipynb)*