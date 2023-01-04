# Projects for testing and learning Machine Learning models
*by Patcharanat P.*

## Table of contents
---
1. Edible or Poisonous Mushroom Classification

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