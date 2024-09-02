# ML Evaluation

## Table of Contents
1. [Regression Task](#1-regression-task)
2. [Classification Task](#2-classification-task)
3. [Baseline Model & Benchmark](#3-baseline-model--benchmark)
4. [Overfitting](#4-overfitting)
5. [Clustering Task](#5-clustering)


## 1. Regression Task
- R^2 [0→1]
    - represents the proportion of the variance in the dependent variable that is explained by the model. with higher values indicating a better fit, R-squared value of 1 means that the model explains all of the variance in the dependent variable.
- MAE (L1 Loss)
    - less sensitive to outlier (Interpretable)
- MSE
    - sensitive to outlier
- RMSE
    - more comprehensive than MSE
- Adjusted R-squared
    - R-squared but is penalty by number of parameters lead to ability of the model to explain the data regarding to complexity of the model (number of variables)
- RMSLE
    ```python
    def RSLE(y_hat,y):
        return np.sqrt((np.log1p(y_hat) - np.log1p(y))**2)

    print("The RMSLE score is %.3f" % RSLE( 400,1000) )
    ```
- other
    - Mean Bias Error (MBE)
    - Pearson Coefficient
    - MAPE (Mean Absolute Percentage Error)


## 2. Classification Task
- Precision
    - ‘*how much the model is right when it says it is right’.*
    - TP/(TP+FP)
- Recall
    - ‘*how much extra right ones, the model missed when it showed the right ones’.*
    - TP/(TP+FN)
- Accuracy (micro-F1)
- F1-Score
    - In general, if you are working with an imbalanced dataset where all classes are equally important, using the **macro** average would be a good choice as it treats all classes equally.
    - If you have an imbalanced dataset but want to assign greater contribution to classes with more examples in the dataset, then the **weighted** average is preferred.
- PR Curves (Precision-Recall)
    - **Due to the absence of TN in the precision-recall equation, they are useful in imbalanced classes**.
    - **Due to the consideration of TN or the negative class in the ROC equation, it is useful when both the classes are important to us.**
- AUC-ROC Curves
- Classification Report Explained: [https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f#:~:text=The macro-averaged F1 score,regardless of their support values](https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f#:~:text=The%20macro%2Daveraged%20F1%20score,regardless%20of%20their%20support%20values).

## 3. Baseline Model & Benchmark
- Baseline model is a simple model that serves as a starting point for comparison establishing a minimum level of performance that a more sophisticated model should be able to exceed.
    - mean, median of the target variable
    - constant model
    - majority class classifier or ZeroR model
    - linear regression
- Example:
    - regression task
        
        ```bash
        import numpy as np
        
        # Assume that 'y' is the target variable and 'X' is the input feature
        # 'X' can be skipped if you don't need any feature to predict the target
        
        # Computing the mean of the target variable
        baseline_pred = np.mean(y)
        
        # Using this value as the prediction for all examples
        predictions = [baseline_pred for _ in range(len(y))]
        ```
        
    - classification task
        
        ```bash
        import numpy as np
        
        # Assume that 'y' is the target variable and 'X' is the input feature
        # 'X' can be skipped if you don't need any feature to predict the target
        
        # Computing the majority class
        majority_class = np.argmax(np.bincount(y))
        
        # Using this class as the prediction for all examples
        predictions = [majority_class for _ in range(len(y))]
        ```
        
        - In classification can also use logistic regression and Naive Bayes as baseline models
            
            ```python
            import sklearn.linear_model import LogisticRegression
            import sklearn.model_selection import train_test_split
            import sklearn.metrics import accuracy_score
            
            X = df[input]
            y  = df[output]
            
            X_train, X_test, y_train, y_test = tran_test_split(X, y, test_size=0.2)
            
            clf = LogisticRegression(random_state=0)
            clf.fit(X_train, y_train)
            baseline_prediction = clf.predict(X_test)
            
            print(accuracy_score(y_test, baseline_prediction))
            ```
            
    - ZeroR
        
        ```bash
        from collections import Counter
        
        # Assume that 'y' is the target variable and 'X' is the input feature
        # 'X' can be skipped if you don't need any feature to predict the target
        
        # Counting the occurence of each class
        counts = Counter(y)
        
        # Finding the most frequent class
        zeroR_class = max(counts, key=counts.get)
        
        # Using this class as the prediction for all examples
        predictions = [zeroR_class for _ in range(len(y))]
        ```
        
    - using more sophisticated model as a baseline model
        
        ```bash
        from sklearn.linear_model import LinearRegression
        
        # Assume that 'y' is the target variable and 'X' is the input feature
        
        # Create an instance of the LinearRegression model
        lin_reg = LinearRegression()
        
        # Fit the model to the data
        lin_reg.fit(X, y)
        
        # Make predictions using the trained model
        predictions = lin_reg.predict(X)
        ```
        
- A benchmark model, on the other hand, is a model that has been previously established as a standard of performance for a given problem and dataset. Benchmark models are often state-of-the-art models for a specific problem and dataset, and they are used as a reference point to evaluate the performance of new models. They are often used to evaluate the performance of new models and to compare the performance of new models to the best models that have been developed so far.
- MAE & MSE & R^2 & RMSE - when to know if a values is acceptable
    - RMSE - Root Mean Squared Error
        - RMSE is a measure of the difference between values predicted by a model and the true values. The smaller the RMSE value, the better the model is at predicting the target variable.
    - MAE - Mean Absolute Error
        - MAE is calculated by taking the average of the absolute differences between the predicted values and the true values. It is less sensitive to outliers compared with MSE since it does not square the errors, and it places equal weight on all errors. This can be useful in situations where the goal is to minimize the overall error, without being too concerned about the impact of outliers.
    - MSE - Mean Squared Error
        - Like the RMSE and MAE, the smaller the MSE value, the better the model is at predicting the target variable. This can be useful in situations where it is important to minimize the impact of large errors.
    - R-squared
        - R² is a value between 0 and 1, where a value of 1 indicates that the model perfectly predicts the target variable. R² is commonly used to evaluate the performance of a model in a regression task.
        - R-squared value does not guarantee that the model is overfitting
    
    Metrics: Use MAE instead of RMSE as a loss function. We can also use truncated loss in dataset that contains outliers

## 4. Overfitting
- comparing performance on a training set and test set
- better performance on the training set means overfitting (failed to generalize to new, unseen data)
- **cross-validation**
    - you can get a sense of how well the model is likely to perform on new, unseen data by splitting the existing data into multiple training and test sets to evaluate
    - k-fold cross validation
        - For a small k, we have a higher selection bias but low variance in the performances.
        - For a large k, we have a small selection bias but high variance in the performances.
        
        *Generally a value of k = 10 is recommended for most purpose.*
        
- regularization
    - helps to prevent overfitting by adding a penalty term to the model's cost function. This penalty term discourages the model from fitting the noise in the data.
- Validation Curve
    - Validation curves allow us to find the sweet spot between underfitting and overfitting a model to build a model that generalizes well.
    - checking a proper region to not overfitting or underfitting
- Learning Curve (≠ learning rate)
    - If our model has **high bias**, we'll observe fairly quick convergence to a high error for the validation and training datasets.
        - A better approach to improving models which suffer from high bias is to consider adding additional features to the dataset so that the model can be more equipped to learn the proper relationships.
    - If our model has **high variance**, we'll see a gap between the training and validation error.
        - In this case, feeding more quantity of data during training can help improve the model's performance.

To Choose the models, consider minimum overfitting (both train and test accuracy are close) and minimum prediction time & tuning time.

## 5. Clustering Task
- Inertia → [0, infinity] close to 0 = better
    - (after fitted) score = model.inertia_
- silhoulette score → [-1, 1] close to 1 = better
- ARI: Adjusted Rand Index (best = 1, worst ≤ 0 → -1)
    - only able to use if we have ground truth (target) to check
    - from sklearn.metrics import adjusted_rand_score
    - score = adjusted_rand_score(y_test, y_pred)