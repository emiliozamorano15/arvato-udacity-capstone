# Data Scientist Nanodegree Capstone Project:
# Customer Segmentation Report for Arvato Financial Services


## Table of Contents

- [Overview](#overview)
- [File Structure](#files)
- [Software Requirements](#requirements)
- [Conclusion](#conclusion)
- [Credits and Acknowledgements](#credits)

***

## 1. Overview <a id='overview'></a>

In this project, you will analyze demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population. You'll use unsupervised learning techniques to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. Then, you'll apply what you've learned on a third dataset with demographics information for targets of a marketing campaign for the company, and use a model to predict which individuals are most likely to convert into becoming customers for the company. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.

## 2. File Structure <a id='files'></a>

    ├── LICENSE
    ├── README.md           
    ├── Arvato Project Workbook.py      <- Main Jupyter notebook
    ├── functions.py                    <- Helper functions for data cleaning, plotting, etc.

## 3. Software Requirement <a id='requirements'></a>

This project was written under Python 3.9.5. The libraries needed to run this project are:

* imbalanced-learn
* scikit-leanr
* pandas
* matplotlib
* seaborn
* scipy

## 4. Conclusion <a id='conclusion'></a>

For Udacity's Capstone Project we addressed a customer segmentation project with real data provided by Bertelsmann/Arvato. The purpose of this is project is manyfold: first, to provide insights into focus customers segments for mailing campaigns, second, to develop a machine learning model to predict customer responses, and third, to gauge the prediction quality of such model in an online competition.

The project itself presented several challenges, as it required the application of following concepts:
* Data cleaning (detecting missing values, transformit column input, ect.)
* Dimension reduction and clustering (e.g., PCA, K-means)
* ML and ML-pipelines (i.e., creating training/test sets, transforming data, selecting algorithms, cross-validation, grid-search, etc.)
* General programming best practices (DRY concepts, code readability, etc.)

As for the customer segmentation, once the data provided by Arvato was processed (i.e. empty values removed and column values transformed) we determined that focus of the campaign could be focused into two customer segments containing therefore addressing more than 50% of the customers population. Regarding the development of a supervised algorithm for response prediciton, on a first place, an imbalance in the response variable was detected and  SMOTE oversampling appraoch was applied for this purpose. With the oversampled data, several classification models were tested being GradientBoostingClassifier the approach with the best validation score. The selected lagorithm was in turn used to train a sequence of models on which different hyper-parameters were tested, in order to determine the best configuration for this algorithm and data structure. Lastly, the resulting optimized model was used to make predictions on a test dataset provided for a Kaggle competition. The results where scored with a 0.70361 score, landing on the top 300 results of the competition on a first submission.

As for possible improvements on the previously described approach we note the following:
* Further exploration on data reduction could be possible. In this case the number of features is still high and some further analyses on this regard might be necessary. Alternatively, there is the possibility of applying feature selection techniques to reduce furthermore the number of features.
* Class imbalance was addressed by using an oversampling approach with synthetic values (i.e. SMOTE). Alternate over- and undersampling approaches need to be evaluated to determine the best option to address this issue.
* Regarding the algorithm selection, alternative algorithms need to be evaluated as, e.g., SVM, XGBoost, etc.

## 3. Credits and Acknowledgements <a id='creadits'></a>

Many thanks to Udacity and Bertelsmann/Arvato for defining this project and making the data available. 