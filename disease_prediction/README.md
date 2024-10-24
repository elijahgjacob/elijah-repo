# Chronic Condition Prediction using Machine Learning

## Overview
This project aims to build a predictive model for classifying individuals based on whether they have been diagnosed with a chronic condition. The categories include:

1. **Yes**: The member has been diagnosed with a chronic condition.
2. **Yes, but female told only during pregnancy**: The member has been diagnosed with a chronic condition during pregnancy.
3. **No**: The member has not been diagnosed with a chronic condition.

Using a dataset containing health information, including demographics, general health, and lifestyle factors, the project involves data preprocessing, model building, and evaluation using machine learning algorithms.

## Project Structure

- `data/` : Contains the dataset for training and testing the models.
- `scripts/` : Scripts for preprocessing, modeling, evaluation, and visualization.
- `README.md` : This file, providing an overview of the project.
- `notebooks/` : Contains Jupyter notebooks with the step-by-step process.
- `results/` : Folder for storing results, visualizations, and analysis.
  
## Data Description
The dataset contains a wide range of health-related features such as:
- **General_Health**: Categories indicating self-reported health (e.g., Excellent, Good, Fair, Poor).
- **Exercise_Status**: Categories indicating the frequency of exercise (e.g., Exercise Regularly, Exercise Sometimes, Inactivity).
- **BMI_Category**: Body mass index classification.
- **Other health indicators**: Kidney Disease Status, Coronary Heart Disease Status, Smoking Status, etc.

## Preprocessing Steps
1. **Missing Values**: Missing and ambiguous values were replaced with `NaN`, and imputation was done using `SimpleImputer`.
2. **Outlier Removal**: Reasonable limits were defined for features like age, height, and weight, and values falling outside these bounds were removed.
3. **Feature Encoding**: Categorical features like `General_Health` were encoded using `OneHotEncoder`.
4. **Handling Imbalanced Data**: **SMOTE (Synthetic Minority Over-sampling Technique)** was used to handle class imbalance by oversampling the minority class in the training data.
5. **Standardization**: Standard scaling was applied to ensure that the features have a mean of 0 and a standard deviation of 1.

## Modeling
- **Model Chosen**: XGBoost Classifier was selected for this task because of its ability to handle complex relationships and provide feature importance.
- **Pipeline**: A pipeline was built for ease of preprocessing and modeling.
  - Steps:
    1. **StandardScaler**: Standardized features.
    2. **XGBClassifier**: The classification model used for prediction.
- **Hyperparameter Tuning**: 
  - **GridSearchCV** was used to tune hyperparameters such as learning rate, max depth, and the number of estimators.
  - The model was evaluated using **5-fold cross-validation** with an accuracy scoring metric.

## Evaluation
- **Metrics**: Accuracy, classification report, and feature importance were used for model evaluation.
- **Results**:
  - Best Model Hyperparameters: The best combination of hyperparameters was identified through grid search.
  - Final Model Accuracy: The model achieved an accuracy of approximately **93%**.
  - Feature Importances: Features like `General_Health`, `BMI_Category`, and `Exercise_Status` were identified as the most important.

## Visualizations
1. **Proportion of Chronic Conditions across General Health**: A stacked bar plot was used to illustrate the proportion of individuals with chronic conditions in different self-reported health categories (`Excellent`, `Good`, `Fair`, `Poor`). 
2. **Proportion of Chronic Conditions across Exercise Status**: Visualizations were created to explore how exercise status impacts the likelihood of chronic conditions. 

## Key Findings
1. **General Health Impact**: Individuals who reported `Poor` or `Fair` general health were more likely to have chronic conditions. 
2. **Exercise and Health**: Those who engaged in regular exercise had a lower proportion of chronic conditions compared to individuals who were inactive.
3. **BMI Influence**: A higher BMI was associated with a higher proportion of chronic conditions.

## Business Insights
1. **Maximizing Customer Value**:
   - Personalized coverage plans can be developed based on key determinants like age, gender, exercise frequency, and self-reported health. This allows for targeted insurance products that provide better customer satisfaction and health management.
2. **Maximizing Shareholder Value**:
   - Price discrimination can be applied using insights from the model to manage risk and optimize the profitability of health insurance plans. Key health indicators can help set premiums that balance expected value and risk.

## How to Run the Project
1. **Dependencies**: Install the required Python packages using `requirements.txt`.
   ```bash
   pip install -r requirements.txt
