Machine Learning model to predict traffic accident severity using classification algorithms
# Traffic Accident Severity Prediction using Machine Learning
## Overview
This project predicts the severity of traffic accidents using machine learning models such as Decision Tree, Logistic Regression, Naive Bayes, and KNN.
## Objective
To classify accident severity into:
* Slight
* Serious
* Severe
## Dataset
* Source: Traffic Accident Dataset
* Records: 60,000 (after cleaning)
* Features Used:
  * Speed Limit
  * Weather Conditions
  * Road Surface Conditions
  * Light Conditions
  * Number of Vehicles
  * Number of Casualties
  * Urban/Rural Area
  
## Methodology
1. Data Cleaning
2. Feature Selection
3. Train-Test Split (80-20)
4. Model Training
5. Evaluation

## Models Used
* Decision Tree (Best)
* Logistic Regression
* Naive Bayes
* K-Nearest Neighbors (KNN)

## Results
| Model               | Accuracy |
| ------------------- | -------- |
| Decision Tree       | 86.91%   |
| Logistic Regression | 86.87%   |
| Naive Bayes         | 83.10%   |
| KNN                 | 85.63%   |

## Evaluation Metrics
* Precision
* Recall
* F1 Score
* Confusion Matrix

## Key Insights
* Number of casualties is the most important feature
* Decision Tree performs best due to non-linear learning
* Dataset is imbalanced (more slight cases)

## Future Work
* Use Deep Learning (ANN, CNN)
* Real-time prediction system
* Web/App deployment

## Conclusion
Machine learning models can effectively predict accident severity and help improve road safety and emergency response systems.

