# DSA_internship
This project is a Machine Learning-based web application that predicts the rating of restaurants listed on Zomato India. The goal of the system is to estimate how well a restaurant is likely to perform based on key attributes such as cuisine, cost, location popularity, and online services.
The application is built using a complete data science pipeline, including data preprocessing, feature engineering, model training, and deployment using Flask. It provides not only rating predictions but also intelligent suggestions to improve restaurant performance.

* **Objective**
  
  -Predict restaurant ratings using historical data

  -Help restaurant owners understand key factors influencing ratings

  -Provide actionable insights for improving customer satisfaction

***Machine Learning Workflow**

**1. Data Cleaning**
   
    -Handled missing values
  
    -Converted rating format
  
**2. Feature Engineering**

    -Cuisine encoding using MultiLabelBinarizer
  
    -Restaurant type encoding using OneHotEncoder
  
    -Created features like cuisine_count, cost_category, city_tier
  
**3.Preprocessing**

    -Scaling using StandardScaler
  
    -Outlier handling
  
**4.Modeling**

    -Selected Random Forest regression models after trying 8 different models
  
**5. Hyperparameter tuning using Gridsearch CV**
   
    - To improve the R^2 score

**6.Evaluation**

    -R² Score
  
    -Mean Absolute Error (MAE)

**GenAI Features**


This project includes a GenAI-inspired recommendation system that provides intelligent suggestions to improve restaurant ratings.

**** How it works:****

Based on user input and predicted rating, the system generates actionable insights such as:

    -Suggest enabling online delivery

    -Recommend adding more cuisine variety

    -Suggest optimizing pricing strategy

    -Recommend adding table booking facilities

    -Provide feedback based on predicted rating level
