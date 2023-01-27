# Red Wine Quality Prediction

The Red Wine Quality Prediction project is a in-class group project at the University of Chicago Data Mining course in 2022 Spring. 

![red_wine](https://user-images.githubusercontent.com/94136772/178409141-fd8b3ab4-54d6-4db0-9f4b-65aa09709469.jpeg)

## Description

The project intends to predict the red wine quality based on its chemical properties and determine which features are the best red wine quality indicators. The data collects features of red wine from Vinho Verde, Portugue. Due to privacy and logistic issues, grape type, wine brand, and selling prices are not included. The data includes 1599 samples with 11 independent variables and 1 responsive variable. Although the responsive variable follows a normal pattern, most of the independent variables are slightly skewed to the right and contain correlations. 

Both supervised and unsupervised methods are employed in the stage of modelling, which includes K-means, linear regression, logistic regression, decision tree classification, random forests and SVM. With a focus on accuracy score, we evaluate models using metrics like MAE, MSE, RMSE, and cross validated prediciton accuracy. The Random Forest method gives us the best accuracy score as high as 67% (which is way beyond industrial average) and the lowest MAE, MSE, RMSE. Feature importance is also calculated from the Random Forest Model. Alcohol, sulphates, volatile acidity are the most relevant features out of 10 features we have.

In the application perspective, with our data mining analysis, manufactuers can have better control over red wine quality and red wine certificate agents can reduce manpower expense and facilitate certification assessment and assurance processes. 

Although we have a decent modeling result, the project still has some limitations and/or areas of improvement. 1. Data is inefficient so that it is hard to generate comprehensive results; 2. data is inbalance so that certain wine quality levels will not be over- or under-representeed. 3. Data accuracy can be further ensured by checking human errors, data drift, and data decay of the original data.  

## Deployment Webpage

https://redwinequalitypred.onrender.com

## Authors

Kaichong (Matt) Zhang: University of Chicago student

Sophie Lyu: University of Chicago student

Michael Wu: University of Chicago student

Michelle Tan: University of Chicago student
