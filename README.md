# Cement-Strength-Prediction-


Objective-
The aim of this project is to develop a machine learning model that can predict the compressive strength of concrete based on its ingredients. This project involves data preprocessing, feature transformation, model training, evaluation, and deployment of a prediction system using a graphical user interface (GUI).

Dataset-
The dataset used in this project is Concrete_Data.xls, which contains the following columns:

Cement 
Blast Furnace Slag 
Fly Ash 
Water 
Superplasticizer 
Coarse Aggregate 
Fine Aggregate 
Age 
Concrete Compressive Strength

Steps Involved:-
-Data Loading and Preprocessing:
-Load the dataset from an Excel file.
-Rename the columns for clarity.
-Check for null values and duplicates.
-Generate summary statistics and a correlation matrix to understand relationships between features.

Data Visualization:-
-Plot distribution and probability plots of features to understand their distributions and identify any deviations from normality.

Feature Transformation:-
-Apply a power transformation to normalize the feature distributions.
-Standardize the features using StandardScaler to ensure they are on the same scale.

Model Training:-
-Split the data into training and testing sets.
-Train several machine learning models (Linear Regression, Ridge, Lasso, Random Forest Regressor, XGBoost Regressor) on the training data.
-Evaluate models using mean squared error (MSE) and R-squared (R²) metrics.
Model Selection:-
-Select the best-performing model (XGBoost Regressor) based on evaluation metrics.

Model Persistence:-
-Save the trained model and the transformers (StandardScaler and PowerTransformer) using pickle for future use.

GUI Development:-
-Develop a graphical user interface using Tkinter to allow users to input the concrete mixture's ingredients and age.
-The GUI takes these inputs, processes them through the trained model, and displays the predicted compressive strength of the concrete.
Code Overview

Data Preparation and Model Training:-
-Load and preprocess data.
-Apply power transformation and standardization.
-Train and evaluate models.
-Save the best model and transformers.

Prediction System:-
-Load the saved model and transformers.
-Define a function to process new input data and predict the compressive strength.

GUI Application:-
-Create a Tkinter window with input fields for each concrete ingredient and age.
-Add a button to trigger the prediction.
-Display the predicted strength in the GUI.

Key Components
-Data Handling: pandas for data manipulation, seaborn and matplotlib for data visualization.
-Modeling: scikit-learn for model training and evaluation, XGBoost for the selected model.
-Transformation: PowerTransformer and StandardScaler for feature preprocessing.
-Persistence: pickle for saving and loading models.
-GUI: Tkinter for creating a user-friendly interface.

Outcome
This project successfully demonstrates how to build, evaluate, and deploy a machine learning model for predicting concrete compressive strength, providing a --practical tool for civil engineers and material scientists. The GUI application makes the prediction system accessible to users without a background in programming or machine learning.








