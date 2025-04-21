import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the wine dataset (or use your preprocessed dataset)
df = pd.read_csv('wine.data.csv', header=None)

# Ensure the dataset columns are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Rename columns for easier access
df.columns = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# Load the trained KNN model and scaler
knn = joblib.load('knn_wine_quality_model.joblib')
scaler = joblib.load('scaler.joblib')

# Streamlit app title
st.title("Wine Quality Prediction")

# Streamlit app description
st.write("""
This app predicts the quality of wine based on various features like alcohol content, acidity, and other chemical properties.
Use the slider to adjust the values and click "Predict" to see the quality prediction.
""")

# Dynamically get min and max values for each feature from the dataset, ensuring they are numeric
alcohol_min, alcohol_max = df['Alcohol'].min(), df['Alcohol'].max()
malic_acid_min, malic_acid_max = df['Malic acid'].min(), df['Malic acid'].max()
ash_min, ash_max = df['Ash'].min(), df['Ash'].max()
magnesium_min, magnesium_max = df['Magnesium'].min(), df['Magnesium'].max()
total_phenols_min, total_phenols_max = df['Total phenols'].min(), df['Total phenols'].max()
flavanoids_min, flavanoids_max = df['Flavanoids'].min(), df['Flavanoids'].max()

# Ensure that the min and max values are numeric
alcohol_min = float(alcohol_min)
alcohol_max = float(alcohol_max)
malic_acid_min = float(malic_acid_min)
malic_acid_max = float(malic_acid_max)
ash_min = float(ash_min)
ash_max = float(ash_max)
magnesium_min = float(magnesium_min)
magnesium_max = float(magnesium_max)
total_phenols_min = float(total_phenols_min)
total_phenols_max = float(total_phenols_max)
flavanoids_min = float(flavanoids_min)
flavanoids_max = float(flavanoids_max)

# Input fields with sliders for the 6 selected features, using min/max from the dataset
alcohol = st.slider('Alcohol', min_value=alcohol_min, max_value=alcohol_max, value=(alcohol_min + alcohol_max) / 2, step=0.1)
malic_acid = st.slider('Malic Acid', min_value=malic_acid_min, max_value=malic_acid_max, value=(malic_acid_min + malic_acid_max) / 2, step=0.1)
ash = st.slider('Ash', min_value=ash_min, max_value=ash_max, value=(ash_min + ash_max) / 2, step=0.1)
magnesium = st.slider('Magnesium', min_value=magnesium_min, max_value=magnesium_max, value=(magnesium_min + magnesium_max) / 2, step=0.1)  # Use step=0.1 for consistency
total_phenols = st.slider('Total Phenols', min_value=total_phenols_min, max_value=total_phenols_max, value=(total_phenols_min + total_phenols_max) / 2, step=0.1)
flavanoids = st.slider('Flavanoids', min_value=flavanoids_min, max_value=flavanoids_max, value=(flavanoids_min + flavanoids_max) / 2, step=0.1)

# When the user clicks the "Predict" button
if st.button('Predict'):
    # Prepare the input for prediction
    input_values = np.array([alcohol, malic_acid, ash, magnesium, total_phenols, flavanoids]).reshape(1, -1)
    
    # Scale the input using the pre-fitted scaler
    input_scaled = scaler.transform(input_values)
    
    # Make the prediction using the trained KNN model
    prediction = knn.predict(input_scaled)
    
    # Map the prediction to a quality label (e.g., 0 -> Low, 1 -> Medium, 2 -> High)
    quality_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    # Display the result
    st.write(f"The predicted wine quality class is: {quality_mapping[int(prediction[0])]}")  # Cast prediction to int
