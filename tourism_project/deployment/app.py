import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing the Wellness Tourism Package
based on their profile and trip preferences.
Enter the customer's details below to get a prediction.
""")

Age                      = st.number_input("Age", 18, 100, 35)
TypeofContact            = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier                 = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch          = st.number_input("Duration of Pitch (minutes)", 1, 60, 15)
Occupation               = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender                   = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting   = st.number_input("Number of Persons Visiting", 1, 10, 3)
NumberOfFollowups        = st.number_input("Number of Follow-ups", 0, 10, 4)
ProductPitched           = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar    = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus            = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips            = st.number_input("Number of Trips per Year", 0, 25, 3)
Passport                 = st.selectbox("Holds a Passport", ["Yes", "No"])
PitchSatisfactionScore   = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar                   = st.selectbox("Owns a Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (under 5)", 0, 5, 1)
Designation               = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome             = st.number_input("Monthly Income", 1000, 100000, 23000, step=500)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
