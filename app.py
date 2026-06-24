import streamlit as st
import pandas as pd
import joblib

# Load saved objects
model = joblib.load("models/knn.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="centered"
)

st.title("👨‍💼 Employee Attrition Prediction")
st.write("Enter employee details below.")

# Numerical Inputs
age = st.number_input("Age", min_value=18, max_value=65, value=30)
daily_rate = st.number_input("Daily Rate", min_value=100, value=800)
distance_from_home = st.number_input("Distance From Home", min_value=0, value=5)
education = st.selectbox("Education", [1, 2, 3, 4, 5])
environment_satisfaction = st.selectbox(
    "Environment Satisfaction", [1, 2, 3, 4]
)
hourly_rate = st.number_input("Hourly Rate", min_value=10, value=50)
job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
monthly_income = st.number_input("Monthly Income", min_value=1000, value=5000)
monthly_rate = st.number_input("Monthly Rate", min_value=1000, value=10000)
num_companies_worked = st.number_input(
    "Number of Companies Worked", min_value=0, value=2
)
percent_salary_hike = st.number_input(
    "Percent Salary Hike", min_value=0, value=15
)
performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
relationship_satisfaction = st.selectbox(
    "Relationship Satisfaction", [1, 2, 3, 4]
)
stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
total_working_years = st.number_input(
    "Total Working Years", min_value=0, value=10
)
training_times_last_year = st.number_input(
    "Training Times Last Year", min_value=0, value=2
)
work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
years_at_company = st.number_input(
    "Years At Company", min_value=0, value=5
)
years_in_current_role = st.number_input(
    "Years In Current Role", min_value=0, value=3
)
years_since_last_promotion = st.number_input(
    "Years Since Last Promotion", min_value=0, value=1
)
years_with_curr_manager = st.number_input(
    "Years With Current Manager", min_value=0, value=3
)

# Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Female"])

business_travel = st.selectbox(
    "Business Travel",
    [
        "Travel_Rarely",
        "Travel_Frequently",
        "Non-Travel"
    ]
)

department = st.selectbox(
    "Department",
    [
        "Research & Development",
        "Sales",
        "Human Resources"
    ]
)

education_field = st.selectbox(
    "Education Field",
    [
        "Life Sciences",
        "Medical",
        "Marketing",
        "Technical Degree",
        "Other",
        "Human Resources"
    ]
)

job_role = st.selectbox(
    "Job Role",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

overtime = st.selectbox(
    "Over Time",
    [
        "Yes",
        "No"
    ]
)

# Predict Button
if st.button("Predict Attrition"):

    input_dict = {
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EnvironmentSatisfaction": environment_satisfaction,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
        "Gender": gender,
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "OverTime": overtime
    }

    # Create dataframe
    input_df = pd.DataFrame([input_dict])

    # Encode categorical columns
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Employee is likely to leave the company.")
    else:
        st.success("✅ Employee is likely to stay.")

    # Probability (if supported)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0]

        st.write(
            f"**Stay Probability:** {prob[0]:.2%}"
        )
        st.write(
            f"**Attrition Probability:** {prob[1]:.2%}"
        )