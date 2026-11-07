import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Dropout Prediction", page_icon="🎓")
st.title("🎓 Student Dropout Prediction - ML")
st.write("Student details enter cheste Dropout avthado / Graduate avthado predict chesthundi.")

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", sep=";")
    return df

df = load_data()
X = df.drop(columns=["Target"])
y = df["Target"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.sidebar.header("Student Details")

# Main inputs - user ki easy ga
marital = st.sidebar.selectbox("Marital Status", [1,2,3,4,5,6], format_func=lambda x: {1:"Single",2:"Married",3:"Widower",4:"Divorced",5:"Facto Union",6:"Legally Separated"}[x])
app_mode = st.sidebar.number_input("Application Mode", 1, 60, 17)
app_order = st.sidebar.number_input("Application Order", 0, 9, 1)
course = st.sidebar.number_input("Course Code", 1, 9999, 171)
daytime = st.sidebar.selectbox("Daytime / Evening", [1,0], format_func=lambda x: "Daytime" if x==1 else "Evening")
prev_grade = st.sidebar.slider("Previous Qualification Grade", 0.0, 200.0, 120.0)
admission_grade = st.sidebar.slider("Admission Grade", 0.0, 200.0, 120.0)
age = st.sidebar.slider("Age at Enrollment", 17, 70, 20)
tuition = st.sidebar.selectbox("Tuition Fees Up To Date?", [1,0], format_func=lambda x: "Yes" if x==1 else "No")
scholarship = st.sidebar.selectbox("Scholarship Holder?", [1,0], format_func=lambda x: "Yes" if x==1 else "No")
debtor = st.sidebar.selectbox("Debtor?", [1,0], format_func=lambda x: "Yes" if x==1 else "No")
gender = st.sidebar.selectbox("Gender", [1,0], format_func=lambda x: "Male" if x==1 else "Female")
sem1_approved = st.sidebar.slider("1st Sem Approved Units", 0, 15, 5)
sem1_grade = st.sidebar.slider("1st Sem Grade", 0.0, 20.0, 12.0)
sem2_approved = st.sidebar.slider("2nd Sem Approved Units", 0, 15, 5)
sem2_grade = st.sidebar.slider("2nd Sem Grade", 0.0, 20.0, 12.0)

if st.button("🔮 Predict"):
    # Default row = median values, manam ichina values replace chestham
    row = X.median(numeric_only=True).to_dict()
    row["Marital status"] = marital
    row["Application mode"] = app_mode
    row["Application order"] = app_order
    row["Course"] = course
    row["Daytime/evening attendance\t"] = daytime
    # Column name lo space undochu, safe ga handle chey
    for col in X.columns:
        if "Previous qualification (grade)" in col:
            row[col] = prev_grade
        if "Admission grade" in col:
            row[col] = admission_grade
        if "Age at enrollment" in col:
            row[col] = age
        if "Tuition fees up to date" in col:
            row[col] = tuition
        if "Scholarship holder" in col:
            row[col] = scholarship
        if "Debtor" in col:
            row[col] = debtor
        if "Gender" in col:
            row[col] = gender
        if "Curricular units 1st sem (approved)" in col:
            row[col] = sem1_approved
        if "Curricular units 1st sem (grade)" in col:
            row[col] = sem1_grade
        if "Curricular units 2nd sem (approved)" in col:
            row[col] = sem2_approved
        if "Curricular units 2nd sem (grade)" in col:
            row[col] = sem2_grade

    input_df = pd.DataFrame([row])[X.columns]
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0].max()*100

    if pred == "Dropout":
        st.error(f"⚠️ **Prediction: Dropout** ({proba:.1f}% confidence)")
        st.write("Ee student ki extra support / counselling avasaram.")
    elif pred == "Graduate":
        st.success(f"✅ **Prediction: Graduate** ({proba:.1f}% confidence)")
    else:
        st.warning(f"ℹ️ **Prediction: Enrolled** ({proba:.1f}% confidence)")

st.markdown("---")
st.write(f"Dataset: {len(df)} students | Model: Random Forest")
