import streamlit as st
import pandas as pd
import pickle

# ------------------ LOAD MODEL ------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# ------------------ UI ------------------
st.set_page_config(page_title="Placement Prediction", layout="centered")

st.title("🎓 Placement Prediction System")

# ------------------ INPUTS ------------------

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)

internships = st.number_input("Internships", 0, 10, 0)
projects = st.number_input("Projects", 0, 10, 0)
workshops = st.number_input("Workshops", 0, 10, 0)
hackathons = st.number_input("Hackathons", 0, 10, 0)

# ------------------ PREDICT BUTTON ------------------

if st.button("Predict"):

    # -------- INPUT DATA --------
    data = {
        'Student_ID': [0],
        'College_ID': [0],
        'Department': ['CSE'],
        'CGPA': [cgpa],
        'Admission_Type': ['Regular'],
        'Skill_Program': [1],
        'Program_Duration': [12],
        'Bootcamps': [1],
        'Workshops': [workshops],
        'Hackathons': [hackathons],
        'Internships': [internships],
        'Projects': [projects],
        'GenAI': [1],
        'ML': [1],
        'College_Tier': [2],
        'NAAC_Grade': ['A+'],
        'NBA_Accredited': ['Yes'],
        'Placement_Probability': [0],
        'Salary_LPA': [0]
    }

    # -------- CONVERT TO DATAFRAME --------
    df = pd.DataFrame(data)

    # -------- PREPROCESS --------
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)

    # -------- PREDICT --------
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    # ------------------ OUTPUT ------------------

    st.subheader(f"📊 Placement Probability: {round(prob*100,2)}%")

    if prediction == 1:
        st.success("🎉 Likely to get placed")
    else:
        st.error("⚠️ Low chance of placement")

    # ------------------ GRAPH (WORKING VERSION) ------------------

    st.subheader("📊 Result Visualization")

    chart_data = pd.DataFrame({
        'Status': ['Not Placed', 'Placed'],
        'Probability': [1 - prob, prob]
    })

    st.bar_chart(chart_data.set_index('Status'))

    # ------------------ SUGGESTIONS ------------------

    if prob < 0.6:
        st.subheader("💡 Improve your profile:")
        st.write("👉 Do internships")
        st.write("👉 Build strong projects")
        st.write("👉 Participate in hackathons")

# ------------------ SIDEBAR ------------------

st.sidebar.title("About")
st.sidebar.info("This app predicts student placement chances using Machine Learning.")

if prob >= 0.7:
    st.success("🔥 Strong profile! High placement chances.")
elif prob >= 0.5:
    st.info("👍 Moderate chances. Improve a few areas.")
else:
    st.warning("⚠️ Low chances. Focus on skill-building.")

    st.subheader("📌 Key Factors Affecting Placement")

st.write("Top influencing factors:")
st.write("- Internships")
st.write("- Projects")
st.write("- Skill Programs")
st.write("- CGPA")

st.markdown("### 🎯 Check your placement chances instantly!")