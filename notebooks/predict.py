import pickle
import pandas as pd

# ✅ load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ✅ load column names
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# ---- USER INPUT (raw form) ----
input_data = {
    'Student_ID': [0],
    'College_ID': [0],
    'Department': ['CSE'],
    'CGPA': [8.5],
    'Admission_Type': ['Regular'],
    'Skill_Program': [1],
    'Program_Duration': [12],
    'Bootcamps': [1],
    'Workshops': [2],
    'Hackathons': [1],
    'Internships': [2],
    'Projects': [3],
    'GenAI': [1],
    'ML': [1],
    'College_Tier': [2],
    'NAAC_Grade': ['A+'],
    'NBA_Accredited': ['Yes'],
    'Placement_Probability': [0],
    'Salary_LPA': [0]
}

# convert to dataframe
df_input = pd.DataFrame(input_data)

# same preprocessing as training
df_input = pd.get_dummies(df_input)

# align columns with training
df_input = df_input.reindex(columns=model_columns, fill_value=0)

# prediction
prediction = model.predict(df_input)
prob = model.predict_proba(df_input)[0][1]

print("Prediction:", prediction[0])
print("Placement Probability:", round(prob * 100, 2), "%")

if prediction[0] == 1:
    print("Likely to get placed")
else:
    print("Low chance of placement")

    st.markdown("---")
st.subheader("📊 Result")

if prob < 0.6:
    st.warning("👉 Improve your profile:")
    st.write("- Do internships")
    st.write("- Build strong projects")
    st.write("- Participate in hackathons")

    st.sidebar.title("About")
st.sidebar.info("This app predicts student placement chances using ML.")