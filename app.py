import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Career Compass AI", page_icon="🎯", layout="wide")

st.title("🎯 Career Compass AI")
st.subheader("Discover Your Interests. Explore Your Future.")
st.write("Your journey towards the right career starts here!")

st.info("Explore possible career paths based on your school stream, interests, and preferred work style.")

data = pd.DataFrame([
["Science", "Technology", "Analytical", "AI Engineer"],
["Science", "Technology", "Creative", "Software Developer"],
["Science", "Healthcare", "Helping People", "Doctor"],
["Science", "Healthcare", "Analytical", "Biomedical Scientist"],
["Science", "Art & Design", "Creative", "Game Designer"],
["Science", "Technology", "Helping People", "AI Ethics Researcher"],
["Science", "Technology", "Analytical", "Data Scientist"],
["Science", "Healthcare", "Helping People", "Nurse"],
["Commerce", "Finance", "Analytical", "Financial Analyst"],
["Commerce", "Business", "Analytical", "Business Analyst"],
["Commerce", "Technology", "Analytical", "Data Analyst"],
["Commerce", "Finance", "Helping People", "Financial Advisor"],
["Commerce", "Business", "Creative", "Entrepreneur"],
["Commerce", "Art & Design", "Creative", "Product Designer"],
["Humanities", "Writing", "Creative", "Journalist"],
["Humanities", "Writing", "Helping People", "Teacher"],
["Humanities", "Art & Design", "Creative", "Graphic Designer"],
["Humanities", "Business", "Creative", "Marketing Specialist"],
["Humanities", "Healthcare", "Helping People", "Counselor"],
["Humanities", "Technology", "Analytical", "UX Researcher"]
], columns=["Subject", "Interest", "Work_Style", "Career"])

features = ["Subject", "Interest", "Work_Style"]

preprocessor = ColumnTransformer([
("categories", OneHotEncoder(handle_unknown="ignore"), features)
])

model = Pipeline([
("preprocessor", preprocessor),
("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
])

model.fit(data[features], data["Career"])

descriptions = {
"AI Engineer": "Develops artificial intelligence systems and applications.",
"Software Developer": "Designs and builds software applications.",
"Doctor": "Diagnoses illnesses and treats patients.",
"Biomedical Scientist": "Studies biology and medical science.",
"Game Designer": "Creates concepts and experiences for video games.",
"AI Ethics Researcher": "Studies the responsible use of artificial intelligence.",
"Data Scientist": "Uses data to solve problems and identify patterns.",
"Nurse": "Supports patient care and medical treatment.",
"Financial Analyst": "Studies financial information to support decisions.",
"Business Analyst": "Helps organizations improve business processes.",
"Data Analyst": "Examines data to find useful patterns.",
"Financial Advisor": "Helps people plan and manage their finances.",
"Entrepreneur": "Creates and manages a business.",
"Product Designer": "Designs useful and user-friendly products.",
"Journalist": "Researches and reports news and stories.",
"Teacher": "Helps students learn and develop their skills.",
"Graphic Designer": "Creates visual designs and communication materials.",
"Marketing Specialist": "Helps promote products and services.",
"Counselor": "Supports people with educational and personal choices.",
"UX Researcher": "Studies user needs to improve digital experiences."
}

st.header("🔎 Find Your Career")

col1, col2 = st.columns(2)

with col1:
subject = st.selectbox(
"🎓 Choose your school stream",
["Science", "Commerce", "Humanities"]
)

interest = st.selectbox(
    "💡 What interests you most?",
    ["Technology", "Finance", "Healthcare", "Writing", "Business", "Art & Design"]
)

with col2:
work_style = st.selectbox(
"🧠 Which work style suits you?",
["Analytical", "Creative", "Helping People"]
)

st.caption("Choose the options that best describe you.")

if st.button("🚀 Find My Career", use_container_width=True):
inputs = pd.DataFrame([{
"Subject": subject,
"Interest": interest,
"Work_Style": work_style
}])

prediction = model.predict(inputs)[0]

st.success("🎓 Your suggested career: " + prediction)

st.subheader("About This Career")
st.write(descriptions[prediction])

st.subheader("🧭 Suggested Next Steps")
st.write("1. Research the qualifications required for this career.")
st.write("2. Explore relevant courses and colleges.")
st.write("3. Talk to teachers or professionals in this field.")
st.write("4. Compare other careers that interest you.")

st.warning(
    "This educational demonstration uses a small sample dataset. "
    "Its predictions are not validated career assessments. "
    "Explore multiple careers before making a decision."
)

st.divider()
st.header("🌟 Explore Career Fields")

career_fields = {
"Technology": "AI Engineer, Software Developer, Data Scientist",
"Healthcare": "Doctor, Nurse, Biomedical Scientist",
"Finance": "Financial Analyst, Financial Advisor, Accountant",
"Business": "Entrepreneur, Business Analyst, Marketing Specialist",
"Writing": "Journalist, Author, Content Writer",
"Art & Design": "Graphic Designer, Product Designer, Game Designer"
}

for field, careers in career_fields.items():
with st.expander("Explore " + field):
st.write(careers)

st.divider()
st.caption("Career Compass AI | Class 12 AI Project")
st.caption("Designed to support career exploration and informed decisions.")
