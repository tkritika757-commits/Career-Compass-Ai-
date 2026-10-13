import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Career Compass AI", page_icon="🎯", layout="wide")

# Application heading

st.title("🎯 Career Compass AI")
st.subheader("Discover Your Interests. Explore Your Future.")
st.write("Your journey towards the right career starts here!")

st.info("Explore possible career paths based on your school stream, interests, and preferred work style.")

Sample dataset

data = pd.DataFrame({
"Subject": [
"Science", "Science", "Science", "Science",
"Science", "Science", "Science", "Science",
"Commerce", "Commerce", "Commerce", "Commerce",
"Commerce", "Commerce", "Humanities", "Humanities",
"Humanities", "Humanities", "Humanities", "Humanities"
],
"Interest": [
"Technology", "Technology", "Healthcare", "Healthcare",
"Art & Design", "Technology", "Technology", "Healthcare",
"Finance", "Business", "Technology", "Finance",
"Business", "Art & Design", "Writing", "Writing",
"Art & Design", "Business", "Healthcare", "Technology"
],
"Work_Style": [
"Analytical", "Creative", "Helping People", "Analytical",
"Creative", "Helping People", "Analytical", "Helping People",
"Analytical", "Analytical", "Analytical", "Helping People",
"Creative", "Creative", "Creative", "Helping People",
"Creative", "Creative", "Helping People", "Analytical"
],
"Career": [
"AI Engineer", "Software Developer", "Doctor",
"Biomedical Scientist", "Game Designer",
"AI Ethics Researcher", "Data Scientist", "Nurse",
"Financial Analyst", "Business Analyst", "Data Analyst",
"Financial Advisor", "Entrepreneur", "Product Designer",
"Journalist", "Teacher", "Graphic Designer",
"Marketing Specialist", "Counselor", "UX Researcher"
]
})

Train the machine-learning model

features = ["Subject", "Interest", "Work_Style"]

preprocessor = ColumnTransformer([
("categories", OneHotEncoder(handle_unknown="ignore"), features)
])

model = Pipeline([
("preprocessor", preprocessor),
("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
])

model.fit(data[features], data["Career"])

Career descriptions

descriptions = {
"AI Engineer": "Develops artificial intelligence systems.",
"Software Developer": "Designs and builds software applications.",
"Doctor": "Diagnoses illnesses and treats patients.",
"Biomedical Scientist": "Studies biology and medical science.",
"Game Designer": "Creates concepts for video games.",
"AI Ethics Researcher": "Studies the responsible use of AI.",
"Data Scientist": "Uses data to solve problems.",
"Nurse": "Supports patient care and treatment.",
"Financial Analyst": "Studies financial information.",
"Business Analyst": "Helps improve business processes.",
"Data Analyst": "Examines data to find useful patterns.",
"Financial Advisor": "Helps people plan their finances.",
"Entrepreneur": "Creates and manages a business.",
"Product Designer": "Designs useful products.",
"Journalist": "Researches and reports news and stories.",
"Teacher": "Helps students learn.",
"Graphic Designer": "Creates visual designs.",
"Marketing Specialist": "Promotes products and services.",
"Counselor": "Supports people with educational choices.",
"UX Researcher": "Studies user needs to improve digital experiences."
}

Career finder

st.header("🔎 Find Your Career")

subject = st.selectbox(
"🎓 Choose your school stream",
["Science", "Commerce", "Humanities"]
)

interest = st.selectbox(
"💡 What interests you most?",
["Technology", "Finance", "Healthcare", "Writing", "Business", "Art & Design"]
)

work_style = st.selectbox(
"🧠 Which work style suits you?",
["Analytical", "Creative", "Helping People"]
)

if st.button("🚀 Find My Career"):
inputs = pd.DataFrame({
"Subject": [subject],
"Interest": [interest],
"Work_Style": [work_style]
})

prediction = model.predict(inputs)[0]

st.success("🎓 Your suggested career: " + prediction)

st.subheader("About This Career")
st.write(descriptions[prediction])

st.subheader("🧭 Suggested Next Steps")
st.write("1. Research the qualifications required.")
st.write("2. Explore relevant courses and colleges.")
st.write("3. Talk to teachers or professionals.")
st.write("4. Explore other careers that interest you.")

st.warning(
    "This is an educational demonstration using a small sample dataset. "
    "Its predictions are not validated career assessments."
)

Career exploration

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

Footer

st.divider()
st.caption("Career Compass AI | Class 12 AI Project")
st.caption("Discover your interests. Explore your future.")
