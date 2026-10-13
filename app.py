import streamlit as st
import pandas as pd
sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(
page_title="Career Compass AI",
page_icon="🎯",
layout="wide"
)

st.markdown("""

<style>
.stApp {
    background-color: #F3F7FF;
}
h1, h2, h3 {
    color: #173B70;
}
.hero {
    background: linear-gradient(120deg, #123B70, #3478D4);
    padding: 28px;
    border-radius: 18px;
    color: white;
    text-align: center;
}
.hero h1 {
    color: white;
}
.hero p {
    font-size: 18px;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #D7E5FA;
    margin-bottom: 12px;
}
</style>""", unsafe_allow_html=True)

st.markdown("""

<div class="hero">
    <h1>🎯 Career Compass AI</h1>
    <p>Discover Your Interests. Explore Your Future.</p>
    <p>Your journey towards the right career starts here.</p>
</div>
""", unsafe_allow_html=True)st.write("")
st.info(
"Explore possible career paths based on your school stream, "
"interests, and preferred work style."
)

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

features = ["Subject", "Interest", "Work_Style"]

preprocessor = ColumnTransformer([
("categories", OneHotEncoder(handle_unknown="ignore"), features)
])

model = Pipeline([
("preprocessor", preprocessor),
("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
])

model.fit(data[features], data["Career"])

st.header("🔎 Find Your Career")

col1, col2 = st.columns(2)

with col1:
subject = st.selectbox(
"🎓 Choose your school stream",
["Science", "Commerce", "Humanities"]
)

interest = st.selectbox(
    "💡 What interests you most?",
    [
        "Technology", "Finance", "Healthcare",
        "Writing", "Business", "Art & Design"
    ]
)

with col2:
work_style = st.selectbox(
"🧠 Which work style suits you?",
["Analytical", "Creative", "Helping People"]
)

st.write("")
st.write("Select the options that best describe you.")

if st.button("🚀 Find My Career", use_container_width=True):
inputs = pd.DataFrame([{
"Subject": subject,
"Interest": interest,
"Work_Style": work_style
}])

prediction = model.predict(inputs)[0]

descriptions = {
    "AI Engineer": "Develops AI systems and intelligent applications.",
    "Software Developer": "Designs, builds, and maintains software.",
    "Doctor": "Diagnoses illnesses and helps treat patients.",
    "Biomedical Scientist": "Studies biology and develops medical knowledge.",
    "Game Designer": "Creates concepts and experiences for video games.",
    "AI Ethics Researcher": "Studies responsible and fair use of AI.",
    "Data Scientist": "Uses data and statistics to solve problems.",
    "Nurse": "Supports patient care and medical treatment.",
    "Financial Analyst": "Studies financial information to support decisions.",
    "Business Analyst": "Uses information to improve business processes.",
    "Data Analyst": "Examines data to identify useful patterns.",
    "Financial Advisor": "Helps people plan and manage their finances.",
    "Entrepreneur": "Builds and manages a business or new venture.",
    "Product Designer": "Designs useful and user-friendly products.",
    "Journalist": "Researches, writes, and reports news and stories.",
    "Teacher": "Helps students learn and develop their skills.",
    "Graphic Designer": "Creates visual designs and communication materials.",
    "Marketing Specialist": "Helps organizations promote products and services.",
    "Counselor": "Supports people as they explore personal or educational choices.",
    "UX Researcher": "Studies user needs to improve digital experiences."
}

st.markdown("---")
st.subheader("🎓 Your Suggested Career")

st.success(f"### {prediction}")

st.markdown(
    f'<div class="card"><h3>About this career</h3>'
    f'<p>{descriptions[prediction]}</p></div>',
    unsafe_allow_html=True
)

st.subheader("🧭 Suggested Next Steps")
st.write("1. Research the qualifications required for this career.")
st.write("2. Explore relevant courses and colleges.")
st.write("3. Talk to teachers or professionals in this field.")
st.write("4. Compare this option with other careers that interest you.")

st.warning(
    "Educational demonstration: this model uses a small, "
    "illustrative dataset. Its predictions are not validated "
    "career assessments. Explore multiple options before deciding."
)

st.markdown("---")

st.subheader("🌟 Explore Career Fields")

career_fields = {
"Technology": "AI Engineer, Software Developer, Data Scientist",
"Healthcare": "Doctor, Nurse, Biomedical Scientist",
"Finance": "Financial Analyst, Financial Advisor, Accountant",
"Business": "Entrepreneur, Business Analyst, Marketing Specialist",
"Writing": "Journalist, Author, Content Writer",
"Art & Design": "Graphic Designer, Product Designer, Game Designer"
}

for field, careers in career_fields.items():
with st.expander(f"Explore {field}"):
st.write(careers)

st.markdown("---")
st.caption("Career Compass AI | Class 12 AI Project")
st.caption("Designed to support career exploration and informed decisions.")
