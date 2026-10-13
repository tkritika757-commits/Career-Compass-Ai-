import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Career Compass AI", page_icon="🎯")

st.title("🎯 Career Compass AI")
st.subheader("Discover Your Interests. Explore Your Future.")

st.write("Find career ideas based on your subject, interests, and work style.")

data = pd.DataFrame({
    "Subject": [
        "Science", "Science", "Science", "Science",
        "Science", "Science", "Commerce", "Commerce",
        "Commerce", "Commerce", "Commerce", "Commerce",
        "Humanities", "Humanities", "Humanities",
        "Humanities", "Humanities", "Humanities"
    ],
    "Interest": [
        "Technology", "Technology", "Healthcare",
        "Healthcare", "Art & Design", "Technology",
        "Finance", "Business", "Technology", "Finance",
        "Business", "Art & Design", "Writing", "Writing",
        "Art & Design", "Business", "Healthcare", "Technology"
    ],
    "Work_Style": [
        "Analytical", "Creative", "Helping People",
        "Analytical", "Creative", "Helping People",
        "Analytical", "Analytical", "Analytical", "Helping People",
        "Creative", "Creative", "Creative", "Helping People",
        "Creative", "Creative", "Helping People", "Analytical"
    ],
    "Career": [
        "AI Engineer", "Software Developer", "Doctor",
        "Biomedical Scientist", "Game Designer",
        "AI Ethics Researcher", "Financial Analyst",
        "Business Analyst", "Data Analyst", "Financial Advisor",
        "Entrepreneur", "Product Designer", "Journalist",
        "Teacher", "Graphic Designer", "Marketing Specialist",
        "Counselor", "UX Researcher"
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

st.header("Find Your Career")

subject = st.selectbox(
    "Choose your school subject",
    ["Science", "Commerce", "Humanities"]
)

interest = st.selectbox(
    "What interests you most?",
    ["Technology", "Finance", "Healthcare", "Writing", "Business", "Art & Design"]
)

work_style = st.selectbox(
    "Which work style suits you?",
    ["Analytical", "Creative", "Helping People"]
)

if st.button("Find My Career"):
    inputs = pd.DataFrame([{
        "Subject": subject,
        "Interest": interest,
        "Work_Style": work_style
    }])

    prediction = model.predict(inputs)[0]

    st.success("Your suggested career:")
    st.subheader("🎓 " + prediction)

    st.info(
        "This is a demonstration using a small sample dataset. "
        "Explore other careers too; this is not a definitive career assessment."
    )

st.divider()
st.caption("Career Compass AI | Class 12 AI Project")
