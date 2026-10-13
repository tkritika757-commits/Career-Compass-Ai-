import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------
st.set_page_config(
    page_title="Career Compass AI",
    page_icon="🎯",
    layout="wide"
)

# -----------------------------------
# APPLICATION HEADING
# -----------------------------------
st.title("🎯 Career Compass AI")
st.subheader("Discover a Career That Matches Your Interests")
st.caption("Career Compass AI | Class 12 AI Project")

st.write(
    "Answer a few questions to discover suitable career options, "
    "your strengths, required skills, and courses after Class 12."
)

st.divider()

# -----------------------------------
# SAMPLE TRAINING DATASET
# -----------------------------------
data = {
    "Subject": [
        "Science", "Science", "Science", "Science",
        "Commerce", "Commerce", "Commerce", "Commerce",
        "Arts", "Arts", "Arts", "Arts",
        "Science", "Commerce", "Arts", "Science",
        "Commerce", "Arts"
    ],
    "Interest": [
        "Technology", "Healthcare", "Research", "Technology",
        "Business", "Finance", "Business", "Finance",
        "Writing", "Design", "Social Work", "Writing",
        "Research", "Business", "Design", "Healthcare",
        "Finance", "Social Work"
    ],
    "Work_Style": [
        "Problem Solving", "Helping People", "Independent",
        "Problem Solving", "Leadership", "Analytical",
        "Leadership", "Analytical", "Creative", "Creative",
        "Helping People", "Creative", "Independent",
        "Leadership", "Creative", "Helping People",
        "Analytical", "Helping People"
    ],
    "Career": [
        "Software Engineer", "Doctor", "Research Scientist",
        "AI Engineer", "Business Manager", "Financial Analyst",
        "Entrepreneur", "Accountant", "Content Writer",
        "Graphic Designer", "Social Worker", "Journalist",
        "Research Scientist", "Entrepreneur", "Graphic Designer",
        "Doctor", "Financial Analyst", "Social Worker"
    ]
}

df = pd.DataFrame(data)

# -----------------------------------
# TRAIN THE AI MODEL
# -----------------------------------
X = df[["Subject", "Interest", "Work_Style"]]
y = df["Career"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Subject", "Interest", "Work_Style"]
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        )
    ]
)

model.fit(X, y)

# -----------------------------------
# CAREER INFORMATION
# -----------------------------------
career_info = {
    "Software Engineer": {
        "strengths": "Logical thinking and problem-solving",
        "skills": "Python, programming, algorithms, and teamwork",
        "courses": "B.Tech in Computer Science, BCA, or B.Sc. Computer Science",
        "explanation": "Your science background, technology interest, "
                       "and problem-solving preference match this career."
    },
    "AI Engineer": {
        "strengths": "Analytical thinking and logical reasoning",
        "skills": "Python, mathematics, machine learning, and data analysis",
        "courses": "B.Tech in AI/Computer Science or B.Sc. Computer Science "
                   "with relevant AI training",
        "explanation": "Your interest in technology and problem-solving "
                       "may suit a career in artificial intelligence."
    },
    "Doctor": {
        "strengths": "Empathy, patience, and scientific thinking",
        "skills": "Biology, communication, observation, and decision-making",
        "courses": "MBBS through the applicable NEET-UG admission process",
        "explanation": "Your interest in healthcare and helping people "
                       "matches the responsibilities of a medical career."
    },
    "Research Scientist": {
        "strengths": "Curiosity, observation, and critical thinking",
        "skills": "Research methods, data analysis, and scientific reasoning",
        "courses": "B.Sc. followed by further study, or an appropriate "
                   "integrated science degree",
        "explanation": "Your interest in research and independent work "
                       "may suit scientific investigation."
    },
    "Business Manager": {
        "strengths": "Leadership, planning, and communication",
        "skills": "Management, teamwork, decision-making, and organization",
        "courses": "BBA, B.Com., or another suitable management-related degree",
        "explanation": "Your business interest and leadership preference "
                       "may suit management."
    },
    "Financial Analyst": {
        "strengths": "Numerical reasoning and analytical thinking",
        "skills": "Excel, accounting, financial analysis, and statistics",
        "courses": "B.Com., BBA in Finance, or another relevant degree",
        "explanation": "Your interest in finance and analytical work "
                       "may suit financial analysis."
    },
    "Entrepreneur": {
        "strengths": "Initiative, creativity, and leadership",
        "skills": "Business planning, communication, budgeting, and marketing",
        "courses": "BBA, B.Com., or another relevant degree",
        "explanation": "Your business interest and leadership preference "
                       "may suit starting or managing a business."
    },
    "Accountant": {
        "strengths": "Accuracy, organization, and numerical reasoning",
        "skills": "Accounting, Excel, taxation, and financial reporting",
        "courses": "B.Com. or a suitable accounting-related qualification",
        "explanation": "Your commerce background and interest in finance "
                       "may suit accounting."
    },
    "Content Writer": {
        "strengths": "Creativity, language, and communication",
        "skills": "Writing, research, editing, and grammar",
        "courses": "BA in English, Journalism, or another relevant degree",
        "explanation": "Your interest in writing and creative work "
                       "may suit content creation."
    },
    "Graphic Designer": {
        "strengths": "Visual creativity and imagination",
        "skills": "Design principles, typography, color theory, and design tools",
        "courses": "B.Des., a suitable design degree, or a relevant diploma",
        "explanation": "Your interest in design and creative work "
                       "may suit graphic design."
    },
    "Social Worker": {
        "strengths": "Empathy, patience, and interpersonal communication",
        "skills": "Communication, community support, and problem-solving",
        "courses": "BSW or another suitable social-science degree",
        "explanation": "Your interest in social work and helping people "
                       "may suit community-focused work."
    },
    "Journalist": {
        "strengths": "Communication, curiosity, and observation",
        "skills": "Writing, interviewing, research, and fact-checking",
        "courses": "BA in Journalism or a relevant mass communication degree",
        "explanation": "Your interest in writing and communication "
                       "may suit journalism."
    }
}

# -----------------------------------
# USER INPUT SECTION
# -----------------------------------
st.header("📝 Tell Us About Yourself")

col1, col2, col3 = st.columns(3)

with col1:
    subject = st.selectbox(
        "Choose your stream",
        ["Science", "Commerce", "Arts"]
    )

with col2:
    interest = st.selectbox(
        "What interests you most?",
        [
            "Technology",
            "Healthcare",
            "Research",
            "Business",
            "Finance",
            "Writing",
            "Design",
            "Social Work"
        ]
    )

with col3:
    work_style = st.selectbox(
        "Which work style do you prefer?",
        [
            "Problem Solving",
            "Helping People",
            "Independent",
            "Leadership",
            "Analytical",
            "Creative"
        ]
    )

st.write("")

# -----------------------------------
# CAREER PREDICTION
# -----------------------------------
if st.button("🔍 Find My Career", type="primary"):

    user_data = pd.DataFrame(
        [{
            "Subject": subject,
            "Interest": interest,
            "Work_Style": work_style
        }]
    )

    probabilities = model.predict_proba(user_data)[0]
    classes = model.named_steps["classifier"].classes_

    ranked_indices = probabilities.argsort()[::-1]

    recommendations = []

    for index in ranked_indices:
        career = classes[index]

        if career not in recommendations:
            recommendations.append(career)

        if len(recommendations) == 3:
            break

    st.divider()
    st.header("🌟 Your Career Recommendations")

    st.info(
        "These are AI-generated suggestions based on your answers. "
        "Use them as a starting point for career exploration."
    )

    for rank, career in enumerate(recommendations, start=1):

        details = career_info[career]

        with st.container(border=True):

            st.subheader(f"{rank}. {career}")

            st.write("💪 **Your potential strengths**")
            st.write(details["strengths"])

            st.write("🛠️ **Skills to develop**")
            st.write(details["skills"])

            st.write("🎓 **Suitable courses after Class 12**")
            st.write(details["courses"])

            st.write("🤖 **AI explanation**")
            st.write(details["explanation"])

            st.write(
                f"**Model score:** {probabilities[classes.tolist().index(career)] * 100:.1f}%"
            )

    st.divider()

    st.subheader("💡 A Helpful Career Tip")

    st.write(
        "Before choosing a career, explore the course syllabus, "
        "eligibility requirements, fees, and future opportunities. "
        "Talk to teachers, counsellors, and professionals in the field."
    )

# -----------------------------------
# FOOTER
# -----------------------------------
st.divider()

st.caption(
    "Career Compass AI | Educational AI Project | Class 12"
)

st.caption(
    "Note: This demonstration uses a small sample dataset. "
    "Recommendations are educational suggestions, not guaranteed "
    "predictions of career success."
)
