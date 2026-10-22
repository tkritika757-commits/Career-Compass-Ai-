import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CareerCompass AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# PREMIUM DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #10131f, #171b2d, #10131f);
    color: #f5f3ee;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #e8c878;
}

p, label, li {
    color: #e5e7eb;
}

.hero {
    background: linear-gradient(135deg, #252b42, #171b2d);
    padding: 38px;
    border-radius: 24px;
    border: 1px solid #74633d;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.subtitle {
    font-size: 17px;
    color: #d1d5db;
}

.card {
    background: #20253a;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #414963;
    margin: 12px 0;
}

.salary {
    font-size: 23px;
    color: #e8c878;
    font-weight: 700;
}

.small-note {
    font-size: 12px;
    color: #b7bdcc;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #c8a75b, #f0d994);
    color: #151827;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    min-height: 44px;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    border: 1px solid #ffffff;
    color: #10131f;
}

[data-testid="stMetric"] {
    background: #20253a;
    border: 1px solid #414963;
    padding: 16px;
    border-radius: 15px;
}

hr {
    border-color: #414963;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CAREER DATABASE
# Salary ranges are illustrative estimates in INR per year.
# Update them with current local labour-market research
# before presenting them as researched figures.
# =========================================================

CAREERS = []


def add_career(field, names, courses, salary,
               interest, work_style, benefits, skills):
    for name in names:
        CAREERS.append({
            "Career": name,
            "Field": field,
            "Qualifications": courses,
            "Salary": salary,
            "Interest": interest,
            "Work Style": work_style,
            "Benefits": benefits,
            "Skills": skills
        })


# MEDICAL AND HEALTHCARE

add_career(
    "Medical & Healthcare",
    ["General Physician", "Surgeon", "Cardiologist",
     "Dermatologist", "Neurologist", "Paediatrician",
     "Psychiatrist", "Radiologist", "Anaesthesiologist",
     "Gynaecologist", "Orthopaedic Doctor",
     "Ophthalmologist", "ENT Specialist",
     "Pathologist", "Emergency Medicine Doctor"],
    "MBBS followed by the required postgraduate training "
    "and registration for the chosen specialty.",
    "₹6–30+ lakh",
    "Healthcare", "Analytical",
    "Patient care, specialised career pathways and opportunities "
    "in hospitals or private practice.",
    "Clinical reasoning, communication, responsibility"
)

add_career(
    "Medical & Healthcare",
    ["Registered Nurse", "Physiotherapist", "Pharmacist",
     "Medical Laboratory Technologist", "Radiology Technologist",
     "Occupational Therapist", "Nutritionist",
     "Audiologist", "Optometrist", "Speech Therapist",
     "Public Health Professional", "Dental Surgeon",
     "Veterinary Doctor", "Medical Research Assistant",
     "Healthcare Administrator"],
    "Relevant recognised degree or diploma, professional "
    "registration where required, and further training as applicable.",
    "₹2–12 lakh",
    "Healthcare", "Practical",
    "Hospitals, clinics, laboratories, community health "
    "and specialised service opportunities.",
    "Patient communication, practical skills, accuracy"
)


# BIOLOGY, BIOTECHNOLOGY AND RESEARCH

add_career(
    "Biology & Research",
    ["Biotechnologist", "Microbiologist", "Geneticist",
     "Molecular Biologist", "Cell Biologist",
     "Biochemist", "Immunologist", "Virologist",
     "Ecologist", "Wildlife Biologist",
     "Marine Biologist", "Botanist", "Zoologist",
     "Biomedical Researcher", "Clinical Research Associate"],
    "Relevant BSc or BS degree; MSc, specialised training "
    "or PhD may be required for advanced research roles.",
    "₹3–15 lakh",
    "Science & Research", "Analytical",
    "Research opportunities, scientific discovery, laboratories "
    "and biotechnology or healthcare industries.",
    "Observation, laboratory methods, data analysis"
)

add_career(
    "Biology & Research",
    ["Bioinformatics Analyst", "Biostatistician",
     "Epidemiologist", "Genetic Counsellor",
     "Clinical Data Coordinator", "Pharmaceutical Researcher",
     "Toxicologist", "Food Scientist",
     "Forensic Science Professional", "Science Teacher"],
    "Relevant degree in life sciences, statistics, computing, "
    "education or forensic science; specialisation as required.",
    "₹3–18 lakh",
    "Science & Research", "Analytical",
    "Interdisciplinary work, research, education and opportunities "
    "in laboratories or public institutions.",
    "Scientific reasoning, documentation, research"
)


# TECHNOLOGY AND AI

add_career(
    "Technology & IT",
    ["Software Developer", "Web Developer",
     "Mobile App Developer", "AI Engineer",
     "Machine Learning Engineer", "Data Scientist",
     "Data Analyst", "Cybersecurity Analyst",
     "Cloud Engineer", "DevOps Engineer",
     "Database Administrator", "Computer Network Engineer",
     "Robotics Engineer", "Computer Vision Engineer",
     "NLP Engineer"],
    "Relevant degree, diploma or equivalent demonstrable skills "
    "in computer science, mathematics or a related area.",
    "₹4–25 lakh",
    "Technology", "Analytical",
    "Opportunities across industries, project-based work "
    "and several possible specialisations.",
    "Programming, problem-solving, continuous learning"
)

add_career(
    "Technology & IT",
    ["UI/UX Designer", "Product Designer",
     "Game Developer", "IT Support Specialist",
     "Systems Analyst", "Software Tester",
     "Quality Assurance Engineer", "IT Project Coordinator",
     "Information Security Engineer", "Blockchain Developer",
     "Embedded Systems Engineer", "IoT Developer",
     "Technical Writer", "Computer Science Educator",
     "Business Intelligence Analyst"],
    "Relevant degree, diploma or portfolio-supported training "
    "depending on the position.",
    "₹3–20 lakh",
    "Technology", "Creative",
    "Diverse employers, practical problem-solving and "
    "opportunities to develop specialist skills.",
    "Technical communication, testing, design or coding"
)


# ENGINEERING AND PHYSICAL SCIENCES

add_career(
    "Engineering & Physical Sciences",
    ["Mechanical Engineer", "Civil Engineer",
     "Electrical Engineer", "Electronics Engineer",
     "Chemical Engineer", "Aerospace Engineer",
     "Automobile Engineer", "Environmental Engineer",
     "Renewable Energy Engineer", "Materials Scientist",
     "Physicist", "Astronomer", "Geologist",
     "Geophysicist", "Nuclear Scientist"],
    "Relevant BTech, BE, BSc or BS degree; advanced "
    "study or professional qualifications may be required.",
    "₹3–18 lakh",
    "Science & Research", "Practical",
    "Work in industry, infrastructure, laboratories, energy "
    "and research organisations.",
    "Mathematics, scientific reasoning, technical design"
)


# COMMERCE, FINANCE AND BUSINESS

add_career(
    "Business & Finance",
    ["Chartered Accountant", "Company Secretary",
     "Cost Accountant", "Accountant", "Auditor",
     "Financial Analyst", "Investment Analyst",
     "Banking Professional", "Tax Consultant",
     "Risk Analyst", "Actuary", "Economist",
     "Business Analyst", "Management Consultant",
     "Financial Planner"],
    "Relevant BCom, BBA, economics, finance or professional "
    "qualification; requirements vary by role.",
    "₹3–20 lakh",
    "Finance", "Analytical",
    "Career paths in organisations, financial services, "
    "consulting and specialist professional practice.",
    "Numeracy, analysis, ethics, communication"
)

add_career(
    "Business & Finance",
    ["Entrepreneur", "Marketing Manager",
     "Human Resources Specialist", "Operations Manager",
     "Supply Chain Analyst", "Logistics Manager",
     "Sales Manager", "Retail Manager",
     "Business Development Executive", "Project Manager",
     "Management Trainee", "Market Research Analyst",
     "E-commerce Manager", "Product Manager",
     "Customer Success Specialist"],
    "Relevant degree, practical experience and role-specific "
    "training; entrepreneurship has no single required degree.",
    "₹3–20 lakh",
    "Business", "Collaborative",
    "Varied industries, leadership development, teamwork "
    "and potential to build or manage projects.",
    "Planning, communication, organisation, decision-making"
)


# ARTS, DESIGN AND MEDIA

add_career(
    "Arts, Design & Media",
    ["Graphic Designer", "Fashion Designer",
     "Interior Designer", "Animation Artist",
     "Illustrator", "Photographer", "Video Editor",
     "Film Director", "Art Director", "Animator",
     "Game Artist", "Creative Writer",
     "Content Creator", "Copywriter", "UX Researcher"],
    "Relevant degree, diploma or portfolio-based training "
    "depending on the profession.",
    "₹2–15 lakh",
    "Art & Design", "Creative",
    "Creative projects, portfolio development, freelance "
    "possibilities and work across different industries.",
    "Creativity, visual communication, storytelling"
)

add_career(
    "Arts, Design & Media",
    ["Journalist", "News Editor", "Radio Presenter",
     "Public Relations Specialist", "Social Media Manager",
     "Digital Marketing Specialist", "Advertising Executive",
     "Brand Strategist", "Translator",
     "Language Teacher", "Editor", "Screenwriter",
     "Documentary Researcher", "Event Manager",
     "Public Relations Manager"],
    "Relevant degree, language skills, professional training "
    "or portfolio depending on the role.",
    "₹2–15 lakh",
    "Writing", "Collaborative",
    "Communication-focused work, varied projects and "
    "opportunities in media or organisations.",
    "Writing, speaking, research, creativity"
)


# LAW AND PUBLIC SERVICE

add_career(
    "Law & Public Service",
    ["Lawyer", "Legal Advisor", "Corporate Lawyer",
     "Judge", "Legal Researcher", "Paralegal",
     "Civil Services Officer", "Diplomat",
     "Policy Analyst", "Government Administrator",
     "Police Service Officer", "Forest Service Officer",
     "Public Policy Researcher", "Human Rights Researcher",
     "Labour Law Specialist"],
    "Law degree and required enrolment or examinations for "
    "legal roles; public-service positions require applicable "
    "competitive examinations and eligibility.",
    "₹3–20 lakh",
    "Writing", "Analytical",
    "Public service, legal practice, research and opportunities "
    "to work on policy or social issues.",
    "Reasoning, research, communication, integrity"
)


# AGRICULTURE AND ENVIRONMENT

add_career(
    "Agriculture & Environment",
    ["Agricultural Scientist", "Agronomist",
     "Horticulturist", "Soil Scientist",
     "Plant Pathologist", "Agricultural Engineer",
     "Food Technologist", "Environmental Scientist",
     "Climate Analyst", "Conservation Scientist",
     "Forestry Professional", "Water Resource Specialist",
     "Sustainability Consultant", "Urban Planner",
     "Waste Management Specialist"],
    "Relevant degree in agriculture, environmental science, "
    "engineering, geography or another related discipline.",
    "₹3–15 lakh",
    "Science & Research", "Practical",
    "Work related to food systems, sustainability, natural "
    "resources, climate and environmental protection.",
    "Observation, analysis, fieldwork, problem-solving"
)


# EDUCATION AND SOCIAL SCIENCES

add_career(
    "Education & Social Sciences",
    ["School Teacher", "College Lecturer",
     "Educational Counsellor", "Instructional Designer",
     "Psychologist", "Social Worker",
     "Sociologist", "Historian", "Geographer",
     "Political Scientist", "Research Assistant",
     "Special Education Teacher", "Academic Coordinator",
     "Education Researcher", "Career Counsellor"],
    "Relevant subject degree and any required teaching, "
    "professional or postgraduate qualification.",
    "₹2–12 lakh",
    "Writing", "Collaborative",
    "Education, research, community service and opportunities "
    "to support learning or social development.",
    "Communication, empathy, research, organisation"
)


# =========================================================
# CREATE TRAINING DATA FOR THE DEMONSTRATION MODEL
# =========================================================

TRAINING_ROWS = []

TRAINING_PATTERNS = [
    ("Science", "Technology", "Analytical", "Technology & IT"),
    ("Science", "Technology", "Creative", "Technology & IT"),
    ("Science", "Technology", "Practical", "Engineering & Physical Sciences"),
    ("Science", "Healthcare", "Analytical", "Medical & Healthcare"),
    ("Science", "Healthcare", "Practical", "Medical & Healthcare"),
    ("Science", "Science & Research", "Analytical", "Biology & Research"),
    ("Science", "Science & Research", "Practical", "Engineering & Physical Sciences"),
    ("Science", "Art & Design", "Creative", "Arts, Design & Media"),
    ("Science", "Finance", "Analytical", "Business & Finance"),
    ("Science", "Business", "Collaborative", "Business & Finance"),
    ("Commerce", "Finance", "Analytical", "Business & Finance"),
    ("Commerce", "Business", "Collaborative", "Business & Finance"),
    ("Commerce", "Technology", "Analytical", "Technology & IT"),
    ("Commerce", "Writing", "Creative", "Arts, Design & Media"),
    ("Commerce", "Art & Design", "Creative", "Arts, Design & Media"),
    ("Commerce", "Science & Research", "Analytical", "Business & Finance"),
    ("Humanities", "Writing", "Creative", "Arts, Design & Media"),
    ("Humanities", "Writing", "Analytical", "Law & Public Service"),
    ("Humanities", "Business", "Collaborative", "Business & Finance"),
    ("Humanities", "Art & Design", "Creative", "Arts, Design & Media"),
    ("Humanities", "Healthcare", "Collaborative", "Education & Social Sciences"),
    ("Humanities", "Science & Research", "Analytical", "Education & Social Sciences"),
    ("Humanities", "Technology", "Creative", "Technology & IT"),
    ("Humanities", "Finance", "Analytical", "Business & Finance"),
]

for subject, interest, style, field in TRAINING_PATTERNS:
    TRAINING_ROWS.append({
        "Subject": subject,
        "Interest": interest,
        "Work Style": style,
        "Field": field
    })

TRAINING_DF = pd.DataFrame(TRAINING_ROWS)

FEATURES = ["Subject", "Interest", "Work Style"]

preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        FEATURES
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ))
])

model.fit(TRAINING_DF[FEATURES], TRAINING_DF["Field"])


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🎯 CareerCompass AI</h1>
    <div class="subtitle">
        Discover Your Interests. Explore Your Future.
    </div>
    <p>
        Explore career pathways, qualifications, estimated salaries,
        and opportunities through an interactive AI-powered
        career exploration demonstration.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD METRICS
# =========================================================

fields = sorted(set(c["Field"] for c in CAREERS))

col1, col2, col3, col4 = st.columns(4)

col1.metric("Career Options", len(CAREERS))
col2.metric("Career Fields", len(fields))
col3.metric("ML Model", "Decision Tree")
col4.metric("Project Level", "Class 12")

st.markdown("---")


# =========================================================
# NAVIGATION
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🎯 AI Career Finder",
    "🌍 Explore Careers",
    "ℹ️ About Project"
])


# =========================================================
# TAB 1: CAREER FINDER
# =========================================================

with tab1:

    st.header("Find Your Career Direction")

    st.write(
        "Select your academic stream, strongest interest, "
        "and preferred work style to explore relevant fields."
    )

    with st.form("career_form"):

        subject = st.selectbox(
            "📚 Your academic stream",
            ["Science", "Commerce", "Humanities"]
        )

        interest = st.selectbox(
            "💡 Your primary interest",
            [
                "Technology",
                "Healthcare",
                "Science & Research",
                "Finance",
                "Business",
                "Art & Design",
                "Writing"
            ]
        )

        work_style = st.selectbox(
            "🧭 Your preferred work style",
            ["Analytical", "Creative", "Practical", "Collaborative"]
        )

        submitted = st.form_submit_button(
            "✦ DISCOVER MY CAREER"
        )

    if submitted:

        user_input = pd.DataFrame([{
            "Subject": subject,
            "Interest": interest,
            "Work Style": work_style
        }])

        predicted_field = model.predict(user_input)[0]

        st.session_state["predicted_field"] = predicted_field
        st.session_state["user_input"] = {
            "Subject": subject,
            "Interest": interest,
            "Work Style": work_style
        }

    if "predicted_field" in st.session_state:

        predicted_field = st.session_state["predicted_field"]

        st.success(
            f"Your suggested career exploration field: {predicted_field}"
        )

        st.info(
            "This is a demonstration prediction from a small, "
            "illustrative training dataset. It is not a validated "
            "career assessment. Explore other fields too."
        )

        matching = [
            c for c in CAREERS
            if c["Field"] == predicted_field
        ]

        st.subheader("Explore Careers in Your Suggested Field")

        for career in matching[:6]:

            st.markdown(
                f"""
                <div class="card">
                    <h3>{career['Career']}</h3>
                    <p><b>Field:</b> {career['Field']}</p>
                    <p><b>Estimated annual salary:</b>
                    <span class="salary">{career['Salary']}</span></p>
                    <p><b>Qualifications:</b>
                    {career['Qualifications']}</p>
                    <p><b>Benefits:</b> {career['Benefits']}</p>
                    <p><b>Skills:</b> {career['Skills']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("Other Fields to Explore")

        other_fields = [f for f in fields if f != predicted_field]

        st.write("You can also explore:")

        st.write(" • " + "\n • ".join(other_fields))


# =========================================================
# TAB 2: CAREER EXPLORER
# =========================================================

with tab2:

    st.header("Explore Career Opportunities")

    st.write(
        "Search careers across different fields and compare "
        "their educational pathways and potential opportunities."
    )

    search_text = st.text_input(
        "🔎 Search by career name or skill",
        placeholder="Example: doctor, engineer, research..."
    )

    selected_field = st.selectbox(
        "Filter by career field",
        ["All Fields"] + fields
    )

    filtered = CAREERS.copy()

    if selected_field != "All Fields":
        filtered = [
            c for c in filtered
            if c["Field"] == selected_field
        ]

    if search_text.strip():

        query = search_text.strip().lower()

        filtered = [
            c for c in filtered
            if query in c["Career"].lower()
            or query in c["Field"].lower()
            or query in c["Qualifications"].lower()
            or query in c["Skills"].lower()
        ]

    st.caption(f"{len(filtered)} careers found")

    if filtered:

        for index, career in enumerate(filtered):

            with st.expander(
                f"✨ {career['Career']} — {career['Field']}"
            ):

                st.markdown(
                    f"### {career['Career']}"
                )

                st.markdown(
                    f'<div class="salary">'
                    f"Estimated annual salary: {career['Salary']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.write("**Qualifications / Courses**")
                st.write(career["Qualifications"])

                st.write("**Potential career benefits**")
                st.write(career["Benefits"])

                st.write("**Important skills**")
                st.write(career["Skills"])

                st.caption(
                    "Salary estimates are illustrative, not "
                    "verified current market statistics. Actual "
                    "earnings vary by location, experience, "
                    "employer and qualifications."
                )

    else:
        st.warning(
            "No matching careers found. Try a different search."
        )


# =========================================================
# TAB 3: ABOUT PROJECT
# =========================================================

with tab3:

    st.header("About CareerCompass AI")

    st.markdown("""
    ### Project Overview

    CareerCompass AI is a Class 12 Artificial Intelligence
    project that demonstrates how machine learning can be
    used to explore possible career fields.

    ### Technology Used

    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - OneHotEncoder
    - Decision Tree Classifier

    ### Input Features

    - Academic stream
    - Primary interest
    - Preferred work style

    ### Output

    A suggested career field and a collection of careers to
    explore within that field.

    ### Machine Learning Workflow

    1. Prepare illustrative training examples.
    2. Convert categorical features into numerical features.
    3. Train a Decision Tree classification model.
    4. Accept a student's selected inputs.
    5. Predict a suggested field.
    6. Display related career information.

    ### Project Limitations

    The model uses a small, manually constructed demonstration
    dataset. Its predictions are not scientifically validated,
    and the results should not be interpreted as a definitive
    assessment of a student's aptitude or future success.

    The career database is intended for exploration. Salary
    ranges require independent verification against current
    Indian labour-market data before being treated as factual
    salary research.
    """)

    st.markdown("---")

    st.subheader("CareerCompass AI")

    st.write(
        "Discover Your Interests. Explore Your Future."
    )

    st.caption(
        "Class 12 AI Project | Career Exploration for Students"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:16px;">
        <p style="color:#e8c878; font-size:18px;">
            🎯 CareerCompass AI
        </p>
        <p class="small-note">
            Explore possibilities. Make informed decisions.
        </p>
        <p class="small-note">
            Educational demonstration — not professional career advice.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
