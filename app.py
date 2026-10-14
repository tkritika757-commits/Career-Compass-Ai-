import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Career Compass AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# LUXURY PREMIUM DESIGN
# ==========================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap'
    );

    :root {
        --gold: #D6B46A;
        --gold-light: #F5DFA5;
        --background: #0B0B0F;
        --card: #15151C;
        --border: #393025;
        --text: #F5F0E6;
        --muted: #AAA6A0;
    }

    .stApp {
        background: radial-gradient(
            ellipse at top,
            #282016 0%,
            #111015 38%,
            #0B0B0F 75%
        );
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--gold-light) !important;
        font-family: 'Playfair Display', serif !important;
    }

    p, label, span, li {
        color: var(--text);
    }

    .brand {
        color: var(--gold);
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #211B15 0%,
            #15141A 55%,
            #211B15 100%
        );
        border: 1px solid #65502D;
        border-radius: 24px;
        padding: 38px 30px;
        margin: 12px 0 28px 0;
        text-align: center;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.35);
    }

    .hero h1 {
        font-size: clamp(34px, 5vw, 54px);
        margin: 8px 0 14px 0;
    }

    .hero p {
        color: #D0C8BA;
        font-size: 16px;
        line-height: 1.8;
        max-width: 700px;
        margin: 0 auto;
    }

    .gold-line {
        height: 2px;
        width: 75px;
        margin: 18px auto;
        background: linear-gradient(
            90deg,
            transparent,
            var(--gold),
            transparent
        );
    }

    .section-label {
        color: var(--gold);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .panel {
        background: rgba(21, 21, 28, 0.95);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 25px;
        margin: 12px 0 20px 0;
    }

    .career-card {
        background: linear-gradient(
            145deg,
            #1D1A17,
            #14141A
        );
        border: 1px solid #5A482B;
        border-radius: 18px;
        padding: 24px;
        margin: 12px 0;
        min-height: 245px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }

    .career-number {
        color: var(--gold);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .career-title {
        color: var(--gold-light);
        font-family: 'Playfair Display', serif;
        font-size: 24px;
        font-weight: 700;
        margin: 12px 0;
    }

    .career-description {
        color: #D0C8BA;
        font-size: 14px;
        line-height: 1.7;
    }

    .match-label {
        display: inline-block;
        border: 1px solid #796235;
        border-radius: 30px;
        padding: 6px 12px;
        margin-top: 14px;
        color: var(--gold-light);
        font-size: 12px;
        font-weight: 600;
    }

    .info-card {
        background: #15151C;
        border: 1px solid #393025;
        border-radius: 14px;
        padding: 20px;
        margin: 10px 0;
    }

    .info-card h4 {
        color: var(--gold-light);
        margin-top: 0;
    }

    .small-note {
        color: #AAA6A0;
        font-size: 12px;
        line-height: 1.7;
    }

    .stButton > button {
        width: 100%;
        min-height: 48px;
        background: linear-gradient(
            135deg,
            #F0D797,
            #B78D43
        );
        color: #17120A !important;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        font-size: 15px;
        transition: 0.25s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #FFE9AD,
            #D6B46A
        );
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(214, 180, 106, 0.2);
    }

    div[data-baseweb="select"] > div {
        background: #1A191D;
        border-color: #65502D;
        border-radius: 10px;
    }

    .stTextInput input {
        background: #1A191D;
        color: white;
        border-color: #65502D;
        border-radius: 10px;
    }

    div[data-testid="stMetric"] {
        background: #17161B;
        border: 1px solid #393025;
        padding: 18px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #C6BCA9;
    }

    div[data-testid="stMetricValue"] {
        color: #E9CC88;
    }

    hr {
        border-color: #393025;
    }

    .footer {
        text-align: center;
        color: #8F887D;
        font-size: 12px;
        padding: 25px 0 10px 0;
        border-top: 1px solid #393025;
        margin-top: 35px;
    }

    @media (max-width: 600px) {
        .block-container {
            padding: 1rem 0.8rem 2rem 0.8rem;
        }

        .hero {
            padding: 28px 18px;
        }

        .panel {
            padding: 18px;
        }

        .career-card {
            padding: 20px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# WEBSITE HEADER
# ==========================================

st.markdown(
    """
    <div class="brand">
        ✦ Career Compass AI
    </div>

    <div class="hero">
        <div class="section-label">
            YOUR FUTURE, REIMAGINED
        </div>

        <h1>Discover Your True Potential</h1>

        <div class="gold-line"></div>

        <p>
            Your interests. Your strengths. Your future.
            Explore career possibilities with an AI-powered
            career guidance experience designed around you.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# CAREER INFORMATION
# ==========================================

career_info = {
    "AI Engineer": {
        "description":
            "Develop intelligent systems, machine learning "
            "applications, and AI-powered solutions.",

        "strengths":
            ["Logical thinking", "Problem-solving", "Curiosity"],

        "skills":
            ["Python", "Mathematics", "Machine Learning"],

        "courses":
            ["B.Tech Computer Science", "B.Sc. Computer Science",
             "AI and Data Science"]
    },

    "Software Developer": {
        "description":
            "Design, build, test, and maintain applications "
            "and software systems.",

        "strengths":
            ["Logical reasoning", "Creativity", "Persistence"],

        "skills":
            ["Python or Java", "Programming", "Debugging"],

        "courses":
            ["B.Tech Computer Science", "BCA", "B.Sc. IT"]
    },

    "Data Analyst": {
        "description":
            "Turn data into useful insights that help "
            "organisations make informed decisions.",

        "strengths":
            ["Attention to detail", "Analytical thinking",
             "Numerical reasoning"],

        "skills":
            ["Excel", "SQL", "Data visualisation"],

        "courses":
            ["B.Sc. Statistics", "BCA", "Data Analytics courses"]
    },

    "Doctor": {
        "description":
            "Diagnose illnesses, support patients, and "
            "contribute to healthcare and medical science.",

        "strengths":
            ["Empathy", "Discipline", "Scientific thinking"],

        "skills":
            ["Biology", "Communication", "Clinical reasoning"],

        "courses":
            ["MBBS", "Relevant healthcare degrees"]
    },

    "Biotechnologist": {
        "description":
            "Apply biological science and technology to "
            "research, medicine, agriculture, and industry.",

        "strengths":
            ["Scientific curiosity", "Patience", "Observation"],

        "skills":
            ["Biology", "Laboratory methods", "Research"],

        "courses":
            ["B.Sc. Biotechnology", "B.Tech Biotechnology",
             "Related life-science degrees"]
    },

    "Business Manager": {
        "description":
            "Coordinate people, projects, and resources "
            "to help organisations achieve their goals.",

        "strengths":
            ["Leadership", "Decision-making", "Organisation"],

        "skills":
            ["Communication", "Planning", "Business analysis"],

        "courses":
            ["BBA", "B.Com", "Business-related degrees"]
    },

    "Graphic Designer": {
        "description":
            "Create visual concepts for brands, websites, "
            "advertising, and digital media.",

        "strengths":
            ["Creativity", "Visual thinking", "Originality"],

        "skills":
            ["Typography", "Layout", "Design software"],

        "courses":
            ["B.Des", "Visual Communication",
             "Graphic Design courses"]
    },

    "Environmental Scientist": {
        "description":
            "Study environmental systems and help address "
            "pollution, conservation, and sustainability.",

        "strengths":
            ["Observation", "Research", "Environmental awareness"],

        "skills":
            ["Biology", "Data analysis", "Field research"],

        "courses":
            ["B.Sc. Environmental Science",
             "Related environmental degrees"]
    }
}


# ==========================================
# TRAINING DATA
# ==========================================

training_rows = [
    ("Computer Science", "Technology", "Analytical",
     "AI Engineer"),
    ("Mathematics", "Technology", "Analytical",
     "AI Engineer"),
    ("Computer Science", "Research", "Independent",
     "AI Engineer"),
    ("Computer Science", "Technology", "Practical",
     "Software Developer"),
    ("Mathematics", "Technology", "Independent",
     "Software Developer"),
    ("Computer Science", "Creative Work", "Analytical",
     "Software Developer"),
    ("Mathematics", "Business", "Analytical",
     "Data Analyst"),
    ("Commerce", "Research", "Organized",
     "Data Analyst"),
    ("Computer Science", "Research", "Analytical",
     "Data Analyst"),
    ("Biology", "Helping People", "People-oriented",
     "Doctor"),
    ("Biology", "Research", "Organized",
     "Doctor"),
    ("Science", "Helping People", "People-oriented",
     "Doctor"),
    ("Biology", "Research", "Independent",
     "Biotechnologist"),
    ("Science", "Nature", "Analytical",
     "Biotechnologist"),
    ("Biology", "Nature", "Practical",
     "Environmental Scientist"),
    ("Science", "Nature", "Independent",
     "Environmental Scientist"),
    ("Arts & Humanities", "Creative Work", "Creative",
     "Graphic Designer"),
    ("Arts & Humanities", "Creative Work", "Independent",
     "Graphic Designer"),
    ("Computer Science", "Creative Work", "Creative",
     "Graphic Designer"),
    ("Commerce", "Business", "Organized",
     "Business Manager"),
    ("Commerce", "Business", "People-oriented",
     "Business Manager"),
    ("Mathematics", "Business", "Organized",
     "Business Manager"),
    ("Science", "Nature", "Research",
     "Environmental Scientist"),
    ("Biology", "Research", "Analytical",
     "Biotechnologist")
]

column_names = [
    "Subject",
    "Interest",
    "Work_Style",
    "Career"
]

data = pd.DataFrame(
    training_rows,
    columns=column_names
)


# ==========================================
# IMPROVED MACHINE LEARNING MODEL
# ==========================================

features = [
    "Subject",
    "Interest",
    "Work_Style"
]

X = data[features]
y = data["Career"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            features
        )
    ]
)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=8,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

model.fit(X, y)

# ==========================================
# STUDENT INPUT FORM
# ==========================================

st.markdown(
    """
    <div class="panel">
        <div class="section-label">01 / YOUR PROFILE</div>
        <h2>Let's Find Your Direction</h2>
        <p>
            Tell us about your academic background,
            interests, and preferred way of working.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("career_form"):

    name = st.text_input(
        "Your name (optional)",
        placeholder="Enter your name"
    )

    col1, col2 = st.columns(2)

    with col1:

        subject = st.selectbox(
            "Favourite subject",
            [
                "Computer Science",
                "Mathematics",
                "Biology",
                "Science",
                "Commerce",
                "Arts & Humanities"
            ]
        )

        interest = st.selectbox(
            "What interests you most?",
            [
                "Technology",
                "Helping People",
                "Business",
                "Creative Work",
                "Research",
                "Nature"
            ]
        )

    with col2:

        work_style = st.selectbox(
            "Preferred working style",
            [
                "Analytical",
                "People-oriented",
                "Creative",
                "Practical",
                "Organized",
                "Independent"
            ]
        )

        study_level = st.selectbox(
            "Your current education level",
            [
                "Class 11",
                "Class 12",
                "Undergraduate",
                "Graduate",
                "Other"
            ]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "✦  DISCOVER MY CAREER"
    )


# ==========================================
# CAREER PREDICTION
# ==========================================

if submitted:

    student = pd.DataFrame(
        [
            {
                "Subject": subject,
                "Interest": interest,
                "Work_Style": work_style
            }
        ]
    )

    probabilities = model.predict_proba(student)[0]

    classes = model.named_steps["classifier"].classes_

    ranking = sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True
    )[:3]

    st.markdown("---")

    greeting = (
        f"Welcome, {name.strip()}!"
        if name.strip()
        else "Your Career Discovery Results"
    )

    st.markdown(
        f"""
        <div class="panel">
            <div class="section-label">
                02 / YOUR RESULTS
            </div>

            <h2>{greeting}</h2>

            <p>
                Explore these career paths based on your
                selected interests and preferences.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Education Level", study_level)
    metric2.metric("Interest", interest)
    metric3.metric("Work Style", work_style)

    st.markdown("## ✦ Your Top 3 Career Matches")

    st.caption(
        "These rankings are produced by a demonstration "
        "machine-learning model trained on a small sample "
        "dataset. They are exploratory suggestions, not "
        "validated probabilities of career success."
    )

    for index, (career, score) in enumerate(ranking):

        info = career_info[career]

        st.markdown(
            f"""
            <div class="career-card">
                <div class="career-number">
                    CAREER OPTION 0{index + 1}
                </div>

                <div class="career-title">
                    {career}
                </div>

                <div class="career-description">
                    {info["description"]}
                </div>

                <div class="match-label">
                    Model score: {score * 100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
# ==========================================
# CAREER PREDICTION
# ==========================================

if submitted:

    student = pd.DataFrame(
        [
            {
                "Subject": subject,
                "Interest": interest,
                "Work_Style": work_style
            }
        ]
    )

    # Get model scores for the selected profile
    probabilities = model.predict_proba(student)[0]
    classes = model.named_steps["classifier"].classes_

    # Additional profile-based relevance scoring
    relevance = {
        career: 0.0
        for career in career_info
    }

    subject_matches = {
        "Computer Science": [
            "AI Engineer",
            "Software Developer",
            "Data Analyst",
            "Graphic Designer"
        ],
        "Mathematics": [
            "AI Engineer",
            "Software Developer",
            "Data Analyst",
            "Business Manager"
        ],
        "Biology": [
            "Doctor",
            "Biotechnologist",
            "Environmental Scientist"
        ],
        "Science": [
            "Doctor",
            "Biotechnologist",
            "Environmental Scientist",
            "AI Engineer"
        ],
        "Commerce": [
            "Business Manager",
            "Data Analyst"
        ],
        "Arts & Humanities": [
            "Graphic Designer",
            "Business Manager"
        ]
    }

    interest_matches = {
        "Technology": [
            "AI Engineer",
            "Software Developer",
            "Data Analyst"
        ],
        "Helping People": [
            "Doctor",
            "Business Manager"
        ],
        "Business": [
            "Business Manager",
            "Data Analyst"
        ],
        "Creative Work": [
            "Graphic Designer",
            "Software Developer"
        ],
        "Research": [
            "Biotechnologist",
            "Data Analyst",
            "Environmental Scientist",
            "AI Engineer"
        ],
        "Nature": [
            "Environmental Scientist",
            "Biotechnologist"
        ]
    }

    work_style_matches = {
        "Analytical": [
            "AI Engineer",
            "Data Analyst",
            "Biotechnologist",
            "Environmental Scientist"
        ],
        "People-oriented": [
            "Doctor",
            "Business Manager"
        ],
        "Creative": [
            "Graphic Designer",
            "Software Developer"
        ],
        "Practical": [
            "Software Developer",
            "Doctor",
            "Environmental Scientist"
        ],
        "Organized": [
            "Business Manager",
            "Data Analyst",
            "Doctor"
        ],
        "Independent": [
            "AI Engineer",
            "Software Developer",
            "Biotechnologist",
            "Graphic Designer"
        ]
    }

    # Award points for matching each preference
    for career in subject_matches.get(subject, []):
        relevance[career] += 3

    for career in interest_matches.get(interest, []):
        relevance[career] += 4

    for career in work_style_matches.get(work_style, []):
        relevance[career] += 2

    # Combine the model output with transparent
    # profile-based relevance scores.
    model_scores = dict(zip(classes, probabilities))

    final_scores = {}

    for career in career_info:

        model_score = model_scores.get(career, 0.0)

        final_scores[career] = (
            relevance[career]
            + model_score
        )

    ranking = sorted(
        final_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:3]

    st.markdown("---")

    greeting = (
        f"Welcome, {name.strip()}!"
        if name.strip()
        else "Your Career Discovery Results"
    )

    st.markdown(
        f"""
        <div class="panel">
            <div class="section-label">
                02 / YOUR RESULTS
            </div>

            <h2>{greeting}</h2>

            <p>
                Explore these career paths based on your
                selected interests and preferences.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Education Level", study_level)
    metric2.metric("Interest", interest)
    metric3.metric("Work Style", work_style)

    st.markdown("## ✦ Your Top 3 Career Matches")

    st.caption(
        "Recommendations combine a demonstration "
        "machine-learning model with profile-based "
        "relevance rules. Scores are ranking values, "
        "not probabilities or guarantees of career success."
    )

    for index, (career, score) in enumerate(ranking):

        info = career_info[career]

        st.markdown(
            f"""
            <div class="career-card">
                <div class="career-number">
                    CAREER OPTION 0{index + 1}
                </div>

                <div class="career-title">
                    {career}
                </div>

                <div class="career-description">
                    {info["description"]}
                </div>

                <div class="match-label">
                    Profile relevance score: {score:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(f"Explore {career}"):

            left, right = st.columns(2)

            with left:

                st.markdown("### ✦ Potential strengths")

                for strength in info["strengths"]:
                    st.write(f"• {strength}")

                st.markdown("### ✦ Skills to develop")

                for skill in info["skills"]:
                    st.write(f"• {skill}")

            with right:

                st.markdown("### ✦ Courses to explore")

                for course in info["courses"]:
                    st.write(f"• {course}")

                st.markdown("### ✦ Suggested next steps")

                st.write(
                    "Explore the course syllabus, check "
                    "eligibility requirements, try a beginner "
                    "project, and speak with a career counsellor."
                )

    st.success(
        "Your career exploration report is ready!"
    )

    st.info(
        "Remember: Your interests can evolve. Explore "
        "different fields before making an important "
        "education or career decision."
    )
# ==========================================
# CAREER EXPLORATION SECTION
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div class="panel">
        <div class="section-label">03 / EXPLORE</div>
        <h2>Explore Your Possibilities</h2>
        <p>
            Every career starts with curiosity.
            Learn about different professional paths
            and the skills they may require.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

career_columns = st.columns(2)

all_careers = list(career_info.items())

for index, (career, info) in enumerate(all_careers):

    with career_columns[index % 2]:

        st.markdown(
            f"""
            <div class="info-card">
                <h4>✦ {career}</h4>
                <p>{info["description"]}</p>
                <p class="small-note">
                    Key skills: {", ".join(info["skills"])}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        <p>
            ✦ CAREER COMPASS AI ✦
        </p>

        <p>
            A student career exploration project
            powered by Python and Machine Learning.
        </p>

        <p>
            Educational guidance only • Explore your future
            with curiosity and confidence.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
