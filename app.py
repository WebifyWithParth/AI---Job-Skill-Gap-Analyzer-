import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- 1) Load data ----------
@st.cache_data
def load_data(path="DataAnalyst.csv"):
    df = pd.read_csv(path)
    return df

# ---------- 2) Skill list + extract ----------
SKILLS = [
    "python","sql","excel","pandas","numpy","power bi","tableau","matplotlib","seaborn",
    "statistics","machine learning","data analysis","data visualization","scikit-learn",
    "tensorflow","pytorch","nlp","git","streamlit","flask","mysql","postgresql","spark"
]

def extract(text):
    L = []
    text = str(text).lower()
    for skill in SKILLS:
        if skill in text:
            L.append(skill)
    return list(set(L))

# ---------- 3) Build TF-IDF (cached) ----------
@st.cache_resource
def build_model(text_series):
    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
    job_vectors = tfidf.fit_transform(text_series)
    return tfidf, job_vectors

# ---------- 4) Recommendation ----------
def recommend(data, tfidf, job_vectors, user_text, top_n=5):
    user_text = str(user_text).lower().strip()
    user_skills = set(extract(user_text))

    user_vec = tfidf.transform([user_text])
    scores = cosine_similarity(user_vec, job_vectors).flatten()

    top_indices = np.argsort(scores)[::-1][:top_n]

    # Ensure job_skills exists
    if "job_skills" not in data.columns:
        data = data.copy()
        data["job_skills"] = data["text"].apply(extract)

    result = data.iloc[top_indices][
        ["Job Title","Company Name","Location","Salary Estimate","job_skills"]
    ].copy()

    result["match_percent"] = (scores[top_indices] * 100).round(2)
    result["missing_skills"] = result["job_skills"].apply(
        lambda js: sorted(list(set(js) - user_skills))
    )

    return result[["Job Title","Company Name","Location","Salary Estimate","match_percent","missing_skills"]]

# ---------- UI ----------
st.set_page_config(page_title="Job Recommender System + Skill Gap", page_icon="💼", layout="wide")
st.title("💼 Job Recommendation System + Skill Gap Analyzer")

df = load_data()

# If your CSV doesn't already have 'text', create it
if "text" not in df.columns:
    df["text"] = (df["Job Title"].astype(str) + " " + df["Job Description"].astype(str)).str.lower()

tfidf, job_vectors = build_model(df["text"])

st.subheader("Enter your skills / interests")
user_text = st.text_input("Example: python sql excel pandas dashboard")

top_n = st.slider("How many recommendations?", 3, 15, 5)

if st.button("Recommend Jobs"):

    if not user_text.strip():
        st.warning("Please enter your skills.")
    
    else:
        results = recommend(df, tfidf, job_vectors, user_text, top_n=top_n)

        st.success("Top matching jobs based on your skills:")

        for index, row in results.iterrows():
            st.markdown("---")

            st.markdown(f"### 🔹 {row['Job Title']}  —  {row['match_percent']}% Match")
            st.write(f"**Company:** {row['Company Name']}")
            st.write(f"**Location:** {row['Location']}")
            st.write(f"**Salary:** {row['Salary Estimate']}")

            if row['missing_skills']:
                st.write("⚠️ **Missing Skills:**")
                st.write(", ".join(row['missing_skills']))
            else:
                st.write("✅ You match all detected skills!")
