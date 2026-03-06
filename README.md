<div align="center">

<!-- ANIMATED HEADER -->
<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=28&pause=1000&color=9B5DE5&center=true&vCenter=true&width=700&lines=AI+Job+%26+Skill+Gap+Analyzer;NLP-Powered+Career+Intelligence;Know+Your+Gaps.+Close+Them." alt="Typing SVG" />

<br/>

<!-- BADGES -->
<a href="https://ai-job-skill-gap-analyzer.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀%20Live%20App-Streamlit-9b5de5?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=09090f" alt="Live App"/>
</a>
&nbsp;
<img src="https://img.shields.io/badge/Python-3.10+-c084fc?style=for-the-badge&logo=python&logoColor=white&labelColor=09090f"/>
&nbsp;
<img src="https://img.shields.io/badge/NLP-TF--IDF-7c3aed?style=for-the-badge&logoColor=white&labelColor=09090f"/>
&nbsp;
<img src="https://img.shields.io/badge/ML-Cosine%20Similarity-4c1d95?style=for-the-badge&logoColor=white&labelColor=09090f"/>
&nbsp;
<img src="https://img.shields.io/badge/Status-Live-22c55e?style=for-the-badge&logoColor=white&labelColor=09090f"/>

<br/><br/>

<!-- APP LINK BUTTON -->
<a href="https://ai-job-skill-gap-analyzer.streamlit.app/">
  <img src="https://img.shields.io/badge/%E2%96%B6%20%20OPEN%20LIVE%20APP%20%E2%86%97-Click%20Here-ffffff?style=for-the-badge&labelColor=7c3aed&color=9b5de5" height="42" alt="Open App"/>
</a>

<br/><br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

</div>

<br/>

## 📌 What Is This?

The **AI Job & Skill Gap Analyzer** is an intelligent career analysis platform that tells you exactly where you stand against any job role — and what to learn next.

You paste your skills. The system does the rest:

- Vectorizes your input and every job description using **TF-IDF**
- Computes **cosine similarity** to rank the best-matching roles for you
- Shows exactly which skills you **already have** ✅ and which ones are **missing** ❌
- Builds a **priority upskilling roadmap** ranked by impact across all matched roles

> **"Don't guess what skills to learn next. Let the data tell you."**

<br/>

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>
</div>

<br/>

## 🖥️ App Screenshots

<br/>

### 🏠 Hero — Landing Screen
> Dark black + purple UI with animated floating orbs, grid lines, and scanline effect

```
╔═════════════════════════════════════════════════════════════════╗
║  ⬤ · NLP-Powered Career Intelligence                           ║
║                                                                  ║
║   Know Your Gaps.                                               ║
║   Close Them.  ░░░░░ ← purple shimmer gradient                 ║
║                                                                  ║
║   Enter your skills and get ranked job matches,                 ║
║   exact skill gap analysis, and a priority upskilling           ║
║   roadmap — powered by TF-IDF & cosine similarity.             ║
║                                          ◉ floating orb        ║
╚═════════════════════════════════════════════════════════════════╝
```

<br/>

### 📝 Step 01 — Skills Input + Options
> Purple blinking cursor in textarea · selected values shown live in pill

```
╔══════════════════════════════════════╗
║  ⬤ Step 01 — Your Profile           ║
║  ────────────────────────────────    ║
║  YOUR SKILLS & TECHNOLOGIES          ║
║  ╔────────────────────────────────╗  ║
║  ║ Python, SQL, Pandas,           ║  ║  ← purple blinking cursor
║  ║ scikit-learn, XGBoost,         ║  ║
║  ║ NLP, matplotlib, git...        ║  ║
║  ╚────────────────────────────────╝  ║
║                                       ║
║  [ Top 5 roles ▾ ]  [ Mode: Std ▾ ]  ║
║                                       ║
║  Selected → [Top 5 roles] [Standard] ║  ← live pill
║                                       ║
║  ╔══════════════════════════════╗    ║
║  ║  🎯  ANALYZE MY SKILLS →    ║    ║  ← glowing purple button
║  ╚══════════════════════════════╝    ║
╚══════════════════════════════════════╝
```

<br/>

### 📊 Step 02 — Results & Metric Strip
> Animated metric cards + ranked job cards with color-coded skill tags

```
╔══════════════════════════════════════════════════╗
║  ⬤ Step 02 — Results                            ║
║  ┌────────┬────────┬────────┬────────┐           ║
║  │  42%   │  31%   │   5    │  18    │           ║
║  │  Best  │  Avg   │ Roles  │  Gaps  │           ║
║  └────────┴────────┴────────┴────────┘           ║
║                                                   ║
║  MATCHED ROLES — RANKED BY FIT                   ║
║  ┌──────────────────────────────────────────┐    ║
║  │ #1  Data Analyst                  42.0%  │    ║
║  │ ████████████████████░░░░░░░░░░           │    ║  ← purple bar
║  │ ✓ python  ✓ sql  ✓ pandas  ✓ numpy      │    ║
║  │ × tableau  × power-bi  × dbt            │    ║
║  └──────────────────────────────────────────┘    ║
║  ┌──────────────────────────────────────────┐    ║
║  │ #2  ML Engineer                   38.5%  │    ║
║  │ ████████████████░░░░░░░░░░░░░░           │    ║
║  │ ✓ scikit-learn  ✓ xgboost                │    ║
║  │ × docker  × mlflow  × kubernetes        │    ║
║  └──────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════╝
```

<br/>

### 🎓 Priority Upskilling Roadmap
> Skills ranked by how many of your matched roles require them

```
╔══════════════════════════════════════════════════════════╗
║  🎓 Priority Upskilling Roadmap                         ║
║  Learn these first — highest ROI                        ║
║  ────────────────────────────────────────────────────   ║
║  tableau       ██████████████████░░░   🔴 High · 4/5   ║
║  docker        ███████████████░░░░░░   🔴 High · 3/5   ║
║  power-bi      ████████████░░░░░░░░░   🟡 Med  · 2/5   ║
║  tensorflow    ██████████░░░░░░░░░░░   🟡 Med  · 2/5   ║
║  kubernetes    ██████░░░░░░░░░░░░░░░   🟢 Low  · 1/5   ║
╚══════════════════════════════════════════════════════════╝
```

<br/>

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>
</div>

<br/>

## ⚙️ How It Works

```
  Your Skills Input
        │
        ▼
  ┌─────────────────────────┐
  │   1. TF-IDF Vectorizer  │  Converts your skill text and each job
  │                         │  description into weighted numeric vectors.
  │   "python sql pandas"   │  Rare/specific skills get higher weights.
  │        ↓                │
  │   [0.42, 0.31, 0.88...] │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   2. Cosine Similarity  │  Measures the angle between your vector
  │                         │  and each job description vector.
  │   your_vec · job_vec    │  Score ranges 0.0 → 1.0 (0% → 100%)
  │   ──────────────────    │
  │   |your| × |job|        │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   3. Ranked Matches     │  Top N roles sorted by similarity score.
  │                         │  Each shown with match % and progress bar.
  │   #1  Data Analyst 42%  │
  │   #2  ML Engineer  38%  │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   4. Gap Analysis       │  Set difference calculation:
  │                         │  job_skills − your_skills = gaps
  │   Missing: tableau,     │  Shown as red × tags on each card.
  │   docker, power-bi...   │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   5. Upskilling Roadmap │  Missing skills ranked by frequency
  │                         │  across ALL matched roles.
  │   tableau  → 4/5 roles  │  High frequency = learn it first.
  │   docker   → 3/5 roles  │
  └─────────────────────────┘
```

<br/>

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>
</div>

<br/>

## 🎯 Key Features

| Feature | Description |
|---|---|
| 🔢 **TF-IDF Vectorization** | Converts skill text into weighted vectors — rare skills score higher |
| 📐 **Cosine Similarity** | Finds most similar job roles to your profile with a match percentage |
| ✅ **Skill Matching** | Shows exactly which skills you already have for each role |
| ❌ **Gap Detection** | Identifies every missing skill per matched role |
| 🎓 **Upskilling Roadmap** | Ranks missing skills by how many roles need them — priority order |
| 📊 **Metric Dashboard** | Best match %, avg score, roles found, unique gaps — all at a glance |
| 🎨 **Animated Dark UI** | Black + purple theme with floating orbs, scanline, glow pulse effects |
| ⚡ **Live Selection Pill** | Dropdown choices shown in a pill below options — always visible |
| 🖊️ **Visible Cursor** | Purple blinking cursor in the skills textarea — no more invisible input |

<br/>

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>
</div>

<br/>

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Frontend** | Streamlit + Custom CSS animations |
| **NLP Engine** | Scikit-learn · TF-IDF Vectorizer |
| **Similarity** | Cosine Similarity (sklearn.metrics.pairwise) |
| **Data Layer** | Pandas · NumPy |
| **Dataset** | DataAnalyst.csv — job roles + required skills |
| **Deployment** | Streamlit Community Cloud |

</div>

<br/>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white"/>
</div>

<br/>

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>
</div>

<br/>

## 📂 Project Structure

```
AI---Job-Skill-Gap-Analyzer-/
│
├── 📄 app.py                ← Main Streamlit app (dark purple animated UI)
├── 📊 DataAnalyst.csv       ← Job roles dataset with skill requirements
├── 📦 requirements.txt      ← Python dependencies
├── 📓 start_01.ipynb        ← Development & exploration notebook
└── 📘 README.md             ← This file
```

<br/>

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>
</div>

<br/>

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/WebifyWithParth/AI---Job-Skill-Gap-Analyzer-.git
cd AI---Job-Skill-Gap-Analyzer-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser. 🎉

<br/>

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>
</div>

<br/>

## 💡 Example Usage

**Input:**
```
Python, SQL, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn,
machine learning, data visualization, Jupyter, Git, statistics, Excel
```

**Output:**
```
┌─────────────────────────────────────────────────────┐
│  🏆 Best Match  42.3%   📊 Avg  31.1%   🔍 Gaps 18  │
└─────────────────────────────────────────────────────┘

#1  Data Analyst              42.3%
    ✓ python  ✓ sql  ✓ pandas  ✓ statistics  ✓ numpy
    × tableau  × power-bi  × dbt  × airflow

#2  Data Scientist             36.8%
    ✓ scikit-learn  ✓ matplotlib  ✓ jupyter
    × r  × scipy  × tensorflow  × research

🎓 Learn Next:
   1. tableau      🔴 High   — appears in 4/5 matched roles
   2. power-bi     🔴 High   — appears in 3/5 matched roles
   3. dbt          🟡 Medium — appears in 2/5 matched roles
```

<br/>

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>
</div>

<br/>

## 🔮 Future Improvements

- [ ] 📄 Resume PDF parsing — upload CV, auto-extract skills
- [ ] 💰 Salary range data per matched role
- [ ] 📍 Location-based job filtering
- [ ] 🤖 LLM-powered skill extraction (BERT / GPT)
- [ ] 🗄️ Database for persistent user skill profiles
- [ ] 📈 Career progression path visualizer

<br/>

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>
</div>

<br/>

## 👨‍💻 Author

<div align="center">

**Parth Tyagi**
B.Tech — Mathematics & Computing · Central University of Karnataka
Machine Learning & AI Engineer

<br/>

<a href="https://github.com/WebifyWithParth">
  <img src="https://img.shields.io/badge/GitHub-WebifyWithParth-9b5de5?style=for-the-badge&logo=github&logoColor=white&labelColor=09090f"/>
</a>
&nbsp;
<a href="https://ai-job-skill-gap-analyzer.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀%20Live%20App-Visit%20Now-7c3aed?style=for-the-badge&logoColor=white&labelColor=09090f"/>
</a>
&nbsp;
<a href="https://parthtyagi-tech.github.io/portfolio/">
  <img src="https://img.shields.io/badge/🌐%20Portfolio-View-4c1d95?style=for-the-badge&logoColor=white&labelColor=09090f"/>
</a>

</div>

<br/>

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

<br/>

### *"Know your gaps. Close them deliberately."*

<br/>

**⭐ Found this useful? Drop a star — it means a lot! ⭐**

<br/>

<img src="https://img.shields.io/github/stars/WebifyWithParth/AI---Job-Skill-Gap-Analyzer-?style=social"/>
&nbsp;
<img src="https://img.shields.io/github/forks/WebifyWithParth/AI---Job-Skill-Gap-Analyzer-?style=social"/>

</div>
