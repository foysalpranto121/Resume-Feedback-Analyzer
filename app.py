# =========================
# IMPORTS
# =========================
## Core libraries
import os
import numpy as np
import streamlit as st

## Vector DB
import chromadb

## NLP
import spacy
from sentence_transformers import SentenceTransformer

## OpenAI (LLM + embeddings)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

## PDF handling
import PyPDF2

## Graphs
import matplotlib.pyplot as plt

## PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

## Load environment variables
from dotenv import load_dotenv


# =========================
# LOAD ENV
# =========================
## Loads .env file (for API key security)
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Resume Analyzer", page_icon="📄")


# =========================
# 🎨 CUSTOM UI STYLE
# =========================
## Adds dark theme + modern card design
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODELS
# =========================

## Load SpaCy model (downloads if missing)
if not os.path.exists("en_core_web_sm"):
    import spacy.cli
    spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

## Semantic similarity model
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

## OpenAI LLM (for feedback)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=api_key
)

## OpenAI embeddings (for vector DB)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)


# =========================
# CHROMADB SETUP
# =========================
## Create local vector database
client = chromadb.Client()
collection = client.get_or_create_collection("resume_db")


# =========================
# FUNCTIONS
# =========================

## -------- PDF TEXT EXTRACTION --------
def extract_text_from_pdf(uploaded_file):
    """Extracts text from uploaded PDF"""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


## -------- EMBEDDING --------
def generate_embeddings(text):
    """Generate embedding using OpenAI"""
    return embeddings.embed_query(text)


## -------- STORE DATA --------
def store_in_chromadb(text, category):
    """Store resume/job in vector DB"""
    collection.add(
        documents=[text],
        metadatas=[{"category": category}],
        embeddings=[generate_embeddings(text)],
        ids=[str(hash(text))]
    )


## -------- KEYWORD EXTRACTION --------
def extract_keywords(text):
    """Extract named entities as keywords"""
    doc = nlp(text)
    return [
        ent.text for ent in doc.ents
        if ent.label_ in ['ORG', 'PERSON', 'WORK_OF_ART', 'MONEY']
    ]


## -------- SEMANTIC SIMILARITY --------
def compare_semantic_similarity(resume, job):
    """Cosine similarity between resume & job"""
    r = semantic_model.encode(resume)
    j = semantic_model.encode(job)
    return np.dot(r, j) / (np.linalg.norm(r) * np.linalg.norm(j))


## -------- ATS SCORE --------
def calculate_ats_score(resume, job):
    """Simple keyword matching score"""
    resume_words = set(resume.lower().split())
    job_words = set(job.lower().split())

    matched = resume_words.intersection(job_words)
    score = len(matched) / len(job_words) * 100

    return round(score, 2)


## -------- SKILL GAP --------
def skill_gap_analysis(resume, job):
    """Find missing skills"""
    resume_words = set(resume.lower().split())
    job_words = set(job.lower().split())

    missing = job_words - resume_words
    return [w for w in missing if len(w) > 3][:20]


## -------- AI FEEDBACK --------
def generate_gpt_feedback(resume, job):
    """Generate HR-style feedback"""
    prompt = f"""
    You are a professional HR recruiter.

    Compare resume with job description.

    Provide:
    - Strengths
    - Weaknesses
    - Missing skills
    - Suggestions
    """

    response = llm.invoke(prompt + resume + job)
    return response.content


## -------- PDF REPORT --------
def create_pdf(report_text):
    """Create downloadable PDF"""
    file_path = "resume_feedback.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []
    for line in report_text.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))

    doc.build(content)
    return file_path


# =========================
# UI HEADER
# =========================
st.markdown("<h1 style='text-align:center;'>📄 Advanced Resume Analyzer</h1>", unsafe_allow_html=True)


# =========================
# INPUT SECTION
# =========================
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])

with col2:
    job_description = st.text_area("📋 Paste Job Description")


resume = ""
if uploaded_file:
    resume = extract_text_from_pdf(uploaded_file)
    st.success("✅ Resume uploaded!")


# =========================
# MAIN BUTTON
# =========================
if st.button("🚀 Analyze Resume"):

    if not resume or not job_description:
        st.warning("⚠ Please provide both inputs")

    else:
        ## Store data
        store_in_chromadb(resume, "resume")
        store_in_chromadb(job_description, "job")

        ## Compute metrics
        similarity = compare_semantic_similarity(resume, job_description)
        ats_score = calculate_ats_score(resume, job_description)
        gaps = skill_gap_analysis(resume, job_description)

        ## AI feedback
        with st.spinner("🤖 AI analyzing..."):
            ai_feedback = generate_gpt_feedback(resume, job_description)

        # =========================
        # DASHBOARD
        # =========================
        st.markdown("## 📊 Dashboard")

        col1, col2 = st.columns(2)

        ## ATS SCORE
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📈 ATS Score")
            st.progress(int(ats_score))
            st.write(f"{ats_score}%")

            fig, ax = plt.subplots()
            ax.bar(["ATS"], [ats_score])
            ax.set_ylim(0, 100)
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        ## SIMILARITY
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🧠 Similarity")

            sim_percent = similarity * 100
            st.progress(int(sim_percent))
            st.write(f"{sim_percent:.2f}%")

            fig2, ax2 = plt.subplots()
            ax2.pie([sim_percent, 100 - sim_percent], labels=["Match", "Gap"], autopct='%1.1f%%')
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

        ## SKILLS & KEYWORDS
        st.subheader("📊 Skill Gaps")
        st.write(gaps)

        st.subheader("🔑 Keywords")
        st.write("Resume:", extract_keywords(resume))
        st.write("Job:", extract_keywords(job_description))

        ## AI Feedback
        st.subheader("🤖 AI Feedback")
        st.write(ai_feedback)

        ## PDF
        report = f"""
        ATS Score: {ats_score}
        Similarity: {similarity}
        Gaps: {gaps}
        Feedback: {ai_feedback}
        """

        pdf_file = create_pdf(report)

        with open(pdf_file, "rb") as f:
            st.download_button("Download PDF", f)
            