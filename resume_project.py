# #Roshan Avatirak 4 week Internship by Edunet 
# import streamlit as st
# from PyPDF2 import PdfReader
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Function to extract text from PDF
# def extract_text_from_pdf(file):
#     pdf = PdfReader(file)
#     text = ""
#     for page in pdf.pages:
#         extracted_text = page.extract_text()
#         if extracted_text:
#             text += extracted_text + "\n"
#     return text

# # Function to rank resumes based on job description
# def rank_resumes(job_desc, resumes):
#     vectorizer = TfidfVectorizer()
#     all_texts = [job_desc] + resumes
#     tfidf_matrix = vectorizer.fit_transform(all_texts)

#     # Compute cosine similarity between job description and resumes
#     job_vector = tfidf_matrix[0]
#     resume_vectors = tfidf_matrix[1:]
#     similarity_scores = cosine_similarity(job_vector, resume_vectors)
    
#     return similarity_scores.flatten()  # Convert to 1D array

# # Streamlit App
# st.title("AI Resume Screening & Candidate Ranking System")

# # Job description input
# st.header("Job Description")
# job_description = st.text_area("Enter the job description")

# # File uploader for resumes
# uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

# if uploaded_files and job_description:
#     st.header("Ranking Resumes")

#     resumes = []
#     for file in uploaded_files:
#         text = extract_text_from_pdf(file)  # Extract text
#         resumes.append(text)

#     # Rank resumes
#     scores = rank_resumes(job_description, resumes)

#     # Display scores
#     results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
#     results = results.sort_values(by="Score", ascending=False)

#     st.write(results)
    
    
    
    # Roshan Avatirak - Enhanced Resume Ranking System

import streamlit as st
from streamlit_lottie import st_lottie
import json
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
from openai import OpenAI

import requests
import streamlit as st


# # Add Bootstrap CDN for styling
# st.markdown("""
#     <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
# """, unsafe_allow_html=True)


# # You can also include Bootstrap JS if needed (optional)
# st.markdown("""
#     <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
#     <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"></script>
# """, unsafe_allow_html=True)



# Use Streamlit's secrets to store your API key securely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_resume_suggestion(job_desc, resume_text):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional resume reviewer."},
                {
                    "role": "user",
                    "content": f"""Analyze this resume against the job description and suggest 3 improvements:

Job Description:
{job_desc}

Resume:
{resume_text}
"""
                },
            ],
            temperature=0.7
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Error getting suggestions:\n\n{e}"



st.set_page_config(page_title="AI Resume Ranker", page_icon="🧠", layout="wide")

# Custom CSS to change the error message background
st.markdown("""
    <style>
    .error-message {
        color: red;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)




# # Display an error message with custom styling
# st.markdown('<div class="error-message">Error getting suggestions: You tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0</div>', unsafe_allow_html=True)

# def get_resume_suggestion(job_description, resume_text):
#     prompt = f"""
#     You're a resume expert. Analyze the following resume against this job description and suggest at least 3 improvements or additions in simple points.

#     Job Description:
#     {job_description}

#     Resume Text:
#     {resume_text}

#     Your Suggestions:
#     """

#     try:
#         response = openai.completion.create(
#             model="gpt-3.5-turbo",  # You can use this with free key
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=300,
#             temperature=0.7
#         )
#         suggestions = response['choices'][0]['message']['content']
#         return suggestions
#     except Exception as e:
#         return f"⚠️ Error getting suggestions: {e}"

# def get_resume_suggestion(job_description, resume_text):
#     prompt = f"""
#     You're a resume expert. Analyze the following resume against this job description and suggest at least 3 improvements or additions in simple points.

#     Job Description:
#     {job_description}

#     Resume Text:
#     {resume_text}

#     Your Suggestions:
#     """

#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant for resume improvement."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=500,
#             temperature=0.7
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"⚠️ Error getting suggestions: {e}"



# ------------------------- PDF TEXT EXTRACTION ------------------------- #
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ------------------------- SIMILARITY SCORING ------------------------- #
def rank_resumes(job_desc, resumes):
    vectorizer = TfidfVectorizer()
    all_texts = [job_desc] + resumes
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(job_vector, resume_vectors)

    return similarity_scores.flatten()

# ------------------------- KEYWORD MATCHING ------------------------- #
def keyword_match_percentage(job_desc, resume_text):
    job_keywords = set(job_desc.lower().split())
    resume_keywords = set(resume_text.lower().split())
    
    matched_keywords = job_keywords.intersection(resume_keywords)
    match_percentage = (len(matched_keywords) / len(job_keywords)) * 100 if job_keywords else 0
    
    return match_percentage, matched_keywords



def add_styles(dark_mode):
    """Add dark/light mode styles based on user preference."""
    if dark_mode:
        st.markdown("""
            <style>
                body {
                    background-color: #121212;
                    color: white;
                }
                .resume-card {
                    background-color: #2a2a2a;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 15px 0;
                    border: 2px solid #444;
                    color: #ccc;  /* Ensure all text is light-colored */
                }
                .resume-card h4 {
                    color: #ffcc00;  /* Bright title color */
                }
                .resume-card p {
                    color: #ccc;  /* Paragraphs clearly visible */
                }
                .resume-card div {
                    background-color: #444; /* Nested divs readable */
                }
                .stProgress > div {
                    background-color: #ffcc00;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                body {
                    background-color: white;
                    color: black;
                }
                .resume-card {
                    background-color: #f9f9f9;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 15px 0;
                    border: 2px solid #ddd;
                    color: #333;  /* Set fallback readable text */
                }
                .resume-card h4 {
                    color: #008080;
                }
                .resume-card p {
                    color: #444;  /* Ensure dark text */
                }
                .resume-card div {
                    background-color: #eee;  /* Light div background */
                }
                .stProgress > div {
                    background-color: #008080;
                }
            </style>
        """, unsafe_allow_html=True)


# ------------------------- MAIN APP ------------------------- #


# 🌗 Theme Toggle
dark_mode = st.toggle("🌗 Toggle Dark/Light Mode", value=True)
add_styles(dark_mode)

# RAO Present branding above the title
st.markdown("""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 5px;">
        <span style="font-size: 18px; font-weight: bold; background: linear-gradient(90deg, #ffd700, #ffcc00); 
                    -webkit-background-clip: text; color: transparent;">
             RAO Present's
        </span>
    </div>
""", unsafe_allow_html=True)

st.markdown(f"<h1 style='text-align:center;'>🧠 AI Resume Screening & Ranking System</h1>", unsafe_allow_html=True)

# Optional Lottie animation
try:
    with open("ai_resume_lottie.json") as f:
        lottie_data = json.load(f)
    st_lottie(lottie_data, height=300)
except:
    st.warning("Lottie animation not loaded.")

st.markdown("---")

# ✍️ Job Description Input
st.markdown("### ✍️ Job Description")
job_description = st.text_area("Paste the job description here...", height=200)

# 📄 Resume Upload
st.markdown("### 📄 Upload Resumes (PDF Only)")
uploaded_files = st.file_uploader("Select one or more resumes", type="pdf", accept_multiple_files=True)

# ⏳ Upload Progress
if uploaded_files:
    st.markdown("### ⏳ Uploading Resumes...")
    progress_bar = st.progress(0)
    for i in range(len(uploaded_files)):
        progress_bar.progress((i + 1) / len(uploaded_files))
        time.sleep(0.1)
    st.success("Resumes uploaded!")

# --- Resume Ranking Section ---
if uploaded_files and job_description:
    st.markdown("### 📊 Resume Match Results")

    # Extract text from resumes
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]

    # Rank resumes
    scores = rank_resumes(job_description, resumes)
    scaled_scores = [round(score * 100, 2) for score in scores]

    # Sort by highest score
    results = sorted(zip(uploaded_files, scaled_scores), key=lambda x: x[1], reverse=True)

    # Show ranked resumes
    for rank, (file, score) in enumerate(results, start=1):
        # Color based on score
        color = "red" if score < 50 else "orange" if score < 70 else "green"
        
        # Extract text from resume to calculate keyword match
        resume_text = extract_text_from_pdf(file)
        
        # Keyword Match %
        match_percentage, matched_keywords = keyword_match_percentage(job_description, resume_text)
        
        # Tips
        if score < 50:
            tip = "❌ Needs significant improvement. Tailor your resume to the job."
        elif score < 70:
            tip = "⚠️ Fair match, but there’s room for improvement."
        else:
            tip = "✅ Great match! Your resume aligns well."

        # Display resume rank, score, and tips
        st.markdown(f"""
        <div class="resume-card">
            <h4>🏆 Rank #{rank}: {file.name}</h4>
            <div style="margin-bottom: 10px;">
                <b>Match Score:</b> {score:.2f}%
                <div style="background-color: lightgray; border-radius: 10px; height: 20px;">
                    <div style="width: {score}%; height: 100%; background-color: {color}; border-radius: 10px;"></div>
                </div>
            </div>
            <b>Keyword Match:</b> {match_percentage:.2f}% 
            <br><b>Matched Keywords:</b> {', '.join(matched_keywords)}
            <p><b>Tips:</b> {tip}</p>
        </div>
        """, unsafe_allow_html=True)
        
        
         # ChatGPT Suggestions
        with st.spinner("Analyzing with ChatGPT..."):
            suggestion = get_resume_suggestion(job_description, resume_text)

        st.markdown(f"""
        <div class="ai-suggestion-card" style="background-color:#121212; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 8px; margin-top: 10px;">
            <h5 style="color: green;"> <img src="https://openai.com/favicon.ico" width="30" height="30" style="border-radius: 50%; margin-right: 12px;" /> AI Suggestions to Improve This Resume</h5>
            <p style="color:#E0E0E0">{suggestion.replace('\n', '<br>')}</p>
        </div>
        """, unsafe_allow_html=True)
        

    st.markdown("---")

    # Download CSV
    df = pd.DataFrame({
        "Rank": list(range(1, len(results)+1)),
        "Resume": [file.name for file, _ in results],
        "Score (%)": [score for _, score in results]
    })
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Ranking Results as CSV",
        data=csv,
        file_name="resume_ranking_results.csv",
        mime="text/csv"
    )




    
    
    # import streamlit as st
# from PyPDF2 import PdfReader
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.metrics.pairwise import cosine_similarity

# #function to extract text from pdf
# def extract_text_from_pdf(file):
#     pdf = PdfReader(file)
#     text = ""
#     for page in pdf.pages:
#         text += page.extract_text()
#         return text
    
#     #streamlit app
# st.title("AI Resume Screening & Candidate Ranking System")
# #job description input
# st.header("job description")
# job_description = st.text_area("Enter the job description")

# if uploaded_files and job_description:
# st.header("Ranking Resumes")

# resumes = []
# for file in uploaded_files:
#     text = extract_text_from_pdf(file) ###function calling is happening
#     resumes.append(text)

#     #Rank resumes
#     scores = rank_resumes(job_description, resumes)

#     #display scores
#     results = pd.DataFrames({"Resume":[file.name for file in uploaded_file], "Score": scores})
#     results = results.sort_values(by="Score", ascending=False)

#     st.write(results)
