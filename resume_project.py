#Roshan Avatirak 4 week Internship by Edunet 
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text

# Function to rank resumes based on job description
def rank_resumes(job_desc, resumes):
    vectorizer = TfidfVectorizer()
    all_texts = [job_desc] + resumes
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Compute cosine similarity between job description and resumes
    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(job_vector, resume_vectors)
    
    return similarity_scores.flatten()  # Convert to 1D array

# Streamlit App
st.title("AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader for resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)  # Extract text
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display scores
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    st.write(results)
    
    
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
