import pandas as pd
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai

# Load API key securely from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

CHUNK_SIZE = 150

# Load dataset and chunk it into smaller parts
def load_and_chunk_data():
    df = pd.read_csv("Training Dataset.csv")

    def row_to_text(row):
        return f"""
        Gender: {row['Gender']}, Married: {row['Married']}, Dependents: {row['Dependents']},
        Education: {row['Education']}, Self_Employed: {row['Self_Employed']}, 
        ApplicantIncome: {row['ApplicantIncome']}, CoapplicantIncome: {row['CoapplicantIncome']},
        LoanAmount: {row['LoanAmount']}, Loan_Amount_Term: {row['Loan_Amount_Term']},
        Credit_History: {row['Credit_History']}, Property_Area: {row['Property_Area']},
        Loan_Status: {row['Loan_Status']}
        """

    docs = df.apply(row_to_text, axis=1).tolist()

    def chunk_text(text, size=CHUNK_SIZE):
        words = text.split()
        return [' '.join(words[i:i + size]) for i in range(0, len(words), size)]

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))

    return all_chunks


# Build FAISS index from chunked data
def build_faiss_index(chunks):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks).toarray().astype('float32')

    dim = tfidf_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(tfidf_matrix)

    return vectorizer, index


# Generate answer using Gemini
def generate_answer(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        chat = model.start_chat()
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}\nNote: This may be due to API quota exhaustion or incorrect API key."
