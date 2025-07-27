# Loan Approval Assistant

### _Week 8 - RAG Q&A chatbot_

Link to working streamlit app: https://rag-q-a-chatbot-f4qggzlbwcf3kaz3b7rnrs.streamlit.app/

A Retrieval-Augmented Generation (RAG) Q&A chatbot that helps users get intelligent answers based on loan applicant data. It combines **document retrieval (FAISS)** with **generative AI** (Gemini via Google GenerativeAI) to generate human-like responses using real loan approval datasets.

## Project Overview

This chatbot is designed to:
- Ingest and chunk a structured loan dataset.
- Build a vector search index using TF-IDF + FAISS.
- Retrieve relevant data chunks based on user queries.
- Use a lightweight generative LLM (Gemini) to formulate intelligent responses.

> Dataset Source: [Kaggle - Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction?select=Training+Dataset.csv)

---

## Features

- **RAG Pipeline**: Uses FAISS for semantic retrieval and Gemini for answer generation.
- **Natural Language Interface**: Ask questions like "What are the common factors in loan rejection?"
- **Lightweight**: Uses TF-IDF for simplicity and performance.
- **Streamlit UI**: Easy-to-use frontend for user queries.

---

## Model Used

Retrieval: TF-IDF + FAISS (faiss.IndexFlatL2)

Generation: Gemini 1.5 Pro via google-generativeai

You can also easily adapt this to use OpenAI, Claude, or any Hugging Face LLM by modifying the generate_answer function in rag_utils.py.

--- 

## Example Questions

  Try asking:
  - "What affects loan approval the most?"
  - "Do self-employed applicants get loans?"
  - "How does credit history impact approval?"

---

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/your-username/loan-rag-chatbot.git
   cd loan-rag-chatbot
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Set up API Key:**
   Create a .env file in the root directory:
   ```
   GOOGLE_API_KEY=your_google_genai_api_key_here
   ```
   
---

## Run the App
  ```
  streamlit run app.py
  ```
  The app will launch in your default web browser.

---

## Using virtual environment:

  ```
  python -m venv venv
  venv\Scripts\activate
  pip install streamlit
  streamlit run app.py
  ```


