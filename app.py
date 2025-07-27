import streamlit as st
from rag_utils import load_and_chunk_data, build_faiss_index, generate_answer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Loan Approval Assistant")
st.title("Loan Approval Assistant")

@st.cache_resource
def setup_rag():
    with st.spinner("Loading data and building index..."):
        chunks = load_and_chunk_data()
        vectorizer, index = build_faiss_index(chunks)
        return chunks, vectorizer, index

chunks, vectorizer, index = setup_rag()

user_query = st.text_input("Ask a question about loan approvals...", "")

if user_query:
    with st.spinner("Generating answer..."):
        # Convert query into vector
        query_vec = vectorizer.transform([user_query]).toarray().astype('float32')

        # Search for similar chunks
        D, I = index.search(query_vec, k=3)
        retrieved_chunks = [chunks[i] for i in I[0]]

        # Build prompt for Gemini
        context = "\n".join(retrieved_chunks)
        prompt = f"""You are an expert loan officer. Based on the following applicant data:\n{context}\n\nAnswer the following question:\n{user_query}"""

        response = generate_answer(prompt)
        st.write("### Response")
        st.write(response)
