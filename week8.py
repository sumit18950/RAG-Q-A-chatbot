# RAG Chatbot on Loan Dataset

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
import streamlit as st


df = pd.read_csv("Training Dataset.csv")
df['text'] = df.apply(lambda row: ' '.join(map(str, row.values)), axis=1)
documents = df['text'].tolist()


embedder = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedder.encode(documents, convert_to_tensor=False)
document_embeddings = np.array(document_embeddings)
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)


def retrieve_docs(query, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]

openai.api_key = "YOUR_OPENAI_API_KEY"

def generate_answer_openai(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    prompt = f"""Answer the following question based on the context:

Context:
{context}

Question: {query}

Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message["content"].strip()

st.title("Loan Dataset RAG Q&A Chatbot")
query = st.text_input("Ask a question about the dataset")

if query:
    docs = retrieve_docs(query)
    answer = generate_answer_openai(query, docs)

    st.subheader("Answer:")
    st.write(answer)

    with st.expander("Retrieved Documents"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Doc {i+1}:** {doc[:500]}...")


