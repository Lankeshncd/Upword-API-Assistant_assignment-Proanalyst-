import os
import requests
import streamlit as st

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# Load API Key
# ==========================================

load_dotenv()

try:
    API_KEY = st.secrets["DEEPINFRA_API_KEY"]
except:
    API_KEY = os.getenv("DEEPINFRA_API_KEY")

# ==========================================
# Load Embedding Model
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# Load FAISS Vector Store
# ==========================================

db = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

# ==========================================
# Retrieve Documents
# ==========================================

def retrieve_documents(query):
    docs = db.similarity_search(query, k=3)
    return docs

# ==========================================
# Call LLM
# ==========================================

def ask_llm(context, question):

    if not API_KEY:
        return "API Error: DEEPINFRA_API_KEY not found."

    prompt = f"""
You are a Senior Upwork API Consultant.

Answer ONLY using the provided documentation.

If the answer is not present in the documentation, respond exactly with:

I'm sorry, but the provided documentation does not contain that information.

Documentation:
{context}

Question:
{question}
"""

    try:

        response = requests.post(
            "https://api.deepinfra.com/v1/openai/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Senior Upwork API Consultant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2
            },
            timeout=60
        )

        result = response.json()

        if "choices" not in result:
            return f"API Error: {result}"

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error calling LLM: {str(e)}"

# ==========================================
# Main RAG Function
# ==========================================

def rag_query(question):

    docs = retrieve_documents(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    answer = ask_llm(
        context,
        question
    )

    return answer, docs