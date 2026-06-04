import os
import requests

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

API_KEY = os.getenv("DEEPINFRA_API_KEY")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector store
db = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)


def retrieve_documents(query):
    docs = db.similarity_search(query, k=3)
    return docs


def ask_llm(context, question):

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
        }
    )

    result = response.json()

    return result["choices"][0]["message"]["content"]


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