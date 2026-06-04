import time
import streamlit as st

from rag import rag_query

st.set_page_config(
    page_title="Upwork API Assistant",
    page_icon="🤖"
)

st.title("🤖 Upwork API Consultant")
st.write("Ask technical questions about the Upwork API documentation.")

question = st.text_input(
    "Ask a question about the Upwork API"
)

if st.button("Submit"):

    start_time = time.time()

    answer, docs = rag_query(question)

    latency = round(
        time.time() - start_time,
        2
    )

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for i, doc in enumerate(docs, start=1):
        with st.expander(f"Source {i}"):
            st.write(doc.page_content)

    st.success(
        f"Latency: {latency} seconds"
    )