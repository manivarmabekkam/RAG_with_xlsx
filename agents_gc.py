import os
import streamlit as st
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_vertexai import VertexAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import vertexai

# Set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Page config
st.set_page_config(page_title="GNI Tool", layout="centered")
st.title("📊 Shopify Inventory QA")

@st.cache_resource
def load_data_and_setup():
    # VertexAI Init
    vertexai.init(project="agents-458316", location="us-central1")
    
    # Load Excel
    df = pd.read_excel("store_inventory.xlsx")

    # Convert rows to Documents
    documents = []
    for _, row in df.iterrows():
        text = " ".join(str(v) for v in row.values)
        documents.append(Document(page_content=text, metadata=dict(row)))

    # Embeddings + VectorStore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # VertexAI model
    llm = VertexAI(
        model_name="gemini-1.5-flash",
        max_output_tokens=256,
        temperature=0.1,
        top_p=0.8,
        top_k=40,
    )

    # RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    )
    return qa_chain

# Load everything
qa_chain = load_data_and_setup()

# UI Input
user_question = st.text_input("Ask a question about the store inventory:", placeholder="e.g., how many toys ordered in p0002?")

if user_question:
    with st.spinner("Generating answer..."):
        answer = qa_chain.run(user_question)
        st.success("✅ Answer:")
        st.write(answer)