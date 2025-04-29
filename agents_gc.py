
import os

import numpy as np
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_vertexai import VertexAI
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import



# Set tokenizer parallelism to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Initialize Vertex AI
import vertexai

vertexai.init(project="agents-458316", location="us-central1")

# 2. Load Excel Data
df = pd.read_excel("store_inventory.xlsx")

# 3. Prepare documents
documents = []
for _, row in df.iterrows():
    # Convert row to string
    text = " ".join(str(v) for v in row.values)
    documents.append(Document(page_content=text, metadata=dict(row)))

# 4. Initialize HuggingFace Embeddings (runs locally)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Create FAISS vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# 6. Create QA Chain with updated VertexAI initialization
llm = VertexAI(
    model_name="gemini-1.5-flash",  # Gemini model name
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
)


# 7. Function: RAG Query
def rag_query(user_query):
    response = qa_chain.run(user_query)
    return response


# 8. Example Usage
if __name__ == "__main__":
    user_question = "which product have higher discount in summer ?"
    answer = rag_query(user_question)
    print("\n---Answer---\n", answer)
