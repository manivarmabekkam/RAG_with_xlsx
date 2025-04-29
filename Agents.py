# Install needed libraries first
# pip install pandas openpyxl google-cloud-aiplatform faiss-cpu

import pandas as pd
import numpy as np
import faiss
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.language_models import TextGenerationModel
import vertexai

# 1. Initialize Vertex AI
vertexai.init(project="agents-458316", location="us-central1")  # e.g., us-central1

# 2. Load Excel Data
df = pd.read_excel('store_inventory.xlsx')  # Or use pd.read_csv('your_file.csv')

# 3. Chunk Data (each row is a chunk)
chunks = df.to_dict(orient='records')

# 4. Embed Each Chunk
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
embeddings = []

for chunk in chunks:
    text = " ".join(str(v) for v in chunk.values())
    embedding = embedding_model.get_embeddings([text])[0].values
    embeddings.append(embedding)

# 5. Create FAISS Vector Store
d = len(embeddings[0])  # Embedding size
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings).astype('float32'))

# 6. Function: RAG Search and Generate Answer
def rag_query(user_query, top_k=5):
    # Step 1: Embed the user query
    query_embedding = embedding_model.get_embeddings([user_query])[0].values
    
    # Step 2: Search Top-K relevant chunks
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]
    
    # Step 3: Prepare context for LLM
    context = "\n".join([str(chunk) for chunk in retrieved_chunks])
    prompt = f"""Use the following context to answer the question.

Context:å
{context}

Question:
{user_query}

Answer:"""
    
    # Step 4: Generate Answer with Vertex LLM
    gen_model = TextGenerationModel.from_pretrained("text-bison")
    response = gen_model.predict(prompt)
    
    return response.text

# 7. Example Usage
if __name__ == "__main__":
    user_question = "What are the top sales regions in January?"
    answer = rag_query(user_question)
    print("\n---Answer---\n", answer)

