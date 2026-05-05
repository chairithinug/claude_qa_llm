# from openai import OpenAI
# client = OpenAI()

# emb = client.embeddings.create(
#     model="text-embedding-3-small",
#     input="Hello World!"
# )

# vector = emb.data[0].embedding
# print(len(vector))  # ~1536 dims

import ollama

response = ollama.embeddings(
    model="nomic-embed-text",
    prompt="RAG is retrieval augmented generation"
)

vector = response["embedding"]
print(len(vector))