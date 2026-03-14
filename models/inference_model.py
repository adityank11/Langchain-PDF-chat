from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from embeddings import Embedding
import pdf_loader
import dotenv
dotenv.load_dotenv()


class InferenceLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
        )

    def generate_response(self, input_text):
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": input_text}],
            model=self.model_name
        )
        return response.choices[0].message.content

    def get_vector_store(self):
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = Embedding(embedding_model)
        loader = pdf_loader.PDFLoader()
        docs = loader.load(file_path="data\Dream Forest Langkawi.pdf")
        vector_store = InMemoryVectorStore(embedding=embeddings.embeddings)
        vector_store.add_documents(docs)
        return vector_store

    def ask_question(self, question):
        print("Getting vector store...")
        vector_store = self.get_vector_store()
        print("Creating retriever...")
        retriever = vector_store.as_retriever()
        print("Retrieving docs...")
        docs = retriever.invoke(question)
        print(f"Retrieved {len(docs)} docs")
        context = "\n".join([doc.page_content for doc in docs])
        print("Context length:", len(context))
        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
        print("Generating response...")
        return self.generate_response(prompt)