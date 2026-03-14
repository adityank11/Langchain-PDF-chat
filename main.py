from pdf_loader import PDFLoader
from models.inference_model import InferenceLLM
from embeddings import Embedding
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import dotenv
dotenv.load_dotenv()



pdf_loader = PDFLoader()
docs = pdf_loader.load(file_path="data\Dream Forest Langkawi.pdf")

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = Embedding(embedding_model)
vector_store = InMemoryVectorStore(embedding=embeddings.embeddings)
vector_store.add_documents(docs)

# For Q&A chatbot
llm = InferenceLLM(model_name="openai/gpt-oss-20b")

if __name__ == "__main__":
    question = "Who is the lead participant?"
    answer = llm.ask_question(question)
    print("Answer:", answer)

