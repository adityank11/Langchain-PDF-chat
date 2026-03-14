from langchain_huggingface import HuggingFaceEmbeddings
import dotenv
dotenv.load_dotenv()

class Embedding:
    def __init__(self, embedding_model):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)