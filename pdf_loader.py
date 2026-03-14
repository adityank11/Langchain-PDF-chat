from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    def __init__(self):
        self.docs = ""
        self.loader = ""

    def load(self, file_path):
        self.loader = PyPDFLoader(file_path, mode="page")
        self.docs = self.loader.load()
        return self.docs