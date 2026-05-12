import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from prompts import POLICY_QA_PROMPT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "vectorstore")


class RAGPipeline:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ✅ Explicitly allow pickle loading (safe: local data you created)
        self.vectorstore = FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        self.llm = Ollama(
            model="mistral",
            temperature=0
        )

    def run(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        #docs = self.retriever.get_relevant_documents(question)

        context = "\n\n".join(
            [f"(Page {d.metadata.get('page', 'N/A')}) {d.page_content}" for d in docs]
        )

        prompt = POLICY_QA_PROMPT.format(
            context=context,
            question=question
        )

        response = self.llm.invoke(prompt)
        return response


def get_rag_chain():
    return RAGPipeline()