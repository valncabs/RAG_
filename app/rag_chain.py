from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.config import settings


PROMPT_TEMPLATE = """
Eres Pet AI, un asistente experto en mascotas.

Utiliza SIEMPRE la información proporcionada en el contexto para responder.

Si el contexto contiene información relacionada, responde usando esa información.

Solo responde que no encontraste información cuando el contexto esté completamente vacío o no tenga ninguna relación con la pregunta.

No inventes información.

Contexto:
{context}

Pregunta:
{question}
"""


RAG_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(embeddings):

    return Chroma(
        persist_directory=settings.CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )


def get_llm():

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=settings.GROQ_API_KEY,
        temperature=0.1,
        max_tokens=512,
    )


def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


_retriever = None
_chain = None


def initialize_chain():

    global _retriever
    global _chain

    settings.validate()

    embeddings = get_embeddings()

    vector_store = get_vector_store(
        embeddings
    )

    llm = get_llm()

    _retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": settings.RETRIEVER_K
        },
    )

    _chain = (
        {
            "context": _retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def get_answer(question: str):

    if _chain is None:
        raise RuntimeError(
            "Pipeline no inicializado."
        )

    source_documents = _retriever.invoke(
        question
    )

    answer = _chain.invoke(question)

    sources = []

    for doc in source_documents:

        sources.append(
            {
                "documento": doc.metadata.get("source", "Desconocido"),
                "pagina": doc.metadata.get("page"),
                "fragmento": doc.page_content[:150] + "..."
            }
        )

    return {
        "answer": answer.strip(),
        "sources": sources,
    }