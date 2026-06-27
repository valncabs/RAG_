"""
Script de ingesta de documentos.

Uso:
python -m app.ingest
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import settings


def load_documents(docs_path: str) -> list:
    """
    Carga todos los PDF y TXT de la carpeta docs.
    """

    documents = []

    docs_folder = Path(docs_path)

    if not docs_folder.exists():
        print(f"La carpeta '{docs_path}' no existe.")
        return documents

    supported_files = list(docs_folder.glob("*.pdf"))
    supported_files += list(docs_folder.glob("*.txt"))

    if not supported_files:
        print(f"No se encontraron documentos en {docs_path}")
        return documents

    for file_path in supported_files:

        try:

            print(f"Cargando {file_path.name}")

            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))

            else:
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8"
                )

            docs = loader.load()

            documents.extend(docs)

            print(f"{len(docs)} páginas cargadas")

        except Exception as e:
            print(f"Error con {file_path.name}: {e}")

    print(f"\nTotal de documentos: {len(documents)}")

    return documents

def split_documents(documents: list) -> list:
    """
    Divide los documentos en fragmentos (chunks).
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Documentos divididos en {len(chunks)} chunks")
    print(f"Tamaño de chunk: {settings.CHUNK_SIZE}")
    print(f"Overlap: {settings.CHUNK_OVERLAP}")

    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """
    Genera embeddings y los almacena en ChromaDB.
    """

    print(f"\nGenerando embeddings con {settings.EMBEDDING_MODEL}")
    print("La primera ejecución puede tardar algunos minutos...")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )

    print(f"\nVector Store creado en:")
    print(settings.CHROMA_DB_PATH)

    print(f"Colección:")
    print(settings.CHROMA_COLLECTION_NAME)

    print(f"Vectores almacenados: {len(chunks)}")

    return vector_store

def run_ingestion():
    """
    Ejecuta el proceso completo de ingesta.
    """

    print("=" * 60)
    print("INICIANDO INGESTA DE DOCUMENTOS")
    print("=" * 60)

    # Validar configuración
    settings.validate()

    # Paso 1
    print("\nPASO 1: Cargando documentos...")
    documents = load_documents(settings.DOCS_PATH)

    if not documents:
        print("\nNo hay documentos para procesar.")
        print("Agrega archivos PDF o TXT dentro de la carpeta docs.")
        sys.exit(1)

    # Paso 2
    print("\nPASO 2: Dividiendo documentos...")
    chunks = split_documents(documents)

    # Paso 3
    print("\nPASO 3: Generando embeddings...")
    create_vector_store(chunks)

    print("\n" + "=" * 60)
    print("INGESTA COMPLETADA CORRECTAMENTE")
    print("=" * 60)

    print("\nAhora puedes iniciar la API con:")

    print("uvicorn app.main:app --reload")


if __name__ == "__main__":
    run_ingestion()