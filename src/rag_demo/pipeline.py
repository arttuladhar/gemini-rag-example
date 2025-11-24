from __future__ import annotations

from typing import Iterable, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from .config import Settings

COLLECTION_NAME = "demo-notes"


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return splitter.split_documents(documents)


def build_vector_store(chunks: List[Document], settings: Settings) -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=settings.gemini_api_key, model=settings.embedding_model
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(settings.persist_directory),
    )
    return vector_store


def _format_docs(docs: Iterable[Document]) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"Source: {source}\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_chain(vector_store: Chroma, settings: Settings):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise assistant. Use the provided context to answer the user's question. "
                "If the answer is not in the context, say you do not know.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        google_api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        temperature=settings.temperature,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": settings.top_k})

    chain = (
        {"context": retriever | RunnableLambda(_format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def run_query(question: str, documents: List[Document], settings: Settings) -> str:
    """Create the vector store and run a single RAG query."""
    chunks = split_documents(documents)
    vector_store = build_vector_store(chunks, settings)
    chain = build_chain(vector_store, settings)
    return chain.invoke(question)
