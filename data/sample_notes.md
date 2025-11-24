# Gemini RAG Demo Notes

## Architecture

- Documents live in the local `data/` folder and are chunked with a recursive splitter.
- Embeddings are generated with `text-embedding-004` from Google Gemini.
- ChromaDB stores the vectors so they can be reused between runs.
- LangChain wires together the retriever, prompt, and Gemini chat model to produce grounded answers.

## Elevator Pitch

This demo illustrates a *hello world* Retrieval Augmented Generation (RAG) pipeline. It is intentionally tiny, but it mirrors the same building blocks used in production systems:

1. Curate and chunk a knowledge base.
2. Embed and persist it in a vector database.
3. Retrieve the most relevant chunks for each incoming query.
4. Ask a multimodal model such as Gemini to answer using only that grounded context.

## Talking Points

- Keep the context window tight (4 chunks) so answers stay focused.
- Swap the markdown files in `data/` with your docs to get a custom demo running in minutes.
- Persisted vectors live under `.rag_cache/` so you can delete that folder to rebuild from scratch.
