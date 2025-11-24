# Building a RAG Application: A Complete Guide Using LangChain, ChromaDB, and Google Gemini

## Table of Contents
1. [Introduction to RAG](#introduction-to-rag)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Implementation Deep Dive](#implementation-deep-dive)
5. [Key Concepts Explained](#key-concepts-explained)
6. [Step-by-Step Setup](#step-by-step-setup)
7. [Code Walkthrough](#code-walkthrough)
8. [Best Practices](#best-practices)
9. [Extensions and Improvements](#extensions-and-improvements)

---

## Introduction to RAG

**Retrieval Augmented Generation (RAG)** combines retrieval and generation to improve LLM outputs with external knowledge.

### Why RAG?
- Reduces hallucinations by grounding answers in source documents
- Keeps models up to date without retraining
- Enables domain-specific knowledge
- Provides source attribution

### How RAG Works
1. **Ingestion**: Load and process documents
2. **Chunking**: Split documents into smaller pieces
3. **Embedding**: Convert chunks into vectors
4. **Storage**: Store vectors in a vector database
5. **Retrieval**: Find relevant chunks for a query
6. **Generation**: Use retrieved context to generate an answer

---

## System Architecture

```
┌─────────────┐
│   User      │
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                         │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │   Document   │───▶│   Text       │───▶│ Embedding│ │
│  │   Loader     │    │   Splitter   │    │  Model   │ │
│  └──────────────┘    └──────────────┘    └────┬─────┘ │
│                                                 │       │
│                                                 ▼       │
│                                          ┌──────────┐  │
│                                          │ ChromaDB │  │
│                                          │  Vector  │  │
│                                          │  Store   │  │
│                                          └────┬─────┘  │
│                                               │        │
│  ┌──────────────┐    ┌──────────────┐       │        │
│  │   Query      │───▶│   Retriever  │◀──────┘        │
│  └──────────────┘    └──────┬───────┘                 │
│                              │                         │
│                              ▼                         │
│                    ┌──────────────────┐                │
│                    │  Prompt Template │                │
│                    │  (with context)  │                │
│                    └────────┬─────────┘                │
│                             │                          │
│                             ▼                          │
│                    ┌──────────────────┐                │
│                    │  Gemini LLM      │                │
│                    │  (Generation)    │                │
│                    └────────┬─────────┘                │
│                             │                          │
└─────────────────────────────┼──────────────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │   Answer     │
                        └──────────────┘
```

---

## Core Components

### 1. **Document Loader** (`data_loader.py`)
Loads and converts files into LangChain `Document` objects.

**Key Features:**
- Supports `.md` and `.txt`
- Recursive directory scanning
- Metadata tracking (source file paths)
- UTF-8 encoding

**Why This Matters:**
- Standardizes input format
- Preserves source information
- Enables batch processing

### 2. **Text Splitter** (`pipeline.py`)
Splits documents into smaller chunks.

**Why Chunking?**
- Embedding models have token limits
- Smaller chunks improve retrieval precision
- Overlap preserves context across boundaries

**Parameters:**
- `chunk_size=600`: Characters per chunk
- `chunk_overlap=100`: Overlapping characters

### 3. **Embedding Model** (`pipeline.py`)
Converts text into dense vectors.

**Google Gemini Embeddings:**
- Model: `text-embedding-004`
- Produces high-dimensional vectors
- Captures semantic meaning

**How Embeddings Work:**
- Similar text → similar vectors
- Enables semantic search
- Distance metrics (cosine similarity) find relevant chunks

### 4. **Vector Database** (`pipeline.py`)
Stores and queries embeddings.

**ChromaDB Features:**
- In-memory and persistent storage
- Fast similarity search
- Collection-based organization
- Automatic persistence

**Why ChromaDB?**
- Lightweight and easy to use
- Good for prototyping
- Local-first (privacy-friendly)
- LangChain integration

### 5. **Retriever** (`pipeline.py`)
Finds relevant chunks for a query.

**Retrieval Strategy:**
- `top_k=4`: Returns top 4 most similar chunks
- Cosine similarity search
- Returns chunks with metadata

### 6. **LLM Chain** (`pipeline.py`)
Orchestrates retrieval and generation.

**Components:**
- **Retriever**: Gets relevant context
- **Prompt Template**: Formats context + question
- **LLM**: Generates answer (Gemini)
- **Output Parser**: Extracts text response

---

## Implementation Deep Dive

### Component 1: Configuration Management (`config.py`)

```python
@dataclass(slots=True)
class Settings:
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-flash-latest"
    embedding_model: str = "models/text-embedding-004"
    temperature: float = 0.2
    top_k: int = 4
    data_dir: Path
    persist_directory: Path
```

**Design Patterns:**
- **Dataclass**: Type-safe configuration
- **Environment Variables**: Secure credential management
- **Default Values**: Sensible defaults
- **Validation**: API key checking

**Key Settings Explained:**
- `temperature=0.2`: Lower = more deterministic
- `top_k=4`: Balance between context and focus
- `persist_directory`: Enables vector reuse

### Component 2: Document Loading (`data_loader.py`)

```python
def load_local_documents(data_dir: Path) -> List[Document]:
    files = [path for path in data_dir.rglob("*") 
             if path.suffix.lower() in TEXT_EXTENSIONS]
    
    documents = []
    for path in sorted(files):
        text = path.read_text(encoding="utf-8").strip()
        metadata = {"source": str(path.relative_to(data_dir))}
        documents.append(Document(page_content=text, metadata=metadata))
    
    return documents
```

**Key Concepts:**
- **Recursive Search**: `rglob("*")` finds all files
- **Document Object**: LangChain's standard format
- **Metadata Preservation**: Tracks source for attribution
- **Encoding Safety**: UTF-8 handling

### Component 3: Text Chunking (`pipeline.py`)

```python
def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=100
    )
    return splitter.split_documents(documents)
```

**RecursiveCharacterTextSplitter Strategy:**
1. Tries to split on paragraphs
2. Falls back to sentences
3. Then to words
4. Finally to characters

**Why Overlap?**
- Prevents context loss at boundaries
- Improves retrieval for edge cases
- Maintains semantic continuity

**Chunk Size Considerations:**
- Too small: Loses context
- Too large: Dilutes relevance
- 600 characters: Good balance for most use cases

### Component 4: Vector Store Creation (`pipeline.py`)

```python
def build_vector_store(chunks: List[Document], settings: Settings) -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=settings.gemini_api_key,
        model=settings.embedding_model
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(settings.persist_directory),
    )
    return vector_store
```

**What Happens Here:**
1. Initialize embedding model
2. Convert all chunks to vectors
3. Store in ChromaDB with metadata
4. Persist to disk for reuse

**Embedding Process:**
- Each chunk → vector (e.g., 768 dimensions)
- Vectors stored with original text and metadata
- Indexed for fast similarity search

### Component 5: RAG Chain Construction (`pipeline.py`)

```python
def build_chain(vector_store: Chroma, settings: Settings):
    # 1. Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise assistant..."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    
    # 2. Initialize LLM
    llm = ChatGoogleGenerativeAI(
        google_api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        temperature=settings.temperature,
    )
    
    # 3. Create retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.top_k}
    )
    
    # 4. Build chain
    chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

**LangChain Expression Language (LCEL):**
- `|` operator chains components
- `RunnablePassthrough()` forwards input
- `RunnableLambda()` applies custom functions
- Enables composable pipelines

**Chain Flow:**
1. Query → Retriever → Top K chunks
2. Chunks → `_format_docs()` → Formatted string
3. Formatted context + question → Prompt template
4. Prompt → LLM → Generated text
5. Generated text → Output parser → Final answer

### Component 6: Document Formatting (`pipeline.py`)

```python
def _format_docs(docs: Iterable[Document]) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"Source: {source}\n{doc.page_content}")
    return "\n\n".join(formatted)
```

**Purpose:**
- Converts retrieved chunks into prompt-ready text
- Includes source attribution
- Separates chunks clearly

**Output Format:**
```
Source: sample_notes.md
[chunk content 1]

Source: sample_notes.md
[chunk content 2]
...
```

---

## Key Concepts Explained

### 1. **Vector Embeddings**

**What They Are:**
- Numerical representations of text
- Dense vectors (hundreds of dimensions)
- Capture semantic meaning

**How They Work:**
- Similar meaning → similar vectors
- Distance = semantic difference
- Cosine similarity measures relatedness

**Example:**
```
"machine learning" → [0.2, -0.1, 0.8, ...]
"artificial intelligence" → [0.3, -0.05, 0.75, ...]
(These would be close in vector space)
```

### 2. **Semantic Search**

**Traditional Search:**
- Keyword matching
- Exact text matching
- Limited understanding

**Semantic Search:**
- Meaning-based matching
- Handles synonyms and paraphrases
- Understands context

**In This System:**
- Query → embedding
- Compare with stored embeddings
- Return most similar chunks

### 3. **Retrieval-Augmented Generation**

**Without RAG:**
- LLM uses only training data
- May hallucinate
- Limited to training cutoff

**With RAG:**
- LLM uses retrieved context
- Grounded in real documents
- Can access recent information

**The Magic:**
```
User Query → Retrieve Relevant Context → Inject into Prompt → Generate Answer
```

### 4. **Prompt Engineering**

**System Prompt:**
```
"You are a concise assistant. Use the provided context to answer 
the user's question. If the answer is not in the context, say 
you do not know."
```

**Key Elements:**
- Role definition
- Instruction to use context
- Fallback behavior

**Human Prompt Template:**
```
Context:
{retrieved_chunks}

Question: {user_question}
```

**Why This Works:**
- Clear separation of context and question
- Explicit instruction to use context
- Prevents hallucination

### 5. **Temperature Control**

**Temperature = 0.2 (Low):**
- More deterministic
- Consistent answers
- Good for factual queries

**Temperature = 1.0 (High):**
- More creative
- Varied responses
- Good for creative tasks

**In RAG:**
- Low temperature preferred
- Focus on accuracy
- Reduce variability

---

## Step-by-Step Setup

### Prerequisites

1. **Python 3.10+**
   ```bash
   python --version
   ```

2. **Google Gemini API Key**
   - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Enable Gemini API access

3. **Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

### Installation

1. **Clone/Download Repository**
   ```bash
   git clone <repository-url>
   cd gemini-rag-example
   ```

2. **Install Dependencies**
   ```bash
   pip install -e .
   ```

   This installs:
   - `langchain`: Core framework
   - `langchain-community`: Community integrations
   - `langchain-google-genai`: Gemini integration
   - `langchain-text-splitters`: Text splitting utilities
   - `chromadb`: Vector database
   - `python-dotenv`: Environment variable management

3. **Configure Environment**
   ```bash
   cp local.env .env  # Or create .env manually
   ```

   Edit `.env`:
   ```env
   GEMINI_API_KEY="your-api-key-here"
   GEMINI_MODEL="gemini-2.5-flash"
   GEMINI_EMBEDDING_MODEL="models/text-embedding-004"
   GEMINI_TEMPERATURE="0.2"
   RETRIEVAL_TOP_K="4"
   RAG_DATA_DIR="./data"
   RAG_VECTOR_CACHE_DIR="./.rag_cache"
   ```

4. **Add Documents**
   - Place `.md` or `.txt` files in `data/` directory
   - Example: `data/sample_notes.md`

5. **Run the Application**
   ```bash
   # Using the CLI script
   rag-demo --query "What is this project about?"
   
   # Or directly
   python -m rag_demo.cli --query "Your question here"
   ```

---

## Code Walkthrough

### Entry Point: `cli.py`

```python
def main() -> None:
    # 1. Parse command-line arguments
    args = parse_args()
    
    # 2. Load configuration
    settings = get_settings()
    
    # 3. Load documents from data directory
    documents = load_local_documents(settings.data_dir)
    
    # 4. Get question (from args or default)
    question = args.query or settings.sample_question
    
    # 5. Run RAG pipeline
    answer = run_query(question, documents, settings)
    
    # 6. Display result
    print(answer)
```

**Flow:**
1. Configuration loading
2. Document ingestion
3. Query processing
4. Result display

### Pipeline Execution: `pipeline.py`

```python
def run_query(question: str, documents: List[Document], settings: Settings) -> str:
    # Step 1: Split documents into chunks
    chunks = split_documents(documents)
    
    # Step 2: Create vector store with embeddings
    vector_store = build_vector_store(chunks, settings)
    
    # Step 3: Build RAG chain
    chain = build_chain(vector_store, settings)
    
    # Step 4: Execute query
    return chain.invoke(question)
```

**What Happens at Each Step:**

**Step 1: Chunking**
- Input: List of full documents
- Process: Split into 600-character chunks with 100-char overlap
- Output: List of smaller document chunks

**Step 2: Embedding & Storage**
- Input: Document chunks
- Process: 
  - Generate embeddings for each chunk
  - Store in ChromaDB with metadata
  - Persist to disk
- Output: Vector store ready for retrieval

**Step 3: Chain Building**
- Input: Vector store, settings
- Process:
  - Create retriever
  - Create prompt template
  - Initialize LLM
  - Compose chain
- Output: Executable RAG chain

**Step 4: Query Execution**
- Input: User question
- Process:
  - Retrieve top K relevant chunks
  - Format chunks with sources
  - Inject into prompt
  - Generate answer
- Output: Final answer

---

## Best Practices

### 1. **Chunk Size Optimization**

**Too Small (< 200 chars):**
- Loses context
- Fragmented information
- Poor retrieval quality

**Too Large (> 1000 chars):**
- Dilutes relevance
- Includes irrelevant information
- Slower processing

**Sweet Spot:**
- 400-800 characters
- Adjust based on document type
- Consider sentence boundaries

### 2. **Overlap Strategy**

**No Overlap:**
- Risk of losing context at boundaries
- May miss relevant information

**Too Much Overlap (> 50%):**
- Redundant information
- Wastes tokens
- Slower processing

**Recommended:**
- 10-20% overlap
- 100 chars for 600-char chunks works well

### 3. **Top-K Selection**

**Too Few (k=1-2):**
- May miss relevant information
- Limited context

**Too Many (k>10):**
- Includes irrelevant chunks
- Dilutes focus
- Higher token costs

**Recommended:**
- Start with k=4-5
- Adjust based on document complexity
- Test with your specific use case

### 4. **Metadata Management**

**Always Include:**
- Source file path
- Chunk index (if needed)
- Timestamp (for versioning)

**Benefits:**
- Source attribution
- Debugging
- Version tracking

### 5. **Error Handling**

**Key Areas:**
- API key validation
- File reading errors
- Network failures
- Empty document sets

**Example:**
```python
try:
    settings = get_settings()
    documents = load_local_documents(settings.data_dir)
except Exception as exc:
    raise SystemExit(str(exc))
```

### 6. **Performance Optimization**

**Vector Store Persistence:**
- Reuse existing embeddings
- Only rebuild when documents change
- Saves API calls and time

**Batch Processing:**
- Process multiple documents together
- More efficient embedding generation

**Caching:**
- Cache embeddings
- Cache retrieval results (if appropriate)

---

## Extensions and Improvements

### 1. **Multi-Modal Support**

**Add Image Processing:**
```python
from langchain_community.document_loaders import UnstructuredImageLoader

def load_images(image_dir: Path):
    loaders = [UnstructuredImageLoader(f) for f in image_dir.glob("*.png")]
    return [loader.load() for loader in loaders]
```

### 2. **Conversation History**

**Add Memory:**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

### 3. **Hybrid Search**

**Combine Semantic + Keyword:**
```python
from langchain.retrievers import BM25Retriever

# Add BM25 retriever
bm25_retriever = BM25Retriever.from_documents(chunks)

# Combine with vector retriever
from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
```

### 4. **Re-Ranking**

**Improve Retrieval Quality:**
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 5. **Streaming Responses**

**Real-Time Output:**
```python
for chunk in chain.stream(question):
    print(chunk, end="", flush=True)
```

### 6. **Web Interface**

**Add FastAPI:**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(query: Query):
    answer = run_query(query.question, documents, settings)
    return {"answer": answer}
```

### 7. **Production Vector Database**

**Upgrade to Managed Service:**
- Pinecone
- Weaviate
- Qdrant
- Milvus

**Example with Pinecone:**
```python
from langchain_community.vectorstores import Pinecone

vector_store = Pinecone.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="rag-index"
)
```

### 8. **Evaluation Framework**

**Add Metrics:**
```python
from langchain.evaluation import QAEvalChain

eval_chain = QAEvalChain.from_llm(llm)
predictions = [{"question": q, "answer": a} for q, a in qa_pairs]
results = eval_chain.evaluate(examples, predictions)
```

---

## Common Pitfalls and Solutions

### 1. **Chunking at Wrong Boundaries**

**Problem:** Splitting mid-sentence
**Solution:** Use `RecursiveCharacterTextSplitter` with proper separators

### 2. **Insufficient Context**

**Problem:** Answer lacks necessary information
**Solution:** Increase `top_k` or chunk size

### 3. **Hallucination**

**Problem:** LLM generates information not in context
**Solution:** Strengthen system prompt, add validation

### 4. **Slow Performance**

**Problem:** Rebuilding vector store every time
**Solution:** Use persistent storage, check for changes

### 5. **API Rate Limits**

**Problem:** Too many API calls
**Solution:** Cache embeddings, batch requests, use persistent storage

---

## Conclusion

This repository demonstrates a production-ready RAG pipeline with:
- Document loading and processing
- Intelligent text chunking
- Vector embeddings and storage
- Semantic retrieval
- Context-aware generation

**Key Takeaways:**
1. RAG combines retrieval and generation
2. Chunking strategy is critical
3. Embeddings enable semantic search
4. Prompt engineering guides behavior
5. Vector databases enable fast retrieval

**Next Steps:**
- Experiment with different chunk sizes
- Try different embedding models
- Add your own documents
- Extend with conversation history
- Deploy to production

This foundation can scale to production systems with proper infrastructure and optimizations.

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://www.trychroma.com/)
- [Google Gemini API](https://ai.google.dev/gemini-api)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
