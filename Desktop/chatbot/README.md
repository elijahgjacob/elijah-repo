# LangChain FastAPI Chatbot with OpenAI & Supabase Vector DB

This project is a minimal Retrieval-Augmented Generation (RAG) chatbot backend using Python, FastAPI, [LangChain](https://python.langchain.com/), OpenAI LLMs, and Supabase as a vector database.

## Features
- FastAPI backend with a `/chat` endpoint
- Retrieval-augmented generation using LangChain
- OpenAI LLM (GPT-3.5/4) for responses
- Supabase (with pgvector) for context retrieval

## Requirements
- Python 3.9+
- Supabase project with `pgvector` extension and `embeddings` table
- OpenAI API key

## Setup

1. **Clone the repo and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   - Copy `.env.example` to `.env` and fill in your credentials:
     ```
     OPENAI_API_KEY=sk-...
     VECTOR_DB_URL=your-supabase-url
     VECTOR_DB_API_KEY=your-supabase-service-role-key
     ```

3. **Supabase Setup:**
   - Ensure you have a table `embeddings` with columns: `id`, `text`, `embedding` (vector/float8[]), `metadata` (jsonb)
   - Enable the `pgvector` extension
   - Create the `match_embeddings` RPC function:
     ```sql
     create or replace function match_embeddings(query_embedding float8[], match_count int)
     returns table(id uuid, text text, embedding float8[], metadata jsonb, similarity float8)
     language sql stable
     as $$
       select id, text, embedding, metadata, 1 - (embedding <#> query_embedding) as similarity
       from embeddings
       order by embedding <#> query_embedding
       limit match_count
     $$;
     ```

## Running the Server

```bash
uvicorn main:app --reload
```

The server will start on `http://localhost:8000`.

## Usage

Send a POST request to `/chat` with JSON body:
```json
{
  "text": "What are your wheelchair options?"
}
```
Response:
```json
{
  "reply": "...LLM-powered answer with context..."
}
```

## Testing
You can use [httpie](https://httpie.io/), [curl](https://curl.se/), or Postman:
```bash
http POST http://localhost:8000/chat text="Tell me about crutches"
```

## License
MIT
