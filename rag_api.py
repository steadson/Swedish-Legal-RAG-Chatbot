import json
import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from dotenv import load_dotenv
import uvicorn
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Swedish Legal RAG API",
    description="Retrieval-Augmented Generation API for Swedish Legal Documents using Gemini",
    version="1.0.0"
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 3
    include_sources: Optional[bool] = True

class SourceDocument(BaseModel):
    title: str
    sfs_number: str
    url: str
    source_link: str
    amendment_register_link: str
    similarity_score: float
    chunk_info: Optional[str] = None

class RAGResponse(BaseModel):
    answer: str
    query: str
    sources: List[SourceDocument]
    timestamp: str
    model_used: str

class SwedishLegalRAG:
    def __init__(self, db_path: str = "./chroma_db_gemini"):
        self.db_path = db_path
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get existing collection
        try:
            self.collection = self.chroma_client.get_collection(
                name="swedish_legal_documents_gemini"
            )
            print(f"✅ Connected to ChromaDB collection with {self.collection.count()} documents")
        except Exception as e:
            raise Exception(f"Failed to connect to ChromaDB collection: {e}")
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for user query using Gemini"""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            raise Exception(f"Failed to get query embedding: {e}")
    
    def search_relevant_documents(self, query: str, max_results: int = 3) -> Dict:
        """Search for relevant documents in ChromaDB"""
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def format_context(self, search_results: Dict) -> str:
        """Format search results into context for Gemini"""
        context_parts = []
        
        for i, (doc, metadata) in enumerate(zip(
            search_results['documents'][0],
            search_results['metadatas'][0]
        )):
            chunk_info = ""
            if metadata.get('is_chunked'):
                chunk_info = f" (Del {metadata['chunk_position']})"
            
            context_part = f"""DOKUMENT {i+1}:
Titel: {metadata['title']}{chunk_info}
SFS-nummer: {metadata['sfs_number']}
Myndighet: {metadata.get('ministry_authority', 'N/A')}
Utfärdad: {metadata.get('issued_date', 'N/A')}

Innehåll:
{doc}

Källa: {metadata['source_link']}
Ändringsregister: {metadata['amendment_register_link']}
"""
            context_parts.append(context_part)
        
        return "\n" + "="*80 + "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Gemini with retrieved context"""
        prompt = f"""Du är en expert på svensk lagstiftning och juridiska dokument. Din uppgift är att svara på frågor baserat på de svenska lagdokument som tillhandahålls.

INSTRUKTIONER:
1. Svara på svenska
2. Basera ditt svar ENDAST på informationen i de tillhandahållna dokumenten
3. Citera specifika SFS-nummer när det är relevant
4. Om informationen inte finns i dokumenten, säg det tydligt
5. Var precis och faktabaserad
6. Inkludera relevanta detaljer från dokumenten

FRÅGA: {query}

TILLHANDAHÅLLNA DOKUMENT:
{context}

SVAR:"""
        
        try:
            # Use Gemini's chat model
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Failed to generate answer: {e}")
    
    def process_query(self, query: str, max_results: int = 3) -> Dict:
        """Complete RAG pipeline: search + generate"""
        # Search for relevant documents
        search_results = self.search_relevant_documents(query, max_results)
        
        if not search_results['documents'][0]:
            return {
                "answer": "Inga relevanta dokument hittades för din fråga.",
                "sources": [],
                "search_results": search_results
            }
        
        # Format context
        context = self.format_context(search_results)
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        # Format sources
        sources = []
        for i, (metadata, distance) in enumerate(zip(
            search_results['metadatas'][0],
            search_results['distances'][0]
        )):
            chunk_info = None
            if metadata.get('is_chunked'):
                chunk_info = f"Del {metadata['chunk_position']}"
            
            source = {
                "title": metadata['title'],
                "sfs_number": metadata['sfs_number'],
                "url": metadata['url'],
                "source_link": metadata['source_link'],
                "amendment_register_link": metadata['amendment_register_link'],
                "similarity_score": round(1 - distance, 3),
                "chunk_info": chunk_info
            }
            sources.append(source)
        
        return {
            "answer": answer,
            "sources": sources,
            "search_results": search_results
        }

# Initialize RAG system
rag_system = SwedishLegalRAG()

@app.get("/")
async def root():
    return {
        "message": "Swedish Legal RAG API with Gemini",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Submit a query to get legal information",
            "/health": "GET - Check API health",
            "/stats": "GET - Get database statistics"
        }
    }

@app.post("/query", response_model=RAGResponse)
async def query_documents(request: QueryRequest):
    """Main RAG endpoint - query Swedish legal documents"""
    try:
        print(f"\n🔍 Processing query: '{request.query}'")
        
        # Process the query through RAG pipeline
        result = rag_system.process_query(
            query=request.query,
            max_results=request.max_results
        )
        
        # Format response
        sources = []
        if request.include_sources:
            sources = [
                SourceDocument(**source) for source in result['sources']
            ]
        
        response = RAGResponse(
            answer=result['answer'],
            query=request.query,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            model_used="gemini-pro + gemini-embedding-001"
        )
        
        print(f"✅ Query processed successfully, {len(sources)} sources found")
        return response
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check ChromaDB connection
        count = rag_system.collection.count()
        
        # Check Gemini API
        test_embedding = rag_system.get_query_embedding("test")
        
        return {
            "status": "healthy",
            "database_documents": count,
            "gemini_api": "connected",
            "embedding_dimensions": len(test_embedding)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.get("/stats")
async def get_stats():
    """Get database and API statistics"""
    try:
        count = rag_system.collection.count()
        
        # Get sample metadata to understand collection structure
        sample = rag_system.collection.peek(limit=1)
        
        stats = {
            "total_documents": count,
            "database_path": rag_system.db_path,
            "collection_name": "swedish_legal_documents_gemini",
            "embedding_model": "gemini-embedding-001",
            "chat_model": "gemini-pro"
        }
        
        if sample['metadatas']:
            sample_meta = sample['metadatas'][0]
            stats["sample_document"] = {
                "title": sample_meta.get('title', 'N/A'),
                "sfs_number": sample_meta.get('sfs_number', 'N/A'),
                "is_chunked": sample_meta.get('is_chunked', False),
                "embedding_model": sample_meta.get('embedding_model', 'N/A')
            }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Swedish Legal RAG API with Gemini...")
    print(f"📊 Database: {rag_system.collection.count()} documents loaded")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )