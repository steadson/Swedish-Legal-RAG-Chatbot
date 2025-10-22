import json
import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from openai import OpenAI
from two_step_retrieval import TwoStepLegalRetrieval

from dotenv import load_dotenv
import uvicorn
from datetime import datetime

# Load environment variables
load_dotenv()
openai_client = OpenAI()
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
    english_mode: Optional[bool] = False
    deep_search: Optional[bool] = False

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
    original_query: Optional[str] = None
    translated_query: Optional[str] = None
    method: Optional[str] = "regular_rag"
    processing_time: Optional[float] = None

class SwedishLegalRAG:

    def __init__(self, db_path: str = "./chroma_db_gemini2", debug_dir: str = "./debug_logs"):
    # def __init__(self, db_path: str = "./chroma_db_gemini", debug_dir: str = "./debug_logs"):
        self.db_path = db_path
        self.debug_dir = debug_dir
        
        # Create debug directory if it doesn't exist
        os.makedirs(self.debug_dir, exist_ok=True)
        
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
                # name="swedish_legal_documents_gemini"
                name="swedish_legal_documents_gemini2"
            )
            print(f"✅ Connected to ChromaDB collection with {self.collection.count()} documents")
        except Exception as e:
            raise Exception(f"Failed to connect to ChromaDB collection: {e}")
    
        # Initialize two-step retrieval system
        try:
            self.two_step_retrieval = TwoStepLegalRetrieval()
            print("✅ Two-step retrieval system initialized")
        except Exception as e:
            print(f"⚠️  Warning: Two-step retrieval system failed to initialize: {e}")
            self.two_step_retrieval = None

    def save_context_to_file(self, query: str, context: str, search_results: Dict, english_mode: bool = False):
        """Save raw retrieved context and metadata to a debug file (no formatting)."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"context_{timestamp}.txt"
            filepath = os.path.join(self.debug_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Query: {query}\n")
                f.write(f"English Mode: {english_mode}\n\n")

                f.write("SEARCH RESULTS RAW:\n")
                f.write("-" * 100 + "\n")
                f.write(str(search_results))  # raw dictionary (no formatting)
                f.write("\n\n")

                f.write("RAW CONTEXT (exactly passed to Gemini):\n")
                f.write("=" * 100 + "\n")
                f.write(context)  # raw context text
                f.write("\n" + "=" * 100 + "\n")

            print(f"📝 Context saved to debug file: {filepath}")
            return filepath

        except Exception as e:
            print(f"⚠️ Warning: Failed to save context to file: {e}")
            return None
    
    def translate_to_swedish(self, english_query: str) -> str:
        """Translate English query to Swedish using Gemini"""
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""Translate the following question from English to Swedish. 
Keep the meaning exact and maintain any legal terminology appropriately.
Only return the Swedish translation, nothing else.

English question: {english_query}

Swedish translation:"""
            
            response = model.generate_content(prompt)
            swedish_query = response.text.strip()
            print(f"🔄 Translated: '{english_query}' → '{swedish_query}'")
            return swedish_query
        except Exception as e:
            raise Exception(f"Failed to translate query to Swedish: {e}")
    
    def translate_to_english(self, swedish_answer: str) -> str:
        """Translate Swedish answer to English using Gemini"""
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = f"""Translate the following answer from Swedish to English.
Maintain all legal terminology, citations, and SFS numbers exactly as they are.
Keep the same structure and formatting.
Only return the English translation, nothing else.

Swedish answer: {swedish_answer}

English translation:"""
            
            response = model.generate_content(prompt)
            english_answer = response.text.strip()
            print(f"🔄 Translated answer to English")
            return english_answer
        except Exception as e:
            raise Exception(f"Failed to translate answer to English: {e}")
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for user query using Gemini"""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            print('result ===>',result)
            return result['embedding']

        except Exception as e:
            raise Exception(f"Failed to get query embedding: {e}")
    
    def search_relevant_documents(self, query: str, max_results: int = 15) -> Dict:
        """Search for relevant documents in ChromaDB"""
        # Get query embedding
        
        query_embedding = self.get_query_embedding(query)
       
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            include=["documents", "metadatas", "distances"]
        )
        print('raw search results been used===>', len(results['documents'][0]))
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
            
            # Get the full metadata from the document
            full_metadata = metadata.get('scraped_metadata', {})
            issued_date = full_metadata.get('Utfärdad', 'N/A')
            
            context_part = f"""
    {'='*80}
    DOKUMENT {i+1}:
    Titel: {metadata['title']}{chunk_info}
    SFS-nummer: {metadata['sfs_number']}
    Myndighet: {metadata.get('ministry_authority', 'N/A')}
    Utfärdad: {issued_date}

    Innehåll:
    {doc}

    Källa: {metadata['source_link']}
    Ändringsregister: {metadata['amendment_register_link']}
    {'='*80}
    """
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
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
            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(prompt)
            return response.text

            # response = openai_client.chat.completions.create(
            # model="gpt-4.1-mini",  # ✅ 1M token model
            # messages=[
            #     {"role": "system", "content": "Du är en juridisk expert som analyserar svenska lagdokument."},
            #     {"role": "user", "content": prompt}
            # ],
            # temperature=0.3,
            # max_completion_tokens=2000  # you can increase if needed
            # )
            # return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"Failed to generate answer: {e}")
    
    def process_query(self, query: str, max_results: int = 3, english_mode: bool = False, deep_search: bool = False) -> Dict:
        """Complete RAG pipeline: search + generate"""
        original_query = None
        translated_query = None
        search_query = query
        # Check if deep search is enabled and two-step retrieval is available
        if deep_search and self.two_step_retrieval:
            print(f"🔍 Using deep search (two-step retrieval) for query: {query}")
            
            # Use the two-step retrieval system
            result = self.two_step_retrieval.process_query(search_query, max_laws=max_results)
            
            # Convert two-step sources to our format
            sources = []
            for source in result.get('sources', []):
                sources.append({
                    "title": source['title'],
                    "sfs_number": "N/A",  # Two-step doesn't have SFS numbers
                    "url": source['url'],
                    "source_link": source['url'],
                    "amendment_register_link": "N/A",
                    "similarity_score": 1.0,  # Two-step uses different scoring
                    "chunk_info": f"File: {source.get('filename', 'N/A')}"
                })
            
            # Translate answer if English mode is enabled
            answer = result['answer']
            if english_mode and answer:
                answer = self.translate_to_english(answer)
            
            return {
                "answer": answer,
                "sources": sources,
                "search_results": {"method": "two_step_retrieval"},
                "original_query": original_query,
                "translated_query": translated_query,
                "method": "two_step_retrieval",
                "processing_time": result.get('processing_time', 0)
            }
        
        # Regular RAG pipeline
        print(f"🔍 Using regular RAG for query: {query}")
        # Step 1: Translate to Swedish if English mode is enabled
        if english_mode:
            original_query = query
            search_query = self.translate_to_swedish(query)
            translated_query = search_query
        
        # Step 2: Search for relevant documents (always in Swedish)
        search_results = self.search_relevant_documents(search_query, max_results)
        
        if not search_results['documents'][0]:
            no_results_msg = "Inga relevanta dokument hittades för din fråga."
            if english_mode:
                no_results_msg = "No relevant documents were found for your query."
            
            return {
                "answer": no_results_msg,
                "sources": [],
                "search_results": search_results,
                "original_query": original_query,
                "translated_query": translated_query
            }
        
        # Step 3: Format context
        context = self.format_context(search_results)
        
        # DEBUG: Save context to file for debugging
        self.save_context_to_file(search_query, context, search_results, english_mode)
        
        # Step 4: Generate answer (in Swedish)
        answer = self.generate_answer(search_query, context)
        
        # Step 5: Translate answer to English if needed
        if english_mode:
            answer = self.translate_to_english(answer)
        
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
            "search_results": search_results,
            "original_query": original_query,
            "translated_query": translated_query,
            "method": "regular_rag",
            "processing_time": None
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
        mode = "English" if request.english_mode else "Swedish"
        search_mode = "Deep Search" if request.deep_search else "Regular RAG"
        print(f"\n🔍 Processing query in {mode} mode with {search_mode}: '{request.query}'")
        # Process the query through RAG pipeline
        result = rag_system.process_query(
            query=request.query,
            max_results=request.max_results,
            english_mode=request.english_mode,
            
            deep_search=request.deep_search
        )
        
        # Format response
        sources = []
        if request.include_sources:
            sources = [
                SourceDocument(**source) for source in result['sources']
            ]
        # Determine model used based on method
        model_used = "gemini-2.0-flash + gemini-embedding-001"
        if result.get('method') == 'two_step_retrieval':
            model_used = "gpt-4o + gpt-4o-mini (Two-Step Retrieval)"
        response = RAGResponse(
            answer=result['answer'],
            query=request.query,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            model_used=model_used,
            original_query=result.get('original_query'),
            translated_query=result.get('translated_query'),
            method=result.get('method', 'regular_rag'),
            processing_time=result.get('processing_time')
        )
        
        print(f"✅ Query processed successfully using {result.get('method', 'regular_rag')}, {len(sources)} sources found")
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
            "collection_name": "swedish_legal_documents_gemini2",
            "embedding_model": "gemini-embedding-001",
            "chat_model": "gemini-2.0-flash",
            "debug_logs_directory": rag_system.debug_dir
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
    print(f"📝 Debug logs will be saved to: {rag_system.debug_dir}")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )