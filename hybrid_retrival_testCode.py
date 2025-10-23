
import json
import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import re
import numpy as np

load_dotenv()

class HybridLegalRetrieval:
    """
    Hybrid retrieval system combining:
    1. AI-based title filtering (with context caching!)
    2. ChromaDB filtered RAG (only on relevant laws)
    3. Gemini answer generation
    
    This avoids sending entire files OR searching entire DB.
    Context caching reduces cost by ~90% on repeated queries!
    """
    
    def __init__(self, 
                 titles_file: str = "titles_only.json",
                 db_path: str = "./chroma_db_gemini",
                 model_name: str = "gemini-2.0-flash-001"):
        """
        Initialize hybrid retrieval system with context caching
        
        Args:
            titles_file: JSON file with law titles for initial filtering
            db_path: Path to ChromaDB database
            model_name: Gemini model to use
        """
        self.titles_file = titles_file
        self.db_path = db_path
        self.model_name = model_name
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Load titles database
        self.laws_titles = self._load_laws_data(titles_file)
        print(f"✅ Loaded {len(self.laws_titles)} law titles for filtering")
        
        # Initialize context cache for law titles
        self.cached_content = None
        self._initialize_cache()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        try:
            self.collection = self.chroma_client.get_collection(
                name="swedish_legal_documents_gemini"
            )
            print(f"✅ Connected to ChromaDB: {self.collection.count()} total documents")
        except Exception as e:
            raise Exception(f"Failed to connect to ChromaDB: {e}")
    
    def _load_laws_data(self, json_file_path: str) -> List[str]:
        """Load law titles from JSON"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load titles from {json_file_path}: {e}")
    
    def _initialize_cache(self):
        """
        Create a cached context with the law titles JSON.
        This runs once and the cache persists for ~1 hour.
        Saves ~90% on input token costs for repeated queries!
        """
        print("🔄 Initializing context cache for law titles...")
        
        laws_context = json.dumps(self.laws_titles, ensure_ascii=False, indent=2)
        
        # System instruction that will be cached
        system_instruction = f"""Du är en svensk juridisk forskningsassistent. Du har tillgång till en omfattande databas med titlar på svenska lagar. Din ENDA uppgift är att agera som ett intelligent filter som identifierar den/de mest relevanta lagtitlarna baserat på en användarfråga.

SVENSK LAGTITEL-DATABAS:
{laws_context}
Följ denna strikta process när du får en fråga:
    1. Läs noggrant användarens fråga.
    2. Välj ut de 1-10 lagtitlar från databasen ovan som är mest sannolika att innehålla svaret.
    3. Returnera ENDAST en JSON-array med exakta lagtitlar. Returnera inga förklaringar eller annan text.
"""
        
        try:
            # Create cached content
            self.cached_content = self.client.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_instruction,
                    ttl='3600s',  # Cache for 1 hour (adjust as needed: 300s-86400s)
                )
            )
            
            print(f"✅ Cache created successfully!")
            print(f"   Cache name: {self.cached_content.name}")
            print(f"   Token count: ~{len(laws_context.split())} tokens cached")
            
        except Exception as e:
            print(f"⚠️  Failed to create cache: {e}")
            print(f"   Will fall back to non-cached mode")
            self.cached_content = None
    
    def find_relevant_law_titles(self, user_query: str) -> List[str]:
        """
        Step 1: Use Gemini with CACHED law titles to identify relevant laws
        
        Cost savings with caching:
        - First request: Pays to write cache + process query
        - Subsequent requests: ~90% discount on reading cached 120K tokens
        """
        print(f"\n🔍 Step 1: Finding relevant law titles for: '{user_query}'")
        
        # The user prompt (this is NOT cached, only the system instruction is)
        user_prompt = f"""USER QUERY: {user_query}

INSTRUCTIONS:
1. Identify the 1-10 most relevant laws that likely contain the answer
2. Return ONLY a JSON array of exact titles from the database
3. Be precise - use exact titles as they appear

Response format:
[
    "exact title 1",
    "exact title 2",
    ...
]

Return [] if no relevant laws found.
CRITICAL: Return ONLY the JSON array, no explanations."""

        try:
            # Use the cached content if available
            if self.cached_content:
                
                print("💰 Using cached law titles (90% cost reduction!)")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        cached_content=self.cached_content.name,
                        temperature=0.1,
                        max_output_tokens=2048,
                    )
                )
            else:
                # Fallback: use regular model without cache
                print("⚠️  Cache not available, using full context (higher cost)")
                laws_context = json.dumps(self.laws_titles, ensure_ascii=False, indent=2)
                full_prompt = f"""You are a Swedish legal research assistant. You have access to a database of Swedish laws.

SWEDISH LAWS DATABASE:
{laws_context}

{user_prompt}"""
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                    )
                )
            
            assistant_message = response.text.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(assistant_message)
                print(f"✅ Step 1 complete: Found {len(result)} relevant law titles")
                return result
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from text
                json_match = re.search(r'\[.*?\]', assistant_message, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        print(f"✅ Step 1 complete: Found {len(result)} relevant law titles (via fallback)")
                        print(f"   Found titles: {result}")
                        return result
                    except json.JSONDecodeError:
                        print(f"❌ Failed to parse JSON from fallback: {json_match.group()}")
                        return []
            
            
        except Exception as e:
            print(f"❌ Error in Step 1: {e}")
            # If cache expired, try to refresh
            if "cache" in str(e).lower() or "expired" in str(e).lower():
                print("🔄 Cache may have expired, refreshing...")
                self._initialize_cache()
            return []
    
    def refresh_cache_if_needed(self):
        """
        Check if cache has expired and refresh it.
        Call this if you get cache-related errors or periodically.
        """
        if not self.cached_content:
            print("🔄 No cache found, creating new cache...")
            self._initialize_cache()
            return
        
        try:
            # Try to access cache - if it fails, cache expired
            _ = self.cached_content.name
            print("✅ Cache still valid")
        except:
            print("🔄 Cache expired or invalid, refreshing...")
            self._initialize_cache()
    
    def filter_chromadb_by_titles(self, titles: List[str]) -> List[str]:
        """
        Step 2: Get all ChromaDB IDs that match the given titles
        This filters the database to only relevant laws
        """
        print(f"\n🔍 Step 2: Filtering ChromaDB by {len(titles)} titles...")
        
        if not titles:
            return []
        
        try:
            # Use ChromaDB's efficient metadata filtering with $in operator
            # This is much faster than retrieving all documents
            matching_ids = []
            
            # ChromaDB supports filtering by metadata using where clause
            where_clause = {"title": {"$in": titles}}
            
            # Get only the IDs of matching documents
            filtered_data = self.collection.get(
                where=where_clause,
                include=[]  # Only get IDs, no content or metadata needed
            )
            
            matching_ids = filtered_data['ids']
            
            print(f"✅ Step 2 complete: Found {len(matching_ids)} matching chunks in ChromaDB")
            print(f"   (From {len(titles)} laws)")
            
            return matching_ids
            
        except Exception as e:
            print(f"❌ Error filtering ChromaDB: {e}")
            return []
    
    def get_embedding(self, text: str) -> List[float]:
        """Get query embedding from Gemini"""
        try:
            result = self.client.models.embed_content(
                model="models/text-embedding-004",
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="retrieval_query"
                )
            )
            return result.embeddings[0].values
        except Exception as e:
            raise Exception(f"Failed to get embedding: {e}")
    
    def _write_debug_chunks(self, results, query):
        """Write debug information about retrieved chunks to a log file."""
        from datetime import datetime
        
        # Create debug_logs directory if it doesn't exist
        debug_dir = "debug_logs"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = os.path.join(debug_dir, f"chunks_debug_{timestamp}.txt")
        
        try:
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"Debug Log - Retrieved Chunks\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Query: {query}\n")
                f.write(f"Number of chunks retrieved: {len(results['documents'][0])}\n")
                f.write("=" * 80 + "\n\n")
                
                # Write each chunk's content and metadata
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    f.write(f"CHUNK {i+1}\n")
                    f.write(f"Distance/Similarity Score: {distance:.4f}\n")
                    f.write(f"Title: {metadata.get('title', 'N/A')}\n")
                    f.write(f"URL: {metadata.get('url', 'N/A')}\n")
                    f.write(f"Content Length: {len(doc)} characters\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Content:\n{doc}\n")
                    f.write("-" * 40 + "\n\n")
            
            print(f"📝 Debug log written to: {debug_file}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not write debug log: {e}")
    
    def search_filtered_chunks(self, query: str, filtered_ids: List[str], top_k: int = 10) -> Dict:
        """
        Step 3: Perform semantic search ONLY on filtered chunks
        This is the key optimization - we only search relevant laws!
        """
        print(f"\n🔍 Step 3: Performing semantic search on {len(filtered_ids)} filtered chunks...")
        
        if not filtered_ids:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Get the filtered documents with their embeddings
            # We need to retrieve in batches if there are too many IDs
            batch_size = 1000  # ChromaDB limit
            all_documents = []
            all_metadatas = []
            all_embeddings = []
            all_ids = []
            
            for i in range(0, len(filtered_ids), batch_size):
                batch_ids = filtered_ids[i:i + batch_size]
                batch_data = self.collection.get(
                    ids=batch_ids,
                    include=["documents", "metadatas", "embeddings"]
                )
                all_documents.extend(batch_data['documents'])
                all_metadatas.extend(batch_data['metadatas'])
                all_embeddings.extend(batch_data['embeddings'])
                all_ids.extend(batch_data['ids'])
            
            # Compute similarities manually
            query_embedding = np.array(query_embedding)
            similarities = []
            
            for embedding in all_embeddings:
                doc_embedding = np.array(embedding)
                # Compute cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append(similarity)
            
            # Sort by similarity (highest first) and get top_k
            sorted_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Format results
            filtered_results = {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
            
            for idx in sorted_indices:
                filtered_results["documents"][0].append(all_documents[idx])
                filtered_results["metadatas"][0].append(all_metadatas[idx])
                # Convert similarity to distance (ChromaDB uses distance, lower is better)
                distance = 1.0 - similarities[idx]
                filtered_results["distances"][0].append(distance)
            
            print(f"✅ Step 3 complete: Retrieved top {len(filtered_results['documents'][0])} most relevant chunks")
            # Debug logging: Write chunks content to file
            self._write_debug_chunks(filtered_results, query)
            
            return filtered_results
            
        except Exception as e:
            print(f"❌ Error during filtered search: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
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
            
            context_part = f"""
{'='*80}
DOKUMENT {i+1}:
Titel: {metadata['title']}{chunk_info}
SFS-nummer: {metadata.get('sfs_number', 'N/A')}
Myndighet: {metadata.get('ministry_authority', 'N/A')}

Innehåll:
{doc}

Källa: {metadata.get('source_link', 'N/A')}
{'='*80}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Step 4: Generate answer using Gemini with filtered context"""
        print(f"\n🤖 Step 4: Generating answer with filtered context...")
        
        prompt = f"""Du är en expert på svensk lagstiftning och juridiska dokument. Du har tillgång till texten från relevanta svenska lagar. Din uppgift är att besvara frågor baserat på de tillhandahållna svenska lagdokumenten, utifrån din förståelse av texten.

INSTRUKTIONER:
1. Läs och analysera noggrant den fullständiga lagtexten som tillhandahålls och se till att du förstår den fullt ut.
2. Besvara användarens fråga utifrån informationen i dessa lagar och din förståelse av lagarna.
3. Citera den specifika lagen och paragrafen du refererar till för att stödja ditt svar.
4. Om svaret inte kan hittas i de tillhandahållna lagarna eller din förståelse av dem, ange tydligt: "Jag kan inte besvara den frågan."
5. Ge ett klart, omfattande och korrekt svar.
6. Inkludera relevanta citat eller specifika avsnitt från lagtexten när det är hjälpsamt.
7. Svara på svenska.

FRÅGA: {query}

TILLHANDAHÅLLNA DOKUMENT:
{context}

SVAR:"""
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=8192,
                )
            )
            print("✅ Step 4 complete: Answer generated")
            return response.text
            
        except Exception as e:
            raise Exception(f"Failed to generate answer: {e}")
    
    def process_query(self, query: str, top_k: int = 50) -> Dict:
        """
        Complete hybrid retrieval pipeline with context caching:
        1. Find relevant law titles using AI (with cached law database!)
        2. Filter ChromaDB to only those laws
        3. Perform semantic search on filtered subset
        4. Generate answer from top results
        
        Args:
            query: User's question
            top_k: Number of top chunks to retrieve from filtered set
            
        Returns:
            Dict with answer, sources, and metadata
        """
        print(f"\n🚀 Starting Hybrid Retrieval Pipeline (with caching)")
        print(f"📝 Query: '{query}'")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Find relevant law titles (using cached context!)
            relevant_titles = self.find_relevant_law_titles(query)
            
            if not relevant_titles:
                return {
                    "answer": "Inga relevanta lagar hittades för din fråga.",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "method": "hybrid_filtered_rag",
                    "stats": {
                        "titles_found": 0,
                        "chunks_filtered": 0,
                        "chunks_retrieved": 0
                    }
                }
            
            # Step 2: Filter ChromaDB by titles
            filtered_ids = self.filter_chromadb_by_titles(relevant_titles)
            
            if not filtered_ids:
                return {
                    "answer": "Lagar identifierades men inga matchande dokument hittades i databasen.",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "method": "hybrid_filtered_rag",
                    "stats": {
                        "titles_found": len(relevant_titles),
                        "chunks_filtered": 0,
                        "chunks_retrieved": 0
                    }
                }
            
            # Step 3: Semantic search on filtered chunks
            search_results = self.search_filtered_chunks(query, filtered_ids, top_k)
            
            if not search_results['documents'][0]:
                return {
                    "answer": "Inga relevanta avsnitt hittades i de identifierade lagarna.",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "method": "hybrid_filtered_rag",
                    "stats": {
                        "titles_found": len(relevant_titles),
                        "chunks_filtered": len(filtered_ids),
                        "chunks_retrieved": 0
                    }
                }
            
            # Step 4: Format context and generate answer
            context = self.format_context(search_results)
            answer = self.generate_answer(query, context)
            
            # Format sources
            sources = []
            for metadata, distance in zip(
                search_results['metadatas'][0],
                search_results['distances'][0]
            ):
                chunk_info = None
                if metadata.get('is_chunked'):
                    chunk_info = f"Del {metadata['chunk_position']}"
                
                sources.append({
                    "title": metadata['title'],
                    "sfs_number": metadata.get('sfs_number', 'N/A'),
                    "url": metadata.get('url', 'N/A'),
                    "source_link": metadata.get('source_link', 'N/A'),
                    "similarity_score": round(1 - distance, 3),
                    "chunk_info": chunk_info
                })
            
            processing_time = time.time() - start_time
            
            print(f"\n🎉 Hybrid Retrieval Complete!")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"📊 Stats:")
            print(f"   - Relevant law titles found: {len(relevant_titles)}")
            print(f"   - Chunks filtered from DB: {len(filtered_ids)}")
            print(f"   - Top chunks retrieved: {len(sources)}")
            print("=" * 80)
            
            return {
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time,
                "method": "hybrid_filtered_rag",
                "stats": {
                    "titles_found": len(relevant_titles),
                    "chunks_filtered": len(filtered_ids),
                    "chunks_retrieved": len(sources),
                    "relevant_laws": relevant_titles
                }
            }
            
        except Exception as e:
            error_msg = f"Ett fel uppstod i hybrid retrieval: {e}"
            print(f"❌ {error_msg}")
            return {
                "answer": error_msg,
                "sources": [],
                "processing_time": time.time() - start_time,
                "method": "hybrid_filtered_rag"
            }
    
    def __del__(self):
        """Clean up cache when object is destroyed"""
        if hasattr(self, 'cached_content') and self.cached_content:
            try:
                self.cached_content.delete()
                print("🗑️  Cache deleted on cleanup")
            except:
                pass

def test_interactive():
    """Interactive test mode - allows user to input their own prompts"""
    print("Swedish Laws Hybrid Retrieval System - Interactive Mode (with Caching!)")
    print("=" * 80)
    print("This system combines:")
    print("1. AI-based title filtering (with 90% cost reduction via caching!)")
    print("2. ChromaDB filtered RAG")
    print("3. Gemini answer generation")
    print("=" * 80)
    
    # Initialize system
    try:
        retrieval_system = HybridLegalRetrieval()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    print("\nType your legal questions in Swedish. Type 'quit' or 'exit' to stop.")
    print("-" * 80)
    
    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                print("Please enter a valid question.")
                continue
            
            print(f"\n{'='*100}")
            print(f"PROCESSING: {query}")
            print('='*100)
            
            # Process the query
            result = retrieval_system.process_query(query, top_k=3)
            
            print(f"\n💬 ANSWER:")
            print("-" * 80)
            print(result['answer'])
            
            print(f"\n🔗 SOURCES ({len(result['sources'])}):")
            print("-" * 80)
            for j, source in enumerate(result['sources'], 1):
                chunk_info = f" [{source['chunk_info']}]" if source['chunk_info'] else ""
                print(f"{j}. {source['title']}{chunk_info}")
                print(f"   Similarity: {source['similarity_score']:.3f}")
                print(f"   URL: {source['url']}")
            
            print(f"\n📊 PROCESS DETAILS:")
            print("-" * 80)
            if 'stats' in result:
                print(f"Relevant law titles found: {result['stats']['titles_found']}")
                print(f"Chunks filtered: {result['stats']['chunks_filtered']}")
                print(f"Chunks retrieved: {result['stats']['chunks_retrieved']}")
            print(f"Processing time: {result['processing_time']:.2f}s")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")

def main():
    """Test the hybrid retrieval system with caching"""
    print("Swedish Laws Hybrid Retrieval System Test (with Context Caching)")
    print("=" * 80)
    # Initialize system
    try:
        retrieval_system = HybridLegalRetrieval()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test queries
    test_queries = [
        "Vad säger lagen om skatt på naturgrus?",
        "Vad säger lagen om diskriminering av deltidsanställda?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*100}")
        print(f"TEST {i}: {query}")
        print('='*100)
        
        result = retrieval_system.process_query(query, top_k=10)
        
        print(f"\n💬 ANSWER:")
        print("-" * 80)
        print(result['answer'])
        
        print(f"\n🔗 SOURCES ({len(result['sources'])}):")
        print("-" * 80)
        for j, source in enumerate(result['sources'], 1):
            chunk_info = f" [{source['chunk_info']}]" if source['chunk_info'] else ""
            print(f"{j}. {source['title']}{chunk_info}")
            print(f"   Similarity: {source['similarity_score']:.3f}")
            print(f"   URL: {source['url']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        test_interactive()
    else:
        main()