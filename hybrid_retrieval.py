import json
import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from dotenv import load_dotenv
import time
import re
import numpy as np

load_dotenv()

class HybridLegalRetrieval:
    """
    Hybrid retrieval system combining:
    1. Regex literal search for law designations
    2. AI-based title filtering (like two-step)
    3. ChromaDB filtered RAG (only on relevant laws)
    4. Gemini answer generation
    
    This avoids sending entire files OR searching entire DB.
    """
    
    def __init__(self, 
                 titles_file: str = "titles_only.json",
                 db_path: str = "./chroma_db_gemini",
                 model_name: str = "gemini-2.0-flash"):
        """
        Initialize hybrid retrieval system
        
        Args:
            titles_file: JSON file with law titles for initial filtering
            db_path: Path to ChromaDB database
            model_name: Gemini model to use (default: gemini-2.0-flash)
        """
        self.titles_file = titles_file
        self.db_path = db_path
        self.model_name = model_name
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Load titles database
        self.laws_titles = self._load_laws_data(titles_file)
        print(f"✅ Loaded {len(self.laws_titles)} law titles for filtering")
        
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
    
    def _extract_law_by_designation(self, user_query: str) -> List[str]:
        """
        Extract law titles from the database by searching for law designations
        in the user query using regex patterns like (yyyy:nnn) or (yyy:nn)
        
        Args:
            user_query: User's question
            
        Returns:
            List of exact law titles that match the designation found in the query
        """
        # Pattern to match Swedish law designations: (yyyy:nnn) or (yyy:nn)
        # Examples: (2022:123), (1999:45), (999:12)
        pattern = r'\((\d{3,4}:\d{1,3})\)'
        
        matches = re.findall(pattern, user_query)
        
        if not matches:
            print(f"🔍 No law designations found in query")
            return []
        
        print(f"🔍 Found {len(matches)} law designation(s) in query: {matches}")
        
        matched_titles = []
        
        # Search for each designation in the titles database
        for designation in matches:
            designation_pattern = f"({designation})"
            
            for title in self.laws_titles:
                if designation_pattern in title:
                    matched_titles.append(title)
                    print(f"   ✅ Regex match: {title}")
        
        return matched_titles
    
    def find_relevant_law_titles(self, user_query: str, model_name: str = None) -> List[str]:
        """
        Step 1: Use Gemini to identify relevant law titles from database
        
        Args:
            user_query: User's question in Swedish
            model_name: Gemini model to use (if None, uses self.model_name)
            
        Returns:
            List of relevant law titles
        """
        # Use provided model_name or fall back to instance default
        model_to_use = model_name if model_name else self.model_name
        
        print(f"\n🔍 Step 1: Finding relevant law titles with {model_to_use} for: '{user_query}'")
        
        laws_context = json.dumps(self.laws_titles, ensure_ascii=False, indent=2)
        
        prompt = f"""You are a Swedish legal research assistant. You have access to a database of Swedish laws.

SWEDISH LAWS DATABASE:
{laws_context}

USER QUERY: {user_query}

INSTRUCTIONS:
1. Read and parse the JSON database above
2. Search through all the laws based on your understanding of Swedish law
3. Find the most relevant laws (1-10) that likely contain the answer to the query 
4. Return ONLY a JSON array of exact titles from the database

Response format should be like:
[
    "exact title 1",
    "exact title 2"
]

Return [] if no relevant laws found.
CRITICAL: Return ONLY the JSON array with exact titles as they appear in the database, no explanations or additional text."""

        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                )
            )
            
            # ✅ Safe extraction of all text parts
            assistant_message = ""
            try:
                # Preferred: use response.parts if available
                if hasattr(response, "parts"):
                    assistant_message = "\n".join(
                        [p.text for p in response.parts if hasattr(p, "text")]
                    ).strip()
                # Otherwise, fall back to candidates
                elif hasattr(response, "candidates") and response.candidates:
                    parts = []
                    for candidate in response.candidates:
                        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                            for part in candidate.content.parts:
                                if hasattr(part, "text"):
                                    parts.append(part.text)
                    assistant_message = "\n".join(parts).strip()
                # Fallback if it's still empty
                if not assistant_message and hasattr(response, "text"):
                    assistant_message = response.text.strip()
            except Exception as parse_err:
                print(f"⚠️ Parsing warning: {parse_err}")
                print("🧾 Dumping raw Gemini response structure:")
                print(response)

            if not assistant_message:
                raise ValueError("Gemini response contained no text parts.")

            # Parse JSON response
            try:
                result = json.loads(assistant_message)
                print(f"✅ Step 1 complete: Found {len(result)} relevant law titles from AI")
                return result
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from text
                json_match = re.search(r'\[.*?\]', assistant_message, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        print(f"✅ Step 1 complete: Found {len(result)} relevant law titles from AI (via fallback)")
                        return result
                    except json.JSONDecodeError:
                        print(f"❌ Failed to parse JSON from fallback: {json_match.group()}")
                        return []
            
        except Exception as e:
            print(f"❌ Error in Step 1 with {model_to_use}: {e}")
            return []
    
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
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            raise Exception(f"Failed to get embedding: {e}")
    
    def _write_debug_chunks(self, results, query):
        """Write debug information about retrieved chunks to a log file."""
        import os
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
    
    def search_filtered_chunks(self, query: str, filtered_ids: List[str], top_k: int = 12) -> Dict:
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
    
    def generate_answer(self, query: str, context: str, model_name: str = None) -> str:
        """
        Step 4: Generate answer using Gemini with filtered context
        
        Args:
            query: User's question
            context: Formatted context from search results
            model_name: Gemini model to use (if None, uses self.model_name)
            
        Returns:
            Generated answer
        """
        # Use provided model_name or fall back to instance default
        model_to_use = model_name if model_name else self.model_name
        
        print(f"\n🤖 Step 4: Generating answer with {model_to_use} using filtered context...")
        
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
            model = genai.GenerativeModel(model_to_use)
            response = model.generate_content(prompt)
            print(f"✅ Step 4 complete: Answer generated with {model_to_use}")
            return response.text
            
        except Exception as e:
            raise Exception(f"Failed to generate answer with {model_to_use}: {e}")
    
    def process_query(self, query: str, top_k: int = 50, model_name: str = None) -> Dict:
        """
        Complete hybrid retrieval pipeline:
        0. Extract laws by designation using regex (literal search)
        1. Find relevant law titles using AI
        2. Merge results and remove duplicates
        3. Filter ChromaDB to only those laws
        4. Perform semantic search on filtered subset
        5. Generate answer from top results
        
        Args:
            query: User's question
            top_k: Number of top chunks to retrieve from filtered set
            model_name: Gemini model to use (if None, uses self.model_name)
            
        Returns:
            Dict with answer, sources, and metadata
        """
        # Use provided model_name or fall back to instance default
        model_to_use = model_name if model_name else self.model_name
        
        print(f"\n🚀 Starting Hybrid Retrieval Pipeline with {model_to_use}")
        print(f"📝 Query: '{query}'")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 0: Extract laws by designation using regex (before AI)
            regex_matched_titles = self._extract_law_by_designation(query)
            
            # Step 1: Find relevant law titles using AI
            ai_matched_titles = self.find_relevant_law_titles(query, model_to_use)
            
            # Merge results and remove duplicates
            all_titles = regex_matched_titles + ai_matched_titles
            unique_titles = []
            seen = set()
            
            for title in all_titles:
                if title not in seen:
                    unique_titles.append(title)
                    seen.add(title)
            
            if regex_matched_titles:
                print(f"📋 Merged results: {len(regex_matched_titles)} from regex + {len(ai_matched_titles)} from AI = {len(unique_titles)} unique titles")
            
            if not unique_titles:
                return {
                    "answer": "Inga relevanta lagar hittades för din fråga.",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "method": "hybrid_filtered_rag",
                    "model_used": model_to_use,
                    "stats": {
                        "regex_matches": 0,
                        "ai_matches": 0,
                        "titles_found": 0,
                        "chunks_filtered": 0,
                        "chunks_retrieved": 0
                    }
                }
            
            # Step 2: Filter ChromaDB by titles
            filtered_ids = self.filter_chromadb_by_titles(unique_titles)
            
            if not filtered_ids:
                return {
                    "answer": "Lagar identifierades men inga matchande dokument hittades i databasen.",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "method": "hybrid_filtered_rag",
                    "model_used": model_to_use,
                    "stats": {
                        "regex_matches": len(regex_matched_titles),
                        "ai_matches": len(ai_matched_titles),
                        "titles_found": len(unique_titles),
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
                    "model_used": model_to_use,
                    "stats": {
                        "regex_matches": len(regex_matched_titles),
                        "ai_matches": len(ai_matched_titles),
                        "titles_found": len(unique_titles),
                        "chunks_filtered": len(filtered_ids),
                        "chunks_retrieved": 0
                    }
                }
            
            # Step 4: Format context and generate answer
            context = self.format_context(search_results)
            answer = self.generate_answer(query, context, model_to_use)
            
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
            
            print(f"\n🎉 Hybrid Retrieval Complete with {model_to_use}!")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"📊 Stats:")
            print(f"   - Regex matches: {len(regex_matched_titles)}")
            print(f"   - AI matches: {len(ai_matched_titles)}")
            print(f"   - Unique relevant law titles: {len(unique_titles)}")
            print(f"   - Chunks filtered from DB: {len(filtered_ids)}")
            print(f"   - Top chunks retrieved: {len(sources)}")
            print("=" * 80)
            
            return {
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time,
                "method": "hybrid_filtered_rag",
                "model_used": model_to_use,
                "stats": {
                    "regex_matches": len(regex_matched_titles),
                    "ai_matches": len(ai_matched_titles),
                    "titles_found": len(unique_titles),
                    "chunks_filtered": len(filtered_ids),
                    "chunks_retrieved": len(sources),
                    "relevant_laws": unique_titles
                }
            }
            
        except Exception as e:
            error_msg = f"Ett fel uppstod i hybrid retrieval: {e}"
            print(f"❌ {error_msg}")
            return {
                "answer": error_msg,
                "sources": [],
                "processing_time": time.time() - start_time,
                "method": "hybrid_filtered_rag",
                "model_used": model_to_use
            }

def test_interactive():
    """Interactive test mode - allows user to input their own prompts"""
    print("Swedish Laws Hybrid Retrieval System - Interactive Mode")
    print("=" * 80)
    print("This system combines:")
    print("1. Regex literal search for law designations")
    print("2. AI-based title filtering")
    print("3. ChromaDB filtered RAG")
    print("4. Gemini answer generation")
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
            result = retrieval_system.process_query(query, top_k=12)
            
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
                print(f"Regex matches: {result['stats']['regex_matches']}")
                print(f"AI matches: {result['stats']['ai_matches']}")
                print(f"Relevant law titles found: {result['stats']['titles_found']}")
                print(f"Chunks filtered: {result['stats']['chunks_filtered']}")
                print(f"Chunks retrieved: {result['stats']['chunks_retrieved']}")
            print(f"Model used: {result.get('model_used', 'N/A')}")
            print(f"Processing time: {result['processing_time']:.2f}s")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")

def main():
    """Test the hybrid retrieval system"""
    print("Swedish Laws Hybrid Retrieval System Test")
    print("=" * 80)
    
    # Initialize system
    try:
        retrieval_system = HybridLegalRetrieval()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test queries - including one with specific law designation
    test_queries = [
        "Vad säger lagen om skatt på naturgrus?",
        "Vad säger lagen om diskriminering av deltidsanställda?",
        "Vad säger Förordning (2020:974) om undantag?"  # Test with specific designation
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*100}")
        print(f"TEST {i}: {query}")
        print('='*100)
        
        result = retrieval_system.process_query(query, top_k=12)
        
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
            print(f"Regex matches: {result['stats']['regex_matches']}")
            print(f"AI matches: {result['stats']['ai_matches']}")
            print(f"Unique titles found: {result['stats']['titles_found']}")
            print(f"Chunks filtered: {result['stats']['chunks_filtered']}")
            print(f"Chunks retrieved: {result['stats']['chunks_retrieved']}")
        print(f"Model used: {result.get('model_used', 'N/A')}")
        print(f"Processing time: {result['processing_time']:.2f}s")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        test_interactive()
    else:
        main()