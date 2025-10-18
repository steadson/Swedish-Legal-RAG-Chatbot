import json
import os
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
import time
import tiktoken
import re

# Load environment variables
load_dotenv()

class SwedishLegalEmbedderChunked:
    def __init__(self, data_file_path: str, db_path: str = "./chroma_db"):
        self.data_file_path = data_file_path
        self.db_path = db_path
        self.max_tokens = 7500  # Leave buffer for metadata
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="swedish_legal_documents_chunked",
            metadata={"description": "Swedish legal documents (SFS) embeddings with chunking"}
        )
    
    def load_scraped_data(self) -> Dict:
        """Load the scraped data from JSON file"""
        print(f"📁 Loading data from: {self.data_file_path}")
        with open(self.data_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data.get('documents', []))} documents")
        return data
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def smart_chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """Intelligently chunk text at sentence boundaries"""
        if self.count_tokens(text) <= max_tokens:
            return [text]
        
        # Split by sentences (Swedish legal text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence too long, split by words
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        test_word_chunk = word_chunk + (" " if word_chunk else "") + word
                        if self.count_tokens(test_word_chunk) <= max_tokens:
                            word_chunk = test_word_chunk
                        else:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                            word_chunk = word
                    if word_chunk:
                        current_chunk = word_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_document_chunks(self, doc: Dict) -> List[Tuple[str, Dict]]:
        """Create chunks for a document, each with full metadata"""
        # Create base metadata (same for all chunks)
        base_metadata = {
            'title': doc.get('title', ''),
            'sfs_number': doc.get('sfs_number', ''),
            'url': doc.get('url', ''),
            'source_link': doc.get('source_link', ''),
            'amendment_register_link': doc.get('amendment_register_link', ''),
            'ministry_authority': doc.get('ministry_authority', ''),
            'scraped_at': doc.get('scraped_at', ''),
            'page_found': str(doc.get('page_found', '')),
        }
        
        # Add issued date if available
        if doc.get('metadata', {}).get('Utfärdad'):
            base_metadata['issued_date'] = doc['metadata']['Utfärdad']
        
        # Create header (title, SFS, metadata)
        header_parts = []
        if doc.get('title'):
            header_parts.append(f"Titel: {doc['title']}")
        if doc.get('sfs_number'):
            header_parts.append(f"SFS-nummer: {doc['sfs_number']}")
        if doc.get('ministry_authority'):
            header_parts.append(f"Myndighet: {doc['ministry_authority']}")
        if doc.get('metadata'):
            for key, value in doc['metadata'].items():
                header_parts.append(f"{key}: {value}")
        
        header_text = "\n".join(header_parts)
        header_tokens = self.count_tokens(header_text)
        
        # Calculate available tokens for content
        available_tokens = self.max_tokens - header_tokens - 100  # Buffer for links
        
        # Get main content
        main_content = doc.get('description', '')
        
        if not main_content:
            # No content to chunk, return single chunk with header
            full_text = header_text
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_id': 0,
                'total_chunks': 1,
                'is_chunked': False,
                'token_count': self.count_tokens(full_text)
            })
            return [(full_text, chunk_metadata)]
        
        # Check if chunking is needed
        full_content_tokens = self.count_tokens(main_content)
        
        if full_content_tokens <= available_tokens:
            # No chunking needed
            full_text = f"{header_text}\n\nInnehåll: {main_content}"
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_id': 0,
                'total_chunks': 1,
                'is_chunked': False,
                'token_count': self.count_tokens(full_text)
            })
            return [(full_text, chunk_metadata)]
        
        # Chunking needed
        print(f"  📄 Document needs chunking: {full_content_tokens} tokens > {available_tokens} available")
        content_chunks = self.smart_chunk_text(main_content, available_tokens)
        
        chunks_with_metadata = []
        total_chunks = len(content_chunks)
        
        for i, content_chunk in enumerate(content_chunks):
            # Create full text for this chunk
            chunk_text = f"{header_text}\n\nInnehåll (Del {i+1}/{total_chunks}): {content_chunk}"
            
            # Add links at the end
            if doc.get('source_link'):
                chunk_text += f"\n\nKälla: {doc['source_link']}"
            if doc.get('amendment_register_link'):
                chunk_text += f"\nÄndringsregister: {doc['amendment_register_link']}"
            
            # Create chunk metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_id': i,
                'total_chunks': total_chunks,
                'is_chunked': True,
                'chunk_position': f"{i+1}/{total_chunks}",
                'token_count': self.count_tokens(chunk_text)
            })
            
            chunks_with_metadata.append((chunk_text, chunk_metadata))
            print(f"    📝 Chunk {i+1}/{total_chunks}: {chunk_metadata['token_count']} tokens")
        
        return chunks_with_metadata
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"  ❌ Error getting embedding: {e}")
            return None
    
    def embed_documents(self, batch_size: int = 3):
        """Embed all documents with chunking support"""
        print(f"\n🚀 Starting embedding process with chunking...")
        data = self.load_scraped_data()
        documents = data.get('documents', [])
        
        print(f"📊 Found {len(documents)} documents to process")
        print(f"📦 Processing in batches of {batch_size}")
        print(f"🔢 Max tokens per chunk: {self.max_tokens}")
        print("=" * 60)
        
        total_embedded = 0
        total_chunks = 0
        
        for i in range(0, len(documents), batch_size):
            batch_num = i // batch_size + 1
            batch = documents[i:i + batch_size]
            
            print(f"\n📦 BATCH {batch_num}/{(len(documents) + batch_size - 1) // batch_size}")
            
            batch_texts = []
            batch_metadatas = []
            batch_ids = []
            batch_embeddings = []
            
            for j, doc in enumerate(batch):
                doc_index = i + j
                print(f"\n  📄 Document {doc_index + 1}: {doc.get('title', 'Unknown')[:60]}...")
                print(f"  🏷️  SFS: {doc.get('sfs_number', 'N/A')}")
                
                # Create chunks for this document
                chunks = self.create_document_chunks(doc)
                
                for chunk_idx, (chunk_text, chunk_metadata) in enumerate(chunks):
                    print(f"    🧩 Processing chunk {chunk_idx + 1}/{len(chunks)}")
                    
                    # Get embedding
                    embedding = self.get_embedding(chunk_text)
                    if embedding is None:
                        print(f"    ⏭️  Skipping chunk due to embedding error")
                        continue
                    
                    # Update metadata with document index
                    chunk_metadata['document_index'] = doc_index
                    
                    # Create unique ID
                    chunk_id = f"doc_{doc.get('sfs_number', doc_index)}_{doc_index}_chunk_{chunk_idx}"
                    
                    batch_texts.append(chunk_text)
                    batch_metadatas.append(chunk_metadata)
                    batch_ids.append(chunk_id)
                    batch_embeddings.append(embedding)
                    
                    print(f"    ✅ Chunk prepared: {chunk_metadata['token_count']} tokens")
                    total_chunks += 1
            
            # Add batch to ChromaDB
            if batch_embeddings:
                print(f"\n  💾 Storing {len(batch_embeddings)} chunks in ChromaDB...")
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                total_embedded += len(batch_embeddings)
                print(f"  ✅ Batch {batch_num} stored successfully!")
                
                # Rate limiting
                time.sleep(1)
        
        print(f"\n🎉 EMBEDDING COMPLETE!")
        print(f"📊 Total documents processed: {len(documents)}")
        print(f"🧩 Total chunks created: {total_chunks}")
        print(f"💾 Total chunks embedded: {total_embedded}")
        print(f"🗄️  Total items in collection: {self.collection.count()}")
        print("=" * 60)
    
    def search_documents(self, query: str, n_results: int = 5) -> Dict:
        """Search for relevant documents/chunks"""
        print(f"\n🔍 Searching for: '{query}'")
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return {"error": "Failed to get query embedding"}
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"✅ Found {len(results['documents'][0])} results")
        return results

# Usage example
if __name__ == "__main__":
    # Initialize embedder with chunking
    embedder = SwedishLegalEmbedderChunked(
        data_file_path="data/scraping_progress.json",
        db_path="./chroma_db_chunked"
    )
    
    # Embed all documents with chunking
    embedder.embed_documents(batch_size=3)
    
    # Test search
    test_query = "skatt på naturgrus"
    results = embedder.search_documents(test_query, n_results=5)
    
    print(f"\n🔍 Search results for '{test_query}':")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0], 
        results['distances'][0]
    )):
        chunk_info = ""
        if metadata.get('is_chunked'):
            chunk_info = f" [Chunk {metadata['chunk_position']}]"
        
        print(f"\n{i+1}. {metadata['title']}{chunk_info}")
        print(f"   SFS: {metadata['sfs_number']}")
        print(f"   Similarity: {1-distance:.3f}")
        print(f"   Tokens: {metadata.get('token_count', 'N/A')}")
        print(f"   URL: {metadata['url']}")
        print(f"   Source: {metadata['source_link']}")
        print(f"   Amendment: {metadata['amendment_register_link']}")
        if metadata.get('is_chunked'):
            print(f"   📄 Part {metadata['chunk_id'] + 1} of {metadata['total_chunks']} chunks")