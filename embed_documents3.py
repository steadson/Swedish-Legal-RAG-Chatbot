import json
import os
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from dotenv import load_dotenv
import time
import tiktoken
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

class SwedishLegalEmbedderGemini:
    def __init__(self, raw_documents_dir: str, db_path: str = "./chroma_db_gemini2"):
        self.raw_documents_dir = raw_documents_dir
        self.db_path = db_path
        self.max_tokens = 800  # Target chunk size

        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize Gemini client
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="swedish_legal_documents_gemini2",
            metadata={"description": "Swedish legal documents (SFS) embeddings with Gemini (800-token chunks)"}
        )

    def load_scraped_data(self) -> Dict:
        """Load document data from raw_documents folder"""
        print(f"📁 Loading documents from: {self.raw_documents_dir}")

        if not os.path.exists(self.raw_documents_dir):
            raise FileNotFoundError(f"Raw documents directory not found: {self.raw_documents_dir}")

        documents = []
        txt_files = [f for f in os.listdir(self.raw_documents_dir) if f.endswith('.txt')]

        for filename in txt_files:
            filepath = os.path.join(self.raw_documents_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                if not content:
                    print(f"⚠️  Skipping empty file: {filename}")
                    continue

                doc_data = self.parse_document_content(content, filename)
                documents.append(doc_data)

            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
                continue

        print(f"✅ Loaded {len(documents)} documents from {len(txt_files)} files")
        return {"documents": documents}

    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate for chunking)"""
        return len(self.tokenizer.encode(text))

    def smart_chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """Intelligently chunk text at §, newlines, or sentence boundaries"""
        if self.count_tokens(text) <= max_tokens:
            return [text]

        # Split by § (sections) first
        sections = re.split(r'(?=\n\d+\s*§)', text)
        chunks = []
        current_chunk = ""

        for section in sections:
            sentences = re.split(r'(?<=[.!?])\s+', section.strip())
            for sentence in sentences:
                test_chunk = current_chunk + (" " if current_chunk else "") + sentence
                if self.count_tokens(test_chunk) <= max_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        # Fallback: split by words if sentence is too long
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

    def parse_document_content(self, content: str, filename: str) -> Dict:
        """Parse document content from .txt file format"""
        lines = content.split('\n')
        doc_data = {
            'title': '',
            'sfs_number': '',
            'url': '',
            'source_link': '',
            'amendment_register_link': '',
            'ministry_authority': '',
            'description': '',
            'metadata': {},
            'scraped_at': '',
            'page_found': '',
            'filename': filename
        }

        content_start_idx = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if line.startswith('Title: '):
                doc_data['title'] = line[7:].strip()
            elif line.startswith('SFS-nummer: '):
                doc_data['sfs_number'] = line[12:].strip()
            elif line.startswith('URL: '):
                doc_data['url'] = line[5:].strip()
            elif line.startswith('Källa: '):
                doc_data['source_link'] = line[7:].strip()
            elif line.startswith('Ändringsregister: '):
                doc_data['amendment_register_link'] = line[18:].strip()
            elif line.startswith('Myndighet: '):
                doc_data['ministry_authority'] = line[11:].strip()
            elif line.startswith('Utfärdad: '):
                doc_data['metadata']['Utfärdad'] = line[10:].strip()
            elif line.startswith('Innehåll:'):
                content_start_idx = i + 1
                break
            elif ':' in line and len(line.split(':', 1)) == 2:
                key, value = line.split(':', 1)
                doc_data['metadata'][key.strip()] = value.strip()

        if content_start_idx < len(lines):
            content_lines = lines[content_start_idx:]
            doc_data['description'] = '\n'.join(content_lines).strip()

        if not doc_data['sfs_number'] and 'SFS' not in filename:
            doc_data['sfs_number'] = filename.replace('.txt', '')

        return doc_data

    def create_document_chunks(self, doc: Dict) -> List[Tuple[str, Dict]]:
        """Create chunks for a document, each with full metadata"""
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

        if doc.get('metadata', {}).get('Utfärdad'):
            base_metadata['issued_date'] = doc['metadata']['Utfärdad']

        header_parts = []
        if doc.get('title'):
            header_parts.append(f"Titel: {doc['title']}")
        if doc.get('sfs_number'):
            header_parts.append(f"SFS-nummer: {doc['sfs_number']}")
        if doc.get('ministry_authority'):
            header_parts.append(f"Myndighet: {doc['ministry_authority']}")
        for key, value in doc.get('metadata', {}).items():
            header_parts.append(f"{key}: {value}")

        header_text = "\n".join(header_parts)
        header_tokens = self.count_tokens(header_text)

        available_tokens = self.max_tokens - header_tokens - 100  # Buffer for links

        main_content = doc.get('description', '')

        if not main_content:
            full_text = header_text
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_id': 0,
                'total_chunks': 1,
                'is_chunked': False,
                'token_count': self.count_tokens(full_text)
            })
            return [(full_text, chunk_metadata)]

        full_content_tokens = self.count_tokens(main_content)

        if full_content_tokens <= available_tokens:
            full_text = f"{header_text}\n\nInnehåll: {main_content}"
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_id': 0,
                'total_chunks': 1,
                'is_chunked': False,
                'token_count': self.count_tokens(full_text)
            })
            return [(full_text, chunk_metadata)]

        content_chunks = self.smart_chunk_text(main_content, available_tokens)

        chunks_with_metadata = []
        total_chunks = len(content_chunks)

        for i, content_chunk in enumerate(content_chunks):
            chunk_text = f"{header_text}\n\nInnehåll (Del {i+1}/{total_chunks}): {content_chunk}"
            if doc.get('source_link'):
                chunk_text += f"\n\nKälla: {doc['source_link']}"
            if doc.get('amendment_register_link'):
                chunk_text += f"\nÄndringsregister: {doc['amendment_register_link']}"

            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_id': i,
                'total_chunks': total_chunks,
                'is_chunked': True,
                'chunk_position': f"{i+1}/{total_chunks}",
                'token_count': self.count_tokens(chunk_text)
            })

            chunks_with_metadata.append((chunk_text, chunk_metadata))

        return chunks_with_metadata

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from Google Gemini"""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"  ❌ Error getting Gemini embedding: {e}")
            return None

    def embed_documents(self, batch_size: int = 32, resume: bool = True, max_workers: int = 8):
        """Embed all documents with chunking support using Gemini (parallelized - FIXED)"""
        print(f"\n🚀 Starting embedding process with Gemini and chunking...")
        data = self.load_scraped_data()
        documents = data.get('documents', [])

        existing_doc_indices = set()
        if resume:
            try:
                existing_items = self.collection.get(include=["metadatas"])
                for metadata in existing_items['metadatas']:
                    if 'document_index' in metadata:
                        existing_doc_indices.add(metadata['document_index'])
                print(f"📋 Found {len(existing_doc_indices)} already embedded documents")
            except Exception as e:
                print(f"⚠️  Could not check existing embeddings: {e}")

        print(f"📊 Found {len(documents)} total documents")
        print(f"⏭️  Skipping {len(existing_doc_indices)} already embedded documents")
        print(f"🔄 Processing {len(documents) - len(existing_doc_indices)} remaining documents")
        print(f"🔢 Max tokens per chunk: {self.max_tokens}")
        print(f"🤖 Using Google Gemini embedding-001 model")
        print(f"👨‍💻 Using {max_workers} parallel workers")
        print("=" * 60)

        total_embedded = 0
        total_chunks = 0

        # Pre-chunk all documents
        all_chunks = []
        for doc_index, doc in enumerate(documents):
            if resume and doc_index in existing_doc_indices:
                continue
            chunks = self.create_document_chunks(doc)
            for chunk_idx, (chunk_text, chunk_metadata) in enumerate(chunks):
                chunk_metadata['document_index'] = doc_index
                chunk_metadata['embedding_model'] = 'gemini-embedding-001'
                chunk_id = f"gemini_doc_{doc.get('sfs_number', doc_index)}_{doc_index}_chunk_{chunk_idx}"
                all_chunks.append((chunk_text, chunk_metadata, chunk_id))

        total_chunks = len(all_chunks)
        print(f"📝 Total chunks to embed: {total_chunks}")

        # Process chunks in parallel batches
        batch_texts = []
        batch_metadatas = []
        batch_ids = []
        batch_embeddings = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and map futures to chunks
            future_to_chunk = {
                executor.submit(self.get_embedding, chunk_text): (chunk_text, chunk_metadata, chunk_id)
                for chunk_text, chunk_metadata, chunk_id in all_chunks
            }

            for i, future in enumerate(as_completed(future_to_chunk)):
                embedding = future.result()
                
                if embedding is not None:
                    chunk_text, chunk_metadata, chunk_id = future_to_chunk[future]
                    batch_texts.append(chunk_text)
                    batch_metadatas.append(chunk_metadata)
                    batch_ids.append(chunk_id)
                    batch_embeddings.append(embedding)

                    # Insert in batches
                    if len(batch_texts) >= batch_size:
                        self.collection.add(
                            embeddings=batch_embeddings,
                            documents=batch_texts,
                            metadatas=batch_metadatas,
                            ids=batch_ids
                        )
                        total_embedded += len(batch_texts)
                        print(f"  💾 Embedded {total_embedded}/{total_chunks} chunks")
                        batch_texts = []
                        batch_metadatas = []
                        batch_ids = []
                        batch_embeddings = []

                # Progress update
                if (i + 1) % 50 == 0:
                    print(f"  🔄 Processed {i + 1}/{total_chunks} chunks")

                time.sleep(0.05)

        # Insert any remaining chunks
        if batch_texts:
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            total_embedded += len(batch_texts)

        print(f"\n🎉 GEMINI EMBEDDING COMPLETE!")
        print(f"📊 Total documents processed: {len(documents)}")
        print(f"🧩 Total chunks created: {total_chunks}")
        print(f"💾 Total chunks embedded: {total_embedded}")
        print(f"🗄️  Total items in collection: {self.collection.count()}")
        print(f"🤖 Embedding model: Google Gemini embedding-001")
        print("=" * 60)

    def search_documents(self, query: str, n_results: int = 5) -> Dict:
        """Search for relevant documents/chunks using Gemini embeddings"""
        print(f"\n🔍 Searching for: '{query}' (using Gemini)")
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = result['embedding']
            print(f"🔍 Query embedding generated: {len(query_embedding)} dimensions")
        except Exception as e:
            print(f"❌ Error getting query embedding: {e}")
            return {"error": "Failed to get query embedding"}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        print(f"✅ Found {len(results['documents'][0])} results")
        return results

if __name__ == "__main__":
    embedder = SwedishLegalEmbedderGemini(
        raw_documents_dir="data/raw_documents",
        db_path="./chroma_db_gemini2"
    )
    
    # Embed with parallel processing (8 workers for speed)
    embedder.embed_documents(batch_size=32, resume=True, max_workers=8)
    
    # Test search
    test_query = "skatt på naturgrus"
    results = embedder.search_documents(test_query, n_results=5)
    
    print(f"\n🔍 Gemini search results for '{test_query}':")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        chunk_info = f" [Chunk {metadata['chunk_position']}]" if metadata.get('is_chunked') else ""
        print(f"\n{i+1}. {metadata['title']}{chunk_info}")
        print(f"   SFS: {metadata['sfs_number']}")
        print(f"   Similarity: {1-distance:.3f}")
        print(f"   Tokens: {metadata.get('token_count', 'N/A')}")
        print(f"   Model: {metadata.get('embedding_model', 'N/A')}")
        print(f"   URL: {metadata['url']}")
        print(f"   Source: {metadata['source_link']}")
        print(f"   Amendment: {metadata['amendment_register_link']}")
        if metadata.get('is_chunked'):
            print(f"   📄 Part {metadata['chunk_id'] + 1} of {metadata['total_chunks']} chunks")