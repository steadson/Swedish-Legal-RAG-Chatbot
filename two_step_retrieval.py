import json
import os
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import time
import re

# Load environment variables
load_dotenv()

class TwoStepLegalRetrieval:
    """
    Two-step retrieval system for Swedish legal documents using Gemini:
    Step 1: Use Gemini to identify relevant laws from titles database
    Step 2: Load actual law texts from files and generate detailed answers
    """
    
    def __init__(self, 
                 titles_file: str = "titles_only.json",
                 full_file: str = "titles_and_urls_readable.json",
                 filenames_file: str = "titles_urls_with_filenames.json",
                 raw_documents_dir: str = "data/raw_documents",
                 model_name: str = "gemini-2.5-flash"):
        """
        Initialize the two-step retrieval system
        
        Args:
            titles_file: JSON file with just law titles for search
            full_file: JSON file with titles and URLs
            filenames_file: JSON file mapping titles to filenames
            raw_documents_dir: Directory containing the actual law text files
            model_name: Gemini model to use
        """
        self.titles_file = titles_file
        self.full_file = full_file
        self.filenames_file = filenames_file
        self.raw_documents_dir = raw_documents_dir
        self.model_name = model_name
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Load the JSON data
        self.laws_titles = self._load_laws_data(titles_file)
        self.full_laws_data = self._load_full_laws_data(full_file)
        self.laws_with_filenames = self._load_laws_with_filenames(filenames_file)
        
        print(f"✅ Two-step retrieval initialized:")
        print(f"   - {len(self.laws_titles)} law titles for search")
        print(f"   - {len(self.full_laws_data)} complete law records")
        print(f"   - {len(self.laws_with_filenames)} laws with filenames")
    
    def _load_laws_data(self, json_file_path: str) -> List[str]:
        """Load the JSON file containing Swedish law titles"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load law titles from {json_file_path}: {e}")
    
    def _load_full_laws_data(self, json_file_path: str) -> List[Dict]:
        """Load the complete JSON file with titles and URLs"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('documents', [])
        except Exception as e:
            raise Exception(f"Failed to load full laws database from {json_file_path}: {e}")
    
    def _load_laws_with_filenames(self, json_file_path: str) -> List[Dict]:
        """Load the JSON file that includes filenames"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('documents', [])
        except Exception as e:
            raise Exception(f"Failed to load laws with filenames from {json_file_path}: {e}")
    
    def _read_law_file_content(self, filename: str) -> Optional[str]:
        """Read the content of a law file from the raw_documents folder"""
        try:
            file_path = os.path.join(self.raw_documents_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️ Error reading file {filename}: {e}")
            return None
    
    def _match_titles_to_urls(self, titles: List[str]) -> List[Dict]:
        """Match AI-returned titles with the full database to get URLs"""
        matched_laws = []
        
        # Create a lookup dictionary for faster matching
        title_lookup = {law['title']: law for law in self.full_laws_data}
        
        for title in titles:
            # Try exact match first
            if title in title_lookup:
                matched_laws.append(title_lookup[title])
            else:
                # Try fuzzy matching if exact match fails
                title_normalized = title.strip().lower()
                for db_title, law in title_lookup.items():
                    if db_title.strip().lower() == title_normalized:
                        matched_laws.append(law)
                        break
        
        return matched_laws
    
    def _get_law_contents(self, matched_laws: List[Dict]) -> List[Dict]:
        """Get file contents for matched laws"""
        laws_with_content = []
        
        # Create lookup by title for laws_with_filenames
        filename_lookup = {law['title']: law for law in self.laws_with_filenames}
        
        for law in matched_laws:
            title = law['title']
            
            # Find the filename
            if title in filename_lookup:
                filename = filename_lookup[title]['filename']
                content = self._read_law_file_content(filename)
                
                if content:
                    laws_with_content.append({
                        'title': title,
                        'url': law['url'],
                        'filename': filename,
                        'content': content
                    })
                else:
                    print(f"⚠️ Could not read content for {title}")
            else:
                print(f"⚠️ No filename found for {title}")
        
        return laws_with_content
    
    def find_relevant_laws(self, user_query: str, model_name: str = "gemini-2.0-flash") -> List[str]:
        """
        Step 1: Use Gemini to identify relevant law titles from the database
        
        Args:
            user_query: User's question in Swedish
            model_name: Gemini model to use
            
        Returns:
            List of relevant law titles
        """
        print(f"\n🔍 Step 1: Finding relevant laws with {model_name} for query: '{user_query}'")
        
        # Prepare the context with laws data
        laws_context = json.dumps(self.laws_titles, ensure_ascii=False, indent=2)
        
        # Create the prompt
        prompt = f"""You are a Swedish legal research assistant. You have access to a database of Swedish laws provided below.

SWEDISH LAWS DATABASE:
{laws_context}

USER QUERY: {user_query}

INSTRUCTIONS:
1. Read and parse all of the JSON database above
2. Search through all the laws based on your understanding of Swedish law
3. Find the most relevant laws (1-10) that likely contain the answer to the query
4. Return ONLY a JSON array of exact titles from the database

Response format should be like:
[
    "exact title 1",
    "exact title 2",
  
]

Return [] if no relevant laws found.
CRITICAL: Return ONLY the JSON array with exact titles as they appear in the database, no explanations or additional text, no makingup text."""

        try:
            # Initialize the model
            model = genai.GenerativeModel(model_name)
            
            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                )
            )
            
            # ✅ FIXED: safe extraction of all text parts
            assistant_message = ""
            # Try new-style response parsing
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
                # Fallback if it’s still empty
                if not assistant_message and hasattr(response, "text"):
                    assistant_message = response.text.strip()
            except Exception as parse_err:
                print(f"⚠️ Parsing warning: {parse_err}")
                print("DEBUG: response dir:", dir(response))
                print("DEBUG: candidates:", getattr(response, "candidates", None))
                print("🧾 Dumping raw Gemini response structure:")
                print(response)

            if not assistant_message:
                raise ValueError("Gemini response contained no text parts.")
            
            # Try to parse JSON response
            try:
                result = json.loads(assistant_message)
                print(f"✅ Step 1 complete: Found {len(result)} relevant law titles")
                print('result',result)
                return result
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from text
                import re
                json_match = re.search(r'\[.*?\]', assistant_message, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        print(f"✅ Step 1 complete: Found {len(result)} relevant law titles")
                        print(f"   Found titles: {result}")
                        return result
                    except:
                        print(f"❌ Failed to parse JSON from fallback: {json_match.group()}")
                        return []
                
        except Exception as e:
            print(f"❌ Error in Step 1: {e}")
            return []
    
    def generate_answer_with_content(self, user_query: str, laws_with_content: List[Dict], 
                                 model_name: str = "gemini-2.5-flash") -> str:
        """
        Step 2: Use Gemini to answer the query based on actual law file contents
        
        Args:
            user_query: User's original question
            laws_with_content: List of law dictionaries with content
            
        Returns:
            AI-generated answer with citations
        """
        print(f"\n🤖 Step 2: Generating answer with {model_name} using {len(laws_with_content)} law texts")
        
        # Prepare context with law contents
        context_parts = []
        for i, law in enumerate(laws_with_content, 1):
            context_parts.append(f"""
===== LAG {i} =====
Titel: {law['title']}
URL: {law['url']}
Filnamn: {law['filename']}

FULLSTÄNDIG LAGTEXT:
{law['content']}

==================
""")
        
        laws_context = "\n".join(context_parts)
        
        prompt = f"""Du är en expert på svensk lagstiftning. Du har fått FULLSTÄNDIG TEXT av relevanta svenska lagar nedan.

ANVÄNDARENS FRÅGA: {user_query}

RELEVANTA SVENSKA LAGAR (FULLSTÄNDIG TEXT):
{laws_context}

INSTRUKTIONER:
1. Läs och analysera den fullständiga lagtexten ovan noggrant
2. Svara på användarens fråga baserat ENDAST på informationen i dessa lagar
3. Citera vilken specifik lag och paragraf du refererar till
4. Om svaret inte finns i de tillhandahållna lagarna, ange det tydligt
5. Ge ett klart, omfattande och korrekt svar
6. Inkludera relevanta citat eller specifika avsnitt när det är hjälpsamt
7. Svara på SVENSKA

Ge ditt svar nu:"""

        try:
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=8192,
                )
            )
            
            answer = response.text.strip()
            print(f"✅ Step 2 complete: Generated detailed answer with {model_name}")
            return answer
            
        except Exception as e:
            error_msg = f"Failed to generate answer with {model_name}: {e}"
            print(f"❌ {error_msg}")
            return f"Tyvärr, jag stötte på ett fel när jag bearbetade din fråga: {error_msg}"

    
    def process_query(self, query: str, max_laws: Optional[int] = None, 
                 model_name: str = "gemini-2.0-flash") -> Dict:
        """
        Complete two-step process: find relevant laws, then generate detailed answer
        
        Args:
            query: User's question in Swedish
            max_laws: Not used (kept for API compatibility)
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        print(f"\n🚀 Starting two-step retrieval with {model_name} for: '{query}'")
        start_time = time.time()
        
        try:
            # Step 1: Find relevant law titles using Gemini
            title_results = self.find_relevant_laws(query, model_name)
            
            if not title_results:
                return {
                    "answer": "Inga relevanta lagar hittades för din fråga.",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "method": "two_step_retrieval",
                    'model_used':model_name
                }
            
            # Match titles to get full objects with URLs
            matched_laws = self._match_titles_to_urls(title_results)
            
            if not matched_laws:
                return {
                    "answer": "Titlar hittades men kunde inte matchas till databasen.",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "method": "two_step_retrieval",
                    "model_used": model_name
                }
            
            print(f"📋 Matched {len(matched_laws)} laws with database")
            
            # Load file contents for matched laws
            laws_with_content = self._get_law_contents(matched_laws)
            
            if not laws_with_content:
                return {
                    "answer": "Relevanta lagar identifierades men deras innehåll kunde inte laddas.",
                    "sources": [{"title": law["title"], "url": law["url"]} for law in matched_laws],
                    "processing_time": time.time() - start_time,
                    "method": "two_step_retrieval",
                    "model_used": model_name
                }
            
            print(f"📄 Successfully loaded {len(laws_with_content)} law file(s):")
            for i, law in enumerate(laws_with_content, 1):
                print(f"   {i}. {law['title']} ({law['filename']})")
            
            # Step 2: Generate detailed answer using law contents
            answer = self.generate_answer_with_content(query, laws_with_content, model_name)
            
            # Format sources
            sources = []
            for law in laws_with_content:
                sources.append({
                    "title": law["title"],
                    "url": law["url"],
                    "filename": law["filename"]
                })
            
            processing_time = time.time() - start_time
            print(f"🎉 Two-step retrieval complete with {model_name} in {processing_time:.2f} seconds")
        
            return {
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time,
                "method": "two_step_retrieval",
                "laws_analyzed": len(laws_with_content),
                "model_used": model_name
            }
            
        except Exception as e:
            error_msg = f"Ett fel uppstod i two-step retrieval: {e}"
            print(f"❌ {error_msg}")
            return {
                "answer": error_msg,
                "sources": [],
                "processing_time": time.time() - start_time,
                "method": "two_step_retrieval",
                "model_used": model_name
            }


def main():
    """Test the two-step retrieval system"""
    print("Swedish Laws Two-Step Retrieval System Test")
    print("=" * 80)
    
    # Initialize the system
    try:
        retrieval_system = TwoStepLegalRetrieval()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Test queries
    test_queries = [
        "Vad säger lagen om skatt på naturgrus?",
        "Vad säger lagen om diskriminering av deltidsanställda?",
        "Förklara lagen om läkemedelsförmåner?"
    ]
    
    print(f"\n📋 Running {len(test_queries)} test queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print("\n" + "=" * 100)
        print(f"TEST QUERY {i}: {query}")
        print("=" * 100)
        
        result = retrieval_system.process_query(query)
        
        print(f"\n⏱️ Processing time: {result['processing_time']:.2f} seconds")
        print(f"📚 Laws analyzed: {result.get('laws_analyzed', 0)}")
        print(f"🔧 Method: {result['method']}")
        
        print(f"\n💬 ANSWER:")
        print("-" * 80)
        print(result['answer'])
        
        print(f"\n🔗 SOURCES ({len(result['sources'])}):")
        print("-" * 80)
        for j, source in enumerate(result['sources'], 1):
            print(f"{j}. {source['title']}")
            print(f"   URL: {source['url']}")
            print(f"   File: {source.get('filename', 'N/A')}")
        
        print("=" * 100)


if __name__ == "__main__":
    main()