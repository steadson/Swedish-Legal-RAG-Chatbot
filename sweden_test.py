# import json
# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()

# # Initialize Gemini
# genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# def load_laws_data(json_file_path):
#     """
#     Load the JSON file containing Swedish laws
#     """
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"Error loading JSON file: {e}")
#         return []

# def load_full_laws_data(json_file_path):
#     """
#     Load the complete JSON file with titles and URLs
#     """
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             return data.get('documents', [])
#     except Exception as e:
#         print(f"Error loading full laws database: {e}")
#         return []

# def match_titles_to_urls(titles, full_laws_data):
#     """
#     Match AI-returned titles with the full database to get URLs
#     """
#     matched_laws = []
    
#     # Create a lookup dictionary for faster matching
#     title_lookup = {law['title']: law for law in full_laws_data}
    
#     for title in titles:
#         # Try exact match first
#         if title in title_lookup:
#             matched_laws.append(title_lookup[title])
#         else:
#             # Try fuzzy matching if exact match fails
#             title_normalized = title.strip().lower()
#             for db_title, law in title_lookup.items():
#                 if db_title.strip().lower() == title_normalized:
#                     matched_laws.append(law)
#                     break
    
#     return matched_laws

# def query_with_gemini(user_query, laws_data, model_name="gemini-2.0-flash-exp"):
#     """
#     Send query to Gemini with laws data in context
#     """
    
#     # Prepare the context with laws data
#     laws_context = json.dumps(laws_data, ensure_ascii=False, indent=2)
    
#     # Create the prompt
#     prompt = f"""You are a Swedish legal research assistant. You have access to a database of Swedish laws provided below.

# SWEDISH LAWS DATABASE:
# {laws_context}

# USER QUERY: {user_query}

# INSTRUCTIONS:
# 1. Read and parse the JSON database above
# 2. Search through all the laws based on your understanding of Swedish law
# 3. Find the most relevant laws (1-10) that likely contain the answer to the query
# 4. Return ONLY a JSON array of exact titles from the database

# Response format:
# [
#     "exact title 1",
#     "exact title 2",
#     ...
# ]

# Return [] if no relevant laws found.
# CRITICAL: Return ONLY the JSON array with exact titles as they appear in the database, no explanations or additional text."""

#     try:
#         # Initialize the model
#         model = genai.GenerativeModel(model_name)
        
#         # Generate response
#         response = model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0.1,
#                 max_output_tokens=2048,
#             )
#         )
        
#         assistant_message = response.text.strip()
        
#         try:
#             result = json.loads(assistant_message)
#             return result
#         except json.JSONDecodeError:
#             import re
#             json_match = re.search(r'\[.*?\]', assistant_message, re.DOTALL)
#             if json_match:
#                 try:
#                     return json.loads(json_match.group())
#                 except:
#                     pass
#             print(f"Could not parse: {assistant_message}")
#             return []
            
#     except Exception as e:
#         print(f"Error querying Gemini: {e}")
#         return []

# def answer_query_with_urls(user_query, matched_laws, model_name="gemini-2.0-flash-exp"):
#     """
#     Use Gemini with grounding to search URLs and answer the query
#     """
    
#     # Prepare the URLs list
#     urls = [law['url'] for law in matched_laws]
    
#     # Create context with law titles and URLs
#     laws_info = "\n".join([f"- {law['title']}: {law['url']}" for law in matched_laws])
    
#     prompt = f"""You are a Swedish legal expert. I have identified the following relevant Swedish laws for the user's question:

# RELEVANT LAWS:
# {laws_info}

# USER QUESTION: {user_query}

# INSTRUCTIONS:
# Please search and read the content from the URLs above to find the answer to the user's question. Then provide a comprehensive answer based on what you find in those law documents.

# Your answer should:
# 1. Be based on the actual content from the provided URLs
# 2. Cite which specific law(s) you're referencing
# 3. Be clear and comprehensive
# 4. State if the information is not found in the provided sources

# Please provide your answer now:"""

#     try:
#         # Initialize model with Google Search grounding
#         model = genai.GenerativeModel(
#             model_name=model_name,
#             tools='google_search_retrieval'
#         )
        
#         # Generate response with grounding
#         response = model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0.3,
#                 max_output_tokens=4096,
#             )
#         )
        
#         return response.text
        
#     except Exception as e:
#         # Fallback: Try without grounding tool if not available
#         try:
#             print(f"Note: Google Search grounding not available, trying alternative method...")
            
#             # Alternative: Pass URLs directly in prompt
#             model = genai.GenerativeModel(model_name=model_name)
            
#             alt_prompt = f"""You are a Swedish legal expert. The user has a question about Swedish law.

# USER QUESTION: {user_query}

# RELEVANT SWEDISH LAW URLS TO REFERENCE:
# {chr(10).join(urls)}

# Based on your knowledge and understanding of Swedish law, provide a comprehensive answer to the user's question. If possible, reference which of the above laws would contain the relevant information.

# Your answer should be clear, comprehensive, and based on Swedish legal principles."""

#             response = model.generate_content(
#                 alt_prompt,
#                 generation_config=genai.types.GenerationConfig(
#                     temperature=0.3,
#                     max_output_tokens=4096,
#                 )
#             )
            
#             return response.text
            
#         except Exception as e2:
#             print(f"Error getting answer from Gemini: {e2}")
#             return "Sorry, I encountered an error while processing your query."

# def main():
#     titles_file_path = 'titles_only.json'
#     full_file_path = 'titles_and_urls_readable.json'
    
#     print("Swedish Laws Query System (Gemini with URL Search)")
#     print("="*80)
#     print("This system will:")
#     print("1. Find relevant laws using AI")
#     print("2. Pass URLs to Gemini to search and read")
#     print("3. Answer your question based on the law content")
#     print("="*80 + "\n")
    
#     # Load both databases
#     print("Loading Swedish laws databases...")
#     laws_titles = load_laws_data(titles_file_path)
#     full_laws_data = load_full_laws_data(full_file_path)
    
#     if not laws_titles or not full_laws_data:
#         print("Error: Could not load laws database. Exiting.")
#         return
    
#     print(f"Loaded {len(laws_titles)} law titles for search.")
#     print(f"Loaded {len(full_laws_data)} complete law records.")
#     print("="*80 + "\n")
    
#     while True:
#         query = input("\nYour query: ").strip()
        
#         if query.lower() in ['quit', 'exit', 'q']:
#             print("Goodbye!")
#             break
        
#         if not query:
#             print("Please enter a valid query.")
#             continue
        
#         print("\n" + "="*80)
#         print("STEP 1: Finding relevant laws...")
#         print("="*80)
        
#         # Get titles from Gemini
#         title_results = query_with_gemini(query, laws_titles)
        
#         if not title_results:
#             print("\nNo relevant laws found.")
#             continue
        
#         # Match titles to get full objects with URLs
#         matched_laws = match_titles_to_urls(title_results, full_laws_data)
        
#         if not matched_laws:
#             print("\nTitles found but could not match to database.")
#             continue
        
#         print(f"\nFound {len(matched_laws)} relevant law(s):")
#         for i, law in enumerate(matched_laws, 1):
#             print(f"  {i}. {law['title']}")
        
#         print("\n" + "="*80)
#         print("STEP 2: Sending URLs to Gemini for search and analysis...")
#         print("="*80 + "\n")
        
#         # Get answer from Gemini with URL grounding
#         answer = answer_query_with_urls(query, matched_laws)
        
#         print("ANSWER:")
#         print("-" * 80)
#         print(answer)
#         print("\n" + "="*80)
#         print("SOURCES:")
#         print("-" * 80)
#         for i, law in enumerate(matched_laws, 1):
#             print(f"{i}. {law['title']}")
#             print(f"   {law['url']}")
#         print("="*80)

# if __name__ == "__main__":
#     main()


import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def load_laws_data(json_file_path):
    """
    Load the JSON file containing Swedish laws
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def load_full_laws_data(json_file_path):
    """
    Load the complete JSON file with titles and URLs
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('documents', [])
    except Exception as e:
        print(f"Error loading full laws database: {e}")
        return []

def load_laws_with_filenames(json_file_path):
    """
    Load the JSON file that includes filenames
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('documents', [])
    except Exception as e:
        print(f"Error loading laws with filenames: {e}")
        return []

def read_law_file_content(filename, folder_path='data/raw_documents'):
    """
    Read the content of a law file from the data/raw_documents folder
    """
    try:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

def match_titles_to_urls(titles, full_laws_data):
    """
    Match AI-returned titles with the full database to get URLs
    """
    matched_laws = []
    
    # Create a lookup dictionary for faster matching
    title_lookup = {law['title']: law for law in full_laws_data}
    
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

def get_law_contents(matched_laws, laws_with_filenames, max_laws=2):
    """
    Get file contents for matched laws (limit to first max_laws)
    """
    laws_with_content = []
    
    # Create lookup by title for laws_with_filenames
    filename_lookup = {law['title']: law for law in laws_with_filenames}
    
    # Process only first max_laws
    for law in matched_laws[:max_laws]:
        title = law['title']
        
        # Find the filename
        if title in filename_lookup:
            filename = filename_lookup[title]['filename']
            content = read_law_file_content(filename)
            
            if content:
                laws_with_content.append({
                    'title': title,
                    'url': law['url'],
                    'filename': filename,
                    'content': content
                })
            else:
                print(f"Warning: Could not read content for {title}")
        else:
            print(f"Warning: No filename found for {title}")
    
    return laws_with_content

def query_with_gemini(user_query, laws_data, model_name="gemini-2.0-flash-exp"):
    """
    Send query to Gemini with laws data in context
    """
    
    # Prepare the context with laws data
    laws_context = json.dumps(laws_data, ensure_ascii=False, indent=2)
    
    # Create the prompt
    prompt = f"""You are a Swedish legal research assistant. You have access to a database of Swedish laws provided below.

SWEDISH LAWS DATABASE:
{laws_context}

USER QUERY: {user_query}

INSTRUCTIONS:
1. Read and parse the JSON database above
2. Search through all the laws based on your understanding of Swedish law
3. Find the most relevant laws (1-10) that likely contain the answer to the query
4. Return ONLY a JSON array of exact titles from the database

Response format:
[
    "exact title 1",
    "exact title 2",
    ...
]

Return [] if no relevant laws found.
CRITICAL: Return ONLY the JSON array with exact titles as they appear in the database, no explanations or additional text."""

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
        
        assistant_message = response.text.strip()
        
        try:
            result = json.loads(assistant_message)
            print(result)
            return result
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\[.*?\]', assistant_message, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            print(f"Could not parse: {assistant_message}")
            return []
            
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return []

def answer_with_law_content(user_query, laws_with_content, model_name="gemini-2.0-flash"):
    """
    Use Gemini to answer the query based on actual law file contents
    """
    
    # Prepare context with law contents
    context_parts = []
    for i, law in enumerate(laws_with_content, 1):
        context_parts.append(f"""
===== LAW {i} =====
Title: {law['title']}
URL: {law['url']}
Filename: {law['filename']}

FULL LAW CONTENT:
{law['content']}

==================
""")
    
    laws_context = "\n".join(context_parts)
    
    prompt = f"""You are a Swedish legal expert. You have been provided with the FULL TEXT of relevant Swedish laws below.

USER QUESTION: {user_query}

RELEVANT SWEDISH LAWS (FULL TEXT):
{laws_context}

INSTRUCTIONS:
1. Read and analyze the complete law text provided above carefully
2. Answer the user's question based ONLY on the information in these laws
3. Cite which specific law(s) and sections you're referencing
4. If the answer is not found in the provided laws, clearly state that
5. Provide a clear, comprehensive, and accurate answer
6. Include relevant quotes or specific sections when helpful

Please provide your answer now:"""

    try:
        model = genai.GenerativeModel(model_name)
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=8192,
            )
        )
        
        return response.text
        
    except Exception as e:
        print(f"Error getting answer from Gemini: {e}")
        return "Sorry, I encountered an error while processing your query."

def main():
    titles_file_path = 'titles_only.json'
    full_file_path = 'titles_and_urls_readable.json'
    filenames_file_path = 'titles_urls_with_filenames.json'
    
    print("Swedish Laws Query System (With Local File Content)")
    print("="*80)
    print("This system will:")
    print("1. Find relevant laws from the database")
    print("2. Load actual law content from local files")
    print("3. Answer your question based on the full law text")
    print("="*80 + "\n")
    
    # Load all databases
    print("Loading Swedish laws databases...")
    laws_titles = load_laws_data(titles_file_path)
    full_laws_data = load_full_laws_data(full_file_path)
    laws_with_filenames = load_laws_with_filenames(filenames_file_path)
    
    if not laws_titles or not full_laws_data or not laws_with_filenames:
        print("Error: Could not load laws database. Exiting.")
        return
    
    print(f"Loaded {len(laws_titles)} law titles for search.")
    print(f"Loaded {len(full_laws_data)} complete law records.")
    print(f"Loaded {len(laws_with_filenames)} laws with filenames.")
    print("="*80 + "\n")
    
    while True:
        query = input("\nYour query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        print("\n" + "="*80)
        print("STEP 1: Finding relevant laws...")
        print("="*80)
        
        # Get titles from Gemini
        title_results = query_with_gemini(query, laws_titles)
        
        if not title_results:
            print("\nNo relevant laws found.")
            continue
        
        # Match titles to get full objects with URLs
        matched_laws = match_titles_to_urls(title_results, full_laws_data)
        
        if not matched_laws:
            print("\nTitles found but could not match to database.")
            continue
        
        print(f"\nFound {len(matched_laws)} relevant law(s)")
        print(f"Processing first 2 laws for detailed analysis...")
        
        print("\n" + "="*80)
        print("STEP 2: Loading law file contents from raw_documents...")
        print("="*80)
        
        # Get file contents (only first 2 laws)
        laws_with_content = get_law_contents(matched_laws, laws_with_filenames, max_laws=2)
        
        if not laws_with_content:
            print("\nCould not load any law file contents.")
            continue
        
        print(f"\nSuccessfully loaded {len(laws_with_content)} law file(s):")
        for i, law in enumerate(laws_with_content, 1):
            print(f"  {i}. {law['title']}")
            print(f"     File: {law['filename']}")
        
        print("\n" + "="*80)
        print("STEP 3: Analyzing law content and generating answer...")
        print("="*80 + "\n")
        
        # Get answer from Gemini based on file content
        answer = answer_with_law_content(query, laws_with_content)
        
        print("ANSWER:")
        print("-" * 80)
        print(answer)
        print("\n" + "="*80)
        print("SOURCES USED:")
        print("-" * 80)
        for i, law in enumerate(laws_with_content, 1):
            print(f"{i}. {law['title']}")
            print(f"   URL: {law['url']}")
            print(f"   File: {law['filename']}")
        print("="*80)

if __name__ == "__main__":
    main()