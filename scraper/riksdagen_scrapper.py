import requests
from bs4 import BeautifulSoup
import time
import json
import os
from urllib.parse import urljoin, quote
import logging
from typing import List, Dict, Optional
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiksdagenScraper:
    def __init__(self, base_url: str = "https://www.riksdagen.se", delay: float = 1.0):
        """
        Initialize the Riksdagen scraper
        
        Args:
            base_url: Base URL for the Riksdagen website
            delay: Delay between requests in seconds (be respectful!)
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create data directories
        self.raw_data_dir = "data/raw_documents"
        self.processed_data_dir = "data/processed_documents"
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Progress tracking
        self.progress_file = "data/scraping_progress.json"
        self.scraped_documents = self.load_progress()
    # Add this new method to your RiksdagenScraper class
    def scrape_missing_pages(self):
        """
        Scrape only the pages that were missed during initial scraping
        This method identifies gaps in completed_pages and fills them
        """
        all_pages = set(range(1, 279))  # Pages 1-278
        completed_pages = set(self.scraped_documents["completed_pages"])
        missing_pages = sorted(all_pages - completed_pages)
        
        if not missing_pages:
            logger.info("No missing pages found. All pages have been scraped.")
            return
        
        logger.info(f"Found {len(missing_pages)} missing pages: {missing_pages}")
        logger.info(f"Starting to scrape missing pages...")
        
        total_new_documents = 0
        
        for page_num in missing_pages:
            logger.info(f"Processing missing page {page_num}")
            
            try:
                # Get the search results page
                soup = self.get_search_page(page_num)
                if not soup:
                    logger.error(f"Failed to fetch missing page {page_num}, skipping")
                    continue
                
                # Extract documents from this page
                page_documents = self.extract_documents_from_page(soup, page_num)
                
                if not page_documents:
                    logger.warning(f"No documents found on missing page {page_num}")
                else:
                    logger.info(f"Found {len(page_documents)} documents on missing page {page_num}")
                
                # Add documents to our collection
                self.scraped_documents["documents"].extend(page_documents)
                self.scraped_documents["completed_pages"].append(page_num)
                
                # Update last_page if this is higher
                if page_num > self.scraped_documents["last_page"]:
                    self.scraped_documents["last_page"] = page_num
                
                total_new_documents += len(page_documents)
                
                # Save progress after each page
                self.save_progress()
                logger.info(f"Missing page {page_num} completed and saved.")
                
            except Exception as e:
                logger.error(f"Error processing missing page {page_num}: {e}")
                continue
        
        # Sort completed_pages for cleaner JSON
        self.scraped_documents["completed_pages"].sort()
        
        # Final save
        self.save_progress()
        logger.info(f"Gap filling completed! Added {total_new_documents} new documents.")
        logger.info(f"Total documents now: {len(self.scraped_documents['documents'])}")

    def get_missing_pages_info(self):
        """
        Get information about missing pages without scraping
        """
        all_pages = set(range(1, 279))  # Pages 1-278
        completed_pages = set(self.scraped_documents["completed_pages"])
        missing_pages = sorted(all_pages - completed_pages)
        
        logger.info(f"Total pages that should exist: {len(all_pages)}")
        logger.info(f"Pages completed: {len(completed_pages)}")
        logger.info(f"Missing pages: {missing_pages}")
        logger.info(f"Estimated missing documents: ~{len(missing_pages) * 20}")  # Rough estimate
        
        return missing_pages
    def load_progress(self) -> Dict:
        """Load scraping progress from file"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"completed_pages": [], "documents": [], "last_page": 0}
    
    def save_progress(self):
        """Save scraping progress to file"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_documents, f, ensure_ascii=False, indent=2)
    
    def get_search_page(self, page_num: int) -> Optional[BeautifulSoup]:
        """
        Get a specific search results page
        
        Args:
            page_num: Page number to fetch (1-278)
            
        Returns:
            BeautifulSoup object of the page or None if failed
        """
        # Construct URL with proper encoding
        url = f"{self.base_url}/sv/sok/?avd=dokument&doktyp=sfs&dokstat=gällande+sfs&p={page_num}"
        
        try:
            logger.info(f"Fetching page {page_num}: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Add delay to be respectful
            time.sleep(self.delay)
            
            return BeautifulSoup(response.content, 'html.parser')
            
        except requests.RequestException as e:
            logger.error(f"Error fetching page {page_num}: {e}")
            return None
    
    def extract_documents_from_page(self, soup: BeautifulSoup, page_num: int) -> List[Dict]:
        """
        Extract document information from a search results page
        
        Args:
            soup: BeautifulSoup object of the page
            page_num: Current page number
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        # Find all document list items
        document_items = soup.find_all('li', class_=['sc-e76b6126-1', 'sc-3f041c0d-0'])
        
        for item in document_items:
            try:
                # Extract document link
                link_element = item.find('a', class_=['sc-680f2911-0', 'sc-d9f50bcf-0'])
                if not link_element:
                    continue
                
                document_url = urljoin(self.base_url, link_element.get('href'))
                title = link_element.get_text(strip=True)
                
                # Extract SFS number from the title or URL
                sfs_element = item.find('h4', class_='sc-680f2911-0')
                sfs_number = sfs_element.get_text(strip=True) if sfs_element else ""
                
                # Extract description/content preview
                content_element = item.find('div', class_='sc-41eb4e8b-0')
                description = content_element.get_text(strip=True) if content_element else ""
                
                # Extract metadata (issued date, changed date, etc.)
                metadata = {}
                dl_element = item.find('dl')
                if dl_element:
                    dt_elements = dl_element.find_all('dt')
                    dd_elements = dl_element.find_all('dd')
                    
                    for dt, dd in zip(dt_elements, dd_elements):
                        key = dt.get_text(strip=True)
                        value = dd.get_text(strip=True)
                        metadata[key] = value
                
                document_info = {
                    'title': title,
                    'url': document_url,
                    'sfs_number': sfs_number,
                    'description': description,
                    'metadata': metadata,
                    'page_found': page_num,
                    'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                documents.append(document_info)
                logger.info(f"Extracted: {title} ({sfs_number})")
                
            except Exception as e:
                logger.error(f"Error extracting document from page {page_num}: {e}")
                continue
        
        return documents
    
    def find_html_document_link(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Find the HTML version link for the full document
        
        Args:
            soup: BeautifulSoup object of the document page
            
        Returns:
            URL to the HTML version of the document or None if not found
        """
        try:
            # Look for links that contain "View the full document" or "html" in the URL
            html_link_selectors = [
                'a[href*="/html/"]',  # Direct HTML links
                'a[class*="sc-d9f50bcf-0"]',  # Based on the class structure seen
                'a[href*="html"]'  # Any link containing "html"
            ]
            
            for selector in html_link_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    link_text = link.get_text(strip=True).lower()
                    
                    # Check if this is likely the HTML document link
                    if (href and 
                        ('/html/' in href or href.endswith('/html')) and 
                        ('view' in link_text or 'document' in link_text or 'hela' in link_text)):
                        return urljoin(self.base_url, href)
            
            # Alternative: look for specific text patterns
            view_document_links = soup.find_all('a', string=re.compile(r'View.*document|hela.*dokument', re.IGNORECASE))
            for link in view_document_links:
                href = link.get('href')
                if href and ('/html/' in href or href.endswith('/html')):
                    return urljoin(self.base_url, href)
                    
        except Exception as e:
            logger.error(f"Error finding HTML document link: {e}")
        
        return None
    
    def extract_html_content(self, html_url: str) -> Optional[str]:
        """
        Extract content from the HTML version of the document
        
        Args:
            html_url: URL to the HTML version of the document
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            logger.info(f"Fetching HTML content: {html_url}")
            response = self.session.get(html_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Content extraction selectors based on the HTML structure
            content_selectors = [
                'div.sc-7f1468f0-0.gdXuYm.sc-4cc907ea-4.edvduf',  # Specific class from rough4.txt
                'main#content',  # Main content area
                'div.sc-7f1468f0-0.gdXuYm',  # General content class
                'div[class*="gdXuYm"]',  # Any div with gdXuYm class
                'main',  # Fallback to main element
                'article'  # Fallback to article element
            ]
            
            content = None
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    # Clean up the content by removing font tags and getting clean text
                    # Remove font tags but keep the text
                    for font_tag in content_element.find_all('font'):
                        font_tag.unwrap()
                    
                    # Get text with proper line breaks
                    content = content_element.get_text(separator='\n', strip=True)
                    
                    # Clean up excessive whitespace and line breaks
                    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
                    content = re.sub(r'[ \t]+', ' ', content)
                    
                    if content and len(content.strip()) > 100:  # Ensure we got substantial content
                        break
            
            time.sleep(self.delay)  # Be respectful
            return content
            
        except requests.RequestException as e:
            logger.error(f"Error fetching HTML content from {html_url}: {e}")
            return None
    
    def get_document_content(self, document_url: str) -> Optional[Dict]:
        """
        Fetch the full content of a specific document, preferring HTML over PDF
        Includes special handling for constitutional documents
        
        Args:
            document_url: URL of the document to fetch
            
        Returns:
            Dictionary containing content and metadata links or None if failed
        """
        try:
            logger.info(f"Fetching document content: {document_url}")
            response = self.session.get(document_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract source and amendment register links
            source_link = None
            amendment_register_link = None
            html_document_link = None
            
            # Find the HTML version link first (priority)
            html_document_link = self.find_html_document_link(soup)
            
            # Find all links in the metadata section
            metadata_links = soup.find_all('a', class_=['sc-680f2911-0', 'sc-d9f50bcf-0'])
            
            for link in metadata_links:
                href = link.get('href', '')
                link_text = link.get_text(strip=True).lower()
                
                # Check for source link (Full text) - but prefer HTML over PDF
                if ('sfst' in href or 'full text' in link_text or 'fulltext' in link_text) and not source_link:
                    # Only use PDF if no HTML link found
                    if not html_document_link or '.pdf' not in href:
                        source_link = href
                
                # Check for amendment register link (SFSR)
                elif 'sfsr' in href or 'sfsr' in link_text:
                    amendment_register_link = href

            # Extract Ministry/Authority from document content
            content_text = soup.get_text()
            # Initialize ministry_authority with default value
            ministry_authority = None
            # Look for "Departement/myndighet" pattern
            ministry_match = re.search(r'Departement/myndighet\s*:\s*([^\n]+)', content_text)
            if ministry_match:
                ministry_authority = ministry_match.group(1).strip()

            # Alternative method: look for specific text patterns
            if not source_link or not amendment_register_link:
                # Find bold text elements that might contain "Source" or "Amendment register"
                bold_elements = soup.find_all('b', class_='sc-680f2911-0')
                
                for bold in bold_elements:
                    text = bold.get_text(strip=True).lower()
                    
                    if 'source' in text or 'källa' in text:
                        # Find the next link after this bold element
                        next_link = bold.find_next('a')
                        if next_link and not source_link:
                            source_link = next_link.get('href')
                    
                    elif 'amendment register' in text or 'ändringsregister' in text:
                        # Find the next link after this bold element
                        next_link = bold.find_next('a')
                        if next_link and not amendment_register_link:
                            amendment_register_link = next_link.get('href')
            
            # Get content - prefer HTML version if available
            content = None
            if html_document_link:
                logger.info(f"Found HTML document link: {html_document_link}")
                content = self.extract_html_content(html_document_link)
                # Update source_link to point to HTML version
                source_link = html_document_link
            
            # Fallback to extracting content from current page
            if not content:
                print('no connect.....')
                content_selectors = [
                    'main#content',
                    '.document-content',
                    '.law-content',
                    'article',
                    'main',
                    'div.sc-7f1468f0-0.gdXuYm',  # Add the specific class
                    'div.sc-81e52487-0.kCiZxu'   # Another content class from the structure
                ]
                
                for selector in content_selectors:
                    content_element = soup.select_one(selector)
                    if content_element:
                        # Clean up font tags
                        for font_tag in content_element.find_all('font'):
                            font_tag.unwrap()
                        
                        content = content_element.get_text(separator='\n', strip=True)
                        
                        # Clean up content
                        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
                        content = re.sub(r'[ \t]+', ' ', content)
                        
                        if content and len(content.strip()) > 100:
                            break
                
                # SPECIAL HANDLING FOR CONSTITUTIONAL DOCUMENTS
                if not content or len(content.strip()) < 100:
                    logger.info("Regular extraction failed, trying constitutional document handling...")
                    content = self.handle_constitutional_document(soup, document_url)
                
                time.sleep(self.delay)  # Be respectful
        
            return {
                'content': content,
                'source_link': source_link,
                'amendment_register_link': amendment_register_link,
                'ministry_authority': ministry_authority,
                'html_document_link': html_document_link
            }
            
        except Exception as e:
            logger.error(f"Error fetching document content from {document_url}: {e}")
            # Return a basic structure to prevent crashes
            return {
                'content': None,
                'source_link': None,
                'amendment_register_link': None,
                'ministry_authority': None,
                'html_document_link': None
            }
    
    def handle_constitutional_document(self, soup: BeautifulSoup, document_url: str) -> Optional[str]:
        """
        Special handler for constitutional documents that have different structure
        Extracts main element content and follows constitution links
        """
        try:
            logger.info("Handling constitutional document with special extraction...")
            
            # Extract main element content
            main_element = soup.find('main', id='content')
            if not main_element:
                main_element = soup.find('main')
            
            if not main_element:
                logger.warning("No main element found in constitutional document")
                return None
            
            # Clean font tags from main element
            for font_tag in main_element.find_all('font'):
                font_tag.unwrap()
            
            # Extract text content from main element
            main_content = main_element.get_text(separator='\n', strip=True)
            
            # Clean up the content
            main_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', main_content)
            main_content = re.sub(r'[ \t]+', ' ', main_content)
            
            # Find links within the main element (looking for constitution links)
            constitution_links = []
            for link in main_element.find_all('a', href=True):
                href = link.get('href')
                link_text = link.get_text(strip=True).lower()
                link_classes = ' '.join(link.get('class', []))
                
                # Look for full constitution links (not PDF)
                if href and '.pdf' not in href.lower():
                    # More specific detection based on actual HTML structure
                    is_constitution_link = (
                        # Check for HTML document links with specific classes
                        ('sc-d9f50bcf-0' in link_classes and 'sc-d383988e-4' in link_classes) or
                        # Check for /html/ in URL path
                        '/html/' in href or href.endswith('/html') or
                        # Check for specific text patterns
                        any(keyword in link_text for keyword in 
                            ['visa hela dokumentet', 'view the full document', 'fulltext', 'full', 'komplett']) or
                        # Check for constitution-related keywords in href
                        any(keyword in href.lower() for keyword in 
                            ['constitution', 'grundlag', 'forfattningssamling', 'dokument']) and
                        # Exclude PDF links more thoroughly
                        not any(exclude in href.lower() for exclude in ['.pdf', 'pdf/', '/pdf'])
                    )
                    
                    if is_constitution_link:
                        constitution_links.append(href)
                        logger.info(f"Found constitution link: {href} with text: {link_text}")
            
            # If we found constitution links, fetch content from the first one
            additional_content = ""
            if constitution_links:
                constitution_url = constitution_links[0]
                
                # Make URL absolute if it's relative
                if constitution_url.startswith('/'):
                    from urllib.parse import urljoin
                    constitution_url = urljoin(document_url, constitution_url)
                
                logger.info(f"Following constitution link: {constitution_url}")
                
                try:
                    const_response = self.session.get(constitution_url, timeout=30)
                    const_response.raise_for_status()
                    
                    const_soup = BeautifulSoup(const_response.content, 'html.parser')
                    
                    # Extract main element from constitution page
                    const_main = const_soup.find('main', id='content')
                    if not const_main:
                        const_main = const_soup.find('main')
                    
                    if const_main:
                        # Clean font tags
                        for font_tag in const_main.find_all('font'):
                            font_tag.unwrap()
                        
                        additional_content = const_main.get_text(separator='\n', strip=True)
                        additional_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', additional_content)
                        additional_content = re.sub(r'[ \t]+', ' ', additional_content)
                        
                        logger.info("Successfully extracted constitution content")
                    
                    time.sleep(self.delay)
                    
                except Exception as e:
                    logger.error(f"Error fetching constitution content from {constitution_url}: {e}")
            
            # Combine both contents
            combined_content = main_content
            if additional_content and additional_content.strip():
                combined_content += "\n\n" + "="*80 + "\n"
                combined_content += "FULL CONSTITUTION CONTENT:\n"
                combined_content += "="*80 + "\n\n"
                combined_content += additional_content
            
            return combined_content if combined_content.strip() else None
            
        except Exception as e:
            logger.error(f"Error in constitutional document handling: {e}")
            return None

    # ... existing code ...
    def scrape_all_documents(self, start_page: int = 1, end_page: int = 278):
        """Scrape all documents from the specified page range"""
        logger.info(f"Starting scrape from page {start_page} to {end_page}")
        
        # Resume from last completed page if available
        if self.scraped_documents["last_page"] > start_page:
            start_page = self.scraped_documents["last_page"] + 1
            logger.info(f"Resuming from page {start_page}")
        
        total_documents = 0
        
        for page_num in range(start_page, end_page + 1):
            # Skip if page already completed
            if page_num in self.scraped_documents["completed_pages"]:
                logger.info(f"Page {page_num} already completed, skipping")
                continue
            
            logger.info(f"Processing page {page_num}/{end_page}")
            
            try:
                # Get the search results page
                soup = self.get_search_page(page_num)
                if not soup:
                    logger.error(f"Failed to fetch page {page_num}, skipping")
                    continue
                
                # Extract documents from this page
                page_documents = self.extract_documents_from_page(soup, page_num)
                
                if not page_documents:
                    logger.warning(f"No documents found on page {page_num}")
                
                # Add documents to our collection
                self.scraped_documents["documents"].extend(page_documents)
                self.scraped_documents["completed_pages"].append(page_num)
                self.scraped_documents["last_page"] = page_num
                
                total_documents += len(page_documents)
                
                # Save progress every 10 pages
                if page_num % 10 == 0:
                    self.save_progress()
                    logger.info(f"Progress saved. Total documents so far: {len(self.scraped_documents['documents'])}")
                
                logger.info(f"Page {page_num} completed. Found {len(page_documents)} documents.")
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue
        
        # Final save
        self.save_progress()
        logger.info(f"Scraping completed! Total documents collected: {len(self.scraped_documents['documents'])}")
    
    def download_document_contents(self, max_documents: Optional[int] = None):
        """Download full content for all scraped documents"""
        documents_to_process = self.scraped_documents["documents"]
        if max_documents:
            documents_to_process = documents_to_process[:max_documents]
        
        logger.info(f"Starting to download content for {len(documents_to_process)} documents")
        
        for i, doc in enumerate(documents_to_process, 1):
            try:
                # logger.info(f"Downloading content {i}/{len(documents_to_process)}: {doc['title']}")
                
                # Create filename from SFS number or title
                base_filename = doc['sfs_number'].replace(':', '_').replace('/', '_') if doc['sfs_number'] else f"doc_{i}"
                filename = base_filename
                filepath = os.path.join(self.raw_data_dir, f"{filename}.txt")
                
                # Handle duplicate filenames by checking if it's the same document
                if os.path.exists(filepath):
                    try:
                        # Check if it's actually the same document by comparing URLs
                        with open(filepath, 'r', encoding='utf-8') as f:
                            existing_content = f.read()
                            if f"URL: {doc['url']}" not in existing_content:
                                # Different document with same SFS number - make unique
                                import hashlib
                                url_hash = hashlib.md5(doc['url'].encode()).hexdigest()[:8]
                                filename = f"{base_filename}_{url_hash}"
                                filepath = os.path.join(self.raw_data_dir, f"{filename}.txt")
                                logger.info(f"Created unique filename for duplicate SFS: {filename}.txt")
                    except Exception as e:
                        # If we can't read the file, use counter-based naming
                        counter = 1
                        while os.path.exists(filepath):
                            filename = f"{base_filename}_{counter}"
                            filepath = os.path.join(self.raw_data_dir, f"{filename}.txt")
                            counter += 1
                        logger.info(f"Created unique filename using counter: {filename}.txt")
                
                # Skip if already downloaded (now checking the correct unique file)
                if os.path.exists(filepath):
                    # logger.info(f"Content already downloaded: {filename}")
                    continue
                
                # Download content
                content_data = self.get_document_content(doc['url'])
                if content_data and content_data['content']:
                    # Save content to file with enhanced metadata
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Title: {doc['title']}\n")
                        f.write(f"SFS Number: {doc['sfs_number']}\n")
                        f.write(f"URL: {doc['url']}\n")
                        f.write(f"Source Link: {content_data.get('source_link', 'N/A')}\n")
                        f.write(f"HTML Document Link: {content_data.get('html_document_link', 'N/A')}\n")
                        f.write(f"Amendment Register Link: {content_data.get('amendment_register_link', 'N/A')}\n")
                        f.write(f"Ministry/Authority: {content_data.get('ministry_authority', 'N/A')}\n")
                        f.write(f"Issued: {doc['metadata'].get('Utfärdad', 'N/A')}\n")
                        f.write(f"Metadata: {json.dumps(doc['metadata'], ensure_ascii=False)}\n")
                        f.write("\n" + "="*50 + "\n\n")
                        f.write(content_data['content'])
                    
                    # Also update the document info in memory
                    doc['source_link'] = content_data.get('source_link')
                    doc['html_document_link'] = content_data.get('html_document_link')
                    doc['amendment_register_link'] = content_data.get('amendment_register_link')
                    doc['ministry_authority'] = content_data.get('ministry_authority')
                    
                    logger.info(f"Content saved: {filename}")
                else:
                    logger.error(f"Failed to download content for: {doc['title']}")
                    
            except Exception as e:
                # logger.error(f"Error downloading content for document {i} ({doc.get('title', 'Unknown')}): {e}")
                continue
                
        # Save updated progress with new metadata
        self.save_progress()
    
    def get_statistics(self) -> Dict:
        """Get scraping statistics"""
        return {
            "total_documents": len(self.scraped_documents["documents"]),
            "completed_pages": len(self.scraped_documents["completed_pages"]),
            "last_page_processed": self.scraped_documents["last_page"],
            "progress_percentage": (len(self.scraped_documents["completed_pages"]) / 278) * 100
        }


def main():
    """Main function to run the scraper"""
    scraper = RiksdagenScraper(delay=1.5)  # 1.5 second delay between requests
    
    # Check for missing pages first
    missing_pages = scraper.get_missing_pages_info()
    
    if missing_pages:
        print(f"\n⚠️  MISSING PAGES DETECTED: {missing_pages}")
        print(f"This means approximately {len(missing_pages) * 20} documents are missing.")
        
        choice = input("\nWhat would you like to do?\n1. Scrape missing pages only\n2. Continue with content download\n3. Exit\nChoice (1/2/3): ")
        
        if choice == "1":
            print("\n🔄 Starting gap-filling process...")
            scraper.scrape_missing_pages()
            print("\n✅ Gap filling completed!")
            # Ask if user wants to download content now
            download_choice = input("\nDo you want to download document content now? (y/n): ")
            if download_choice.lower() == 'y':
                print("\n📥 Starting content download...")
                scraper.download_document_contents()
            
            return
        elif choice == "3":
            print("Exiting...")
            return
    
    try:
        # First, scrape all document metadata (fast)
        logger.info("Starting metadata scraping...")
        scraper.scrape_all_documents()
        
        # Print statistics
        stats = scraper.get_statistics()
        logger.info(f"Scraping Statistics: {stats}")
        
        # Then download full content (slower)
        logger.info("Starting content download...")
        scraper.download_document_contents()
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        scraper.save_progress()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        scraper.save_progress()


if __name__ == "__main__":
    main()