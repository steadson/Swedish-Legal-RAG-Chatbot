import requests
from bs4 import BeautifulSoup
import time
import json
import os
from urllib.parse import urljoin, quote
import logging
from typing import List, Dict, Optional

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
    
    def get_document_content(self, document_url: str) -> Optional[str]:
        """
        Fetch the full content of a specific document
        
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
            
            # Find all links in the metadata section
            metadata_links = soup.find_all('a', class_=['sc-680f2911-0', 'sc-d9f50bcf-0'])
            
            for link in metadata_links:
                href = link.get('href', '')
                link_text = link.get_text(strip=True).lower()
                
                # Check for source link (Full text)
                if 'sfst' in href or 'full text' in link_text or 'fulltext' in link_text:
                    source_link = href
                
                # Check for amendment register link (SFSR)
                elif 'sfsr' in href or 'sfsr' in link_text:
                    amendment_register_link = href

            # Extract Ministry/Authority from document content
            content_text = soup.get_text()
            import re
            
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
            
            # Find the main content area (this may need adjustment based on actual document structure)
            content_selectors = [
                'main#content',
                '.document-content',
                '.law-content',
                'article',
                'main'
            ]
            
            content = None
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    content = content_element.get_text(separator='\n', strip=True)
                    break
            
            if not content:
                # Fallback: get all text from body
                body = soup.find('body')
                if body:
                    content = body.get_text(separator='\n', strip=True)
            
            time.sleep(self.delay)  # Be respectful
        
            return {
                'content': content,
                'source_link': source_link,
                'amendment_register_link': amendment_register_link,
                'ministry_authority': ministry_authority
            }
            
        except requests.RequestException as e:
            logger.error(f"Error fetching document content from {document_url}: {e}")
            return None
    
    def scrape_all_documents(self, start_page: int = 1, end_page: int = 2):
        """
        Scrape all documents from the specified page range
        
        Args:
            start_page: Starting page number (default: 1)
            end_page: Ending page number (default: 278)
        """
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
        
        # Final save
        self.save_progress()
        logger.info(f"Scraping completed! Total documents collected: {len(self.scraped_documents['documents'])}")
    
    def download_document_contents(self, max_documents: Optional[int] = None):
        """
        Download full content for all scraped documents
        
        Args:
            max_documents: Maximum number of documents to download (None for all)
        """
        documents_to_process = self.scraped_documents["documents"]
        if max_documents:
            documents_to_process = documents_to_process[:max_documents]
        
        logger.info(f"Starting to download content for {len(documents_to_process)} documents")
        
        for i, doc in enumerate(documents_to_process, 1):
            logger.info(f"Downloading content {i}/{len(documents_to_process)}: {doc['title']}")
            
            # Create filename from SFS number or title
            filename = doc['sfs_number'].replace(':', '_').replace('/', '_') if doc['sfs_number'] else f"doc_{i}"
            filename = f"{filename}.txt"
            filepath = os.path.join(self.raw_data_dir, filename)
            
            # Skip if already downloaded
            if os.path.exists(filepath):
                logger.info(f"Content already downloaded: {filename}")
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
                    f.write(f"Amendment Register Link: {content_data.get('amendment_register_link', 'N/A')}\n")
                    f.write(f"Ministry/Authority: {content_data.get('ministry_authority', 'N/A')}\n")
                    f.write(f"Issued: {doc['metadata'].get('Utfärdad', 'N/A')}\n")
                    f.write(f"Metadata: {json.dumps(doc['metadata'], ensure_ascii=False)}\n")
                    f.write("\n" + "="*50 + "\n\n")
                    f.write(content_data['content'])
                
                # Also update the document info in memory
                doc['source_link'] = content_data.get('source_link')
                doc['amendment_register_link'] = content_data.get('amendment_register_link')
                doc['ministry_authority'] = content_data.get('ministry_authority')
                
                logger.info(f"Content saved: {filename}")
            else:
                logger.error(f"Failed to download content for: {doc['title']}")
        # Save updated progress with new metadata
        self.save_progress()
    
    def get_statistics(self) -> Dict:
        """Get scraping statistics"""
        return {
            "total_documents": len(self.scraped_documents["documents"]),
            "completed_pages": len(self.scraped_documents["completed_pages"]),
            "last_page_processed": self.scraped_documents["last_page"],
            "progress_percentage": (len(self.scraped_documents["completed_pages"]) / 2) * 100
        }


def main():
    """Main function to run the scraper"""
    scraper = RiksdagenScraper(delay=1.5)  # 1.5 second delay between requests
    
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