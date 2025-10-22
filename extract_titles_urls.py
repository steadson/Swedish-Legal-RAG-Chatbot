import os
import json
import re
from pathlib import Path
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction_log.txt'),
        logging.StreamHandler()
    ]
)

# Compile regex patterns once for better performance
TITLE_PATTERN = re.compile(r'Title:\s*(.*?)(?=SFS Number:)', re.DOTALL)
URL_PATTERN = re.compile(r'^URL:\s*(.+)$', re.MULTILINE)

def extract_title_and_url_fast(file_path):
    """
    Fast extraction of title and URL from a document file.
    Optimized for speed with minimal file reading.
    """
    try:
        filename = os.path.basename(file_path)
        
        # Read only the first few lines to find title and URL (they're at the top)
        with open(file_path, 'r', encoding='utf-8', buffering=8192) as file:
            # Read first 1024 characters (should contain title and URL)
            header_content = file.read(1024)
        
        # Extract title and URL using pre-compiled patterns
        title_match = TITLE_PATTERN.search(header_content)
        if title_match:
            # Clean up the title: remove extra whitespace and newlines
            title = ' '.join(title_match.group(1).strip().split())
        else:
            title = None
        
        url_match = URL_PATTERN.search(header_content)
        url = url_match.group(1).strip() if url_match else None
        
        return {
            'title': title,
            'url': url,
            'filename': filename
        }
    
    except Exception as e:
        return {
            'filename': os.path.basename(file_path),
            'title': None,
            'url': None,
            'status': 'error',
            'error': str(e)
        }

def process_batch(file_paths):
    """Process a batch of files in a single process."""
    results = []
    for file_path in file_paths:
        result = extract_title_and_url_fast(file_path)
        results.append(result)
    return results

def load_progress():
    """Load progress from JSON file if it exists."""
    progress_file = 'extraction_progress.json'
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Loaded progress: {len(data.get('processed_files', []))} files already processed")
                return data
        except Exception as e:
            print(f"Error loading progress file: {e}")
            return {'processed_files': [], 'results': []}
    return {'processed_files': [], 'results': []}

def save_progress(data):
    """Save current progress to JSON file."""
    try:
        with open('extraction_progress.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Progress saved: {len(data['results'])} files processed")
    except Exception as e:
        print(f"Error saving progress: {e}")

def save_final_results(results, file_paths):
    """Save final results to JSON file."""
    try:
        # Filter out entries with missing title or URL and create clean structure
        clean_results = []
        detailed_results = []
        successful_count = 0
        failed_count = 0
        
        # Create a mapping from results to filenames
        # for i, result in enumerate(results):
        #     filename = os.path.basename(file_paths[i]) if i < len(file_paths) else f"unknown_file_{i}"
        # Process results (each result now includes its own filename)
        for result in results:
            if result.get('title') and result.get('url'):
                # For compact version (AI processing)
                clean_results.append({
                    'title': result['title'],
                    'url': result['url']
                })
                # For detailed version (with filenames)
                detailed_results.append({
                    'title': result['title'],
                    'url': result['url'],
                    'filename': result.get('filename', 'unknown_file')
                })
                successful_count += 1
            else:
                failed_count += 1
        # Save compact version for AI processing (no indentation, minimal metadata)
        compact_data = {
            'total': len(clean_results),
            'documents': clean_results
        }
        
        with open('titles_and_urls.json', 'w', encoding='utf-8') as f:
            json.dump(compact_data, f, ensure_ascii=False, separators=(',', ':'))
        
        # Save readable version with full metadata (no filenames)
        readable_data = {
            'total_files': len(results),
            'successful_extractions': successful_count,
            'failed_extractions': failed_count,
            'documents': clean_results
        }
        
        with open('titles_and_urls_readable.json', 'w', encoding='utf-8') as f:
            json.dump(readable_data, f, ensure_ascii=False, indent=2)
        # Save detailed version with filenames
        detailed_data = {
            'total_files': len(results),
            'successful_extractions': successful_count,
            'failed_extractions': failed_count,
            'documents': detailed_results
        }
        
        with open('titles_urls_with_filenames.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
            # Save titles-only array
            titles_only = [result['title'] for result in clean_results]
            
            with open('titles_only.json', 'w', encoding='utf-8') as f:
                json.dump(titles_only, f, ensure_ascii=False, indent=2)
        print(f"Results saved:")
        print(f"- titles_and_urls.json (compact for AI): {len(clean_results)} documents")
        print(f"- titles_and_urls_readable.json (human-readable): {len(clean_results)} documents")
        print(f"- titles_urls_with_filenames.json (with filenames): {len(detailed_results)} documents")
        print(f"- titles_only.json (titles array only): {len(titles_only)} titles")
        print(f"Total files: {len(results)}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {failed_count}")
        
    except Exception as e:
        print(f"Error saving final results: {e}")

def create_batches(file_list, batch_size):
    """Create batches of files for processing."""
    for i in range(0, len(file_list), batch_size):
        yield file_list[i:i + batch_size]

def main():
    """Main function to extract titles and URLs from all documents using multiprocessing."""
    
    start_time = time.time()
    
    # Define the raw documents directory
    raw_docs_dir = Path("data/raw_documents")
    
    if not raw_docs_dir.exists():
        print(f"Directory {raw_docs_dir} does not exist!")
        return
    
    # Load existing progress
    progress_data = load_progress()
    processed_files = set(progress_data['processed_files'])
    results = progress_data['results']
    
    # Get all .txt files in the directory
    txt_files = list(raw_docs_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} text files in {raw_docs_dir}")
    
    # Filter out already processed files
    remaining_files = [f for f in txt_files if f.name not in processed_files]
    print(f"Remaining files to process: {len(remaining_files)}")
    
    if not remaining_files:
        print("All files have been processed!")
        save_final_results(results, txt_files)
        return
    
    # Determine optimal number of processes and batch size
    num_processes = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overwhelming the system
    batch_size = max(50, len(remaining_files) // (num_processes * 4))  # Adaptive batch size
    
    print(f"Using {num_processes} processes with batch size {batch_size}")
    print(f"Estimated processing time: {len(remaining_files) / (num_processes * 100):.1f} minutes")
    
    # Create batches of files
    file_batches = list(create_batches(remaining_files, batch_size))
    total_batches = len(file_batches)
    
    # Process files in parallel
    completed_batches = 0
    
    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(process_batch, batch): i 
                for i, batch in enumerate(file_batches)
            }
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                completed_batches += 1
                
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    
                    # Add processed files to the set (using batch file names since results no longer contain filename)
                    batch = file_batches[batch_idx]
                    for file_path in batch:
                        processed_files.add(file_path.name)
                    
                    # Progress update
                    progress_pct = (completed_batches / total_batches) * 100
                    elapsed_time = time.time() - start_time
                    files_processed = len(results) - len(progress_data['results'])
                    
                    if files_processed > 0:
                        files_per_second = files_processed / elapsed_time
                        remaining_files_count = len(remaining_files) - files_processed
                        eta_seconds = remaining_files_count / files_per_second if files_per_second > 0 else 0
                        eta_minutes = eta_seconds / 60
                        
                        print(f"Progress: {progress_pct:.1f}% ({completed_batches}/{total_batches} batches) "
                              f"- {files_per_second:.1f} files/sec - ETA: {eta_minutes:.1f} min")
                    
                    # Save progress every 10% or every 10 batches
                    if completed_batches % max(1, total_batches // 10) == 0 or completed_batches % 10 == 0:
                        progress_data = {
                            'processed_files': list(processed_files),
                            'results': results
                        }
                        save_progress(progress_data)
                
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    # Add error results for the failed batch
                    batch = file_batches[batch_idx]
                    for file_path in batch:
                        error_result = {
                            
                            'title': None,
                            'url':None
                        }
                        results.append(error_result)
                        processed_files.add(file_path.name)
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
        progress_data = {
            'processed_files': list(processed_files),
            'results': results
        }
        save_progress(progress_data)
        return
    
    # Calculate final statistics
    total_time = time.time() - start_time
    files_processed = len(results) - len(progress_data['results'])
    
    print(f"\nProcessing completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Files processed: {files_processed}")
    print(f"Average speed: {files_processed/total_time:.1f} files/second")
    
    # Save final results
    save_final_results(results, txt_files)
    
    # Clean up progress file
    try:
        if os.path.exists('extraction_progress.json'):
            os.remove('extraction_progress.json')
            print("Cleaned up progress file")
    except Exception as e:
        print(f"Could not remove progress file: {e}")

if __name__ == "__main__":
    # Ensure multiprocessing works on Windows
    mp.freeze_support()
    
    try:
        main()
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Unexpected error in main: {e}")