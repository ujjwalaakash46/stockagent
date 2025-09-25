# Fixed Queue Manager with Better Error Handling
# ==============================================

import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import queue
import uuid
import traceback

class StockAnalysisQueueManager:
    """
    Enhanced queue manager with better                 # Run analysis with retry mechanism
                max_retries = 3
                retry_count = 0
                last_error = None

                while retry_count < max_retries:
                    try:
                        result = analyze_stock_with_optimized_agents(
                            next_item.get("ticker"),
                            next_item.get("company_name")
                        )
                        
                        # Check if result has error
                        if isinstance(result, dict) and result.get('error'):
                            raise Exception(result['error'])
                        
                        completed_result = {
                            "id": next_item.get("id"),
                            "ticker": next_item.get("ticker"),
                            "company_name": next_item.get("company_name"),
                            "status": "completed",
                            "created_at": next_item.get("created_at"),
                            "started_at": next_item.get("started_at"),
                            "completed_at": datetime.now().isoformat(),
                            "analysis_result": result,
                            "success": True,
                            "error": None,
                            "execution_time": result.get("execution_time", 0),
                            "retry_count": retry_count
                        }
                        break
                    except Exception as e:
                        retry_count += 1
                        last_error = str(e)
                        print(f"Attempt {retry_count} failed: {last_error}")
                        if retry_count < max_retries:
                            time.sleep(5 * retry_count)  # Incremental backoff
                        continue
                
                if retry_count >= max_retries:
                    completed_result = {
                        "id": next_item.get("id"),
                        "ticker": next_item.get("ticker"),
                        "company_name": next_item.get("company_name"),
                        "status": "failed",
                        "created_at": next_item.get("created_at"),
                        "started_at": next_item.get("started_at"),
                        "completed_at": datetime.now().isoformat(),
                        "analysis_result": None,
                        "success": False,
                        "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
                        "retry_count": retry_count
                    }validation
    """
    
    def __init__(self, queue_file="stock_queue.json", results_file="stock_results.json"):
        self.queue_file = queue_file
        self.results_file = results_file
        self.processing_queue = queue.Queue()
        self.is_processing = False
        self.worker_thread = None
        self.file_lock = threading.Lock()
        
        # Initialize files with proper validation
        self._initialize_files()
        self._validate_and_repair_files()
    
    def _initialize_files(self):
        """Create and initialize JSON files with proper structure"""
        for file_path in [self.queue_file, self.results_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump([], f, indent=2)
                print(f"Created {file_path}")
    
    def _validate_and_repair_files(self):
        """Validate and repair corrupted JSON files"""
        for file_path in [self.queue_file, self.results_file]:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Ensure it's a list
                if not isinstance(data, list):
                    print(f"Warning: {file_path} is not a list, resetting to empty list")
                    with open(file_path, 'w') as f:
                        json.dump([], f, indent=2)
                
                # Validate each item has required fields
                if file_path == self.queue_file:
                    self._validate_queue_items(data, file_path)
                else:
                    self._validate_result_items(data, file_path)
                    
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {file_path}: {e}. Resetting to empty list.")
                with open(file_path, 'w') as f:
                    json.dump([], f, indent=2)
    
    def _validate_queue_items(self, items, file_path):
        """Validate queue items have required fields"""
        required_fields = ['id', 'ticker', 'status', 'created_at']
        fixed_items = []
        
        for item in items:
            if not isinstance(item, dict):
                continue
                
            # Check and fix missing fields
            if all(field in item for field in required_fields):
                fixed_items.append(item)
            else:
                print(f"Warning: Invalid queue item found, skipping: {item}")
        
        # Save fixed items if changes were made
        if len(fixed_items) != len(items):
            with open(file_path, 'w') as f:
                json.dump(fixed_items, f, indent=2, default=str)
            print(f"Fixed {file_path}: removed {len(items) - len(fixed_items)} invalid items")
    
    def _validate_result_items(self, items, file_path):
        """Validate result items have required fields"""
        required_fields = ['id', 'ticker', 'status', 'created_at', 'completed_at']
        fixed_items = []
        
        for item in items:
            if not isinstance(item, dict):
                continue
                
            if all(field in item for field in required_fields):
                fixed_items.append(item)
            else:
                print(f"Warning: Invalid result item found, skipping: {item}")
        
        if len(fixed_items) != len(items):
            with open(file_path, 'w') as f:
                json.dump(fixed_items, f, indent=2, default=str)
            print(f"Fixed {file_path}: removed {len(items) - len(fixed_items)} invalid items")
    
    def _load_json_file(self, file_path: str) -> List[Dict]:
        """Safely load JSON file with comprehensive error handling"""
        with self.file_lock:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Ensure it's always a list
                if not isinstance(data, list):
                    print(f"Warning: {file_path} contains non-list data, returning empty list")
                    return []
                
                return data
                
            except FileNotFoundError:
                print(f"File {file_path} not found, creating empty list")
                with open(file_path, 'w') as f:
                    json.dump([], f, indent=2)
                return []
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error in {file_path}: {e}. Resetting to empty list.")
                with open(file_path, 'w') as f:
                    json.dump([], f, indent=2)
                return []
                
            except Exception as e:
                print(f"Unexpected error loading {file_path}: {e}")
                return []
    
    def _save_json_file(self, file_path: str, data: List[Dict]):
        """Safely save JSON file with validation"""
        with self.file_lock:
            try:
                # Validate data before saving
                if not isinstance(data, list):
                    print(f"Error: Attempting to save non-list data to {file_path}")
                    return False
                
                # Create backup
                backup_path = f"{file_path}.backup"
                if os.path.exists(file_path):
                    import shutil
                    shutil.copy2(file_path, backup_path)
                
                # Save with proper JSON serialization
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str, ensure_ascii=False)
                
                return True
                
            except Exception as e:
                print(f"Error saving {file_path}: {e}")
                
                # Restore from backup if save failed
                backup_path = f"{file_path}.backup"
                if os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(backup_path, file_path)
                    print(f"Restored {file_path} from backup")
                
                return False
    
    def add_to_queue(self, ticker: str, company_name: str = None) -> str:
        """
        Enhanced add to queue with better validation and error handling
        """
        try:
            # Validate inputs
            if not ticker or not isinstance(ticker, str):
                raise ValueError("Ticker must be a non-empty string")
            
            ticker = ticker.upper().strip()
            company_name = (company_name or ticker).strip()
            
            request_id = str(uuid.uuid4())
            
            # Create well-structured request object
            request = {
                "id": request_id,
                "ticker": ticker,
                "company_name": company_name,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "position_in_queue": 0,
                "priority": 1,  # Default priority
                "retry_count": 0
            }
            
            # Load current queue
            current_queue = self._load_json_file(self.queue_file)
            
            # Check for duplicates
            for item in current_queue:
                if (item.get("ticker") == ticker and 
                    item.get("status") == "pending"):
                    print(f"Ticker {ticker} already in queue")
                    return item.get("id", "unknown")
            
            # Add to queue
            current_queue.append(request)
            
            # Update positions
            self._update_queue_positions(current_queue)
            
            # Save queue
            if self._save_json_file(self.queue_file, current_queue):
                print(f"Added {ticker} to queue successfully")
                return request_id
            else:
                raise Exception("Failed to save queue to file")
                
        except Exception as e:
            error_msg = f"Failed to add {ticker} to queue: {str(e)}"
            print(f"Error: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            raise Exception(error_msg)
    
    def _update_queue_positions(self, queue_items):
        """Update position numbers for pending items"""
        position = 1
        for item in queue_items:
            if item.get("status") == "pending":
                item["position_in_queue"] = position
                position += 1
            else:
                item["position_in_queue"] = 0
    
    def get_queue_status(self) -> Dict:
        """
        Enhanced queue status with better error handling
        """
        try:
            current_queue = self._load_json_file(self.queue_file)
            current_results = self._load_json_file(self.results_file)
            
            # Filter items by status with validation
            pending_items = []
            processing_items = []
            
            for item in current_queue:
                if not isinstance(item, dict):
                    continue
                    
                status = item.get("status", "unknown")
                if status == "pending":
                    pending_items.append(item)
                elif status == "processing":
                    processing_items.append(item)
            
            # Sort pending items by position
            pending_items.sort(key=lambda x: x.get("position_in_queue", 999))
            
            return {
                "total_pending": len(pending_items),
                "total_processing": len(processing_items),
                "total_completed": len(current_results),
                "pending_items": pending_items,
                "processing_items": processing_items,
                "queue_health": "healthy",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting queue status: {e}")
            return {
                "total_pending": 0,
                "total_processing": 0,
                "total_completed": 0,
                "pending_items": [],
                "processing_items": [],
                "queue_health": "error",
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    def get_results(self, limit: int = None) -> List[Dict]:
        """Get results with enhanced filtering"""
        try:
            results = self._load_json_file(self.results_file)
            
            # Validate and clean results
            valid_results = []
            for result in results:
                if isinstance(result, dict) and result.get("ticker"):
                    valid_results.append(result)
            
            # Sort by completion time (newest first)
            valid_results.sort(
                key=lambda x: x.get("completed_at", ""), 
                reverse=True
            )
            
            if limit:
                return valid_results[:limit]
            return valid_results
            
        except Exception as e:
            print(f"Error getting results: {e}")
            return []
    
    def get_result_by_ticker(self, ticker: str) -> Optional[Dict]:
        """Find most recent result for a ticker"""
        try:
            results = self.get_results()
            ticker = ticker.upper().strip()
            
            for result in results:
                if result.get("ticker", "").upper() == ticker:
                    return result
            return None
            
        except Exception as e:
            print(f"Error finding result for {ticker}: {e}")
            return None
    
    def _process_next_item(self):
        """Enhanced processing with better error handling"""
        try:
            # Import stockai module
            try:
                from stockai import analyze_stock_with_optimized_agents
            except ImportError as e:
                print(f"Could not import stockai module: {e}")
                return False
            
            current_queue = self._load_json_file(self.queue_file)
            
            # Find next pending item
            next_item = None
            for item in current_queue:
                if item.get("status") == "pending":
                    next_item = item
                    break
            
            if not next_item:
                return False
            
            print(f"Processing {next_item.get('ticker')}...")
            
            # Update status to processing
            for item in current_queue:
                if item.get("id") == next_item.get("id"):
                    item["status"] = "processing"
                    item["started_at"] = datetime.now().isoformat()
                    break
            
            self._save_json_file(self.queue_file, current_queue)
            
            # Run analysis
            try:
                result = analyze_stock_with_optimized_agents(
                    next_item.get("ticker"),
                    next_item.get("company_name")
                )
                
                completed_result = {
                    "id": next_item.get("id"),
                    "ticker": next_item.get("ticker"),
                    "company_name": next_item.get("company_name"),
                    "status": "completed",
                    "created_at": next_item.get("created_at"),
                    "started_at": next_item.get("started_at"),
                    "completed_at": datetime.now().isoformat(),
                    "analysis_result": result,
                    "success": True,
                    "error": None,
                    "execution_time": result.get("execution_time", 0)
                }
                
                print(f"Analysis completed for {next_item.get('ticker')}")
                
            except Exception as e:
                completed_result = {
                    "id": next_item.get("id"),
                    "ticker": next_item.get("ticker"),
                    "company_name": next_item.get("company_name"),
                    "status": "failed",
                    "created_at": next_item.get("created_at"),
                    "started_at": next_item.get("started_at"),
                    "completed_at": datetime.now().isoformat(),
                    "analysis_result": None,
                    "success": False,
                    "error": str(e),
                    "execution_time": 0
                }
                
                print(f"Analysis failed for {next_item.get('ticker')}: {e}")
            
            # Save result
            results = self._load_json_file(self.results_file)
            results.append(completed_result)
            self._save_json_file(self.results_file, results)
            
            # Remove from queue
            current_queue = [
                item for item in current_queue 
                if item.get("id") != next_item.get("id")
            ]
            
            # Update positions
            self._update_queue_positions(current_queue)
            self._save_json_file(self.queue_file, current_queue)
            
            return True
            
        except Exception as e:
            print(f"Error processing queue item: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def start_background_processing(self):
        """Enhanced background processing"""
        if self.is_processing:
            print("Background processing already running")
            return
        
        def worker():
            self.is_processing = True
            print("Started background queue processing")
            
            consecutive_errors = 0
            
            while self.is_processing:
                try:
                    if self._process_next_item():
                        consecutive_errors = 0
                        time.sleep(5)  # Short wait between items
                    else:
                        time.sleep(15)  # Longer wait when queue is empty
                        
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Background processing error #{consecutive_errors}: {e}")
                    
                    # Exponential backoff for errors
                    wait_time = min(30 * (2 ** consecutive_errors), 300)  # Max 5 minutes
                    time.sleep(wait_time)
                    
                    # Stop if too many consecutive errors
                    if consecutive_errors >= 5:
                        print("Too many consecutive errors, stopping background processing")
                        self.is_processing = False
                        break
        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
    
    def stop_background_processing(self):
        """Stop background processing"""
        if self.is_processing:
            self.is_processing = False
            print("Background processing stopped")
    
    def clear_completed_results(self, older_than_days: int = None):
        """Clear results with optional age filter"""
        try:
            results = self._load_json_file(self.results_file)
            
            if older_than_days is None:
                self._save_json_file(self.results_file, [])
                print(f"Cleared all {len(results)} results")
            else:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)
                cutoff_str = cutoff_date.isoformat()
                
                filtered_results = [
                    result for result in results
                    if result.get("completed_at", "") > cutoff_str
                ]
                
                removed_count = len(results) - len(filtered_results)
                self._save_json_file(self.results_file, filtered_results)
                print(f"Cleared {removed_count} results older than {older_than_days} days")
                
        except Exception as e:
            print(f"Error clearing results: {e}")

# Global instance
queue_manager = None

def get_queue_manager():
    """Get or create global queue manager"""
    global queue_manager
    if queue_manager is None:
        queue_manager = StockAnalysisQueueManager()
    return queue_manager

def initialize_queue_system():
    """Initialize and start the queue system"""
    try:
        manager = get_queue_manager()
        manager.start_background_processing()
        return manager
    except Exception as e:
        print(f"Failed to initialize queue system: {e}")
        raise

# Wrapper functions
def add_stock_to_queue(ticker: str, company_name: str = None) -> str:
    """Add stock to analysis queue"""
    manager = get_queue_manager()
    return manager.add_to_queue(ticker, company_name)

def get_queue_info() -> Dict:
    """Get current queue status"""
    manager = get_queue_manager()
    return manager.get_queue_status()

def get_all_results(limit: int = None) -> List[Dict]:
    """Get completed analysis results"""
    manager = get_queue_manager()
    return manager.get_results(limit)

def find_result_by_ticker(ticker: str) -> Optional[Dict]:
    """Find result by ticker symbol"""
    manager = get_queue_manager()
    return manager.get_result_by_ticker(ticker)