# Stock Analysis Persistent Queue System
# =====================================
# This module provides a persistent queue system for stock analysis that works
# independently of Streamlit sessions and continues processing in the background

import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import queue
import uuid

class StockAnalysisQueueManager:
    """
    Manages a persistent queue system for stock analysis requests.
    Uses JSON files for persistence and supports background processing.
    """
    
    def __init__(self, queue_file="stock_queue.json", results_file="stock_results.json"):
        """
        Initialize the queue manager with JSON file paths
        
        Args:
            queue_file (str): Path to store pending analysis requests
            results_file (str): Path to store completed analysis results
        """
        self.queue_file = queue_file
        self.results_file = results_file
        self.processing_queue = queue.Queue()
        self.is_processing = False
        self.worker_thread = None
        
        # Initialize JSON files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Create JSON files if they don't exist"""
        for file_path in [self.queue_file, self.results_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump([], f, indent=2)
                print(f"âœ… Created {file_path}")
    
    def _load_json_file(self, file_path: str) -> List[Dict]:
        """Safely load JSON file with error handling"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_json_file(self, file_path: str, data: List[Dict]):
        """Safely save JSON file with error handling"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"âŒ Error saving {file_path}: {str(e)}")
    
    def add_to_queue(self, ticker: str, company_name: str = None) -> str:
        """
        Add a new stock analysis request to the queue
        
        Args:
            ticker (str): Stock ticker symbol
            company_name (str): Optional company name
            
        Returns:
            str: Unique request ID
        """
        request_id = str(uuid.uuid4())
        
        # Create request object
        request = {
            "id": request_id,
            "ticker": ticker.upper(),
            "company_name": company_name or ticker,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "position_in_queue": None
        }
        
        # Load existing queue
        current_queue = self._load_json_file(self.queue_file)
        
        # Check if already in queue
        for item in current_queue:
            if item["ticker"] == ticker.upper() and item["status"] == "pending":
                print(f"âš ï¸ {ticker} is already in the queue with ID: {item['id']}")
                return item["id"]
        
        # Add to queue
        current_queue.append(request)
        
        # Update positions
        for i, item in enumerate(current_queue):
            if item["status"] == "pending":
                item["position_in_queue"] = len([x for x in current_queue[:i+1] if x["status"] == "pending"])
        
        # Save updated queue
        self._save_json_file(self.queue_file, current_queue)
        
        print(f"âœ… Added {ticker} to analysis queue with ID: {request_id}")
        return request_id
    
    def get_queue_status(self) -> Dict:
        """
        Get current queue status including pending items and processing info
        
        Returns:
            Dict: Queue status information
        """
        current_queue = self._load_json_file(self.queue_file)
        current_results = self._load_json_file(self.results_file)
        
        pending_items = [item for item in current_queue if item["status"] == "pending"]
        processing_items = [item for item in current_queue if item["status"] == "processing"]
        
        return {
            "total_pending": len(pending_items),
            "total_processing": len(processing_items),
            "total_completed": len(current_results),
            "pending_items": pending_items,
            "processing_items": processing_items,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_results(self, limit: int = None) -> List[Dict]:
        """
        Get completed analysis results
        
        Args:
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict]: Completed analysis results
        """
        results = self._load_json_file(self.results_file)
        
        # Sort by completion time (newest first)
        results.sort(key=lambda x: x.get("completed_at", ""), reverse=True)
        
        if limit:
            return results[:limit]
        return results
    
    def get_result_by_id(self, request_id: str) -> Optional[Dict]:
        """Get specific result by request ID"""
        results = self._load_json_file(self.results_file)
        
        for result in results:
            if result.get("id") == request_id:
                return result
        return None
    
    def remove_from_queue(self, request_id: str) -> bool:
        """
        Remove a request from the queue (only if pending)
        
        Args:
            request_id (str): Request ID to remove
            
        Returns:
            bool: True if removed successfully
        """
        current_queue = self._load_json_file(self.queue_file)
        
        for i, item in enumerate(current_queue):
            if item["id"] == request_id:
                if item["status"] == "pending":
                    current_queue.pop(i)
                    
                    # Update positions for remaining items
                    pending_count = 1
                    for remaining_item in current_queue:
                        if remaining_item["status"] == "pending":
                            remaining_item["position_in_queue"] = pending_count
                            pending_count += 1
                    
                    self._save_json_file(self.queue_file, current_queue)
                    print(f"âœ… Removed {item['ticker']} from queue")
                    return True
                else:
                    print(f"âš ï¸ Cannot remove {item['ticker']} - already {item['status']}")
                    return False
        
        print(f"âŒ Request ID {request_id} not found in queue")
        return False
    
    def _process_next_item(self):
        """Process the next item in the queue"""
        try:
            from stockai import analyze_stock_with_multi_agents
        except ImportError:
            print("âŒ Could not import stockai module")
            return False
        
        current_queue = self._load_json_file(self.queue_file)
        
        # Find next pending item
        next_item = None
        for item in current_queue:
            if item["status"] == "pending":
                next_item = item
                break
        
        if not next_item:
            return False
        
        print(f"ðŸš€ Starting analysis for {next_item['ticker']}...")
        
        # Update status to processing
        for item in current_queue:
            if item["id"] == next_item["id"]:
                item["status"] = "processing"
                item["started_at"] = datetime.now().isoformat()
                break
        
        self._save_json_file(self.queue_file, current_queue)
        
        try:
            # Run the analysis
            result = analyze_stock_with_multi_agents(
                next_item["ticker"], 
                next_item["company_name"]
            )
            
            # Prepare result object
            completed_result = {
                "id": next_item["id"],
                "ticker": next_item["ticker"],
                "company_name": next_item["company_name"],
                "status": "completed",
                "created_at": next_item["created_at"],
                "started_at": next_item["started_at"],
                "completed_at": datetime.now().isoformat(),
                "analysis_result": result,
                "success": True,
                "error": None
            }
            
            print(f"âœ… Analysis completed for {next_item['ticker']}")
            
        except Exception as e:
            # Handle errors
            error_msg = str(e)
            completed_result = {
                "id": next_item["id"],
                "ticker": next_item["ticker"],
                "company_name": next_item["company_name"],
                "status": "failed",
                "created_at": next_item["created_at"],
                "started_at": next_item["started_at"],
                "completed_at": datetime.now().isoformat(),
                "analysis_result": None,
                "success": False,
                "error": error_msg
            }
            
            print(f"âŒ Analysis failed for {next_item['ticker']}: {error_msg}")
        
        # Save result
        results = self._load_json_file(self.results_file)
        results.append(completed_result)
        self._save_json_file(self.results_file, results)
        
        # Remove from queue
        current_queue = [item for item in current_queue if item["id"] != next_item["id"]]
        
        # Update positions for remaining items
        pending_count = 1
        for item in current_queue:
            if item["status"] == "pending":
                item["position_in_queue"] = pending_count
                pending_count += 1
        
        self._save_json_file(self.queue_file, current_queue)
        
        return True
    
    def start_background_processing(self):
        """Start background processing of the queue"""
        if self.is_processing:
            print("âš ï¸ Background processing is already running")
            return
        
        def worker():
            self.is_processing = True
            print("ðŸ”„ Started background queue processing...")
            
            while self.is_processing:
                try:
                    # Check if there are items to process
                    if self._process_next_item():
                        # Wait a bit before processing next item
                        time.sleep(5)
                    else:
                        # No items to process, wait longer
                        time.sleep(10)
                        
                except Exception as e:
                    print(f"âŒ Background processing error: {str(e)}")
                    time.sleep(30)  # Wait longer on errors
        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        print("âœ… Background processing thread started")
    
    def stop_background_processing(self):
        """Stop background processing"""
        if self.is_processing:
            self.is_processing = False
            print("ðŸ›‘ Stopping background processing...")
            if self.worker_thread:
                self.worker_thread.join(timeout=5)
            print("âœ… Background processing stopped")
        else:
            print("â„¹ï¸ Background processing was not running")
    
    def clear_completed_results(self, older_than_days: int = None):
        """
        Clear completed results, optionally only older than specified days
        
        Args:
            older_than_days (int): Only clear results older than this many days
        """
        results = self._load_json_file(self.results_file)
        
        if older_than_days is None:
            # Clear all results
            self._save_json_file(self.results_file, [])
            print(f"ðŸ—‘ï¸ Cleared all {len(results)} results")
        else:
            # Clear only old results
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            cutoff_str = cutoff_date.isoformat()
            
            filtered_results = [
                result for result in results 
                if result.get("completed_at", "") > cutoff_str
            ]
            
            removed_count = len(results) - len(filtered_results)
            self._save_json_file(self.results_file, filtered_results)
            print(f"ðŸ—‘ï¸ Cleared {removed_count} results older than {older_than_days} days")

# Global queue manager instance
queue_manager = None

def get_queue_manager():
    """Get or create the global queue manager instance"""
    global queue_manager
    if queue_manager is None:
        queue_manager = StockAnalysisQueueManager()
    return queue_manager

def initialize_queue_system():
    """Initialize the queue system and start background processing"""
    manager = get_queue_manager()
    manager.start_background_processing()
    return manager

# Utility functions for easy integration
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
    """Find the most recent result for a specific ticker"""
    manager = get_queue_manager()
    results = manager.get_results()
    
    for result in results:
        if result.get("ticker", "").upper() == ticker.upper():
            return result
    return None