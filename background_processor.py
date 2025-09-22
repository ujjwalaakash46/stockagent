# Background Queue Processor (Standalone Script)
# ==============================================
# This script can run independently to process the queue even when Streamlit is closed

import time
import sys
import os
from datetime import datetime

def main():
    """Main function to run the background queue processor"""
    
    print("ðŸš€ Starting Background Stock Analysis Queue Processor")
    print("=" * 60)
    
    try:
        # Import the queue manager
        from queue_manager import get_queue_manager
        
        # Get the queue manager instance
        manager = get_queue_manager()
        
        print(f"âœ… Queue Manager initialized")
        print(f"ðŸ“ Queue file: {manager.queue_file}")
        print(f"ðŸ“ Results file: {manager.results_file}")
        
        # Start background processing
        manager.start_background_processing()
        
        print("\nðŸ”„ Background processing started!")
        print("ðŸ“Š Queue status will be shown every 60 seconds...")
        print("Press Ctrl+C to stop the processor\n")
        
        # Keep the script running and show periodic status
        try:
            while True:
                time.sleep(60)  # Wait 1 minute
                
                # Show current status
                status = manager.get_queue_status()
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"\n[{current_time}] Queue Status:")
                print(f"  ðŸ“‹ Pending: {status['total_pending']}")
                print(f"  ðŸ”„ Processing: {status['total_processing']}")
                print(f"  âœ… Completed: {status['total_completed']}")
                
                if status['processing_items']:
                    for item in status['processing_items']:
                        print(f"  ðŸ”¥ Currently processing: {item['ticker']}")
                
                if status['pending_items']:
                    next_items = status['pending_items'][:3]  # Show next 3
                    print(f"  â³ Next in queue: {', '.join([item['ticker'] for item in next_items])}")
                
        except KeyboardInterrupt:
            print("\n\nðŸ›‘ Stopping background processor...")
            manager.stop_background_processing()
            print("âœ… Background processor stopped successfully")
            
    except ImportError:
        print("âŒ Error: Could not import queue_manager.py")
        print("   Make sure queue_manager.py is in the same directory as this script")
        sys.exit(1)
        
    except Exception as e:
        print(f"âŒ Error starting background processor: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()