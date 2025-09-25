# Enhanced Background Queue Processor
# ==================================
# Improved standalone script for processing stock analysis queue

import time
import sys
import os
from datetime import datetime
import signal
import argparse

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True

def main():
    """Enhanced main function with better error handling and monitoring"""
    
    print("🚀 Enhanced Stock Analysis Queue Processor v2.0")
    print("=" * 65)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Analysis Queue Processor')
    parser.add_argument('--status-interval', type=int, default=60,
                       help='Status report interval in seconds (default: 60)')
    parser.add_argument('--max-errors', type=int, default=10,
                       help='Maximum consecutive errors before stopping (default: 10)')
    parser.add_argument('--queue-file', type=str, default='stock_queue.json',
                       help='Queue file path (default: stock_queue.json)')
    parser.add_argument('--results-file', type=str, default='stock_results.json',
                       help='Results file path (default: stock_results.json)')
    
    args = parser.parse_args()
    
    try:
        # Import the queue manager
        from queue_manager import get_queue_manager
        print("✅ Queue manager imported successfully")
        
        # Get the queue manager instance
        manager = get_queue_manager()
        print(f"✅ Queue Manager initialized")
        print(f"📁 Queue file: {manager.queue_file}")
        print(f"📁 Results file: {manager.results_file}")
        
        # Validate files exist and are accessible
        for file_path in [manager.queue_file, manager.results_file]:
            if os.path.exists(file_path):
                print(f"✅ Found {file_path}")
            else:
                print(f"⚠️  {file_path} will be created")
        
        # Start background processing
        manager.start_background_processing()
        print("\n🔄 Background processing started!")
        print(f"📊 Status reports every {args.status_interval} seconds")
        print("Press Ctrl+C to stop gracefully\n")
        
        # Initialize monitoring variables
        consecutive_errors = 0
        last_status_time = time.time()
        start_time = time.time()
        
        # Main monitoring loop
        try:
            while not shutdown_requested:
                current_time = time.time()
                
                # Check if it's time for status report
                if current_time - last_status_time >= args.status_interval:
                    try:
                        # Get current status
                        status = manager.get_queue_status()
                        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        uptime = int(current_time - start_time)
                        
                        print(f"\n[{current_time_str}] System Status (Uptime: {uptime}s):")
                        print(f"  📋 Pending: {status['total_pending']}")
                        print(f"  🔄 Processing: {status['total_processing']}")
                        print(f"  ✅ Completed: {status['total_completed']}")
                        
                        # Show currently processing items
                        processing_items = status.get('processing_items', [])
                        if processing_items:
                            for item in processing_items:
                                ticker = item.get('ticker', 'Unknown')
                                started_at = item.get('started_at', '')
                                if started_at:
                                    try:
                                        start_time_obj = datetime.fromisoformat(started_at)
                                        elapsed = datetime.now() - start_time_obj
                                        print(f"  🔥 Processing: {ticker} (elapsed: {elapsed})")
                                    except:
                                        print(f"  🔥 Processing: {ticker}")
                        
                        # Show next pending items
                        pending_items = status.get('pending_items', [])
                        if pending_items:
                            next_items = pending_items[:3]  # Show next 3
                            tickers = [item.get('ticker', 'Unknown') for item in next_items]
                            print(f"  ⏳ Next in queue: {', '.join(tickers)}")
                            
                            if len(pending_items) > 3:
                                print(f"     ... and {len(pending_items) - 3} more")
                        
                        # System health check
                        health = status.get('queue_health', 'unknown')
                        if health == 'healthy':
                            print("  💚 System health: GOOD")
                        elif health == 'error':
                            print("  ❤️  System health: ERROR")
                            consecutive_errors += 1
                        else:
                            print("  💛 System health: UNKNOWN")
                        
                        # Check if processing is still active
                        if not manager.is_processing:
                            print("  ⚠️  WARNING: Background processing has stopped!")
                            consecutive_errors += 1
                        else:
                            consecutive_errors = 0  # Reset error count on success
                        
                        last_status_time = current_time
                        
                    except Exception as e:
                        consecutive_errors += 1
                        print(f"  ❌ Error getting status: {str(e)}")
                
                # Check for too many consecutive errors
                if consecutive_errors >= args.max_errors:
                    print(f"\n💥 Too many consecutive errors ({consecutive_errors}). Stopping...")
                    break
                
                # Sleep for a short interval
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received")
        
        finally:
            print("\n🛑 Stopping background processor...")
            manager.stop_background_processing()
            
            # Final status report
            try:
                final_status = manager.get_queue_status()
                print(f"📊 Final Status:")
                print(f"  📋 Pending: {final_status['total_pending']}")
                print(f"  ✅ Completed: {final_status['total_completed']}")
                print(f"  ⏱️  Total uptime: {int(time.time() - start_time)} seconds")
            except:
                pass
            
            print("✅ Background processor stopped successfully")
            
    except ImportError as e:
        print("❌ Error: Could not import queue_manager.py")
        print("   Make sure queue_manager.py is in the same directory as this script")
        print(f"   Import error: {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Critical error starting background processor: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        sys.exit(1)

def check_system_status():
    """Quick system status check without starting processor"""
    try:
        from queue_manager import get_queue_manager
        
        manager = get_queue_manager()
        status = manager.get_queue_status()
        
        print("📊 Current System Status:")
        print(f"  📋 Pending: {status['total_pending']}")
        print(f"  🔄 Processing: {status['total_processing']}")
        print(f"  ✅ Completed: {status['total_completed']}")
        print(f"  💚 Health: {status.get('queue_health', 'unknown')}")
        print(f"  🔄 Background processing: {'Active' if manager.is_processing else 'Stopped'}")
        
    except Exception as e:
        print(f"❌ Error checking status: {str(e)}")

if __name__ == "__main__":
    # Check if user wants status only
    if len(sys.argv) > 1 and sys.argv[1] == '--status-only':
        check_system_status()
    else:
        main()