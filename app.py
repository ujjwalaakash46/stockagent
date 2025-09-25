# Enhanced Streamlit App with Better Queue Integration
# ===================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import time
import traceback
import json

# Enhanced queue manager import with better error handling
QUEUE_AVAILABLE = False
try:
    from queue_manager import (
        initialize_queue_system, 
        add_stock_to_queue, 
        get_queue_info, 
        get_all_results,
        find_result_by_ticker,
        get_queue_manager
    )
    QUEUE_AVAILABLE = True
    st.success("Queue manager loaded successfully!")
except ImportError as e:
    st.error(f"Queue manager not available: {str(e)}")
    st.error("Please ensure queue_manager.py is in the same directory.")
except Exception as e:
    st.error(f"Unexpected error loading queue manager: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="AI Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize queue system
if QUEUE_AVAILABLE and 'queue_initialized' not in st.session_state:
    try:
        with st.spinner("Initializing queue system..."):
            initialize_queue_system()
            st.session_state.queue_initialized = True
            st.session_state.last_refresh = datetime.now()
    except Exception as e:
        st.error(f"Failed to initialize queue system: {str(e)}")
        st.error("Please check your configuration and restart the application.")

# Header
st.title("🤖 AI-Powered Stock Analysis Pro")
st.markdown("**Advanced Multi-Agent Analysis for Indian Stocks**")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Auto-refresh settings
    auto_refresh_enabled = st.checkbox(
        "🔄 Auto-refresh", 
        value=False, 
        help="Automatically refresh queue status every 30 seconds"
    )
    
    refresh_interval = st.select_slider(
        "Refresh interval (seconds)",
        options=[10, 20, 30, 60, 120],
        value=30
    )
    
    st.markdown("---")
    
    # Queue statistics (if available)
    if QUEUE_AVAILABLE:
        st.header("📊 Quick Stats")
        try:
            status = get_queue_info()
            st.metric("Pending", status.get("total_pending", 0))
            st.metric("Processing", status.get("total_processing", 0))
            st.metric("Completed", status.get("total_completed", 0))
            
            # Health indicator
            health = status.get("queue_health", "unknown")
            if health == "healthy":
                st.success("System healthy")
            elif health == "error":
                st.error("System error detected")
            else:
                st.warning("System status unknown")
                
        except Exception as e:
            st.error(f"Error getting stats: {str(e)}")
    
    st.markdown("---")
    
    # Manual actions
    st.header("🔧 Actions")
    if st.button("🔄 Force Refresh", type="secondary"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# Auto-refresh logic
if auto_refresh_enabled:
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
    if time_since_refresh >= refresh_interval:
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # Show countdown
    remaining = refresh_interval - time_since_refresh
    st.sidebar.text(f"Next refresh in: {remaining}s")

# Main content area
if not QUEUE_AVAILABLE:
    st.error("❌ Queue system is not available. Please check your installation.")
    st.stop()

# Layout: Two main columns
col1, col2 = st.columns([1, 2])

# Left column: Input and Queue Status
with col1:
    st.header("📋 Add Analysis Request")
    
    # Enhanced input form
    with st.form("enhanced_stock_form", clear_on_submit=False):
        st.subheader("Stock Information")
        
        # Ticker input with validation
        ticker = st.text_input(
            "Stock Ticker Symbol *",
            placeholder="e.g., RELIANCE, TCS, HDFCBANK",
            help="Enter NSE ticker symbol (without .NS suffix)"
        ).upper().strip()
        
        # Company name input
        company_name = st.text_input(
            "Company Name (Optional)",
            placeholder="e.g., Reliance Industries Ltd",
            help="Full company name for better analysis context"
        ).strip()
        
        # Priority selection
        priority = st.selectbox(
            "Analysis Priority",
            options=["Normal", "High", "Low"],
            index=0,
            help="High priority requests are processed first"
        )
        
        # Submit button
        col_submit, col_reset = st.columns(2)
        with col_submit:
            submitted = st.form_submit_button("🚀 Add to Queue", type="primary")
        with col_reset:
            reset = st.form_submit_button("🗑️ Reset Form")
    
    # Handle form submission
    if submitted:
        if not ticker:
            st.error("⚠️ Please enter a stock ticker symbol")
        else:
            try:
                with st.spinner(f"Adding {ticker} to analysis queue..."):
                    request_id = add_stock_to_queue(
                        ticker=ticker, 
                        company_name=company_name or ticker
                    )
                
                st.success(f"✅ Successfully added {ticker} to queue!")
                st.info(f"🆔 Request ID: {request_id[:8]}...")
                
                # Auto-refresh after successful submission
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to add to queue: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
    
    if reset:
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced Queue Status
    st.header("📊 Queue Status")
    
    try:
        with st.spinner("Loading queue status..."):
            queue_status = get_queue_info()
        
        # Status metrics with better styling
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            st.metric(
                label="⏳ Pending",
                value=queue_status.get("total_pending", 0),
                delta=None
            )
        
        with metric_cols[1]:
            st.metric(
                label="🔄 Processing", 
                value=queue_status.get("total_processing", 0),
                delta=None
            )
        
        with metric_cols[2]:
            st.metric(
                label="✅ Completed",
                value=queue_status.get("total_completed", 0),
                delta=None
            )
        
        # Show pending items
        pending_items = queue_status.get("pending_items", [])
        if pending_items:
            st.subheader("⏳ Pending Analysis")
            for i, item in enumerate(pending_items[:5]):  # Show top 5
                ticker = item.get("ticker", "Unknown")
                position = item.get("position_in_queue", i + 1)
                created_time = datetime.fromisoformat(
                    item.get("created_at", datetime.now().isoformat())
                ).strftime("%H:%M")
                
                st.text(f"{position}. {ticker} - Added at {created_time}")
            
            if len(pending_items) > 5:
                st.text(f"... and {len(pending_items) - 5} more")
        
        # Show processing items
        processing_items = queue_status.get("processing_items", [])
        if processing_items:
            st.subheader("🔄 Currently Processing")
            for item in processing_items:
                ticker = item.get("ticker", "Unknown")
                started_time = datetime.fromisoformat(
                    item.get("started_at", datetime.now().isoformat())
                ).strftime("%H:%M")
                st.text(f"🔥 {ticker} - Started at {started_time}")
        
        # Last updated info
        last_updated = queue_status.get("last_updated")
        if last_updated:
            update_time = datetime.fromisoformat(last_updated).strftime("%H:%M:%S")
            st.caption(f"Last updated: {update_time}")
    
    except Exception as e:
        st.error(f"❌ Error loading queue status: {str(e)}")
        with st.expander("🔍 Debug Information"):
            st.code(traceback.format_exc())

# Right column: Results and Management
with col2:
    st.header("📈 Analysis Results & Management")
    
    # Enhanced tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Recent Results", 
        "🔍 Search Results", 
        "⚙️ Queue Management",
        "📊 Analytics Dashboard"
    ])
    
    with tab1:
        st.subheader("📋 Recent Analysis Results")
        
        try:
            # Add result limit selector
            col_limit, col_refresh = st.columns([2, 1])
            with col_limit:
                result_limit = st.selectbox(
                    "Show results:",
                    [5, 10, 20, 50],
                    index=1,
                    key="result_limit"
                )
            with col_refresh:
                if st.button("🔄 Refresh Results"):
                    st.rerun()
            
            with st.spinner("Loading recent results..."):
                results = get_all_results(limit=result_limit)
            
            if results:
                st.write(f"Showing {len(results)} most recent results:")
                
                for result in results:
                    ticker = result.get("ticker", "Unknown")
                    success = result.get("success", False)
                    completed_at = result.get("completed_at", "")
                    
                    # Format completion time
                    try:
                        completed_time = datetime.fromisoformat(completed_at)
                        time_str = completed_time.strftime("%m/%d %H:%M")
                    except:
                        time_str = "Unknown"
                    
                    # Create expander with status indicator
                    status_icon = "✅" if success else "❌"
                    expander_title = f"{status_icon} {ticker} - {time_str}"
                    
                    with st.expander(expander_title):
                        col_info, col_result = st.columns([1, 3])
                        
                        with col_info:
                            st.write(f"**Ticker:** {ticker}")
                            st.write(f"**Company:** {result.get('company_name', 'N/A')}")
                            st.write(f"**Status:** {result.get('status', 'Unknown')}")
                            
                            # Timing information
                            created_at = result.get("created_at")
                            if created_at and completed_at:
                                try:
                                    created = datetime.fromisoformat(created_at)
                                    completed = datetime.fromisoformat(completed_at)
                                    duration = completed - created
                                    st.write(f"**Duration:** {duration}")
                                except:
                                    st.write("**Duration:** Unknown")
                            
                            # Execution time if available
                            exec_time = result.get("execution_time")
                            if exec_time:
                                st.write(f"**Exec Time:** {exec_time:.1f}s")
                        
                        with col_result:
                            if success and result.get('analysis_result'):
                                analysis = result.get('analysis_result')
                                if isinstance(analysis, dict) and 'analysis_result' in analysis:
                                    st.markdown("**Analysis Summary:**")
                                    analysis_text = str(analysis['analysis_result'])
                                    # Show first 400 characters with expand option
                                    if len(analysis_text) > 400:
                                        st.write(analysis_text[:400] + "...")
                                        with st.expander("Show Full Analysis"):
                                            st.markdown(analysis_text)
                                    else:
                                        st.markdown(analysis_text)
                                else:
                                    st.markdown("**Analysis Summary:**")
                                    st.write(str(analysis)[:400] + "..." if len(str(analysis)) > 400 else str(analysis))
                            else:
                                error_msg = result.get('error', 'Unknown error occurred')
                                st.error(f"Analysis failed: {error_msg}")
            else:
                st.info("No completed analyses yet. Add stocks to the queue to get started!")
        
        except Exception as e:
            st.error(f"Error loading results: {str(e)}")
            with st.expander("Debug Information"):
                st.code(traceback.format_exc())
    
    with tab2:
        st.subheader("🔍 Search for Stock Results")
        
        # Enhanced search
        col_search, col_filter = st.columns([2, 1])
        
        with col_search:
            search_ticker = st.text_input(
                "Enter ticker to search:",
                placeholder="e.g., RELIANCE, TCS",
                key="search_ticker"
            ).upper().strip()
        
        with col_filter:
            search_filter = st.selectbox(
                "Filter by:",
                ["All", "Successful Only", "Failed Only"],
                key="search_filter"
            )
        
        if search_ticker:
            try:
                with st.spinner(f"Searching for {search_ticker}..."):
                    result = find_result_by_ticker(search_ticker)
                
                if result:
                    st.success(f"Found result for {search_ticker}")
                    
                    # Enhanced result display
                    col_meta, col_analysis = st.columns([1, 3])
                    
                    with col_meta:
                        st.write(f"**Company:** {result.get('company_name', 'N/A')}")
                        st.write(f"**Status:** {result.get('status', 'Unknown')}")
                        
                        completed_at = result.get('completed_at')
                        if completed_at:
                            try:
                                completed_time = datetime.fromisoformat(completed_at)
                                st.write(f"**Completed:** {completed_time.strftime('%Y-%m-%d %H:%M')}")
                            except:
                                st.write(f"**Completed:** {completed_at}")
                        
                        success = result.get('success', False)
                        st.write(f"**Success:** {'Yes' if success else 'No'}")
                        
                        exec_time = result.get('execution_time')
                        if exec_time:
                            st.write(f"**Execution Time:** {exec_time:.1f}s")
                    
                    with col_analysis:
                        if result.get('success') and result.get('analysis_result'):
                            analysis = result.get('analysis_result')
                            if isinstance(analysis, dict) and 'analysis_result' in analysis:
                                st.markdown("**Full Analysis Result:**")
                                st.markdown(str(analysis['analysis_result']))
                            else:
                                st.markdown("**Full Analysis Result:**")
                                st.markdown(str(analysis))
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            st.error(f"Analysis failed: {error_msg}")
                            
                            # Show error details if available
                            if 'traceback' in result:
                                with st.expander("Error Details"):
                                    st.code(result['traceback'])
                
                else:
                    st.warning(f"No results found for {search_ticker}")
            
            except Exception as e:
                st.error(f"Error searching results: {str(e)}")
    
    with tab3:
        st.subheader("⚙️ Queue Management")
        
        # System status
        try:
            manager = get_queue_manager()
            
            col_status, col_controls = st.columns(2)
            
            with col_status:
                st.write("**System Status:**")
                if manager.is_processing:
                    st.success("🔄 Background processing is ACTIVE")
                else:
                    st.warning("⏸️ Background processing is STOPPED")
                
                # Show system health
                queue_status = get_queue_info()
                health = queue_status.get("queue_health", "unknown")
                if health == "healthy":
                    st.success("System health: GOOD")
                elif health == "error":
                    st.error("System health: ERROR")
                else:
                    st.warning("System health: UNKNOWN")
            
            with col_controls:
                st.write("**Controls:**")
                
                # Start/Stop processing
                if not manager.is_processing:
                    if st.button("▶️ Start Background Processing", type="primary"):
                        manager.start_background_processing()
                        st.success("Started background processing")
                        time.sleep(1)
                        st.rerun()
                else:
                    if st.button("⏹️ Stop Background Processing", type="secondary"):
                        manager.stop_background_processing()
                        st.success("Stopped background processing")
                        time.sleep(1)
                        st.rerun()
        
        except Exception as e:
            st.error(f"Error checking system status: {str(e)}")
        
        st.markdown("---")
        
        # Data management
        st.subheader("🗑️ Data Management")
        
        col_clear1, col_clear2 = st.columns(2)
        
        with col_clear1:
            if st.button("🗑️ Clear All Results", type="secondary"):
                try:
                    manager = get_queue_manager()
                    manager.clear_completed_results()
                    st.success("✅ Cleared all completed results")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing results: {str(e)}")
        
        with col_clear2:
            days_old = st.number_input(
                "Clear results older than (days):",
                min_value=1,
                value=7,
                key="days_old"
            )
            if st.button(f"🗑️ Clear Old Results ({days_old}d+)"):
                try:
                    manager = get_queue_manager()
                    manager.clear_completed_results(older_than_days=days_old)
                    st.success(f"✅ Cleared results older than {days_old} days")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing old results: {str(e)}")
        
        # File system info
        st.markdown("---")
        st.subheader("📁 File System Info")
        
        try:
            import os
            queue_file = "stock_queue.json"
            results_file = "stock_results.json"
            
            col_files = st.columns(2)
            
            with col_files[0]:
                if os.path.exists(queue_file):
                    size = os.path.getsize(queue_file)
                    st.write(f"**Queue file:** {size} bytes")
                else:
                    st.write("**Queue file:** Not found")
            
            with col_files[1]:
                if os.path.exists(results_file):
                    size = os.path.getsize(results_file)
                    st.write(f"**Results file:** {size} bytes")
                else:
                    st.write("**Results file:** Not found")
        
        except Exception as e:
            st.write(f"Error getting file info: {str(e)}")
    
    with tab4:
        st.subheader("📊 Analytics Dashboard")
        
        try:
            with st.spinner("Loading analytics..."):
                results = get_all_results()
            
            if results:
                # Success rate
                successful = len([r for r in results if r.get('success')])
                total = len(results)
                success_rate = (successful / total * 100) if total > 0 else 0
                
                col_metrics = st.columns(4)
                
                with col_metrics[0]:
                    st.metric("Total Analyses", total)
                
                with col_metrics[1]:
                    st.metric("Successful", successful)
                
                with col_metrics[2]:
                    st.metric("Failed", total - successful)
                
                with col_metrics[3]:
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Execution time analysis
                exec_times = [r.get('execution_time', 0) for r in results if r.get('execution_time')]
                if exec_times:
                    avg_time = sum(exec_times) / len(exec_times)
                    min_time = min(exec_times)
                    max_time = max(exec_times)
                    
                    st.subheader("⏱️ Execution Time Analysis")
                    
                    time_cols = st.columns(3)
                    with time_cols[0]:
                        st.metric("Average", f"{avg_time:.1f}s")
                    with time_cols[1]:
                        st.metric("Fastest", f"{min_time:.1f}s")
                    with time_cols[2]:
                        st.metric("Slowest", f"{max_time:.1f}s")
                
                # Most analyzed stocks
                ticker_counts = {}
                for result in results:
                    ticker = result.get('ticker', 'Unknown')
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
                
                if ticker_counts:
                    st.subheader("📈 Most Analyzed Stocks")
                    sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    for ticker, count in sorted_tickers:
                        st.write(f"**{ticker}:** {count} analyses")
            
            else:
                st.info("No analysis data available yet.")
        
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>AI Stock Analysis Pro - Multi-Agent Stock Analysis System</p>
        <p>Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
    """,
    unsafe_allow_html=True
)