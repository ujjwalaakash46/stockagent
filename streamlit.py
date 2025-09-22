# Enhanced Streamlit App with Persistent Queue System
# ====================================================
# This version integrates with the queue manager for persistent analysis queue

import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Import the queue manager
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
except ImportError:
    QUEUE_AVAILABLE = False
    st.error("❌ Queue manager not available. Please ensure queue_manager.py is in the same directory.")

# Page configuration
st.set_page_config(
    page_title="Stock AI Analyzer with Queue",
    page_icon="📈",
    layout="wide"
)

# Initialize queue system on app start
if QUEUE_AVAILABLE and 'queue_initialized' not in st.session_state:
    try:
        initialize_queue_system()
        st.session_state.queue_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize queue system: {str(e)}")

# Title and description
st.title("🤖 AI-Powered Stock Analysis with Persistent Queue")
st.markdown("Multi-agent analysis of Indian stocks with background processing and persistent results.")

# Auto-refresh checkbox
auto_refresh = st.checkbox("🔄 Auto-refresh queue status (every 30 seconds)", value=False)

if auto_refresh:
    # Auto-refresh every 30 seconds
    time.sleep(30)
    st.rerun()

# Main layout with columns
col1, col2 = st.columns([1, 2])

# Left column - Input and Queue Status
with col1:
    st.header("📋 Add to Analysis Queue")
    
    # Input form
    with st.form("stock_form"):
        company_name = st.text_input(
            "Company Name",
            placeholder="e.g., Reliance Industries"
        )
        
        ticker = st.text_input(
            "Stock Ticker",
            placeholder="e.g., RELIANCE"
        ).upper()
        
        submitted = st.form_submit_button("🚀 Add to Queue", type="primary")
    
    # Handle form submission
    if submitted and ticker and QUEUE_AVAILABLE:
        try:
            request_id = add_stock_to_queue(ticker, company_name)
            st.success(f"✅ Added {ticker} to analysis queue!")
            st.info(f"Request ID: {request_id[:8]}...")
            
            # Auto-refresh to show updated queue
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to add to queue: {str(e)}")
    
    elif submitted and not ticker:
        st.warning("⚠️ Please enter a stock ticker symbol.")
    
    # Queue Status Section
    if QUEUE_AVAILABLE:
        st.header("📊 Queue Status")
        
        try:
            queue_status = get_queue_info()
            
            # Queue metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Pending", queue_status["total_pending"])
            with col_b:
                st.metric("Processing", queue_status["total_processing"])
            with col_c:
                st.metric("Completed", queue_status["total_completed"])
            
            # Show pending items
            if queue_status["pending_items"]:
                st.subheader("⏳ Pending Analysis")
                for item in queue_status["pending_items"][:5]:  # Show first 5
                    position = item.get("position_in_queue", "?")
                    created_time = datetime.fromisoformat(item["created_at"]).strftime("%H:%M")
                    st.write(f"**{position}.** {item['ticker']} - Added at {created_time}")
            
            # Show processing items
            if queue_status["processing_items"]:
                st.subheader("🔄 Currently Processing")
                for item in queue_status["processing_items"]:
                    started_time = datetime.fromisoformat(item["started_at"]).strftime("%H:%M")
                    st.write(f"🔥 **{item['ticker']}** - Started at {started_time}")
            
            # Manual refresh button
            if st.button("🔄 Refresh Status"):
                st.rerun()
                
        except Exception as e:
            st.error(f"Error loading queue status: {str(e)}")

# Right column - Results
with col2:
    st.header("📈 Analysis Results")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Recent Results", "🔍 Search Results", "⚙️ Queue Management"])
    
    with tab1:
        if QUEUE_AVAILABLE:
            try:
                results = get_all_results(limit=10)  # Get last 10 results
                
                if results:
                    st.write(f"Showing {len(results)} most recent results:")
                    
                    for result in results:
                        with st.expander(
                            f"{'✅' if result['success'] else '❌'} {result['ticker']} - "
                            f"{datetime.fromisoformat(result['completed_at']).strftime('%m/%d %H:%M')}"
                        ):
                            col_x, col_y = st.columns([1, 3])
                            
                            with col_x:
                                st.write(f"**Ticker:** {result['ticker']}")
                                st.write(f"**Company:** {result['company_name']}")
                                st.write(f"**Status:** {result['status']}")
                                
                                # Timing information
                                created = datetime.fromisoformat(result['created_at'])
                                completed = datetime.fromisoformat(result['completed_at'])
                                duration = completed - created
                                st.write(f"**Duration:** {duration}")
                            
                            with col_y:
                                if result['success'] and result['analysis_result']:
                                    # Show analysis result
                                    analysis = result['analysis_result']
                                    if isinstance(analysis, dict) and 'analysis_result' in analysis:
                                        st.markdown("**Analysis:**")
                                        st.write(str(analysis['analysis_result'])[:500] + "...")
                                    else:
                                        st.markdown("**Analysis:**")
                                        st.write(str(analysis)[:500] + "...")
                                else:
                                    st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                else:
                    st.info("No completed analyses yet. Add stocks to the queue to get started!")
                    
            except Exception as e:
                st.error(f"Error loading results: {str(e)}")
    
    with tab2:
        st.subheader("🔍 Search for Stock Results")
        
        search_ticker = st.text_input("Enter ticker to search:", placeholder="e.g., RELIANCE").upper()
        
        if search_ticker and QUEUE_AVAILABLE:
            try:
                result = find_result_by_ticker(search_ticker)
                
                if result:
                    st.success(f"Found result for {search_ticker}")
                    
                    # Show result details
                    col_p, col_q = st.columns([1, 3])
                    
                    with col_p:
                        st.write(f"**Company:** {result['company_name']}")
                        st.write(f"**Completed:** {datetime.fromisoformat(result['completed_at']).strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Status:** {result['status']}")
                    
                    with col_q:
                        if result['success'] and result['analysis_result']:
                            analysis = result['analysis_result']
                            if isinstance(analysis, dict) and 'analysis_result' in analysis:
                                st.markdown("**Full Analysis Result:**")
                                st.markdown(str(analysis['analysis_result']))
                            else:
                                st.markdown("**Full Analysis Result:**")
                                st.markdown(str(analysis))
                        else:
                            st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                else:
                    st.warning(f"No results found for {search_ticker}")
                    
            except Exception as e:
                st.error(f"Error searching results: {str(e)}")
    
    with tab3:
        st.subheader("⚙️ Queue Management")
        
        if QUEUE_AVAILABLE:
            # Clear results button
            if st.button("🗑️ Clear All Completed Results", type="secondary"):
                try:
                    manager = get_queue_manager()
                    manager.clear_completed_results()
                    st.success("✅ Cleared all completed results")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing results: {str(e)}")
            
            # Clear old results
            days_old = st.number_input("Clear results older than (days):", min_value=1, value=7)
            if st.button(f"🗑️ Clear Results Older Than {days_old} Days"):
                try:
                    manager = get_queue_manager()
                    manager.clear_completed_results(older_than_days=days_old)
                    st.success(f"✅ Cleared results older than {days_old} days")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing old results: {str(e)}")
            
            # Show background processing status
            try:
                manager = get_queue_manager()
                if manager.is_processing:
                    st.success("🔄 Background processing is ACTIVE")
                else:
                    st.warning("⏸️ Background processing is STOPPED")
                    if st.button("▶️ Start Background Processing"):
                        manager.start_background_processing()
                        st.success("Started background processing")
                        st.rerun()
            except Exception as e:
                st.error(f"Error checking processing status: {str(e)}")
        
        else:
            st.error("Queue management not available - queue_manager.py not found")

# Footer information
st.markdown("---")
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("### 🛠️ How It Works")
    st.markdown("""
    1. **Add stocks** to the analysis queue
    2. **Background processing** runs continuously
    3. **Results persist** even when Streamlit is closed
    4. **Email notifications** sent when analysis completes
    5. **View results** anytime, from any session
    """)

with col_info2:
    st.markdown("### 📁 File Storage")
    st.markdown("""
    - `stock_queue.json` - Pending analysis requests
    - `stock_results.json` - Completed analysis results
    - Background processing continues independently
    - Data persists across browser sessions
    """)
