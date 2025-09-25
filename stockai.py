# -*- coding: utf-8 -*-
"""
Optimized Multi-Agent Stock Analysis System
==========================================
- Fixed data serialization issues
- Reduced from 6 agents to 4 agents
- Optimized data handling and caching
- Faster execution (target: 2-3 minutes vs 5+ minutes)
- Better error handling and reliability
"""

# STEP 1: IMPORTS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from textblob import TextBlob
import requests
import os
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# CrewAI Multi-Agent Framework
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel

# Custom Local Imports
from mailing import StockAIEmailNotifier
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Configure the LLM
gemini_llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.2  # Lower for more consistent analysis
)

print("Multi-Agent Environment Setup Complete!")

# STEP 2: OPTIMIZED TOOLS WITH FIXED DATA HANDLING

class OptimizedStockDataTool(BaseTool):
    name: str = "Optimized Indian Stock Data Fetcher"
    description: str = "Fetches comprehensive Indian stock data with optimized data handling"

    @lru_cache(maxsize=50)
    def _get_cached_data(self, symbol: str, cache_key: str):
        """Simple in-memory caching to avoid repeated API calls"""
        return None  # Cache implementation can be added later

    def _run(self, symbol: str, period: str = "6mo") -> dict:  # Reduced from 1y to 6mo
        """
        Optimized data fetching with better error handling
        """
        try:
            # FIX: Validate the 'period' parameter to prevent yfinance errors
            VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"]
            if period not in VALID_PERIODS:
                print(f"Warning: Invalid period '{period}' provided. Defaulting to '6mo'.")
                period = "6mo"

            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            print(f"Fetching data for {symbol} with period '{period}'...")
            
            # Create ticker and fetch data
            ticker = yf.Ticker(symbol)
            
            # First try to get basic info
            info = {}
            try:
                info = ticker.info
            except Exception as e:
                print(f"Warning: Could not fetch company info: {str(e)}")
            
            # Get historical data with multiple attempts
            attempts = 0
            max_attempts = 3
            hist_data = None
            
            while attempts < max_attempts and (hist_data is None or hist_data.empty):
                try:
                    hist_data = ticker.history(period=period, interval="1d")
                    if not hist_data.empty:
                        break
                except Exception as e:
                    print(f"Attempt {attempts + 1} failed: {str(e)}")
                attempts += 1
                time.sleep(2)  # Wait before retry
            
            if hist_data is None or hist_data.empty:
                raise ValueError(f"Could not fetch data for {symbol} after {max_attempts} attempts")
            
            if hist_data.empty:
                raise ValueError(f"No historical data found for {symbol}")

            # Calculate basic metrics
            current_price = float(hist_data['Close'].iloc[-1])
            
            # Moving averages (with error handling)
            ma_20 = float(hist_data['Close'].rolling(20).mean().iloc[-1]) if len(hist_data) >= 20 else current_price
            ma_50 = float(hist_data['Close'].rolling(50).mean().iloc[-1]) if len(hist_data) >= 50 else current_price
            
            # Volatility calculation
            returns = hist_data['Close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 1 else 0
            
            # Prepare historical data for technical analysis (FIXED FORMAT)
            hist_subset = hist_data.tail(60)  # Last 60 days
            
            # Convert to a format that works with technical analysis
            historical_data_processed = {
                'dates': hist_subset.index.strftime('%Y-%m-%d').tolist(),
                'open': hist_subset['Open'].tolist(),
                'high': hist_subset['High'].tolist(), 
                'low': hist_subset['Low'].tolist(),
                'close': hist_subset['Close'].tolist(),
                'volume': hist_subset['Volume'].tolist()
            }

            result = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol.replace('.NS', '')),
                'current_price': current_price,
                'currency': info.get('currency', 'INR'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'return_on_equity': info.get('returnOnEquity'),
                'profit_margin': info.get('profitMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'ma_20': ma_20,
                'ma_50': ma_50,
                'volatility': volatility,
                'current_volume': int(hist_data['Volume'].iloc[-1]),
                'avg_volume': int(hist_data['Volume'].mean()),
                'data_points': len(hist_data),
                'historical_data': historical_data_processed,  # FIXED FORMAT
                'fetched_at': datetime.now().isoformat()
            }
            
            print(f"Successfully fetched data for {symbol} ({len(hist_data)} data points)")
            return result
            
        except Exception as e:
            error_msg = f"Error fetching data for {symbol}: {str(e)}"
            print(error_msg)
            return {'error': error_msg, 'symbol': symbol}


class FastTechnicalAnalysisTool(BaseTool):
    name: str = "Fast Technical Analysis Calculator" 
    description: str = "Calculates technical indicators with optimized processing"

    def _run(self, symbol: str, historical_data: dict) -> dict:
        """
        Fast technical analysis with proper data handling
        """
        try:
            print(f"Calculating technical indicators for {symbol}...")
            
            # Handle the new data format
            if 'error' in historical_data:
                return {'error': f"No data available for technical analysis: {historical_data['error']}", 'symbol': symbol}
            
            # Extract price data from the processed format
            dates = historical_data.get('dates', [])
            closes = historical_data.get('close', [])
            highs = historical_data.get('high', [])
            lows = historical_data.get('low', [])
            volumes = historical_data.get('volume', [])
            
            if not closes or len(closes) < 20:
                return {'error': 'Insufficient data for technical analysis', 'symbol': symbol}
            
            # Convert to pandas Series for calculations
            close_series = pd.Series(closes)
            high_series = pd.Series(highs)
            low_series = pd.Series(lows)
            volume_series = pd.Series(volumes)
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(close_series, high_series, low_series, volume_series)
            
            # Generate trading signals
            signals = self._generate_signals(indicators, close_series.iloc[-1])
            
            result = {
                'symbol': symbol,
                'current_price': float(close_series.iloc[-1]),
                'indicators': indicators,
                'signals': signals,
                'data_points_analyzed': len(closes),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            print(f"Technical analysis complete for {symbol}")
            return result
            
        except Exception as e:
            error_msg = f"Technical analysis error for {symbol}: {str(e)}"
            print(error_msg)
            return {'error': error_msg, 'symbol': symbol}
    
    def _calculate_indicators(self, closes, highs, lows, volumes):
        """Calculate all technical indicators"""
        
        # RSI (14-period)
        rsi = self._calculate_rsi(closes, 14)
        
        # Moving Averages
        ma_20 = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else closes.iloc[-1]
        ma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else closes.iloc[-1]
        
        # MACD
        exp1 = closes.ewm(span=12).mean()
        exp2 = closes.ewm(span=26).mean()
        macd_line = exp1 - exp2
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal
        
        # Bollinger Bands
        sma_20 = closes.rolling(20).mean()
        std_20 = closes.rolling(20).std()
        bb_upper = sma_20 + (2 * std_20)
        bb_lower = sma_20 - (2 * std_20)
        
        # Support and Resistance
        support = lows.tail(30).min() if len(lows) >= 30 else lows.min()
        resistance = highs.tail(30).max() if len(highs) >= 30 else highs.max()
        
        # Volume analysis
        avg_volume = volumes.mean()
        volume_ratio = volumes.iloc[-1] / avg_volume if avg_volume > 0 else 1
        
        return {
            'rsi': float(rsi),
            'ma_20': float(ma_20),
            'ma_50': float(ma_50), 
            'macd_line': float(macd_line.iloc[-1]),
            'macd_signal': float(macd_signal.iloc[-1]),
            'macd_histogram': float(macd_histogram.iloc[-1]),
            'bb_upper': float(bb_upper.iloc[-1]) if len(bb_upper) > 0 and pd.notna(bb_upper.iloc[-1]) else None,
            'bb_lower': float(bb_lower.iloc[-1]) if len(bb_lower) > 0 and pd.notna(bb_lower.iloc[-1]) else None,
            'support_level': float(support),
            'resistance_level': float(resistance),
            'volume_ratio': float(volume_ratio)
        }
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        if loss.iloc[-1] == 0:
            return 100 # Avoid division by zero if there are no losses
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 and pd.notna(rsi.iloc[-1]) else 50  # Default to neutral
    
    def _generate_signals(self, indicators, current_price):
        """Generate buy/sell/hold signals"""
        signals = []
        signal_score = 0
        
        # RSI signals
        rsi = indicators['rsi']
        if rsi < 30:
            signals.append("RSI oversold - potential buy signal")
            signal_score += 1
        elif rsi > 70:
            signals.append("RSI overbought - potential sell signal") 
            signal_score -= 1
        else:
            signals.append(f"RSI neutral at {rsi:.1f}")
        
        # Moving average signals
        ma_20 = indicators['ma_20']
        ma_50 = indicators['ma_50']
        
        if current_price > ma_20 > ma_50:
            signals.append("Price above both MAs - bullish trend")
            signal_score += 1
        elif current_price < ma_20 < ma_50:
            signals.append("Price below both MAs - bearish trend")
            signal_score -= 1
        
        # MACD signals
        if indicators['macd_histogram'] > 0:
            signals.append("MACD bullish")
            signal_score += 0.5
        else:
            signals.append("MACD bearish")
            signal_score -= 0.5
        
        # Volume signals
        volume_ratio = indicators['volume_ratio']
        if volume_ratio > 1.5:
            signals.append("High volume - strong interest")
            signal_score += 0.5
        elif volume_ratio < 0.5:
            signals.append("Low volume - weak interest")
        
        # Overall signal
        if signal_score >= 2:
            overall = "BUY"
        elif signal_score <= -2:
            overall = "SELL"
        else:
            overall = "HOLD"
        
        return {
            'individual_signals': signals,
            'overall_signal': overall,
            'signal_strength': float(signal_score),
            'confidence': min(abs(signal_score) / 3.0, 1.0)  # 0-1 scale
        }


class OptimizedNewsSentimentTool(BaseTool):
    name: str = "Optimized News Sentiment Analyzer"
    description: str = "Fast news sentiment analysis with timeout handling"

    def _run(self, symbol: str, company_name: str) -> dict:
        """
        Optimized news sentiment analysis with timeout
        """
        if not NEWS_API_KEY:
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'overall_sentiment': 'Neutral',
                'articles_analyzed': 0,
                'error': 'NEWS_API_KEY not configured'
            }
        
        try:
            print(f"Fetching news sentiment for {company_name}...")
            
            # Clean company name and symbol for search
            symbol_clean = symbol.replace(".NS", "").strip()
            company_clean = company_name.replace(' Ltd', '').replace(' Limited', '').replace('.NS', '').strip()
            
            # MODIFICATION: Simplified query for broader results
            query = f'"{company_clean}" OR "{symbol_clean}"'
            
            # MODIFICATION: Added a date range and commented out the restrictive 'domains' parameter
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # Use the News API endpoint with better parameters
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 20, # Increased page size slightly
                "apiKey": NEWS_API_KEY,
                "from": from_date,
                "searchIn": "title,description",
                # FIX: Commenting out 'domains' to broaden the search. This is the main reason for 0 articles.
                # "domains": "moneycontrol.com,economictimes.indiatimes.com,ndtv.com,livemint.com,business-standard.com"
            }
            
            # Make request with proper error handling
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            
            if not articles:
                print(f"No news articles found for query: {query}")
                return {
                    'symbol': symbol,
                    'sentiment_score': 0.0,
                    'overall_sentiment': 'Neutral',
                    'articles_analyzed': 0
                }
            
            # Analyze sentiment
            sentiment_scores = []
            for article in articles[:15]:  # Limit to first 15 for speed
                if article.get('title'):
                    try:
                        blob = TextBlob(article['title'])
                        sentiment_scores.append(blob.sentiment.polarity)
                    except:
                        continue
            
            if not sentiment_scores:
                avg_sentiment = 0.0
            else:
                avg_sentiment = np.mean(sentiment_scores)
            
            # Classify sentiment
            if avg_sentiment > 0.15:
                overall_sentiment = 'Bullish'
            elif avg_sentiment < -0.15:
                overall_sentiment = 'Bearish'
            else:
                overall_sentiment = 'Neutral'
            
            result = {
                'symbol': symbol,
                'sentiment_score': float(avg_sentiment),
                'overall_sentiment': overall_sentiment,
                'articles_analyzed': len(sentiment_scores),
                'confidence': min(abs(avg_sentiment) * 2, 1.0),  # 0-1 scale
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            print(f"Sentiment analysis complete for {symbol}: {overall_sentiment} ({len(sentiment_scores)} articles)")
            return result
            
        except requests.Timeout:
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'overall_sentiment': 'Neutral',
                'articles_analyzed': 0,
                'error': 'News API timeout'
            }
        except Exception as e:
            error_msg = f"Sentiment analysis error for {symbol}: {str(e)}"
            print(error_msg)
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'overall_sentiment': 'Neutral', 
                'articles_analyzed': 0,
                'error': error_msg
            }


# STEP 3: OPTIMIZED AGENTS (4 instead of 6)

# Agent 1: Data Collection
data_collector_agent = Agent(
    role='Senior Data Collection Specialist',
    goal='Efficiently fetch comprehensive Indian stock market data',
    backstory="Expert at sourcing and validating financial data for Indian markets with focus on speed and accuracy.",
    tools=[OptimizedStockDataTool()],
    verbose=True,
    llm=gemini_llm
)

# Agent 2: Technical Analyst  
technical_analyst_agent = Agent(
    role='Senior Technical Analysis Expert',
    goal='Perform fast and comprehensive technical analysis with clear trading signals',
    backstory="Experienced technical analyst who quickly identifies trends, patterns and generates actionable trading signals for Indian stocks.",
    tools=[FastTechnicalAnalysisTool()],
    verbose=True,
    llm=gemini_llm
)

# Agent 3: Market Sentiment Analyst
sentiment_analyst_agent = Agent(
    role='Market Sentiment and Risk Specialist', 
    goal='Analyze news sentiment and assess market risks efficiently',
    backstory="Expert in behavioral finance who rapidly processes news sentiment and market psychology to assess investment risks.",
    tools=[OptimizedNewsSentimentTool()],
    verbose=True,
    llm=gemini_llm
)

# Agent 4: Investment Strategist
investment_strategist_agent = Agent(
    role='Chief Investment Strategist',
    goal='Synthesize all analysis into clear, actionable investment recommendations',
    backstory="Top-tier strategist who combines technical, fundamental, and sentiment analysis into precise investment decisions with specific targets and risk management.",
    verbose=True,
    llm=gemini_llm
)

print("4 Optimized AI Agents Created Successfully!")

# STEP 4: STREAMLINED TASKS

def create_optimized_analysis_tasks(symbol, company_name=None):
    """Create optimized task sequence"""
    
    company_name = company_name or symbol.replace('.NS', '')
    
    # Task 1: Data Collection (30-45 seconds)
    data_task = Task(
        description=f"""
        Collect comprehensive stock data for {symbol} ({company_name}):
        - Use the Optimized Indian Stock Data Fetcher tool.
        - Fetch data for a period of '6mo' to get sufficient historical context.
        - Ensure data quality and completeness for analysis.
        
        Focus on efficiency and accuracy.
        """,
        agent=data_collector_agent,
        expected_output="A dictionary containing complete stock data, including fundamental metrics and historical prices in a processed format for the last 6 months."
    )
    
    # Task 2: Technical Analysis (45-60 seconds)
    technical_task = Task(
        description=f"""
        Perform comprehensive technical analysis for {symbol}:
        - Use the Fast Technical Analysis Calculator with the historical data from the previous step.
        - Calculate key indicators like RSI, MACD, Moving Averages, and Bollinger Bands.
        - Identify important support/resistance levels and analyze volume patterns.
        - Generate clear BUY/HOLD/SELL signals with associated confidence levels.
        
        Provide specific trading signals and a clear technical outlook.
        """,
        agent=technical_analyst_agent,
        expected_output="A dictionary of technical analysis results, including indicator values, calculated signals, and trading recommendations.",
        context=[data_task]
    )
    
    # Task 3: Sentiment Analysis (30-45 seconds)  
    sentiment_task = Task(
        description=f"""
        Analyze the latest market sentiment for {symbol} ({company_name}):
        - Use the Optimized News Sentiment Analyzer tool.
        - Process recent news and headlines to gauge public opinion.
        - Determine the overall market sentiment (Bullish/Bearish/Neutral).
        - Assess how the current sentiment might impact the stock's performance.
        
        Focus on recent news (last 7 days) and the general market mood.
        """,
        agent=sentiment_analyst_agent,
        expected_output="A dictionary with sentiment analysis results, including the overall sentiment classification and an assessment of its potential impact.",
        context=[data_task]
    )
    
    # Task 4: Final Recommendation (60-90 seconds)
    recommendation_task = Task(
        description=f"""
        Create a final, consolidated investment recommendation for {symbol} ({company_name}).
        
        Synthesize all the information gathered from the previous analyses:
        - The stock's fundamental data and key metrics.
        - The signals and trends from the technical analysis.
        - The impact of market sentiment and recent news.
        
        Provide a structured, actionable output with the following sections:
        1. Final Recommendation: A clear verdict (e.g., STRONG BUY, HOLD, SPECULATIVE SELL).
        2. Price Target: A realistic target price for a 3-6 month outlook.
        3. Stop-Loss: A suggested price level to exit the position to manage risk.
        4. Confidence Score: A rating from 1 to 10 on the strength of this recommendation.
        5. Summary & Rationale: A brief paragraph explaining the key reasons (from fundamental, technical, and sentiment analysis) that support your recommendation.
        6. Risk Assessment: Briefly mention potential risks or headwinds.
        
        Format the output as a clean, easy-to-read investment report.
        """,
        agent=investment_strategist_agent,
        expected_output="A final, structured investment recommendation report containing a clear verdict, price target, stop-loss, and a detailed rationale.",
        context=[data_task, technical_task, sentiment_task]
    )
    
    return [data_task, technical_task, sentiment_task, recommendation_task]


# STEP 5: OPTIMIZED EXECUTION ENGINE

def analyze_stock_with_optimized_agents(symbol, company_name=None):
    """
    Optimized multi-agent stock analysis - Target: 2-3 minutes
    """
    start_time = time.time()
    
    print(f"\nStarting Optimized Multi-Agent Analysis for {symbol}")
    print("=" * 60)
    
    try:
        # Create tasks
        tasks = create_optimized_analysis_tasks(symbol, company_name)
        
        # Create optimized crew
        analysis_crew = Crew(
            agents=[
                data_collector_agent,
                technical_analyst_agent, 
                sentiment_analyst_agent,
                investment_strategist_agent
            ],
            tasks=tasks,
            llm=gemini_llm,
            process=Process.sequential,
            verbose=True,
            max_execution_time=300  # 5 minute timeout
        )
        
        print(f"Deploying {len(analysis_crew.agents)} optimized agents...")
        
        # Execute analysis
        result = analysis_crew.kickoff()
        
        execution_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("OPTIMIZED MULTI-AGENT ANALYSIS COMPLETE!")
        print(f"Execution Time: {execution_time:.1f} seconds")
        print("=" * 60)
        print(result)
        
        # Prepare structured result
        final_result = {
            'symbol': symbol,
            'company_name': company_name,
            'analysis_result': result,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'agents_used': len(analysis_crew.agents),
            'tasks_completed': len(tasks),
            'optimization_version': '2.1-fixed'
        }
        
        # Send email notification
        try:
            email_notifier = StockAIEmailNotifier()
            email_notifier.send_email(final_result, symbol)
            print("Email notification sent successfully!")
        except Exception as e:
            print(f"Email notification failed: {str(e)}")
        
        return final_result
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Analysis failed for {symbol}: {str(e)}"
        print(error_msg)
        
        return {
            'symbol': symbol,
            'error': error_msg,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'optimization_version': '2.1-fixed'
        }


# STEP 6: BATCH ANALYSIS FOR MULTIPLE STOCKS

def analyze_multiple_stocks(stock_list, delay_seconds=15):
    """
    Analyze multiple stocks with rate limiting
    stock_list: [('SYMBOL', 'Company Name'), ...]
    """
    results = []
    
    print(f"Starting batch analysis of {len(stock_list)} stocks...")
    
    for i, (symbol, company_name) in enumerate(stock_list, 1):
        print(f"\n[{i}/{len(stock_list)}] Analyzing {symbol}...")
        
        result = analyze_stock_with_optimized_agents(symbol, company_name)
        results.append(result)
        
        # Rate limiting between stocks
        if i < len(stock_list):
            print(f"Waiting {delay_seconds} seconds before next analysis...")
            time.sleep(delay_seconds)
    
    return results


# EXAMPLE USAGE AND TESTING

# if __name__ == '__main__':
#     print("Optimized Multi-Agent Stock Analysis System Ready!")
    
#     # Test with a single stock
#     result = analyze_stock_with_optimized_agents("RELIANCE", "Reliance Industries")
    
#     # Test with multiple stocks
#     # sample_stocks = [
#     #     ("HDFCBANK", "HDFC Bank Ltd"),
#     #     ("TCS", "Tata Consultancy Services"), 
#     #     ("INFY", "Infosys Ltd")
#     # ]
#     # batch_results = analyze_multiple_stocks(sample_stocks, delay_seconds=10)
    
#     pass