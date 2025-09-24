# -*- coding: utf-8 -*-
# 🔸 STEP 1: PROPER IMPORTS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
from textblob import TextBlob
import requests
import os

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

# Configure the LLM (ensure GOOGLE_API_KEY is set in your .env file)
gemini_llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.3 # Lower temperature for more factual financial analysis
)

print("🚀 Multi-Agent Environment Setup Complete!")

# 🔸 STEP 2: CUSTOM TOOLS (UNCHANGED)

class YFinanceIndianStockTool(BaseTool):
    name: str = "Indian Stock Data Fetcher"
    description: str = "Fetches comprehensive Indian stock data using yfinance with .NS suffix"

    def _run(self, symbol: str, period: str = "1y") -> dict:
        """
        Fetch Indian stock data with proper .NS suffix
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        """
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            print(f"📊 Fetching data for {symbol}...")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist_data = ticker.history(period=period)
            if hist_data.empty:
                raise ValueError("No historical data found.")

            current_price = hist_data['Close'].iloc[-1]
            ma_20 = hist_data['Close'].rolling(20).mean().iloc[-1] if len(hist_data) >= 20 else None
            volatility = hist_data['Close'].pct_change().std() * np.sqrt(252) * 100 if len(hist_data) > 1 else 0

            result = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'current_price': float(current_price),
                'currency': info.get('currency', 'INR'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'return_on_equity': info.get('returnOnEquity'),
                'profit_margin': info.get('profitMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'ma_20': float(ma_20) if ma_20 else None,
                'volatility': float(volatility),
                'volume': int(hist_data['Volume'].iloc[-1]),
                'avg_volume': int(hist_data['Volume'].mean()),
                'historical_data': hist_data.tail(60).to_dict(),
                'fetched_at': datetime.now().isoformat()
            }
            print(f"✅ Successfully fetched data for {symbol}")
            return result
        except Exception as e:
            error_msg = f"❌ Error fetching data for {symbol}: {str(e)}"
            print(error_msg)
            return {'error': error_msg, 'symbol': symbol}

class TechnicalAnalysisTool(BaseTool):
    name: str = "Technical Analysis Calculator"
    description: str = "Calculates comprehensive technical indicators for stock analysis"

    def _run(self, symbol: str, historical_data: dict) -> dict:
        try:
            print(f"🔧 Calculating technical indicators for {symbol}...")
            df = pd.DataFrame(historical_data)
            if len(df) < 50: # Ensure enough data for 50-day MA
                return {'error': 'Insufficient data for comprehensive technical analysis'}

            closes = df['Close']
            # RSI
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            # MACD
            exp1 = closes.ewm(span=12, adjust=False).mean()
            exp2 = closes.ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            result = {
                'symbol': symbol,
                'rsi': float(rsi.iloc[-1]),
                'ma_20': float(closes.rolling(20).mean().iloc[-1]),
                'ma_50': float(closes.rolling(50).mean().iloc[-1]),
                'macd_line': float(macd_line.iloc[-1]),
                'macd_signal_line': float(signal_line.iloc[-1]),
                'support_level': float(df['Low'].tail(30).min()),
                'resistance_level': float(df['High'].tail(30).max()),
            }
            print(f"✅ Technical analysis complete for {symbol}")
            return result
        except Exception as e:
            error_msg = f"❌ Technical analysis error for {symbol}: {str(e)}"
            print(error_msg)
            return {'error': error_msg, 'symbol': symbol}

class NewsSentimentTool(BaseTool):
    name: str = "Live News Sentiment Analyzer"
    description: str = "Fetches and analyzes real-time news sentiment for Indian stocks using NewsAPI."

    def _run(self, symbol: str, company_name: str) -> dict:
        if not NEWS_API_KEY:
            return {'error': 'NEWS_API_KEY is not configured.'}
        
        print(f"📰 Fetching and analyzing live news for {company_name}...")
        query = f'"{company_name}" OR "{symbol.replace(".NS", "")}"'
        url = (f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=20&apiKey={NEWS_API_KEY}")

        try:
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            if not articles:
                return {'average_sentiment_score': 0.0, 'overall_sentiment': 'Neutral', 'articles_analyzed': 0}

            sentiment_scores = [TextBlob(a['title']).sentiment.polarity for a in articles if a['title']]
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0

            if avg_sentiment > 0.15: overall_sentiment = 'Bullish'
            elif avg_sentiment < -0.15: overall_sentiment = 'Bearish'
            else: overall_sentiment = 'Neutral'

            result = {
                'symbol': symbol,
                'articles_analyzed': len(sentiment_scores),
                'average_sentiment_score': float(avg_sentiment),
                'overall_sentiment': overall_sentiment,
            }
            print(f"✅ Sentiment analysis complete for {symbol}: {overall_sentiment}")
            return result
        except Exception as e:
            error_msg = f"❌ News/Sentiment analysis error for {symbol}: {str(e)}"
            print(error_msg)
            return {'error': error_msg, 'symbol': symbol}

# 🔸 STEP 3: CREATE CONSOLIDATED AI AGENTS

data_collection_agent = Agent(
    role='Senior Data Collection Specialist',
    goal='Fetch comprehensive and accurate stock market data for Indian equities using yfinance',
    backstory="An expert in sourcing real-time and historical financial data for the Indian stock market, ensuring accuracy and completeness.",
    tools=[YFinanceIndianStockTool()],
    verbose=True,
    llm=gemini_llm
)

financial_analyst_agent = Agent(
    role='Senior Financial Analyst',
    goal='Perform comprehensive technical and fundamental analysis to assess a stock\'s value and price trends',
    backstory="A seasoned analyst skilled at interpreting financial statements, ratios, and chart patterns to form a holistic view of a stock's health and potential.",
    tools=[TechnicalAnalysisTool()],
    verbose=True,
    llm=gemini_llm
)

market_risk_agent = Agent(
    role='Market Sentiment and Risk Specialist',
    goal='Analyze news sentiment and assess all investment risks to provide a clear risk profile',
    backstory="An expert in behavioral finance who understands how news, market sentiment, and volatility impact stock prices, providing crucial risk management insights.",
    tools=[NewsSentimentTool()],
    verbose=True,
    llm=gemini_llm
)

investment_strategist_agent = Agent(
    role='Chief Investment Strategist',
    goal='Synthesize all analyses into a final, actionable investment recommendation',
    backstory="A top-tier investment strategist who combines technical, fundamental, and risk analysis to craft clear, concise investment advice with specific targets and risk-management levels.",
    verbose=True,
    llm=gemini_llm,
    allow_delegation=False
)

print("🤖 All 4 Consolidated AI Agents Created Successfully!")

# 🔸 STEP 4: CREATE STREAMLINED TASKS

def create_optimized_tasks(symbol, company_name=None):
    company_name = company_name or symbol
    
    data_collection_task = Task(
        description=f"Collect comprehensive, up-to-date stock data for {symbol} ({company_name}) using the Indian Stock Data Fetcher tool. Include current price, historical data, financial ratios, and volume.",
        agent=data_collection_agent,
        expected_output="A JSON object containing complete stock data, including 'historical_data' for the last year."
    )

    financial_analysis_task = Task(
        description=f"Using the collected data for {symbol}, perform a thorough financial analysis. First, use the Technical Analysis Calculator tool on the 'historical_data' to get key indicators. Then, analyze the fundamental ratios (P/E, Debt-to-Equity, Profit Margin, Revenue Growth). Conclude with a valuation assessment (e.g., undervalued, fairly valued, overvalued).",
        agent=financial_analyst_agent,
        context=[data_collection_task],
        expected_output="A detailed report covering both technical signals (RSI, MACD, MAs, support/resistance) and a fundamental assessment of financial health and valuation."
    )

    market_risk_task = Task(
        description=f"Assess the market sentiment and overall risk for investing in {symbol}. Use the News Sentiment Analyzer tool to gauge market mood. Evaluate the stock's volatility, liquidity (from volume data), and any sector-specific risks. Provide a final risk assessment (e.g., Low, Moderate, High).",
        agent=market_risk_agent,
        context=[data_collection_task, financial_analysis_task],
        expected_output="A risk and sentiment report detailing the overall market sentiment, a risk level classification, and key risks to consider."
    )

    final_recommendation_task = Task(
        description=f"""
        Synthesize all analyses for {symbol} ({company_name}) from the previous tasks into a final, structured investment recommendation.
        The final output MUST be a single block of text and follow this exact format, replacing placeholders with your analysis:

        ### Investment Recommendation: {symbol} ({company_name})
        
        **Date:** {datetime.now().strftime('%B %d, %Y')}
        **Current Price:** [Specify current price from data]

        ---
        
        #### 1. Clear Recommendation: **BUY/HOLD/SELL**
        
        ---
        
        #### 2. Target Price (3-6 month outlook): **[Your Price Target]**
        
        ---
        
        #### 3. Stop-Loss Level for Risk Management: **[Your Stop-Loss Price]**
        
        ---
        
        #### 4. Confidence Level (1-10 scale): **[Your Confidence Score]/10**
        
        ---
        
        #### 5. Key Reasons Supporting the Recommendation:
        
        **A. Technical Analysis ([Bullish/Bearish/Neutral]):**
        * [Brief summary of key technical signals and indicators]
        
        **B. Fundamental Valuation & Financial Health ([Undervalued/Fairly Valued/Overvalued]):**
        * [Brief summary of valuation, P/E, debt, and profitability]
        
        **C. Market Sentiment & News Impact ([Bullish/Bearish/Neutral]):**
        * [Brief summary of news sentiment and its potential impact]
        
        **D. Risk Assessment ([Low/Moderate/High]):**
        * [Brief summary of volatility, liquidity, and other key risks]
        
        **Overall Synthesis:**
        [A concise paragraph bringing all the points together to justify the final recommendation.]
        
        ---
        
        #### 6. Position Sizing Recommendation (%% of portfolio): **[Recommended %% Allocation]**
        
        ---
        
        #### 7. Time Horizon for the Investment: **[e.g., Short-Term, Medium to Long-Term]**
        """,
        agent=investment_strategist_agent,
        context=[financial_analysis_task, market_risk_task],
        expected_output="The final, fully formatted investment recommendation as a single string."
    )

    return [data_collection_task, financial_analysis_task, market_risk_task, final_recommendation_task]

print("📋 Optimized Task Creation Function Ready!")

# 🔸 STEP 5: CREATE AND EXECUTE THE OPTIMIZED CREW

def analyze_stock_with_multi_agents(symbol, company_name=None):
    """
    Run the complete, optimized multi-agent stock analysis.
    """
    print(f"\n🚀 Starting Optimized Multi-Agent Analysis for {symbol}")
    print("=" * 80)

    tasks = create_optimized_tasks(symbol, company_name)

    stock_analysis_crew = Crew(
        agents=[
            data_collection_agent,
            financial_analyst_agent,
            market_risk_agent,
            investment_strategist_agent
        ],
        tasks=tasks,
        llm=gemini_llm,
        process=Process.sequential,
        verbose=True,
    )

    print(f"🤖 Deploying {len(stock_analysis_crew.agents)} AI agents for analysis...")
    
    try:
        result = stock_analysis_crew.kickoff()

        print("\n" + "=" * 80)
        print("🎯 MULTI-AGENT ANALYSIS COMPLETE!")
        print("=" * 80)
        print(result)

        # Prepare final structured result for storage and email
        final_result = {
            'symbol': symbol,
            'analysis_result': result,
            'timestamp': datetime.now().isoformat(),
            'agents_used': len(stock_analysis_crew.agents),
            'tasks_completed': len(tasks)
        }

        # Send email notification
        email_notifier = StockAIEmailNotifier()
        email_notifier.send_email(final_result, symbol)
        
        return final_result

    except Exception as e:
        error_msg = f"❌ Crew execution error: {str(e)}"
        print(error_msg)
        return {'error': error_msg, 'symbol': symbol}

# # 🔸 EXAMPLE USAGE

# if __name__ == '__main__':
#     print("🔥 STARTING MULTI-AGENT STOCK ANALYSIS")
    
#     # Example: Analyze a single stock
#     # The script will run this analysis when executed directly.
#     # To use this function from your queue manager, the queue manager will call
#     # analyze_stock_with_multi_agents(ticker, company_name)
    
#     result = analyze_stock_with_multi_agents("RELIANCE", "Reliance Industries")

#     # To analyze multiple stocks, you can loop through them
#     # indian_stocks = {
#     #     "TATAMOTORS": "TATA Motors Ltd",
#     #     "HDFCBANK": "HDFC Bank Ltd",
#     # }
#     # for symbol, name in indian_stocks.items():
#     #     analyze_stock_with_multi_agents(symbol, name)
#     #     time.sleep(10) # Optional delay between analyses