import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
from openai import OpenAI
from data_models import AnalysisResult


class MarketDataAgent:
    """
    Agent specialized in market data analysis using LLMs
    
    Analyzes:
    - Price trends (bullish/bearish/neutral)
    - Volatility assessment
    - Valuation opinion
    - Investment recommendations
    
    LLM Model: GPT-4o-mini for cost-effective analysis
    """
    
    def __init__(self, llm_client: Optional[OpenAI] = None):
        self.name = "Market Data Agent"
        
        if llm_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key required for LLM agent. "
                    "Set OPENAI_API_KEY environment variable. "
                    "Get key at: https://platform.openai.com/api-keys"
                )
            self.llm = OpenAI(api_key=api_key)
        else:
            self.llm = llm_client
        
        print(f"{self.name} initialized with LLM")
    
    def analyze(self, symbol: str, data: Dict[str, Any]) -> AnalysisResult:
        """
        Analyze market data using LLM reasoning
        
        Args:
            symbol: Stock ticker symbol
            data: Market data from Yahoo Finance
        
        Returns:
            AnalysisResult with LLM-powered analysis
        """
        print(f"\n[{self.name}] Analyzing {symbol} with LLM...")
        
        # Prepare comprehensive market prompt for LLM
        market_prompt = self._create_market_prompt(symbol, data)
        
        try:
            # Call OpenAI LLM for analysis
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert market analyst. Provide analysis in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": market_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            llm_analysis = response.choices[0].message.content
            
            # Parse LLM response
            try:
                findings = json.loads(llm_analysis)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                findings = {
                    "raw_analysis": llm_analysis,
                    "price_trend": "Analysis provided in raw format",
                    "recommendations": ["Review raw analysis for insights"]
                }
            
            print(f"  LLM analysis complete")
            print(f"  Trend: {findings.get('price_trend', 'N/A')}")
            
            # Extract the recommendations
            recommendations = findings.get("recommendations", [])
            if isinstance(recommendations, str):
                recommendations = [recommendations]
            
            return AnalysisResult(
                agent_name=self.name,
                timestamp=datetime.now().isoformat(),
                data_source="Yahoo Finance AOI + OpenAI GPT-4o-mini",
                findings=findings,
                confidence_score=0.85,
                recommendations=recommendations,
                llm_reasoning=llm_analysis
            )
            
        except Exception as e:
            print(f"  LLM analysis error: {str(e)}")
            raise RuntimeError(f"Failed to analyze with LLM: {str(e)}")
    
    def _create_market_prompt(self, symbol: str, data: Dict[str, Any]) -> str:
        """Create comprehensive prompt for LLM analysis"""
        
        # Extract key metrics with safe defaults
        company = data.get('company_name', 'N/A')
        sector = data.get('sector', 'N/A')
        current_price = data.get('current_price', 0)
        prev_close = data.get('previous_close', 0)
        market_cap = data.get('market_cap', 0)
        pe_ratio = data.get('pe_ratio', 'N/A')
        week_52_high = data.get('52_week_high', 0)
        week_52_low = data.get('52_week_low', 0)
        beta = data.get('beta', 'N/A')
        volume = data.get('volume', 0)
        
        # Calculate price change
        price_change = current_price - prev_close if current_price and prev_close else 0
        price_change_pct = (price_change / prev_close * 100) if prev_close else 0
        
        # Calculate position in 52-week range
        if week_52_high and week_52_low and week_52_high > week_52_low:
            range_position = ((current_price - week_52_low) / (week_52_high - week_52_low)) * 100
        else:
            range_position = None
        
        prompt = f"""
You are an expert market analyst. Analyze this stock's market data and provide investment insights.

STOCK INFORMATION:
Symbol: {symbol}
Company: {company}
Sector: {sector}

PRICE METRICS:
Current Price: ${current_price:.2f}
Previous Close: ${prev_close:.2f}
Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
52-Week High: ${week_52_high:.2f}
52-Week Low: ${week_52_low:.2f}
52-Week Range Position: {f"{range_position:.1f}%" if range_position else "N/A"}

VALUATION & RISK:
Market Cap: ${market_cap:,}
PE Ratio: {pe_ratio}
Beta (Volatility): {beta}
Volume: {volume:,}

ANALYSIS REQUIRED:
Provide your analysis in JSON format with these fields:
{{
    "price_trend": "bullish/bearish/neutral with 2-3 sentence explanation",
    "volatility_assessment": "high/moderate/low with reasoning based on beta and price action",
    "valuation_opinion": "overvalued/undervalued/fairly valued with reasoning based on PE and market position",
    "technical_position": "analysis of 52-week range position and recent price action",
    "key_observations": ["observation 1", "observation 2", "observation 3"],
    "recommendations": ["specific recommendation 1", "specific recommendation 2", "specific recommendation 3"]
}}

Be specific, data-driven, and actionable. Focus on the metrics provided.
"""
        
        return prompt


if __name__ == "__main__":
    print("\nTesting Market Data Agent with LLM...")
    print("="*60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        print("\nSet your OpenAI API key:")
        print("  export OPENAI_API_KEY=sk-your-key-here")
        print("\nGet key at: https://platform.openai.com/api-keys")
        exit(1)
    
    try:
        # Initialize agent
        agent = MarketDataAgent()
        
        # Test data
        test_data = {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "current_price": 175.43,
            "previous_close": 174.22,
            "market_cap": 2750000000000,
            "pe_ratio": 28.5,
            "52_week_high": 199.62,
            "52_week_low": 164.08,
            "beta": 1.24,
            "volume": 52000000
        }
        
        print("\nAnalyzing AAPL with real LLM...")
        result = agent.analyze("AAPL", test_data)
        
        print("\n" + "="*60)
        print("ANALYSIS RESULT:")
        print("="*60)
        print(result.summary())
        
        print("\nFINDINGS:")
        for key, value in result.findings.items():
            if key != "recommendations" and key != "key_observations":
                print(f"  {key}: {value}")
        
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*60)
        print("Market Data Agent with LLM: WORKING!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. OPENAI_API_KEY is set")
        print("  2. API key is valid")
        print("  3. You have API credits")
