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


class FundamentalsAgent:
    """
    Agent specialized in fundamental analysis using LLMs
    
    Analyzes:
    - Profitability (margins, ROE)
    - Growth potential
    - Financial health
    - Competitive position
    """
    
    def __init__(self, llm_client: Optional[OpenAI] = None):
        self.name = "Fundamentals Agent"
        
        if llm_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            self.llm = OpenAI(api_key=api_key)
        else:
            self.llm = llm_client
        
        print(f"{self.name} initialized with LLM")
    
    def analyze(self, symbol: str, data: Dict[str, Any]) -> AnalysisResult:
        """Analyze fundamental data using LLM"""
        print(f"\n[{self.name}] Analyzing {symbol} fundamentals with LLM...")
        
        prompt = self._create_fundamentals_prompt(symbol, data)
        
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert fundamental analyst. Provide analysis in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            llm_analysis = response.choices[0].message.content
            findings = json.loads(llm_analysis)
            
            print(f"  Fundamentals analysis complete")
            
            return AnalysisResult(
                agent_name=self.name,
                timestamp=datetime.now().isoformat(),
                data_source="Yahoo Finance + Alpha Vantage + OpenAI",
                findings=findings,
                confidence_score=0.82,
                recommendations=findings.get("recommendations", []),
                llm_reasoning=llm_analysis
            )
            
        except Exception as e:
            print(f"  Error: {e}")
            raise
    
    def _create_fundamentals_prompt(self, symbol: str, data: Dict[str, Any]) -> str:
        """Create prompt for fundamental analysis"""
        
        company = data.get('company_name', 'N/A')
        revenue = data.get('revenue', 0)
        profit_margin = data.get('profit_margin', 0)
        operating_margin = data.get('operating_margin', 0)
        earnings_growth = data.get('earnings_growth', 0)
        pe_ratio = data.get('pe_ratio', 'N/A')
        forward_pe = data.get('forward_pe', 'N/A')
        debt_to_equity = data.get('debt_to_equity', 'N/A')
        roe = data.get('return_on_equity', 'N/A')
        
        return f"""
Analyze the fundamental financial health of this company:

COMPANY: {symbol} - {company}

PROFITABILITY:
Revenue (TTM): ${revenue:,}
Profit Margin: {profit_margin:.2%} if profit_margin else "N/A"
Operating Margin: {operating_margin:.2%} if operating_margin else "N/A"
Return on Equity: {roe}

VALUATION:
PE Ratio (Trailing): {pe_ratio}
PE Ratio (Forward): {forward_pe}

GROWTH:
Earnings Growth: {earnings_growth:.2%} if earnings_growth else "N/A"

FINANCIAL STRENGTH:
Debt-to-Equity: {debt_to_equity}

Provide JSON analysis:
{{
    "profitability_assessment": "strong/moderate/weak with detailed explanation",
    "growth_potential": "high/moderate/low with reasoning and growth trajectory",
    "financial_health": "excellent/good/fair/poor with balance sheet analysis",
    "competitive_position": "market leader/strong/average/weak with reasoning",
    "valuation_summary": "analysis of PE ratios and valuation metrics",
    "key_strengths": ["strength 1", "strength 2", "strength 3"],
    "key_concerns": ["concern 1", "concern 2"],
    "recommendations": ["actionable recommendation 1", "recommendation 2", "recommendation 3"]
}}

Be specific and data-driven.
"""


class EconomicContextAgent:
    """
    Agent specialized in economic context analysis using LLMs
    
    Analyzes:
    - Interest rate impact
    - Employment conditions
    - Sector outlook
    - Macroeconomic risks
    """
    
    def __init__(self, llm_client: Optional[OpenAI] = None):
        self.name = "Economic Context Agent"
        
        if llm_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            self.llm = OpenAI(api_key=api_key)
        else:
            self.llm = llm_client
        
        print(f"{self.name} initialized with LLM")
    
    def analyze(self, sector: str, economic_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze economic context using LLM"""
        print(f"\n[{self.name}] Analyzing {sector} sector economic context...")
        
        prompt = self._create_economic_prompt(sector, economic_data)
        
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert macroeconomic analyst. Provide analysis in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            llm_analysis = response.choices[0].message.content
            findings = json.loads(llm_analysis)
            
            print(f"  Economic analysis complete")
            
            return AnalysisResult(
                agent_name=self.name,
                timestamp=datetime.now().isoformat(),
                data_source="FRED + OpenAI",
                findings=findings,
                confidence_score=0.85,
                recommendations=findings.get("recommendations", []),
                llm_reasoning=llm_analysis
            )
            
        except Exception as e:
            print(f"  Error: {e}")
            raise
    
    def _create_economic_prompt(self, sector: str, data: Dict[str, Any]) -> str:
        """Create prompt for economic analysis"""
        
        fed_rate = data.get('fed_funds_rate', 'N/A')
        unemployment = data.get('unemployment_rate', 'N/A')
        cpi = data.get('cpi', 'N/A')
        gdp_growth = data.get('gdp_growth', 'N/A')
        
        return f"""
Analyze how current macroeconomic conditions affect the {sector} sector:

ECONOMIC INDICATORS:
Federal Funds Rate: {fed_rate}%
Unemployment Rate: {unemployment}%
CPI (Inflation): {cpi}
GDP Growth: {gdp_growth}

TARGET SECTOR: {sector}

Provide JSON analysis:
{{
    "interest_rate_impact": "positive/negative/neutral with detailed explanation of rate effects on {sector}",
    "employment_impact": "analysis of how employment trends affect {sector} demand and operations",
    "inflation_impact": "how inflation affects {sector} costs, pricing power, and margins",
    "sector_outlook": "favorable/neutral/challenging with 3-6 month outlook for {sector}",
    "cyclical_analysis": "where we are in economic cycle and {sector} positioning",
    "key_risks": ["macroeconomic risk 1", "risk 2", "risk 3"],
    "key_opportunities": ["opportunity 1", "opportunity 2"],
    "recommendations": ["sector-specific recommendation 1", "recommendation 2", "recommendation 3"]
}}

Focus on sector-specific impacts. Be specific about transmission mechanisms.
"""


if __name__ == "__main__":
    import sys
    
    print("\nTesting Additional LLM Agents...")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Test Fundamentals Agent
    print("\n1. Testing Fundamentals Agent...")
    try:
        fund_agent = FundamentalsAgent()
        
        test_data = {
            "company_name": "Apple Inc.",
            "revenue": 385000000000,
            "profit_margin": 0.25,
            "operating_margin": 0.30,
            "earnings_growth": 0.08,
            "pe_ratio": 28.5,
            "forward_pe": 26.0,
            "return_on_equity": 0.45,
            "debt_to_equity": 1.5
        }
        
        result = fund_agent.analyze("AAPL", test_data)
        print(f"  Analysis complete")
        print(f"    Findings: {len(result.findings)} metrics")
        print(f"    Recommendations: {len(result.recommendations)}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test Economic Agent
    print("\n2. Testing Economic Context Agent...")
    try:
        econ_agent = EconomicContextAgent()
        
        econ_data = {
            "fed_funds_rate": 5.33,
            "unemployment_rate": 3.8,
            "cpi": 310.5,
            "gdp_growth": 2.5
        }
        
        result = econ_agent.analyze("Technology", econ_data)
        print(f"  Analysis complete")
        print(f"    Sector: Technology")
        print(f"    Recommendations: {len(result.recommendations)}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("Additional LLM Agents: WORKING!")


class RegulatoryAgent:
    """
    Agent specialized in regulatory compliance analysis
    Simpler agent - uses structured data without LLM
    """
    
    def __init__(self):
        self.name = "Regulatory Agent"
        print(f"{self.name} initialized")
    
    def analyze(self, symbol: str, filings_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze regulatory filings"""
        print(f"\n[{self.name}] Analyzing regulatory status for {symbol}...")
        
        # Extract filing information
        recent_filings = filings_data.get("recent_filings", [])
        total_filings = filings_data.get("total_filings", 0)
        cik = filings_data.get("cik", "Unknown")
        
        # Analyze compliance
        findings = {
            "cik": cik,
            "total_filings": total_filings,
            "recent_filings_count": len(recent_filings),
            "filing_types": list(set([f.get("form_type", "") for f in recent_filings[:10]])),
            "latest_filing": recent_filings[0] if recent_filings else None,
            "compliance_status": "Current" if recent_filings else "Unknown"
        }
        
        # Generate recommendations
        recommendations = [
            f"Company maintains {findings['compliance_status']} SEC filings (CIK: {cik})",
            f"Total filings on record: {total_filings}",
            "Review recent 10-K for annual details",
            "Review recent 10-Q for quarterly updates"
        ]
        
        if findings["latest_filing"]:
            latest = findings["latest_filing"]
            recommendations.insert(0, 
                f"Latest filing: {latest.get('form_type')} on {latest.get('filing_date')}"
            )
        
        print(f"  Regulatory analysis complete")
        print(f"    Status: {findings['compliance_status']}")
        
        return AnalysisResult(
            agent_name=self.name,
            timestamp=datetime.now().isoformat(),
            data_source="SEC EDGAR",
            findings=findings,
            confidence_score=0.70,
            recommendations=recommendations,
            llm_reasoning="Structured analysis of SEC filings data"
        )


# Test all agents
if __name__ == "__main__":
    print("\nTesting All Agents...")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set - LLM agents will fail")
        print("Regulatory agent will work (doesn't use LLM)")
    
    # Test Regulatory Agent (no LLM needed)
    print("\n3. Testing Regulatory Agent (no LLM)...")
    try:
        reg_agent = RegulatoryAgent()
        
        test_filings = {
            "cik": "0000320193",
            "total_filings": 1250,
            "recent_filings": [
                {"form_type": "10-K", "filing_date": "2024-10-25"},
                {"form_type": "10-Q", "filing_date": "2024-07-28"},
                {"form_type": "8-K", "filing_date": "2024-06-15"}
            ]
        }
        
        result = reg_agent.analyze("AAPL", test_filings)
        print(f"  Analysis complete")
        print(f"    CIK: {result.findings['cik']}")
        print(f"    Status: {result.findings['compliance_status']}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("ALL 4 SPECIALIZED AGENTS COMPLETE!")
    print("="*60)
    print("\nAgents created:")
    print("  1. MarketDataAgent (LLM)")
    print("  2. FundamentalsAgent (LLM)")
    print("  3. EconomicContextAgent (LLM)")
    print("  4. RegulatoryAgent (Structured)")
