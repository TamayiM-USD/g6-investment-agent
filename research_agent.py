import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
from openai import OpenAI

from data_models import ResearchPlan, AnalysisResult, AgentMemory
from api_clients import YahooFinanceClient, AlphaVantageClient, FREDClient, SECEdgarClient
from agents import MarketDataAgent, FundamentalsAgent, EconomicContextAgent, RegulatoryAgent
from workflows import PromptChainWorkflow, RoutingWorkflow, EvaluatorOptimizerWorkflow


class InvestmentResearchAgent:
    """
    Main autonomous research agent with LLM integration
    
    Demonstrates agentic AI capabilities:
    - Autonomous planning using LLM
    - Dynamic tool selection and usage
    - Self-reflection on output quality
    - Learning across multiple runs
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.name = "Investment Research Agent"
        
        # Initialize LLM
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required for autonomous agent. "
                "Set OPENAI_API_KEY environment variable."
            )
        
        self.llm = OpenAI(api_key=api_key)
        print(f"{self.name} initialized with LLM")
        
        # Initialize API clients
        print("  Initializing API clients...")
        self.yahoo_client = YahooFinanceClient()
        
        try:
            self.alpha_vantage = AlphaVantageClient()
        except ValueError:
            print("    ⚠ Alpha Vantage API key not set (optional)")
            self.alpha_vantage = None
        
        try:
            self.fred_client = FREDClient()
        except ValueError:
            print("    ⚠ FRED API key not set (optional)")
            self.fred_client = None
        
        self.sec_client = SECEdgarClient()
        
        # Initialize agents
        print("  Initializing specialized agents...")
        self.market_agent = MarketDataAgent(self.llm)
        self.fundamentals_agent = FundamentalsAgent(self.llm)
        self.economic_agent = EconomicContextAgent(self.llm)
        self.regulatory_agent = RegulatoryAgent()
        
        # Initialize workflows
        print("  Initializing workflows...")
        self.prompt_chain = PromptChainWorkflow(self.llm)
        self.router = RoutingWorkflow(self.llm)
        self.evaluator_optimizer = EvaluatorOptimizerWorkflow(self.llm)
        
        # Memory system
        self.memory = []
        
        print(f"{self.name} fully initialized!")
    
    def plan_research(self, symbol: str) -> ResearchPlan:
        """
        AGENT FUNCTION 1: Planning
        
        Uses LLM to autonomously generate comprehensive research plan
        
        Args:
            symbol: Stock ticker to research
        
        Returns:
            ResearchPlan with LLM-generated objectives and steps
        """
        print(f"\n{'='*60}")
        print(f"[AGENT FUNCTION 1: PLANNING]")
        print(f"Creating research plan for {symbol} using LLM...")
        print(f"{'='*60}\n")
        
        planning_prompt = f"""
You are an expert financial research planner. Create a comprehensive research plan for analyzing {symbol} stock.

Generate a detailed plan in JSON format:
{{
    "objectives": [
        "Clear, specific objective 1",
        "Specific objective 2",
        "Specific objective 3",
        "Specific objective 4",
        "Specific objective 5"
    ],
    "data_sources": [
        "Yahoo Finance - real-time stock data",
        "Alpha Vantage - company fundamentals",
        "FRED - economic indicators",
        "SEC EDGAR - regulatory filings",
        "News sources - recent developments"
    ],
    "analysis_steps": [
        "Step 1: Fetch current market data and price trends",
        "Step 2: Analyze financial health and profitability metrics",
        "Step 3: Evaluate macroeconomic context and sector conditions",
        "Step 4: Review regulatory compliance and recent filings",
        "Step 5: Synthesize findings using multi-agent analysis",
        "Step 6: Generate investment recommendations",
        "Step 7: Assess analysis quality and confidence"
    ],
    "expected_outputs": [
        "Market trend analysis with price targets",
        "Fundamental health assessment",
        "Economic risk analysis",
        "Regulatory compliance status",
        "Investment recommendation with rationale"
    ],
    "reasoning": "Detailed explanation of why this research plan is appropriate for {symbol}, considering its sector, market cap, and typical investor interest. 2-3 sentences."
}}

Be specific and actionable. Focus on what makes {symbol} analysis unique.
"""
        
        try:
            print("  Calling LLM to generate research plan...")
            
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial research planner. Create detailed, actionable research plans in JSON format."
                    },
                    {
                        "role": "user",
                        "content": planning_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            plan_data = json.loads(response.choices[0].message.content)
            
            plan = ResearchPlan(
                stock_symbol=symbol,
                objectives=plan_data.get("objectives", []),
                data_sources=plan_data.get("data_sources", []),
                analysis_steps=plan_data.get("analysis_steps", []),
                expected_outputs=plan_data.get("expected_outputs", []),
                reasoning=plan_data.get("reasoning", "")
            )
            
            print(f"\nLLM-generated research plan created!")
            print(f"  Objectives: {len(plan.objectives)}")
            print(f"  Analysis Steps: {len(plan.analysis_steps)}")
            print(f"  Expected Outputs: {len(plan.expected_outputs)}")
            print(f"\nReasoning: {plan.reasoning}\n")
            
            return plan
            
        except Exception as e:
            print(f"\n✗ Error in LLM planning: {e}")
            print("  Using fallback plan...")
            
            return ResearchPlan(
                stock_symbol=symbol,
                objectives=[
                    "Analyze current market position and trends",
                    "Evaluate financial health and profitability",
                    "Assess macroeconomic context",
                    "Review regulatory compliance"
                ],
                data_sources=[
                    "Yahoo Finance", "Alpha Vantage", "FRED", "SEC EDGAR"
                ],
                analysis_steps=[
                    "Gather market data",
                    "Analyze fundamentals",
                    "Evaluate economic environment",
                    "Review filings",
                    "Synthesize findings"
                ],
                expected_outputs=[
                    "Market analysis",
                    "Fundamental assessment",
                    "Economic context",
                    "Investment recommendation"
                ],
                reasoning="Standard comprehensive financial analysis plan"
            )


if __name__ == "__main__":
    print("\nTesting Investment Research Agent - Planning Function...")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY required")
        exit(1)
    
    try:
        # Initialize agent
        print("\n1. Initializing agent...")
        agent = InvestmentResearchAgent()
        
        # Test planning function
        print("\n2. Testing plan_research() - Agent Function 1...")
        plan = agent.plan_research("AAPL")
        
        print("\n" + "="*60)
        print("RESEARCH PLAN GENERATED:")
        print("="*60)
        print(plan.summary())
        
        print("Objectives:")
        for i, obj in enumerate(plan.objectives, 1):
            print(f"  {i}. {obj}")
        
        print("\nAnalysis Steps:")
        for i, step in enumerate(plan.analysis_steps, 1):
            print(f"  {i}. {step}")
        
        print("\n" + "="*60)
        print("Agent Function 1 (Planning): WORKING! ")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    def execute_research(self, symbol: str) -> Dict[str, Any]:
        """
        AGENT FUNCTION 2: Tool Usage
        
        Dynamically uses APIs and coordinates agents to execute research
        
        Args:
            symbol: Stock ticker to analyze
        
        Returns:
            Complete research results with all agent analyses
        """
        print(f"\n{'='*60}")
        print(f"[AGENT FUNCTION 2: TOOL USAGE]")
        print(f"Executing research for {symbol} using real APIs...")
        print(f"{'='*60}\n")
        
        # Create research plan
        plan = self.plan_research(symbol)
        
        # Gather data from real APIs
        print("\n[Data Collection Phase]")
        print("Fetching from real financial APIs...")
        
        print("  1. Yahoo Finance...")
        stock_data = self.yahoo_client.get_stock_info(symbol)
        if "error" in stock_data:
            raise RuntimeError(f"Failed to fetch stock data: {stock_data['error']}")
        print(f"     Got data for {stock_data.get('company_name', symbol)}")
        
        print("  2. Yahoo Finance - News...")
        news = self.yahoo_client.get_news(symbol, limit=3)
        print(f"     Got {len(news)} news articles")
        
        # Alpha Vantage (optional)
        company_overview = None
        if self.alpha_vantage:
            try:
                print("  3. Alpha Vantage...")
                company_overview = self.alpha_vantage.get_company_overview(symbol)
                print(f"     Got company overview")
            except Exception as e:
                print(f"     ⚠ Alpha Vantage skipped: {str(e)[:50]}")
        
        # FRED (optional)
        fed_rate = None
        unemployment = None
        if self.fred_client:
            try:
                print("  4. FRED - Economic indicators...")
                fed_rate = self.fred_client.get_economic_indicator("DFF", limit=3)
                unemployment = self.fred_client.get_economic_indicator("UNRATE", limit=3)
                print(f"     Got economic data")
            except Exception as e:
                print(f"     ⚠ FRED skipped: {str(e)[:50]}")
        
        print("  5. SEC EDGAR...")
        try:
            sec_filings = self.sec_client.get_company_submissions(symbol)
            print(f"     Got SEC filings (CIK: {sec_filings.get('cik', 'N/A')})")
        except Exception as e:
            print(f"     ⚠ SEC data limited: {str(e)[:50]}")
            sec_filings = {"error": str(e)}
        
        # Prepare economic context
        economic_data = {
            "fed_funds_rate": fed_rate.get("latest_value") if fed_rate else "5.33",
            "unemployment_rate": unemployment.get("latest_value") if unemployment else "3.8",
            "cpi": "310.5"
        }
        
        # Store results
        results = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "research_plan": plan.to_dict(),
            "raw_data": {
                "stock_info": stock_data,
                "company_overview": company_overview,
                "news": news,
                "economic_indicators": economic_data,
                "sec_filings": sec_filings
            },
            "agent_analyses": {},
            "workflow_results": {}
        }
        
        # Execute agent analyses
        print("\n[Agent Analysis Phase]")
        print("Running LLM-powered specialized agents...")
        
        print("  1. Market Data Agent...")
        market_analysis = self.market_agent.analyze(symbol, stock_data)
        results["agent_analyses"]["market"] = market_analysis.to_dict()
        
        print("  2. Fundamentals Agent...")
        fundamentals_analysis = self.fundamentals_agent.analyze(symbol, stock_data)
        results["agent_analyses"]["fundamentals"] = fundamentals_analysis.to_dict()
        
        print("  3. Economic Context Agent...")
        sector = stock_data.get("sector", "Unknown")
        economic_analysis = self.economic_agent.analyze(sector, economic_data)
        results["agent_analyses"]["economic"] = economic_analysis.to_dict()
        
        print("  4. Regulatory Agent...")
        regulatory_analysis = self.regulatory_agent.analyze(symbol, sec_filings)
        results["agent_analyses"]["regulatory"] = regulatory_analysis.to_dict()
        
        # Execute workflows
        print("\n[Workflow Execution Phase]")
        print("Running LLM-powered workflow patterns...")
        
        print("  1. Prompt Chain Workflow...")
        chain_result = self.prompt_chain.execute(symbol, stock_data)
        results["workflow_results"]["prompt_chain"] = chain_result.to_dict()
        
        print("  2. Routing Workflow...")
        routing_result = self.router.execute(
            f"What's the investment outlook for {symbol}?",
            ["MarketDataAgent", "FundamentalsAgent", "EconomicContextAgent", "RegulatoryAgent"]
        )
        results["workflow_results"]["routing"] = routing_result
        
        print("  3. Evaluator-Optimizer Workflow...")
        eval_result = self.evaluator_optimizer.execute(results["agent_analyses"])
        results["workflow_results"]["evaluator_optimizer"] = eval_result
        
        print("\nResearch execution complete!")
        print(f"  Agents run: {len(results['agent_analyses'])}")
        print(f"  Workflows executed: {len(results['workflow_results'])}")
        
        return results


if __name__ == "__main__":
    print("\nTesting Investment Research Agent - Execution...")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY required")
        exit(1)
    
    try:
        agent = InvestmentResearchAgent()
        
        print("\n" + "="*60)
        print("Testing execute_research() - Agent Function 2")
        print("="*60)
        
        results = agent.execute_research("AAPL")
        
        print("\n" + "="*60)
        print("EXECUTION RESULTS:")
        print("="*60)
        print(f"Symbol: {results['symbol']}")
        print(f"Agents analyzed: {len(results['agent_analyses'])}")
        print(f"Workflows executed: {len(results['workflow_results'])}")
        
        print("\nAgent Analyses:")
        for agent_name, analysis in results['agent_analyses'].items():
            print(f"  {agent_name}: {len(analysis.get('recommendations', []))} recommendations")
        
        print("\nWorkflow Results:")
        for workflow_name in results['workflow_results'].keys():
            print(f"  {workflow_name}")
        
        print("\n" + "="*60)
        print("Agent Functions 1 & 2: WORKING! ")
        print("  1. Planning: LLM generates research plan ")
        print("  2. Tool Usage: Coordinates APIs and agents ")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
