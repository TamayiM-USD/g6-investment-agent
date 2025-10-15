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
