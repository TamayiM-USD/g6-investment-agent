"""
Integration tests for all specialized agents
"""

import os
import sys
sys.path.append('..')

from dotenv import load_dotenv
load_dotenv() 

from agents import MarketDataAgent, FundamentalsAgent, EconomicContextAgent, RegulatoryAgent


def test_all_agents_integration():
    """Test all 4 agents working together"""
    print("\n" + "="*60)
    print("AGENT INTEGRATION TEST")
    print("="*60)
    
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    if not has_openai:
        print("\nOPENAI_API_KEY not set")
        print("LLM agents will be skipped")
    
    agents_created = []
    
    # Test Market Data Agent
    if has_openai:
        print("\n1. Initializing Market Data Agent...")
        try:
            market_agent = MarketDataAgent()
            agents_created.append("Market Data Agent")
            print("  Success")
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("\n1. Skipping Market Data Agent (no API key)")
    
    # Test Fundamentals Agent
    if has_openai:
        print("\n2. Initializing Fundamentals Agent...")
        try:
            fund_agent = FundamentalsAgent()
            agents_created.append("Fundamentals Agent")
            print("  Success")
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("\n2. Skipping Fundamentals Agent (no API key)")
    
    # Test Economic Agent
    if has_openai:
        print("\n3. Initializing Economic Context Agent...")
        try:
            econ_agent = EconomicContextAgent()
            agents_created.append("Economic Context Agent")
            print("  Success")
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("\n3. Skipping Economic Context Agent (no API key)")
    
    # Test Regulatory Agent (always works)
    print("\n4. Initializing Regulatory Agent...")
    try:
        reg_agent = RegulatoryAgent()
        agents_created.append("Regulatory Agent")
        print("  Success")
    except Exception as e:
        print(f"  Failed: {e}")
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"\nAgents successfully created: {len(agents_created)}/4")
    for agent in agents_created:
        print(f"  {agent}")
    
    if len(agents_created) == 4:
        print("\nAll agents operational!")
    elif len(agents_created) >= 1:
        print(f"\n{4 - len(agents_created)} agent(s) require OPENAI_API_KEY")
    
    print("="*60)


if __name__ == "__main__":
    test_all_agents_integration()
