"""
Tests LLM integration and output format for the for Market Data Agent
"""

import os
import sys
sys.path.append('..')

from dotenv import load_dotenv
load_dotenv() 

from agents import MarketDataAgent
from data_models import AnalysisResult


def test_agent_initialization():
    """Test agent can be initialized with API key"""
    print("\n1. Testing agent initialization...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("  Skipping (OPENAI_API_KEY not set)")
        return False
    
    try:
        agent = MarketDataAgent()
        assert agent.name == "Market Data Agent"
        assert agent.llm is not None
        print("  Agent initialized successfully")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def test_analyze_with_real_data():
    """Test analysis with realistic stock data"""
    print("\n2. Testing analysis with real data...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("  Skipping (OPENAI_API_KEY not set)")
        return False
    
    try:
        agent = MarketDataAgent()
        
        test_data = {
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
        
        result = agent.analyze("AAPL", test_data)
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.agent_name == "Market Data Agent"
        assert result.confidence_score > 0
        assert len(result.recommendations) > 0
        assert "findings" in result.to_dict()
        
        print("  Analysis completed successfully")
        print(f"    Confidence: {result.confidence_score}")
        print(f"    Recommendations: {len(result.recommendations)}")
        return True
        
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def test_llm_output_format():
    """Test that LLM output is properly formatted"""
    print("\n3. Testing LLM output format...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("  Skipping (OPENAI_API_KEY not set)")
        return False
    
    try:
        agent = MarketDataAgent()
        
        test_data = {
            "company_name": "Microsoft Corporation",
            "sector": "Technology",
            "current_price": 380.00,
            "previous_close": 378.50,
            "market_cap": 2800000000000,
            "pe_ratio": 32.5,
            "52_week_high": 390.00,
            "52_week_low": 310.00,
            "beta": 0.95,
            "volume": 18000000
        }
        
        result = agent.analyze("MSFT", test_data)
        
        # Check for expected fields in findings
        required_fields = ["price_trend", "volatility_assessment", "recommendations"]
        findings = result.findings
        
        has_required = any(field in findings for field in required_fields)
        assert has_required, "Missing required analysis fields"
        
        # Check LLM reasoning is captured
        assert result.llm_reasoning is not None
        assert len(result.llm_reasoning) > 0
        
        print("  LLM output properly formatted")
        print(f"    Fields found: {list(findings.keys())}")
        return True
        
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("MARKET DATA AGENT TEST SUITE")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY not set")
        print("Some tests will be skipped")
        print("\nTo run all tests:")
        print("  export OPENAI_API_KEY=sk-your-key-here")
    
    results = []
    
    results.append(("Initialization", test_agent_initialization()))
    results.append(("Real Data Analysis", test_analyze_with_real_data()))
    results.append(("LLM Output Format", test_llm_output_format()))
    
    print("\n" + "="*60)
    print("TEST RESULTS:")
    print("="*60)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\nAll tests passed!")
    else:
        print(f"\n{total_count - passed_count} test(s) failed or skipped")
    
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
