from dataclasses import dataclass, asdict
from typing import Dict, Any, List
from datetime import datetime


@dataclass
class ResearchPlan:
    """
    This class represents an AI-generated research plan from the LLM
    The class holds all the data related to the research plan
    It's gotten from calls to InvestmentResearchAgent.plan_research()
    """
    stock_symbol: str
    objectives: List[str]
    data_sources: List[str]
    analysis_steps: List[str]
    expected_outputs: List[str]
    reasoning: str  # LLM's reasoning for the plan
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def summary(self) -> str:
        """Get human-readable summary"""
        return f"""
Research Plan for {self.stock_symbol}
Objectives: {len(self.objectives)} goals
Data Sources: {', '.join(self.data_sources)}
Analysis Steps: {len(self.analysis_steps)} steps
Generated: {self.timestamp}
"""


@dataclass
class AnalysisResult:
    """
    Standard format for agent analysis outputs
    """
    agent_name: str
    timestamp: str
    data_source: str
    findings: Dict[str, Any]
    confidence_score: float  # 0.0 to 1.0
    recommendations: List[str]
    llm_reasoning: str  # Full LLM response/reasoning
    
    def __post_init__(self):
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def summary(self) -> str:
        """Get human-readable summary"""
        return f"""
{self.agent_name} Analysis
Confidence: {self.confidence_score:.2f}
Recommendations: {len(self.recommendations)}
Source: {self.data_source}
Timestamp: {self.timestamp}
"""


@dataclass
class AgentMemory:
    """
    Stores learning across runs
    """
    stock_symbol: str
    timestamp: str
    insights: List[str]
    quality_scores: Dict[str, float]
    recommendations: List[str]
    analysis_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def update_quality(self, new_score: float):
        """Update quality score with running average"""
        current_avg = self.quality_scores.get("overall", new_score)
        self.analysis_count += 1
        self.quality_scores["overall"] = (
            (current_avg * (self.analysis_count - 1) + new_score) / self.analysis_count
        )
    
    def add_insight(self, insight: str):
        """Add new insight to memory"""
        if insight not in self.insights:
            self.insights.append(insight)
            # Keep only most recent 10 insights
            if len(self.insights) > 10:
                self.insights = self.insights[-10:]


@dataclass
class WorkflowResult:
    """
    Standard format for workflow outputs
    Used by: PromptChainWorkflow, RoutingWorkflow, EvaluatorOptimizerWorkflow
    """
    workflow_name: str
    timestamp: str
    steps_completed: int
    final_output: Any
    intermediate_results: List[Dict[str, Any]]
    execution_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


if __name__ == "__main__":
    print("Testing Data Models...")
    print("="*60)
    
    # Test ResearchPlan
    print("\n1. Testing ResearchPlan...")
    plan = ResearchPlan(
        stock_symbol="AAPL",
        objectives=["Analyze market trends", "Evaluate fundamentals"],
        data_sources=["Yahoo Finance", "Alpha Vantage"],
        analysis_steps=["Fetch data", "Analyze", "Report"],
        expected_outputs=["Market analysis", "Investment recommendation"],
        reasoning="Comprehensive analysis needed for tech stock"
    )
    print(f"Created plan for {plan.stock_symbol}")
    print(plan.summary())
    
    # Test AnalysisResult
    print("\n2. Testing AnalysisResult...")
    result = AnalysisResult(
        agent_name="Market Data Agent",
        timestamp=datetime.now().isoformat(),
        data_source="Yahoo Finance",
        findings={"trend": "bullish", "price": 175.43},
        confidence_score=0.85,
        recommendations=["Consider buying", "Monitor volatility"],
        llm_reasoning="Stock shows strong upward momentum"
    )
    print(f"Created analysis with confidence {result.confidence_score}")
    print(result.summary())
    
    # Test AgentMemory
    print("\n3. Testing AgentMemory...")
    memory = AgentMemory(
        stock_symbol="AAPL",
        timestamp=datetime.now().isoformat(),
        insights=["Strong growth", "High valuation"],
        quality_scores={"overall": 0.85},
        recommendations=["Monitor closely"]
    )
    memory.add_insight("Positive sentiment")
    memory.update_quality(0.90)
    print(f"Created memory with {len(memory.insights)} insights")
    print(f"  Updated quality: {memory.quality_scores['overall']:.2f}")
    
    print("\n" + "="*60)
    print("All data models working!")
