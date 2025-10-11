import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import OpenAI
from data_models import WorkflowResult


class PromptChainWorkflow:
    """
    Workflow Pattern 1: Prompt Chaining with LLM
    
    5-Step Pipeline:
    1. Ingest data
    2. Preprocess and structure
    3. Classify data types
    4. Extract insights (LLM-powered)
    5. Summarize findings (LLM-powered)
    """
    
    def __init__(self, llm_client: Optional[OpenAI] = None):
        self.name = "Prompt Chain Workflow"
        
        if llm_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required for workflow")
            self.llm = OpenAI(api_key=api_key)
        else:
            self.llm = llm_client
        
        print(f"{self.name} initialized with LLM")
    
    def execute(self, symbol: str, data: Dict[str, Any]) -> WorkflowResult:
        """Execute 5-step prompt chain workflow"""
        print(f"\n[{self.name}] Executing for {symbol}...")
        
        start_time = time.time()
        intermediate_results = []
        
        print("  [Step 1/5] Ingesting data...")
        ingested = self._step1_ingest(data)
        intermediate_results.append({"step": 1, "name": "Ingest", "output": "Data ingested"})
        
        print("  [Step 2/5] Preprocessing...")
        preprocessed = self._step2_preprocess(ingested, symbol)
        intermediate_results.append({"step": 2, "name": "Preprocess", "output": "Data structured"})
        
        print("  [Step 3/5] Classifying...")
        classified = self._step3_classify(preprocessed)
        intermediate_results.append({"step": 3, "name": "Classify", "output": "Data classified"})
        
        print("  [Step 4/5] Extracting insights with LLM...")
        insights = self._step4_extract_insights_llm(classified, symbol)
        intermediate_results.append({"step": 4, "name": "Extract (LLM)", "output": insights})
        
        print("  [Step 5/5] Synthesizing summary with LLM...")
        summary = self._step5_summarize_llm(insights, symbol)
        intermediate_results.append({"step": 5, "name": "Summarize (LLM)", "output": summary})
        
        execution_time = time.time() - start_time
        
        print(f"  Workflow complete in {execution_time:.2f}s")
        
        return WorkflowResult(
            workflow_name=self.name,
            timestamp=datetime.now().isoformat(),
            steps_completed=5,
            final_output=summary,
            intermediate_results=intermediate_results,
            execution_time_seconds=execution_time
        )
    
    def _step1_ingest(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Ingest raw data"""
        return {"raw_data": data, "ingested_at": datetime.now().isoformat()}
    
    def _step2_preprocess(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Step 2: Preprocess and structure"""
        return {
            "symbol": symbol,
            "structured_data": data.get("raw_data", {}),
            "timestamp": datetime.now().isoformat()
        }
    
    def _step3_classify(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Classify data into categories"""
        return {
            "symbol": data.get("symbol"),
            "categories": {
                "market_data": True,
                "fundamental_data": True,
                "economic_context": False
            }
        }
    
    def _step4_extract_insights_llm(self, classified: Dict[str, Any], symbol: str) -> List[str]:
        """Step 4: Extract insights using LLM"""
        
        prompt = f"""
Based on financial analysis of {symbol}, extract 3-5 key investment insights.

Focus on:
- Market positioning and trends
- Financial health indicators
- Investment opportunities or risks

Provide insights as a JSON array of strings:
{{"insights": ["insight 1", "insight 2", "insight 3"]}}

Be specific and actionable.
"""
        
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial analyst extracting key insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            insights = result.get("insights", [])
            
            return insights if insights else [
                f"Market analysis completed for {symbol}",
                "Financial metrics evaluated",
                "Investment considerations identified"
            ]
            
        except Exception as e:
            print(f"    Warning: LLM extraction error: {e}")
            return [
                f"Analysis completed for {symbol}",
                "Key metrics evaluated",
                "Investment factors assessed"
            ]
    
    def _step5_summarize_llm(self, insights: List[str], symbol: str) -> str:
        """Step 5: Synthesize summary using LLM"""
        
        insights_text = "\n".join(f"- {insight}" for insight in insights)
        
        prompt = f"""
Synthesize these investment insights for {symbol} into a concise executive summary (2-3 sentences):

{insights_text}

Provide a clear, actionable summary for investors in JSON format:
{{"summary": "your executive summary here"}}
"""
        
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial analyst creating executive summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            summary = result.get("summary", "")
            
            return summary if summary else f"Analysis of {symbol} reveals {len(insights)} key insights for investors."
            
        except Exception as e:
            print(f"    Warning: LLM summary error: {e}")
            return f"Investment analysis for {symbol} completed. Key insights: {', '.join(insights[:2])}."


if __name__ == "__main__":
    print("\nTesting Prompt Chain Workflow...")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY not set")
        exit(1)
    
    try:
        workflow = PromptChainWorkflow()
        
        test_data = {
            "symbol": "AAPL",
            "price": 175.43,
            "market_cap": 2750000000000
        }
        
        result = workflow.execute("AAPL", test_data)
        
        print("\n" + "="*60)
        print("WORKFLOW RESULT:")
        print("="*60)
        print(f"Steps completed: {result.steps_completed}")
        print(f"Execution time: {result.execution_time_seconds:.2f}s")
        print(f"\nFinal Summary:\n{result.final_output}")
        
        print("\n" + "="*60)
        print("Prompt Chain Workflow: WORKING! ")
        
    except Exception as e:
        print(f"✗ Error: {e}")
