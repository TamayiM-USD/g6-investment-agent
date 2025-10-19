# Multi-Agent Investment Research System

**AAI-520 Final Team Project - Group 6**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)](https://openai.com/)
[![License](https://img.shields.io/badge/License-Academic-yellow)](LICENSE)

---

## Team Members

- **Tamayi Mlanda** - tmlanda@sandiego.edu
- **Guruganesh Hegde** - ghegde@sandiego.edu
- **Arunkumar Rajaganapathy** - arajaganapathy@sandiego.edu

---

## Project Overview

An autonomous Investment Research AI that integrates real-world financial data, coordinates multiple specialized agents, and demonstrates advanced agentic AI workflows, self-reflect on outputs, and learn across runs - delivering a professional, reproducible system for adaptive financial analysis.

## Objectives

The objectives of the project are to:

-	Build financial analysis pipelines that demonstrate Prompt Chaining, Routing, and Evaluator–Optimizer approaches.
-	Integrate diverse financial data sources (e.g., Yahoo Finance, FRED, SEC EDGAR, Alpha Vantage) and financial news datasets to enable real-world analysis.
-	Showcase multi-agent collaboration by coordinating specialized LLM agents (e.g., earnings analyzer, news analyzer, market trend analyzer) to complete complex financial tasks.
-	Highlight agentic AI capabilities such as reasoning, planning, and iterative improvement, moving beyond scripted pipelines toward adaptive, self-correcting workflows.
-	Apply software engineering best practices, including GitHub-based collaboration, PEP8-compliant code, and clear documentation of agent design and workflows.
-	Deliver a professional final artifact in the form of a reproducible and well-documented code notebook.


## Data Sources

Our project integrates multiple reputable financial and economic data sources to ensure robust and real-world analysis. Access will be through public APIs where available, but will also rely on offline data samples where necessary to avoid incurring costs. Some data is available through “unofficial” means / wrappers / scraping. These data sources include:

-	**Yahoo Finance (Yahoo Finance Platform)**  
  A widely used financial data service offering real-time and historical stock prices, company financials, and market news. It serves as a primary source for stock-level data and basic market indicators.

-	**FRED (Federal Reserve Economic Data)**  
  A comprehensive database maintained by the Federal Reserve Bank of St. Louis, providing thousands of U.S. and international economic and financial time series. It is particularly valuable for incorporating macroeconomic indicators into financial analyses.

-	**SEC EDGAR (U.S. Securities and Exchange Commission – Electronic Data Gathering, Analysis, and Retrieval system)**  
  The official repository for public company filings, including 10-K annual reports, 10-Q quarterly reports, and other regulatory disclosures. It is essential for extracting structured company financials and compliance information.

-	**Alpha Vantage (Alpha Vantage Inc.)**  
  A provider of APIs that supply real-time and historical data for equities, forex, and cryptocurrencies. It enables programmatic access to financial data at scale, complementing other datasets with broad coverage.


### Key Features

- **Agentic AI**: Autonomous planning, reasoning, and decision-making  
- **Multi-Agent Architecture**: 4 specialized LLM-powered agents  
- **Real API Integration**: Live financial data from multiple sources  
- **Self-Reflection**: AI evaluates its own output quality  
- **Continuous Learning**: Memory system improves over time  

---

## Agent Functions (33.8%)

### 1. Planning 
**LLM generates comprehensive research plans**
- Defines research objectives
- Identifies data sources
- Plans analysis steps
- Explains reasoning

### 2. Tool Usage 
**Dynamically coordinates APIs and agents**
- Yahoo Finance - Real-time stock data
- Alpha Vantage - Company fundamentals
- FRED - Economic indicators
- SEC EDGAR - Regulatory filings

### 3. Self-Reflection 
**LLM assesses output quality**
- Quality scores (0.0-1.0)
- Identifies strengths
- Identifies weaknesses
- Suggests improvements

### 4. Learning 
**Memory system for continuous improvement**
- Stores insights from analyses
- Tracks quality over time
- Learns from repeated analyses
- Applies past learnings

---

## Workflow Patterns (33.8%)

### 1. Prompt Chaining 
**5-step LLM pipeline**
- Ingest → Preprocess → Classify → Extract (LLM) → Summarize (LLM)

### 2. Routing 
**LLM-based intelligent routing**
- Analyzes query content
- Selects appropriate specialist
- Provides routing reasoning

### 3. Evaluator-Optimizer 
**Iterative quality improvement**
- LLM evaluates quality
- Suggests improvements
- Optimizes up to 3 iterations

---

## System Architecture

```
InvestmentResearchAgent (Main)
    ├── Planning (LLM)
    ├── API Clients
    │   ├── Yahoo Finance
    │   ├── Alpha Vantage
    │   ├── FRED
    │   └── SEC EDGAR
    ├── Specialized Agents
    │   ├── MarketDataAgent (LLM)
    │   ├── FundamentalsAgent (LLM)
    │   ├── EconomicContextAgent (LLM)
    │   └── RegulatoryAgent
    ├── Workflow Patterns
    │   ├── PromptChainWorkflow (LLM)
    │   ├── RoutingWorkflow (LLM)
    │   └── EvaluatorOptimizerWorkflow (LLM)
    └── Memory System
```

---

## Quick Start

### 1. Install Dependencies

```bash
uv init                         # initialize environment for uv
uv venv                         # create virtual environment
source .venv/Scripts/activate   # activate the virtual environment
uv sync                         # install dependencies from pyproject.toml
```

### 2. Configure API Keys

Create `.env` file:
```bash
# REQUIRED
OPENAI_API_KEY=sk-your-openai-key-here
ALPHA_VANTAGE_API_KEY=your-alphavantage-key
FRED_API_KEY=your-fred-key
```

**Get API Keys:**
- OpenAI: https://platform.openai.com/api-keys (Free $5 credit)
- Alpha Vantage: https://www.alphavantage.co/support/#api-key (Free)
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html (Free)

### 3. Run Analysis

```bash
python research_agent.py
```

**Or import in Python:**
```python
from research_agent import InvestmentResearchAgent

agent = InvestmentResearchAgent()
report = agent.conduct_research("AAPL")

print(f"Quality Score: {report['self_reflection']['overall_quality_score']:.2f}")
```

---

## Example Output

```
============================================================
# AUTONOMOUS RESEARCH: AAPL
# LLM-Powered Multi-Agent System
============================================================

[AGENT FUNCTION 1: PLANNING]
LLM-generated research plan created!
  Objectives: 5
  Analysis Steps: 7

[AGENT FUNCTION 2: TOOL USAGE]
  Yahoo Finance data fetched
  Market Data Agent analysis complete
  Fundamentals Agent analysis complete
  Economic Context Agent analysis complete
  Regulatory Agent analysis complete
  All 3 workflow patterns executed

[AGENT FUNCTION 3: SELF-REFLECTION]
Self-reflection complete!
  Overall Quality Score: 0.87/1.00
  Strengths identified: 3
  Improvements suggested: 3

[AGENT FUNCTION 4: LEARNING]
Learning recorded!
  Insights captured: 5
  Total memory entries: 1

============================================================
AUTONOMOUS RESEARCH COMPLETE
Quality Score: 0.87/1.00
All 4 Agent Functions: COMPLETE
============================================================
```

---

## Project Structure

```
.
├── api_clients.py           # Real API clients
├── agents.py                # Specialized agents
├── workflows.py             # Workflow patterns
├── data_models.py           # Data structures
├── research_agent.py        # Main agent
├── pyproject.toml           # Dependencies
├── .env.example             # API key template
├── tests/
│   ├── test_market_agent.py
│   └── test_all_agents.py
└── README.md
```

---

## Testing

### Test Individual Components

```bash
# Test API clients
python api_clients.py

# Test agents
python agents.py

# Test workflows
python workflows.py

# Test complete system
python research_agent.py
```

### Test Suite

```bash
# Test market agent
python tests/test_market_agent.py

# Test all agents
python tests/test_all_agents.py
```

---

## Project Requirements Met

### Agent Functions (33.8%)
- [x] **Planning**: LLM generates comprehensive research plans
- [x] **Tool Usage**: Coordinates 4 real APIs + 4 agents dynamically
- [x] **Self-Reflection**: LLM evaluates quality (0.0-1.0 scale)
- [x] **Learning**: Memory system stores insights for improvement

### Workflow Patterns (33.8%)
- [x] **Prompt Chaining**: 5-step pipeline with LLM
- [x] **Routing**: LLM intelligently routes to specialists
- [x] **Evaluator-Optimizer**: Iterative quality improvement

### Code Quality (32.4%)
- [x] Clean, documented code
- [x] PEP8 compliant
- [x] Comprehensive error handling
- [x] Test suite included
- [x] GitHub integration

---

## Repository

https://github.com/TamayiM-USD/g6-investment-agent

---

## Key Achievements

1. **Agentic AI**: System autonomously plans, executes, reflects, and learns
2. **LLM Integration**: OpenAI GPT-4o-mini throughout for intelligent reasoning
3. **Real Data**: All API calls are real - no demo/fallback data
4. **Multi-Agent**: 4 specialized agents coordinate intelligently
5. **Self-Aware**: Agent evaluates its own performance
6. **Continuous Learning**: Memory system improves with each analysis

---

## License

Academic use only - AAI-520 Final Project  
University of San Diego

---

## Acknowledgments

- OpenAI for GPT-4o-mini API
- Yahoo Finance for financial data
- Alpha Vantage for company fundamentals
- FRED for economic indicators
- SEC for regulatory data

---

## Contact

For questions about this project:
- Tamayi Mlanda - tmlanda@sandiego.edu
- Guruganesh Hegde - ghegde@sandiego.edu
- Arunkumar Rajaganapathy - arajaganapathy@sandiego.edu
