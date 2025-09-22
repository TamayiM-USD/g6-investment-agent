# Autonomous Investment Research Agent

# Overview

An autonomous Investment Research AI that integrates real-world financial data, coordinates multiple specialized agents, and demonstrates advanced agentic AI workflows, self-reflect on outputs, and learn across runs - delivering a professional, reproducible system for adaptive financial analysis.

# Objectives

The objectives of the project are to:

-	Build financial analysis pipelines that demonstrate Prompt Chaining, Routing, and Evaluator–Optimizer approaches.
-	Integrate diverse financial data sources (e.g., Yahoo Finance, FRED, SEC EDGAR, Alpha Vantage) and financial news datasets to enable real-world analysis.
-	Showcase multi-agent collaboration by coordinating specialized LLM agents (e.g., earnings analyzer, news analyzer, market trend analyzer) to complete complex financial tasks.
-	Highlight agentic AI capabilities such as reasoning, planning, and iterative improvement, moving beyond scripted pipelines toward adaptive, self-correcting workflows.
-	Apply software engineering best practices, including GitHub-based collaboration, PEP8-compliant code, and clear documentation of agent design and workflows.
-	Deliver a professional final artifact in the form of a reproducible and well-documented code notebook.


# Data Sources

Our project integrates multiple reputable financial and economic data sources to ensure robust and real-world analysis. Access will be through public APIs where available, but will also rely on offline data samples where necessary to avoid incurring costs. Some data is available through “unofficial” means / wrappers / scraping. These data sources include:

-	**Yahoo Finance (Yahoo Finance Platform)**  
  A widely used financial data service offering real-time and historical stock prices, company financials, and market news. It serves as a primary source for stock-level data and basic market indicators.

-	**FRED (Federal Reserve Economic Data)**  
  A comprehensive database maintained by the Federal Reserve Bank of St. Louis, providing thousands of U.S. and international economic and financial time series. It is particularly valuable for incorporating macroeconomic indicators into financial analyses.

-	**SEC EDGAR (U.S. Securities and Exchange Commission – Electronic Data Gathering, Analysis, and Retrieval system)**  
  The official repository for public company filings, including 10-K annual reports, 10-Q quarterly reports, and other regulatory disclosures. It is essential for extracting structured company financials and compliance information.

-	**Alpha Vantage (Alpha Vantage Inc.)**  
  A provider of APIs that supply real-time and historical data for equities, forex, and cryptocurrencies. It enables programmatic access to financial data at scale, complementing other datasets with broad coverage.
