import yfinance as yf
import requests
import re
from typing import Dict, Any, List, Optional
from cache_manager import cache_response


class YahooFinanceClient:
    """Yahoo Finance API client - Real data only"""
    
    def __init__(self):
        self.name = "Yahoo Finance"
    
    @cache_response("Yahoo Finance")
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch comprehensive stock information
        Returns real data from Yahoo Finance API
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key financial metrics
            stock_data = {
                "symbol": symbol,
                "company_name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
                "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose", 0),
                "open_price": info.get("open") or info.get("regularMarketOpen", 0),
                "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh", 0),
                "day_low": info.get("dayLow") or info.get("regularMarketDayLow", 0),
                "volume": info.get("volume") or info.get("regularMarketVolume", 0),
                "pe_ratio": info.get("trailingPE", None),
                "forward_pe": info.get("forwardPE", None),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "beta": info.get("beta", None),
                "dividend_yield": info.get("dividendYield", None),
                "profit_margin": info.get("profitMargins", None),
                "operating_margin": info.get("operatingMargins", None),
                "revenue": info.get("totalRevenue", 0),
                "earnings_growth": info.get("earningsGrowth", None),
                "revenue_growth": info.get("revenueGrowth", None),
                "ebitda": info.get("ebitda", None),
                "debt_to_equity": info.get("debtToEquity", None),
                "return_on_equity": info.get("returnOnEquity", None),
                "currency": info.get("currency", "USD"),
            }
            
            return stock_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    @cache_response("Yahoo Finance")
    def get_news(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch recent news articles
        Returns real news from Yahoo Finance
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return []
            
            news_items = []
            for article in news[:limit]:
                news_items.append({
                    "title": article.get("title", ""),
                    "publisher": article.get("publisher", ""),
                    "link": article.get("link", ""),
                    "published_date": article.get("providerPublishTime", ""),
                    "type": article.get("type", "")
                })
            
            return news_items
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch news for {symbol}: {str(e)}")
    
    @cache_response("Yahoo Finance")
    def get_historical_data(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """
        Fetch historical price data
        Period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {"error": "No historical data available"}
            
            return {
                "symbol": symbol,
                "period": period,
                "data_points": len(hist),
                "latest_close": float(hist['Close'].iloc[-1]),
                "period_high": float(hist['High'].max()),
                "period_low": float(hist['Low'].min()),
                "average_volume": int(hist['Volume'].mean()),
                "price_change": float(hist['Close'].iloc[-1] - hist['Close'].iloc[0]),
                "price_change_percent": float(
                    ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                )
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch historical data for {symbol}: {str(e)}")


if __name__ == "__main__":
    # Test the client
    print("Testing Yahoo Finance Client...")
    print("="*50)
    
    client = YahooFinanceClient()
    
    # Test stock info
    print("\n1. Testing get_stock_info('AAPL')...")
    try:
        info = client.get_stock_info("AAPL")
        print(f"Company: {info['company_name']}")
        print(f"Price: ${info['current_price']}")
        print(f"Market Cap: ${info['market_cap']:,}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test news
    print("\n2. Testing get_news('AAPL')...")
    try:
        news = client.get_news("AAPL", limit=3)
        print(f"Found {len(news)} news articles")
        if news:
            print(f"  Latest: {news[0]['title'][:60]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test historical data
    print("\n3. Testing get_historical_data('AAPL')...")
    try:
        hist = client.get_historical_data("AAPL", period="1mo")
        print(f"Data points: {hist['data_points']}")
        print(f"Price change: {hist['price_change_percent']:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50)
    print("Yahoo Finance Client: READY ")


class AlphaVantageClient:
    """Alpha Vantage API client - Real API calls with key"""
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key required. "
                "Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter. "
                "Get free key at: https://www.alphavantage.co/support/#api-key"
            )
        self.base_url = "https://www.alphavantage.co/query"
        self.name = "Alpha Vantage"
    
    @cache_response("Alpha Vantage", ttl=86400)  # 24 hours to maximize 25/day limit
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive company fundamentals
        Requires valid API key - no demo mode
        """
        import requests
        
        try:
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Note" in data:
                raise RuntimeError(
                    "API rate limit reached. Alpha Vantage free tier: 25 requests/day. "
                    "Wait or upgrade at: https://www.alphavantage.co/premium/"
                )
            
            if "Error Message" in data:
                raise RuntimeError(f"API Error: {data['Error Message']}")
            
            if not data or "Symbol" not in data:
                raise RuntimeError(f"No data returned for {symbol}")
            
            # Return comprehensive overview
            return {
                "symbol": symbol,
                "name": data.get("Name", ""),
                "description": data.get("Description", ""),
                "sector": data.get("Sector", ""),
                "industry": data.get("Industry", ""),
                "market_cap": data.get("MarketCapitalization", ""),
                "pe_ratio": data.get("PERatio", ""),
                "peg_ratio": data.get("PEGRatio", ""),
                "book_value": data.get("BookValue", ""),
                "dividend_yield": data.get("DividendYield", ""),
                "eps": data.get("EPS", ""),
                "revenue_per_share": data.get("RevenuePerShareTTM", ""),
                "profit_margin": data.get("ProfitMargin", ""),
                "operating_margin": data.get("OperatingMarginTTM", ""),
                "return_on_assets": data.get("ReturnOnAssetsTTM", ""),
                "return_on_equity": data.get("ReturnOnEquityTTM", ""),
                "revenue": data.get("RevenueTTM", ""),
                "gross_profit": data.get("GrossProfitTTM", ""),
                "ebitda": data.get("EBITDA", ""),
                "analyst_target_price": data.get("AnalystTargetPrice", ""),
                "52_week_high": data.get("52WeekHigh", ""),
                "52_week_low": data.get("52WeekLow", ""),
            }
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error fetching data from Alpha Vantage: {str(e)}")
    
    @cache_response("Alpha Vantage", ttl=86400)  # 24 hours to maximize 25/day limit
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote data"""
        import requests
        
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "Note" in data:
                raise RuntimeError("API rate limit reached")
            
            quote = data.get("Global Quote", {})
            if not quote:
                raise RuntimeError(f"No quote data for {symbol}")
            
            return {
                "symbol": quote.get("01. symbol", ""),
                "price": float(quote.get("05. price", 0)),
                "volume": int(quote.get("06. volume", 0)),
                "latest_trading_day": quote.get("07. latest trading day", ""),
                "previous_close": float(quote.get("08. previous close", 0)),
                "change": float(quote.get("09. change", 0)),
                "change_percent": quote.get("10. change percent", ""),
            }
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error: {str(e)}")



if __name__ == "__main__":
    import os
    
    print("\nTesting Alpha Vantage Client...")
    print("="*50)
    
    # Check for API key
    if not os.getenv("ALPHA_VANTAGE_API_KEY"):
        print("ALPHA_VANTAGE_API_KEY not set")
        print("Get free key: https://www.alphavantage.co/support/#api-key")
        print("Then set: export ALPHA_VANTAGE_API_KEY=your-key")
    else:
        try:
            client = AlphaVantageClient()
            
            print("\n1. Testing get_company_overview('IBM')...")
            overview = client.get_company_overview("IBM")
            print(f"Company: {overview['name']}")
            print(f"Sector: {overview['sector']}")
            print(f"PE Ratio: {overview['pe_ratio']}")
            
            print("\n2. Testing get_quote('IBM')...")
            quote = client.get_quote("IBM")
            print(f"Price: ${quote['price']}")
            print(f"Change: {quote['change_percent']}")
            
            print("\n" + "="*50)
            print("Alpha Vantage Client: READY ")
            
        except Exception as e:
            print(f"Error: {e}")
            print("\nNote: Free tier has 25 requests/day limit")


class FREDClient:
    """FRED API client for economic indicators - Real API only"""
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED API key required. "
                "Set FRED_API_KEY environment variable. "
                "Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.name = "FRED"
    
    @cache_response("FRED", ttl=604800)  # 7 days - economic data updates slowly
    def get_economic_indicator(self, series_id: str, limit: int = 12) -> Dict[str, Any]:
        """
        Fetch economic indicator data from FRED
        Real API call - requires valid key

        Common series:
        - DFF: Federal Funds Rate
        - UNRATE: Unemployment Rate
        - CPIAUCSL: Consumer Price Index
        - GDP: Gross Domestic Product
        """
        import requests
        
        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "limit": limit,
                "sort_order": "desc"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "error_code" in data:
                raise RuntimeError(f"FRED API Error: {data.get('error_message', 'Unknown error')}")
            
            observations = data.get("observations", [])
            if not observations:
                raise RuntimeError(f"No data available for series {series_id}")
            
            # Get series metadata
            series_info = self._get_series_info(series_id)
            
            return {
                "series_id": series_id,
                "name": series_info["name"],
                "units": series_info["units"],
                "frequency": series_info["frequency"],
                "observations": [
                    {
                        "date": obs.get("date"),
                        "value": obs.get("value"),
                        "is_current": i == 0
                    }
                    for i, obs in enumerate(observations)
                ],
                "latest_value": observations[0].get("value") if observations else None,
                "latest_date": observations[0].get("date") if observations else None,
                "data_points": len(observations)
            }
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error fetching FRED data: {str(e)}")
    
    def _get_series_info(self, series_id: str) -> Dict[str, str]:
        """Get metadata about a series"""
        series_map = {
            "DFF": {
                "name": "Federal Funds Effective Rate",
                "units": "Percent",
                "frequency": "Daily"
            },
            "UNRATE": {
                "name": "Unemployment Rate",
                "units": "Percent",
                "frequency": "Monthly"
            },
            "CPIAUCSL": {
                "name": "Consumer Price Index for All Urban Consumers",
                "units": "Index 1982-1984=100",
                "frequency": "Monthly"
            },
            "GDP": {
                "name": "Gross Domestic Product",
                "units": "Billions of Dollars",
                "frequency": "Quarterly"
            },
            "MORTGAGE30US": {
                "name": "30-Year Fixed Rate Mortgage Average",
                "units": "Percent",
                "frequency": "Weekly"
            }
        }
        
        return series_map.get(series_id, {
            "name": series_id,
            "units": "See FRED documentation",
            "frequency": "Varies"
        })
    
    def get_multiple_indicators(self, series_ids: List[str]) -> Dict[str, Any]:
        """Fetch multiple economic indicators"""
        results = {}
        
        for series_id in series_ids:
            try:
                results[series_id] = self.get_economic_indicator(series_id, limit=5)
            except Exception as e:
                results[series_id] = {"error": str(e)}
        
        return results


if __name__ == "__main__":
    import os
    
    print("\nTesting FRED Client...")
    print("="*50)
    
    if not os.getenv("FRED_API_KEY"):
        print("FRED_API_KEY not set")
        print("Get free key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("Then set: export FRED_API_KEY=your-key")
    else:
        try:
            client = FREDClient()
            
            print("\n1. Testing Federal Funds Rate...")
            dff = client.get_economic_indicator("DFF")
            print(f"{dff['name']}")
            print(f"Latest: {dff['latest_value']}% ({dff['latest_date']})")
            
            print("\n2. Testing Unemployment Rate...")
            unrate = client.get_economic_indicator("UNRATE")
            print(f"{unrate['name']}")
            print(f"Latest: {unrate['latest_value']}% ({unrate['latest_date']})")
            
            print("\n3. Testing Multiple Indicators...")
            multi = client.get_multiple_indicators(["DFF", "UNRATE", "GDP"])
            print(f"Fetched {len([k for k, v in multi.items() if 'error' not in v])} indicators")
            
            print("\n" + "="*50)
            print("FRED Client: READY ")
            
        except Exception as e:
            print(f"Error: {e}")


class SECEdgarClient:
    """SEC EDGAR API client for regulatory filings"""
    
    def __init__(self):
        self.base_url = "https://data.sec.gov"
        self.name = "SEC EDGAR"
        self.headers = {
            "User-Agent": "University of San Diego AAI-520 Research tmlanda@sandiego.edu",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }
    
    @cache_response("SEC EDGAR", ttl=2592000)  # 30 days - filings are historical
    def get_company_submissions(self, ticker: str) -> Dict[str, Any]:
        """
        Get company CIK and recent filings
        Real SEC EDGAR API call
        """
        import requests
        
        try:
            # First, get CIK from ticker
            cik = self._get_cik_from_ticker(ticker)
            if not cik:
                raise RuntimeError(f"Could not find CIK for ticker {ticker}")
            
            # Fetch submissions
            url = f"{self.base_url}/submissions/CIK{cik}.json"
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Extract recent filings
            recent_filings = data.get("filings", {}).get("recent", {})
            
            filings_list = []
            if recent_filings:
                forms = recent_filings.get("form", [])
                dates = recent_filings.get("filingDate", [])
                accessions = recent_filings.get("accessionNumber", [])
                
                for i in range(min(10, len(forms))):
                    filings_list.append({
                        "form_type": forms[i],
                        "filing_date": dates[i],
                        "accession_number": accessions[i],
                        "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={forms[i]}&dateb=&owner=exclude"
                    })
            
            return {
                "ticker": ticker,
                "cik": cik,
                "company_name": data.get("name", ""),
                "sic": data.get("sic", ""),
                "sic_description": data.get("sicDescription", ""),
                "recent_filings": filings_list,
                "total_filings": len(forms) if forms else 0
            }
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch SEC data: {str(e)}")
    
    def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK number from ticker symbol using SEC's search API"""

        try:
            # Use SEC's search/browse functionality
            ticker_upper = ticker.upper()
            search_url = "https://www.sec.gov/cgi-bin/browse-edgar"

            params = {
                "action": "getcompany",
                "company": ticker_upper,
                "type": "",
                "dateb": "",
                "owner": "exclude",
                "output": "atom",
                "count": "1"
            }

            response = requests.get(search_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()

            # Extract CIK from XML response
            # Look for CIK in the response
            cik_match = re.search(r'<CIK>(\d+)</CIK>', response.text)
            if cik_match:
                cik = cik_match.group(1).zfill(10)
                return cik

            return None

        except Exception as e:
            print(f"Warning: Could not fetch CIK for {ticker}: {e}")

            # Fallback: Use a hardcoded mapping for common tickers
            # This is a last resort for well-known companies
            common_tickers = {
                "AAPL": "0000320193",
                "MSFT": "0000789019",
                "GOOGL": "0001652044",
                "GOOG": "0001652044",
                "AMZN": "0001018724",
                "TSLA": "0001318605",
                "META": "0001326801",
                "NVDA": "0001045810",
                "JPM": "0000019617",
                "V": "0001403161",
                "IBM": "0000051143",
                "NFLX": "0001065280",
                "DIS": "0001001039",
                "BA": "0000012927",
                "INTC": "0000050863"
            }

            cik = common_tickers.get(ticker.upper())
            if cik:
                print(f"Info: Using cached CIK for {ticker}")
                return cik

            return None
    
    def get_filing_content(self, accession_number: str, cik: str) -> str:
        """
        Get the text content of a specific filing
        Note: Returns URL - parsing full filing is complex
        """
        accession_no_dashes = accession_number.replace("-", "")
        url = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession_number}"
        return url


if __name__ == "__main__":
    print("\nTesting SEC EDGAR Client...")
    print("="*50)
    
    try:
        client = SECEdgarClient()
        
        print("\n1. Testing get_company_submissions('AAPL')...")
        submissions = client.get_company_submissions("AAPL")
        print(f"Company: {submissions['company_name']}")
        print(f"CIK: {submissions['cik']}")
        print(f"Total filings: {submissions['total_filings']}")
        print(f"Recent filings: {len(submissions['recent_filings'])}")
        
        if submissions['recent_filings']:
            latest = submissions['recent_filings'][0]
            print(f"  Latest: {latest['form_type']} on {latest['filing_date']}")
        
        print("\n" + "="*50)
        print("SEC EDGAR Client: READY ")
        print("\nNote: SEC enforces rate limits (10 requests/second)")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Ensure proper User-Agent header is set")
