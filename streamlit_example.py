import streamlit as st
import yfinance as yf
import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()  # take environment variables


# Set your OpenAI API key
# It's best practice to set this as an environment variable
# st.secrets can also be used in Streamlit Cloud
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.error("⚠️ OPENAI_API_KEY not found in .env file or Streamlit secrets!")
    st.info("Please add your OpenAI API key to your .env file:\n\n`OPENAI_API_KEY=your_actual_api_key_here`")
    st.stop()

# --- LangChain Agent Setup ---

@tool
def get_stock_info(ticker: str) -> str:
    """Fetches real-time stock information for a given ticker symbol from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant information
        short_name = info.get('shortName', 'N/A')
        current_price = info.get('currentPrice', 'N/A')
        previous_close = info.get('previousClose', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        currency = info.get('currency', 'N/A')

        return (
            f"Stock: {short_name} ({ticker})\n"
            f"Current Price: {current_price} {currency}\n"
            f"Previous Close: {previous_close} {currency}\n"
            f"Market Cap: {market_cap}\n"
            f"Data Source: Yahoo Finance"
        )
    except Exception as e:
        return f"Error fetching stock information for {ticker}: {e}"

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo") # You can specify a model

tools = [get_stock_info]

prompt = PromptTemplate.from_template('''
You are a friendly and knowledgeable AI stock advisor chatbot. You help users get real-time stock information and provide insights about the market.

You have access to the following tools:
{tools}

Guidelines for responses:
- Be conversational and helpful
- Provide clear, actionable insights
- Format numbers and data nicely for readability
- If a user asks about multiple stocks, get information for each one
- Always provide context and explain what the data means
- Use emojis occasionally to make responses more engaging

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
''')

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Streamlit Chat UI ---

st.set_page_config(
    page_title="Stock Advisor Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Advisor Agent")
st.markdown("💬 Chat with your AI stock advisor powered by real-time market data!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I'm your AI stock advisor. I can help you get real-time stock information, analyze market data, and answer questions about any publicly traded company. What would you like to know?"
        }
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me about any stock..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing market data..."):
            try:
                # Invoke the agent
                result = agent_executor.invoke({"input": prompt})
                response = result['output']
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"I apologize, but I encountered an error while processing your request: {str(e)}"
                st.error(error_message)
                # Add error to chat history
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar with additional features
st.sidebar.header("🔧 Chat Controls")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I'm your AI stock advisor. I can help you get real-time stock information, analyze market data, and answer questions about any publicly traded company. What would you like to know?"
        }
    ]
    st.rerun()

# Example queries
st.sidebar.header("💡 Example Questions")
st.sidebar.markdown("""
- What's the current price of Apple (AAPL)?
- Compare Tesla and Ford stock prices
- What's Amazon's market cap?
- How is Microsoft performing today?
- Get me info on NVIDIA stock
- What's the latest on Google's stock?
""")

st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "This chat application uses a LangChain agent powered by OpenAI's GPT-3.5 Turbo "
    "to fetch real-time stock information from Yahoo Finance. "
    "Simply type your question in the chat input below!"
)

# Display current session info
st.sidebar.header("📊 Session Info")
st.sidebar.metric("Messages in Chat", len(st.session_state.messages))