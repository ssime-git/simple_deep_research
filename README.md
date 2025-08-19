# Simple Deep Research Agent

A powerful research agent built with LangGraph, deepagents, and Kimi-K2-Instruct model via Groq for lightning-fast inference and massive context windows.

## Features

- 🧠 **Kimi-K2-Instruct Model**: 131,072 token context window via Groq
- 🔍 **Web Search Integration**: Tavily search for real-time information
- 🤖 **Multi-Agent System**: Research and critique sub-agents
- ⚡ **Fast Inference**: Groq infrastructure for speed
- 🎪 **LangGraph Studio**: Visual interface for agent interaction

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- API Keys:
  - [Groq API Key](https://console.groq.com/keys)
  - [Tavily API Key](https://tavily.com/)

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd simple_deep_research
```

### 2. Install Dependencies

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
# Required API Keys
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Install Project in Editable Mode

```bash
# This enables proper import resolution for LangGraph
uv pip install -e .
```

### 5. Start the LangGraph Server

```bash
# For local access only
uv run langgraph dev

# For remote access (tunneled)
uv run langgraph dev --tunnel

# For network access (bind to all interfaces)
uv run langgraph dev --host 0.0.0.0
```

## Remote Access Setup (SSH Port Forwarding)

If you're running the server on a remote machine (WSL, VM, etc.) and want to access it from another device:

### 1. Start Server on Remote Machine
```bash
uv run langgraph dev
```

### 2. SSH Port Forwarding from Local Machine
```bash
# Forward local port 2024 to remote port 2024
ssh -L 2024:localhost:2024 username@remote_machine_ip

# Keep this SSH connection open while using the Studio
```

### 3. Access Studio UI
Open in your local browser:
```
https://smith.langchain.com/studio/?baseUrl=http://localhost:2024
```

## API Access

### Local API Endpoints
- **API Server**: http://127.0.0.1:2024
- **API Documentation**: http://127.0.0.1:2024/docs
- **Studio UI**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

### Remote API Access (with SSH forwarding)
- **API Server**: http://localhost:2024 (forwarded)
- **API Documentation**: http://localhost:2024/docs
- **Studio UI**: https://smith.langchain.com/studio/?baseUrl=http://localhost:2024

## Project Structure

```
simple_deep_research/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── research_agent.py     # Main agent definition
│   └── prompts.py           # Agent prompts and instructions
├── .env                     # Environment variables (create this)
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Model Configuration

The agent uses **Kimi-K2-Instruct** through Groq:
- **Model**: `moonshotai/kimi-k2-instruct`
- **Context Window**: 131,072 tokens
- **Provider**: Groq (fast inference)
- **Temperature**: 0.1 (focused research)

## Usage Examples

### Via LangGraph Studio
1. Open the Studio UI in your browser
2. Find the "research" graph
3. Start a new conversation
4. Ask research questions like:
   - "Research the latest developments in AI safety"
   - "Compare renewable energy adoption rates globally"
   - "Analyze recent trends in cryptocurrency markets"

### Via API
```python
import requests

# Example API call
response = requests.post(
    "http://localhost:2024/threads/runs/stream",
    json={
        "input": "Research the impact of AI on job markets",
        "graph_id": "research"
    }
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure editable installation
   uv pip install -e .
   ```

2. **Connection Refused (Remote Access)**
   ```bash
   # Check SSH port forwarding is active
   ssh -L 2024:localhost:2024 user@host
   ```

3. **Model Not Found**
   - Verify GROQ_API_KEY is correct
   - Check Groq API status

4. **No Search Results**
   - Verify TAVILY_API_KEY is correct
   - Check internet connection

### Server Options

```bash
# Development server with hot reload
uv run langgraph dev

# Custom port
uv run langgraph dev --port 3000

# External access
uv run langgraph dev --host 0.0.0.0

# Public tunnel (easiest for remote access)
uv run langgraph dev --tunnel

# No browser auto-open
uv run langgraph dev --no-browser
```

## Development

### Adding New Tools
Add functions to `research_agent.py` and include them in the tools list:

```python
def new_tool(query: str):
    # Your tool logic here
    return results

# Add to agent creation
agent = create_deep_agent(
    [internet_search, new_tool],  # Include new tool
    RESEARCH_INSTRUCTIONS,
    model=kimi_model,
    subagents=[critique_sub_agent, research_sub_agent],
)
```

### Modifying Prompts
Edit prompts in `src/prompts.py`. Changes are automatically reloaded in dev mode.

### Custom Models
To use a different model, update `research_agent.py`:

```python
# Example: Switch to different Groq model
kimi_model = ChatGroq(
    model="llama-3.3-70b-versatile",  # Different model
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.1,
)
```

## Performance Tips

- **Context Window**: Kimi-K2 supports 131k tokens - great for large documents
- **Temperature**: Lower values (0.1) for factual research, higher (0.7) for creative tasks
- **Parallel Sub-agents**: The system automatically parallelizes research tasks
- **Caching**: Tavily results are automatically cached during the session

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `uv run langgraph dev`
5. Submit a pull request

