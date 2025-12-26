# open-gemini-cli

AI coding agent for the terminal - with local LLM support.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Based on Google's [gemini-cli](https://github.com/google-gemini/gemini-cli), extended with local LLM support and additional providers for offline/private use.

## Added Features

- Local LLM support (MLX, llama.cpp, vLLM)
- Additional cloud providers (OpenAI, Anthropic)
- JSON-configurable provider system
- Optional authentication for local use

## Supported Providers

**Local LLMs**

- MLX Server
- llama.cpp
- vLLM

**Cloud Providers**

- OpenAI (GPT-4, GPT-5, Codex)
- Anthropic (Claude Opus, Sonnet, Haiku)
- Google Gemini (original functionality preserved)

## Installation

```bash
git clone https://github.com/limkcreply/open-gemini-cli
cd open-gemini-cli
npm install
npm run build
```

**Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

**Run:**
```bash
npx kaidex
# OR link globally:
npm link
kaidex
```

Requirements: Node.js 18+

## Quick Start (Local LLM)

```bash
# 1. Start your local LLM server (MLX, llama.cpp, etc.)

# 2. Configure .env
cp .env.example .env
# Set: BYPASS_AUTH=true
# Set: LLM_PROVIDER=local-mlx (or llamacpp, vllm)
# Set: KAIDEX_SERVER_URL=http://localhost:11435 (your server URL)

# 3. Run
npx kaidex
```

## Usage

### With Local LLM

```bash
# .env
BYPASS_AUTH=true
LLM_PROVIDER=llamacpp
LLAMACPP_SERVER_URL=http://localhost:8080
```

### With OpenAI

```bash
# .env
OPENAI_API_KEY=your-key
LLM_PROVIDER=gpt-4.1-mini
```

### With Anthropic

```bash
# .env
ANTHROPIC_API_KEY=your-key
LLM_PROVIDER=claude-sonnet
```

### With Google Gemini

```bash
# .env
GEMINI_API_KEY=your-key
LLM_PROVIDER=gemini-flash
```

Then run: `npx kaidex`

## Provider Configuration

Providers are configured in `bundle/llmProviders.json`. Add your own:

```json
{
  "providers": {
    "my-local-model": {
      "name": "My Local Model",
      "baseURL": "http://localhost:8080",
      "endpoint": "/v1/chat/completions",
      "format": "openai",
      "streaming": true
    }
  }
}
```

## Features

- File operations (read, write, edit)
- Shell command execution
- Web search (3-tier fallback: Tavily → Google Custom Search → Gemini)
- Web fetching
- MCP (Model Context Protocol) support
- Conversation checkpointing
- Project context via KAIDEX.md files
- Conductor: Context-driven development (optional)

## Conductor Extension (Optional)

Conductor is a bundled extension that adds a "plan before you code" workflow. It's disabled by default.

**Enable:**
```bash
# In .env
CONDUCTOR_ENABLED=true
```

**Commands:**
- `/conductor:setup` - Initialize project context (product, tech stack, workflow)
- `/conductor:newTrack <name>` - Create a feature/bug track with spec and plan
- `/conductor:implement <track-id>` - Execute the plan step by step
- `/conductor:status` - Show progress across all tracks
- `/conductor:revert <track-id>` - Undo a track's changes

**How it works:**

1. **Setup once**: Define your project (what you're building, tech stack, workflow)
2. **Create tracks**: For each feature, create a spec + implementation plan
3. **Implement**: AI follows the plan, checking off tasks as completed

Files are stored in `conductor/` directory and committed to git for team sharing.

## Web Search Setup

Web search works with all LLM providers using automatic fallback:

1. **Tavily AI** (recommended) - Free 1000 searches/month at [tavily.com](https://tavily.com)
2. **Google Custom Search** - [Setup guide](https://programmablesearchengine.google.com/)
3. **Google Gemini** - Built-in (Gemini provider only)

```bash
# In your .env file:
TAVILY_API_KEY=your-tavily-key
# OR
GOOGLE_API_KEY=your-google-key
GOOGLE_SEARCH_ENGINE_ID=your-search-engine-id
```

## Documentation

See the [docs](./docs) folder for detailed documentation on:

- [Configuration](./docs/cli/configuration.md)
- [Built-in Tools](./docs/tools/index.md)
- [MCP Servers](./docs/tools/mcp-server.md)

## License

Apache 2.0 - Same as the original gemini-cli.

## Credits

Based on [gemini-cli](https://github.com/google-gemini/gemini-cli) by Google.
