# Claude Code with OpenAI Models - Easy Setup Guide

## What is this?
This tool lets you use Claude Code (Anthropic's command line assistant) but with OpenAI's models like GPT-4o and the reasoning model o3-mini instead. It's like connecting Claude's interface to OpenAI's brain!

### Benefits:
- Use Claude Code's nice interface with different AI models
- Easy switching between models through a simple web UI
- See what's happening with your requests in real-time
- No coding needed!

## Setup Guide

### For Mac Users

#### 1. Install Python (if you don't have it already)
- Download and install Python from [python.org](https://python.org)
- Make sure to select "Add Python to PATH" during installation

#### 2. Install Git (for downloading the code)
- Download and install Git from [git-scm.com](https://git-scm.com)
- Or use Homebrew if you have it: `brew install git`

#### 3. Open Terminal
- Press Command+Space, type "Terminal" and press Enter

#### 4. Download the project
```bash
git clone https://github.com/1rgs/claude-code-openai.git
cd claude-code-openai
```

#### 5. Set up your environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install httpx fastapi uvicorn litellm python-dotenv python-multipart
```

#### 6. Create a .env file with your OpenAI API key
```bash
echo "OPENAI_API_KEY=your-openai-key" > .env
```
(Replace "your-openai-key" with your actual OpenAI API key)

#### 7. Start the server
```bash
uvicorn server:app --host 0.0.0.0 --port 8082
```

#### 8. Install Claude Code (in a new Terminal window)
```bash
npm install -g @anthropic-ai/claude-code
```

### For Linux Users

#### 1. Install Python (if you don't have it already)
- **Ubuntu/Debian**: `sudo apt update && sudo apt install python3 python3-venv python3-pip`
- **Fedora**: `sudo dnf install python3 python3-pip`
- **Arch**: `sudo pacman -S python python-pip`

#### 2. Install Git (for downloading the code)
- **Ubuntu/Debian**: `sudo apt install git`
- **Fedora**: `sudo dnf install git`
- **Arch**: `sudo pacman -S git`

#### 3. Open Terminal
- Open your terminal application

#### 4. Download the project
```bash
git clone https://github.com/1rgs/claude-code-openai.git
cd claude-code-openai
```

#### 5. Set up your environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install httpx fastapi uvicorn litellm python-dotenv python-multipart
```

#### 6. Create a .env file with your OpenAI API key
```bash
echo "OPENAI_API_KEY=your-openai-key" > .env
```
(Replace "your-openai-key" with your actual OpenAI API key)

#### 7. Start the server
```bash
uvicorn server:app --host 0.0.0.0 --port 8082
```

#### 8. Install Claude Code (in a new terminal window)
```bash
npm install -g @anthropic-ai/claude-code
```

## How to Use

### Step 1: Start the Server
Make sure the server is running from the steps above. Keep this window open!

### Step 2: Access the Web Interface
Open your web browser and go to:
```
http://localhost:8082
```
You'll see a friendly user interface where you can:
- Choose which models to use
- View request history
- Monitor connections in real-time
- See how models are mapped

### Step 3: Configure Your Models
- **Big Model** (used for Claude Sonnet): Choose from options like gpt-4o or o3-mini
- **Small Model** (used for Claude Haiku): Usually gpt-4o-mini for speed

> **Note**: o3-mini and o1 are "reasoning models" that will automatically get special settings (reasoning_effort="high") for better results.

### Step 4: Connect Claude Code
Open a new terminal window and run:
```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it! You're now using Claude Code with OpenAI models!

## Understanding the UI

### Configuration Panel
- **Model Selection**: Choose which models to use for big and small requests
- **Automatic Reasoning**: The proxy automatically detects reasoning models (o3-mini, o1) and adds the required parameters
- **Save Configuration**: Your choices are saved to the .env file for persistence

### Connection Info
- Shows how to connect Claude to the proxy
- Displays the current model mappings

### Request History
- Real-time view of all requests processed by the proxy
- Shows original model, mapped model, message count, and status
- Includes a refresh button to update the view

## About the Different Models

| Model | Description |
|-------|-------------|
| **gpt-4o** | Good all-around model, very capable |
| **gpt-4o-mini** | Faster and cheaper, but less powerful |
| **o3-mini** | Special reasoning model, great for coding and complex thinking |
| **o1** | The most powerful reasoning model, but slower and more expensive |
