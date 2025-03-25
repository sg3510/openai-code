# Claude Code Proxy: Use OpenAI & Google Gemini Models

![Claude Code Proxy UI](https://github.com/user-attachments/assets/afa0074c-6f43-4c34-87e2-d0d3bd3ad2af)

## What is this?

This tool acts as a local proxy server that allows you to use the **Claude Code** command-line interface (Anthropic's agentic coding assistant) with different powerful AI models from **OpenAI** (like GPT-4o, o3-mini) or **Google Gemini** (like Gemini 1.5 Pro, Gemini 1.5 Flash).

It intercepts requests from the `claude` CLI, translates them using LiteLLM, sends them to your chosen provider (OpenAI or Google), and returns the response, effectively swapping the "brain" behind the Claude Code interface.

### Key Features & Benefits:

*   **Model Flexibility:** Use Claude Code's excellent terminal interface with state-of-the-art models like GPT-4o or Gemini 1.5 Pro.
*   **Easy Configuration:** Switch between OpenAI, Google Gemini, or even original Anthropic models via a simple web UI.
*   **Provider Agnostic:** Leverages LiteLLM to seamlessly interact with different model providers.
*   **Real-time Monitoring:** The web UI shows request history, model mappings, and status updates.
*   **Simple Setup:** Get up and running quickly with minimal dependencies.
*   **No Cloud Hosting:** Runs entirely on your local machine.

## Prerequisites

1.  **Python:** Version 3.10 or higher.
2.  **Git:** For cloning the project repository.
3.  **Node.js & npm:** Version 18 or higher (for installing the `claude-code` CLI itself).
4.  **API Keys:** You **must** have API keys for the model providers you intend to use:
    *   **Google Gemini:** Get a key from [Google AI Studio](https://makersuite.google.com/app/apikey). (Required if selecting Gemini models).
    *   **OpenAI:** Get a key from [OpenAI Platform](https://platform.openai.com/api-keys). (Required if selecting OpenAI models).
    *   **Anthropic (Optional):** Get a key from [Anthropic Console](https://console.anthropic.com). (Only required if you want to select *actual* Claude models via the proxy).

## Setup Guide

Follow the steps for your operating system.

### For macOS Users

1.  **Install Python:** If you don't have it, download from [python.org](https://python.org) (ensure "Add Python to PATH" is checked if applicable) or use Homebrew (`brew install python`).
2.  **Install Git:** If you don't have it, download from [git-scm.com](https://git-scm.com) or use Homebrew (`brew install git`).
3.  **Install Node.js & npm:** If you don't have them, download from [nodejs.org](https://nodejs.org/) or use Homebrew (`brew install node`).
4.  **Open Terminal:** (Applications > Utilities > Terminal or use Spotlight).
5.  **Clone the Project:**
    ```bash
    # Replace with the actual repo URL if different
    git clone https://github.com/1rgs/claude-code-openai.git
    cd claude-code-openai
    ```
6.  **Set Up Python Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # Install required Python packages
    pip install httpx fastapi uvicorn litellm python-dotenv python-multipart Jinja2
    ```
7.  **Create `.env` File for API Keys:**
    Create a file named `.env` in the `claude-code-openai` directory. Add the API keys for *all* providers you might want to use. The server will pick the correct key based on your UI selection.

    ```dotenv
    # .env file contents:

    # Required if using OpenAI models (gpt-4o, o3-mini, etc.)
    OPENAI_API_KEY=your-openai-key-here

    # Required if using Google Gemini models (gemini-1.5-pro, etc.)
    GOOGLE_API_KEY=your-google-gemini-key-here

    # Optional: Only required if selecting original Claude models in the UI
    # ANTHROPIC_API_KEY=your-anthropic-key-here
    ```
    *(Replace `your-...-key-here` with your actual keys. Leave a key blank or commented out if you don't have/need it, but the server will error if you select a model without its corresponding key present).*

8.  **Start the Proxy Server:**
    ```bash
    uvicorn server:app --host 0.0.0.0 --port 8082
    ```
    Keep this terminal window running. You'll see logs here.

9.  **Install Claude Code CLI (in a *new* Terminal window):**
    ```bash
    npm install -g @anthropic-ai/claude-code
    ```
    *(If you encounter permission errors, **do not use `sudo`**. Refer to the official [Claude Code Installation Guide](https://docs.anthropic.com/en/code#install-and-authenticate) for troubleshooting permissions safely).*

### For Linux Users

1.  **Install Prerequisites:**
    *   **Ubuntu/Debian:** `sudo apt update && sudo apt install python3 python3-venv python3-pip git nodejs npm`
    *   **Fedora:** `sudo dnf install python3 python3-pip git nodejs npm`
    *   **Arch:** `sudo pacman -S python python-pip git nodejs npm`
    *(Ensure Node.js version is 18+)*
2.  **Open Terminal.**
3.  **Clone the Project:**
    ```bash
    # Replace with the actual repo URL if different
    git clone https://github.com/1rgs/claude-code-openai.git
    cd claude-code-openai
    ```
4.  **Set Up Python Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # Install required Python packages
    pip install httpx fastapi uvicorn litellm python-dotenv python-multipart Jinja2
    ```
5.  **Create `.env` File for API Keys:**
    Create a file named `.env` in the `claude-code-openai` directory. Add the API keys for *all* providers you might want to use.

    ```dotenv
    # .env file contents:

    # Required if using OpenAI models (gpt-4o, o3-mini, etc.)
    OPENAI_API_KEY=your-openai-key-here

    # Required if using Google Gemini models (gemini-1.5-pro, etc.)
    GOOGLE_API_KEY=your-google-gemini-key-here

    # Optional: Only required if selecting original Claude models in the UI
    # ANTHROPIC_API_KEY=your-anthropic-key-here
    ```
    *(Replace `your-...-key-here` with your actual keys).*

6.  **Start the Proxy Server:**
    ```bash
    uvicorn server:app --host 0.0.0.0 --port 8082
    ```
    Keep this terminal window running.

7.  **Install Claude Code CLI (in a *new* Terminal window):**
    ```bash
    npm install -g @anthropic-ai/claude-code
    ```
    *(If you encounter permission errors, **do not use `sudo`**. Refer to the official [Claude Code Installation Guide](https://docs.anthropic.com/en/code#install-and-authenticate) for troubleshooting permissions safely).*

## How to Use

1.  **Run the Proxy Server:** Make sure the `uvicorn server:app ...` command from the setup steps is running in a terminal.
2.  **Access the Web UI:** Open your web browser and navigate to `http://localhost:8082`.
3.  **Configure Models:**
    *   Use the dropdown menus in the UI to select the specific models you want to use.
    *   **Big Model (for Sonnet):** Choose the model to handle requests originally intended for `claude-3-sonnet`. Examples: `gemini-1.5-pro`, `gpt-4o`, `o3-mini`.
    *   **Small Model (for Haiku):** Choose the model to handle requests originally intended for `claude-3-haiku`. Examples: `gemini-1.5-flash`, `gpt-4o-mini`.
    *   > **Important:** Ensure the API key for the provider of your selected models (Google, OpenAI, or Anthropic) exists and is correctly set in your `.env` file. If the key is missing, you will get an error when making requests.
4.  **Save Configuration:** Click the "Save Configuration" button in the UI.
5.  **Connect Claude Code CLI:** Open a **new** terminal window (do not reuse the server window) and run the `claude` command, prefixing it with the `ANTHROPIC_BASE_URL` environment variable pointing to your proxy server:
    ```bash
    export ANTHROPIC_BASE_URL=http://localhost:8082
    claude "Your prompt here, e.g., explain this project"
    # Or just run `claude` to enter its interactive mode
    ```
    *(You can add the `export ANTHROPIC_BASE_URL...` line to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.) to make it permanent).*

Now, any commands you run with the `claude` CLI will be processed by the proxy server and routed to the Google Gemini or OpenAI models you selected!

## Understanding the UI (`http://localhost:8082`)

*   **Configuration Panel:**
    *   Select the specific backend models (Gemini, OpenAI, Claude) to map to Claude Code's internal 'Sonnet' (Big) and 'Haiku' (Small) requests.
    *   Reminds you about the API key requirement in `.env`.
    *   "Save Configuration" applies your choices immediately on the running server.
*   **Connection Info:**
    *   Shows the command needed to point the `claude` CLI to this proxy.
    *   Displays the *current* mapping (e.g., "Claude Sonnet Requests -> `gemini-1.5-pro (Google)`").
*   **Request History:**
    *   Shows a table of recent requests processed by the proxy.
    *   **Columns:** Timestamp, Original Model (from `claude` CLI), Mapped Model (actual model used), Provider (deduced), Message Count, Stream (Yes/No), Status (Success/Error/Processing), Info (Error message snippet).
    *   Auto-refreshes every 10 seconds; manual refresh button available.
    *   Error snippets in the 'Info' column are clickable to show the full error in a modal.
    *   Model names are color-coded by provider (Blue=Gemini, Teal=OpenAI, Orange=Anthropic).

## Supported Models (Examples)

The UI dropdown lists available models. Here's a reference for common choices:

| Model                       | Provider | Description                                       | Notes                                                 |
| :-------------------------- | :------- | :------------------------------------------------ | :---------------------------------------------------- |
| **`gemini-1.5-pro`**        | Google   | Highly capable multimodal model, large context  | Excellent reasoning, good for complex coding tasks    |
| **`gemini-1.5-flash`**      | Google   | Faster, lower-cost multimodal model             | Great balance of performance and cost                 |
| **`gemini-2.0-flash`**      | Google   | Newer Flash variant                               | Fast multimodal option                                |
| `gemini-2.5-pro-exp-03-25`  | Google   | Experimental advanced reasoning model             | May change, potentially powerful                      |
| **`gpt-4o`**                | OpenAI   | Powerful all-around model, multimodal           | Good balance of capability and speed                  |
| **`gpt-4o-mini`**           | OpenAI   | Faster and cheaper, good for simpler tasks      | Less powerful than gpt-4o                             |
| **`o3-mini`**               | OpenAI   | Specialized reasoning model                       | Great for coding, `reasoning_effort="high"` applied |
| **`o1`**                    | OpenAI   | Most powerful OpenAI reasoning model            | Slower/expensive, `reasoning_effort="high"` applied |
| `claude-3-opus-...`         | Anthropic| Original Anthropic models (Opus, Sonnet, Haiku) | Requires `ANTHROPIC_API_KEY` if selected              |
| `claude-3-sonnet-...`       | Anthropic| "                                                 | "                                                     |
| `claude-3-haiku-...`        | Anthropic| "                                                 | "                                                     |

*Note: The `reasoning_effort="high"` parameter is automatically applied by this proxy *only* when using OpenAI's `o1` or `o3-mini` models.*

## Troubleshooting

*   **Error: `API key for provider 'google'/'openai' not configured...`**: Make sure the required API key (`GOOGLE_API_KEY` or `OPENAI_API_KEY`) is correctly set in your `.env` file and matches the model provider you selected in the UI. Restart the server after editing `.env`.
*   **Error: `Port 8082 is already in use`**: Another application is using port 8082. Stop that application or change the port in the `uvicorn` command (e.g., `--port 8083`) and update `ANTHROPIC_BASE_URL` accordingly.
*   **`claude` command not found:** Ensure you installed `@anthropic-ai/claude-code` globally with `npm` and that npm's global bin directory is in your system's `PATH`.
*   **Permission errors during `npm install`:** Follow the official Claude Code documentation for setting up npm permissions without using `sudo`.
*   **Web UI not loading:** Ensure the `uvicorn` server is running and accessible (check firewall if necessary). Verify dependencies like `Jinja2` are installed (`pip install Jinja2`).