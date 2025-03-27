from fastapi import FastAPI, Request, HTTPException, Form
import uvicorn
import logging
import json
import traceback
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google import genai
from google.genai import types
import tiktoken
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
import webbrowser
import threading
from index_template import index_html

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.ERROR,  # Only show errors
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Remove default stdout handler by only defining a NullHandler
        logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]

        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Set default models, user can change them in the UI
BIG_MODEL = "gemini-2.5-pro"
SMALL_MODEL = "gemini-2.0-flash"

# Store request history for UI display
REQUEST_HISTORY = []  # Items will be inserted at the front, so newest requests are always at the top
MAX_HISTORY = 50  # Maximum number of requests to keep in history

# Create directory for templates if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Set up templates directory for the UI
templates = Jinja2Templates(directory="templates")

# Flag to enable model swapping between Anthropic and OpenAI
# Set based on the selected models
if "claude" in BIG_MODEL.lower() and "claude" in SMALL_MODEL.lower():
    USE_OPENAI_MODELS = False
    logger.debug(f"Using Claude models exclusively - disabling OpenAI model swapping")
else:
    USE_OPENAI_MODELS = True
    logger.debug(f"Using non-Claude models - enabling model swapping")

app = FastAPI()

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    reasoning_effort: Optional[str] = None  # Added for OpenAI o1 and o3-mini models
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('model')
    def validate_model(cls, v, info):
        # Store the original model name
        original_model = v

        # Check if we're using OpenAI models and need to swap
        if USE_OPENAI_MODELS:
            # Remove anthropic/ prefix if it exists
            if v.startswith('anthropic/'):
                v = v[10:]  # Remove 'anthropic/' prefix

            # Swap Haiku with small model (default: gpt-4o-mini)
            if 'haiku' in v.lower():
                # If small model starts with "claude", keep original model with anthropic/ prefix
                if SMALL_MODEL.startswith("claude"):
                    # Ensure we use the anthropic/ prefix for Claude models
                    if not original_model.startswith("anthropic/"):
                        new_model = f"anthropic/{v}"
                    else:
                        new_model = original_model  # Keep the original model as-is

                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} (CLAUDE)")
                    v = new_model
                else:
                    # Use OpenAI model
                    new_model = f"openai/{SMALL_MODEL}"
                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                    v = new_model

            # Swap any Sonnet model with big model (default: gpt-4o)
            elif 'sonnet' in v.lower():
                # If big model starts with "claude", keep original model with anthropic/ prefix
                if BIG_MODEL.startswith("claude"):
                    # Ensure we use the anthropic/ prefix for Claude models
                    if not original_model.startswith("anthropic/"):
                        new_model = f"anthropic/{v}"
                    else:
                        new_model = original_model  # Keep the original model as-is

                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} (CLAUDE)")
                    v = new_model
                else:
                    # Use OpenAI model
                    new_model = f"openai/{BIG_MODEL}"
                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                    v = new_model

            # Keep the model as is but add openai/ prefix if not already present
            elif not v.startswith('openai/') and not v.startswith('anthropic/'):
                new_model = f"openai/{v}"
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                v = new_model

            # Store the original model in the values dictionary
            # This will be accessible as request.original_model
            values = info.data
            if isinstance(values, dict):
                values['original_model'] = original_model

            return v
        else:
            # Original behavior - ensure anthropic/ prefix
            original_model = v
            if not v.startswith('anthropic/'):
                new_model = f"anthropic/{v}"
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")

                # Store original model
                values = info.data
                if isinstance(values, dict):
                    values['original_model'] = original_model

                return new_model
            return v

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None  # Added for OpenAI o1 and o3-mini models
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('model')
    def validate_model(cls, v, info):
        # Store the original model name
        original_model = v

        # Same validation as MessagesRequest
        if USE_OPENAI_MODELS:
            # Remove anthropic/ prefix if it exists
            if v.startswith('anthropic/'):
                v = v[10:]

            # Swap Haiku with small model (default: gpt-4o-mini)
            if 'haiku' in v.lower():
                # If small model starts with "claude", keep original model with anthropic/ prefix
                if SMALL_MODEL.startswith("claude"):
                    # Ensure we use the anthropic/ prefix for Claude models
                    if not original_model.startswith("anthropic/"):
                        new_model = f"anthropic/{v}"
                    else:
                        new_model = original_model  # Keep the original model as-is

                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} (CLAUDE)")
                    v = new_model
                else:
                    # Use OpenAI model
                    new_model = f"openai/{SMALL_MODEL}"
                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                    v = new_model

            # Swap any Sonnet model with big model (default: gpt-4o)
            elif 'sonnet' in v.lower():
                # If big model starts with "claude", keep original model with anthropic/ prefix
                if BIG_MODEL.startswith("claude"):
                    # Ensure we use the anthropic/ prefix for Claude models
                    if not original_model.startswith("anthropic/"):
                        new_model = f"anthropic/{v}"
                    else:
                        new_model = original_model  # Keep the original model as-is

                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model} (CLAUDE)")
                    v = new_model
                else:
                    # Use OpenAI model
                    new_model = f"openai/{BIG_MODEL}"
                    logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                    v = new_model

            # Keep the model as is but add openai/ prefix if not already present
            elif not v.startswith('openai/') and not v.startswith('anthropic/'):
                new_model = f"openai/{v}"
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")
                v = new_model

            # Store the original model in the values dictionary
            values = info.data
            if isinstance(values, dict):
                values['original_model'] = original_model

            return v
        else:
            # Original behavior - ensure anthropic/ prefix
            if not v.startswith('anthropic/'):
                new_model = f"anthropic/{v}"
                logger.debug(f"📌 MODEL MAPPING: {original_model} ➡️ {new_model}")

                # Store original model
                values = info.data
                if isinstance(values, dict):
                    values['original_model'] = original_model

                return new_model
            return v

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path

    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")

    # Process the request and get the response
    response = await call_next(request)

    return response

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)

    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_gemini(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to Gemini format."""
    gemini_request = {"contents": [], "config": {}}
    
    # System instructions (if present)
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            system_text = anthropic_request.system
        elif isinstance(anthropic_request.system, list):
            system_text = " ".join(
                [blk.text if hasattr(blk, "text") else blk.get("text", "") for blk in anthropic_request.system]
            )
        gemini_request["contents"].append({
            "role": "user",
            "parts": [{"text": f"System instructions: {system_text}"}]
        })
    
    # Process conversation messages
    for msg in anthropic_request.messages:
        role = "user" if msg.role == "user" else "model"
        text = ""
        if isinstance(msg.content, str):
            text = msg.content
        elif isinstance(msg.content, list):
            for block in msg.content:
                if hasattr(block, "type") and block.type == "text":
                    text += block.text + "\n"
        gemini_request["contents"].append({
            "role": role,
            "parts": [{"text": text.strip()}]
        })
    
    # Map configuration parameters
    gemini_request["config"]["temperature"] = anthropic_request.temperature
    gemini_request["config"]["maxOutputTokens"] = anthropic_request.max_tokens
    if anthropic_request.top_p:
        gemini_request["config"]["topP"] = anthropic_request.top_p

    # Map tools / function calling (if any)
    if anthropic_request.tools:
        function_declarations = []
        for tool in anthropic_request.tools:
            tool_dict = tool.dict() if hasattr(tool, "dict") else tool
            function_declarations.append({
                "name": tool_dict["name"],
                "description": tool_dict.get("description", ""),
                "parameters": tool_dict["input_schema"]
            })
        if function_declarations:
            gemini_request["config"]["tools"] = [{
                "function_declarations": function_declarations
            }]
    return gemini_request


def convert_gemini_to_anthropic(gemini_response: Dict[str, Any], original_request: MessagesRequest) -> MessagesResponse:
    """Convert Gemini API response format to Anthropic format."""
    content = []
    if "text" in gemini_response and gemini_response["text"]:
        content.append({"type": "text", "text": gemini_response["text"]})
    
    # If Gemini returned a function call, convert it
    if "function_calls" in gemini_response and gemini_response["function_calls"]:
        function_call = gemini_response["function_calls"][0]
        try:
            arguments = json.loads(function_call.get("arguments", "{}"))
        except Exception:
            arguments = {}
        content.append({
            "type": "tool_use",
            "id": f"tu_{os.urandom(4).hex()}",
            "name": function_call.get("name", ""),
            "input": arguments
        })
    
    usage = gemini_response.get("usage", {})
    return MessagesResponse(
        id=f"msg_{os.urandom(4).hex()}",
        model=original_request.model,
        role="assistant",
        content=content if content else [{"type": "text", "text": ""}],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0)
        )
    )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from Gemini and convert to Anthropic event-stream format."""
    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"

        accumulated_text = ""
        # Process each streaming chunk from Gemini
        async for chunk in response_generator:
            delta_text = getattr(chunk, "text", "") or chunk.get("text", "")
            if delta_text:
                accumulated_text += delta_text
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_text}})}\n\n"

        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': len(accumulated_text.split())}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in streaming: {str(e)}")
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None, 'error_message': str(e)}, 'usage': {'output_tokens': 0}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    global REQUEST_HISTORY
    print(f"REQUEST: {request}")
    try:
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        display_model = original_model.split("/")[-1] if "/" in original_model else original_model

        logger.debug(f" PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")

        # Convert Anthropic request to Gemini format
        gemini_request = convert_anthropic_to_gemini(request)

        # Record request history (same as before)...
        request_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_model": original_model,
            "mapped_model": gemini_request["config"].get("model", request.model),
            "num_messages": len(gemini_request["contents"]),
            "num_tools": len(request.tools) if request.tools else 0,
            "stream": request.stream,
            "status": "success"
        }
        REQUEST_HISTORY.insert(0, request_info)
        if len(REQUEST_HISTORY) > MAX_HISTORY:
            REQUEST_HISTORY = REQUEST_HISTORY[:MAX_HISTORY]

        if not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(status_code=401, detail="Missing GEMINI_API_KEY in environment variables.")

        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )

        if request.stream:
            response_generator = client.models.generate_content_stream(
                model=request.model,
                contents=gemini_request["contents"],
                config=gemini_request["config"],
            )
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            gemini_response = client.models.generate_content(
                model=request.model,
                contents=gemini_request["contents"],
                config=gemini_request["config"],
            )
            anthro_response = convert_gemini_to_anthropic(gemini_response, request)
            return anthro_response

    except Exception as e:
        error_traceback = traceback.format_exc()
        # (Error logging/handling remains the same)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        original_model = request.original_model or request.model
        display_model = original_model.split("/")[-1] if "/" in original_model else original_model

        # Concatenate all message text for token counting
        combined_text = ""
        for msg in request.messages:
            if isinstance(msg.content, str):
                combined_text += msg.content + " "
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text"):
                        combined_text += block.text + " "

        encoder = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoder.encode(combined_text))
        return TokenCountResponse(input_tokens=token_count)

    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def ui_root(request: Request):
    # Get available models with display labels for the dropdown
    available_models = [
        # Gemini models
        {"value": "gemini-2.5-pro", "label": "Gemini 2.5 Pro"},
        {"value": "gemini-2.0-flash", "label": "Gemini 2.0 Flash"},

        # OpenAI models
        {"value": "gpt-4o", "label": "gpt-4o"},
        {"value": "gpt-4o-mini", "label": "gpt-4o-mini"},
        {"value": "gpt-3.5-turbo", "label": "gpt-3.5-turbo"},

        # OpenAI reasoning models
        {"value": "o1", "label": "o1"},
        {"value": "o3-mini", "label": "o3-mini"},

        # Anthropic models
        {"value": "claude-3-7-sonnet-20240229", "label": "claude-3-7-sonnet-20240229"},
        {"value": "claude-3-haiku-20240307", "label": "claude-3-haiku-20240307"},
    ]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "big_model": BIG_MODEL,
            "small_model": SMALL_MODEL,
            "available_models": available_models,
            "request_history": REQUEST_HISTORY
        }
    )

@app.post("/update_models")
async def update_models(big_model: str = Form(...), small_model: str = Form(...)):
    global BIG_MODEL, SMALL_MODEL, USE_OPENAI_MODELS

    # Check for appropriate API keys
    if ("claude" in big_model.lower() or "claude" in small_model.lower()) and not ANTHROPIC_API_KEY:
        return {"status": "error", "message": "Missing Anthropic API key. Please set ANTHROPIC_API_KEY in your environment variables."}

    if (big_model.startswith("openai/") or small_model.startswith("openai/") or
        (not big_model.startswith("anthropic/")) or
        (not small_model.startswith("anthropic/"))) and not OPENAI_API_KEY:
        return {"status": "error", "message": "Missing OpenAI API key. Please set OPENAI_API_KEY in your environment variables."}

    # Update the model settings
    BIG_MODEL = big_model
    SMALL_MODEL = small_model

    # Refresh environment - this is important for the model swap to take effect
    if "claude" in BIG_MODEL.lower() and "claude" in SMALL_MODEL.lower():
        USE_OPENAI_MODELS = False
        logger.debug(f"Using Claude models exclusively - disabling OpenAI model swapping")
    else:
        USE_OPENAI_MODELS = True
        logger.debug(f"Using non-Claude models - enabling model swapping")

    logger.warning(f"MODEL CONFIGURATION UPDATED: Big Model = {BIG_MODEL}, Small Model = {SMALL_MODEL}, USE_OPENAI_MODELS = {USE_OPENAI_MODELS}")

    return {"status": "success", "big_model": BIG_MODEL, "small_model": SMALL_MODEL, "use_openai_models": USE_OPENAI_MODELS}

@app.get("/api/history")
async def get_history():
    return {"history": REQUEST_HISTORY}

# Create the HTML template for the UI
@app.on_event("startup")
async def create_templates():
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)

    # Minimal server startup message with styling
    print(f"\n{Colors.GREEN}{Colors.BOLD}SERVER STARTED SUCCESSFULLY!{Colors.RESET}")
    print(f"{Colors.CYAN}Access the web UI at: {Colors.BOLD}http://localhost:8082{Colors.RESET}")
    print(f"{Colors.CYAN}Connect Claude Code with: {Colors.BOLD}ANTHROPIC_BASE_URL=http://localhost:8082 claude{Colors.RESET}\n")

    # Create index.html template
    
    # Write the HTML template to the file
    with open("templates/index.html", "w") as f:
        f.write(index_html)

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # This function has been modified to disable terminal logging
    # The web view still has access to these logs

    # Simply return without printing anything to the terminal
    return

    # Format the Claude model name nicely (code kept for reference but not executed)
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)

    # Configure uvicorn to run with minimal logs
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8082,
        log_level="critical",  # Only show critical errors
        access_log=False       # Disable access logs completely
    )
