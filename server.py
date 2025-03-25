from fastapi import FastAPI, Request, HTTPException, Form
import uvicorn
import logging
import json
import traceback
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import litellm
# Configure litellm to drop unsupported parameters
litellm.drop_params = True
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
import webbrowser
import threading

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
# Basic config first
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Get specific loggers
logger = logging.getLogger(__name__)
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_error_logger = logging.getLogger("uvicorn.error")
litellm_logger = logging.getLogger("LiteLLM") # Get LiteLLM's logger

# Set levels for verbosity control
logger.setLevel(logging.DEBUG) # Main app logger - show debug messages
uvicorn_access_logger.setLevel(logging.WARNING) # Quieter access logs
uvicorn_error_logger.setLevel(logging.ERROR) # Show errors
litellm_logger.setLevel(logging.WARNING) # Quieter LiteLLM, show only warnings and above

# Create handlers (console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG) # Handler level

# Create formatters
basic_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings and errors"""
    GREY = "\x1b[38;20m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD_RED = "\x1b[31;1m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    LOG_FORMAT = f"{BOLD}%(levelname)s:{RESET} %(message)s" # Simplified format

    FORMATS = {
        logging.DEBUG: GREY + LOG_FORMAT + RESET,
        logging.INFO: CYAN + LOG_FORMAT + RESET,
        logging.WARNING: YELLOW + LOG_FORMAT + RESET,
        logging.ERROR: RED + LOG_FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + LOG_FORMAT + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        message = formatter.format(record)
        if "📌 MODEL MAPPING" in record.getMessage():
            # Apply special colors for mapping
             message = message.replace("📌 MODEL MAPPING:", f"{self.BOLD}{self.MAGENTA}📌 MODEL MAPPING:{self.RESET}{self.YELLOW}")
        elif "✅ RESPONSE RECEIVED" in record.getMessage():
             message = message.replace("✅ RESPONSE RECEIVED:", f"{self.BOLD}{self.GREEN}✅ RESPONSE RECEIVED:{self.RESET}")
        elif "PROCESSING REQUEST" in record.getMessage():
             message = message.replace("PROCESSING REQUEST:", f"{self.BOLD}{self.BLUE}PROCESSING REQUEST:{self.RESET}")
        return message

# Set formatters for handlers
console_handler.setFormatter(ColorizedFormatter())

# Add handlers ONLY to the loggers we want to control output for
logger.addHandler(console_handler)
logger.propagate = False # Prevent root logger from handling messages again

litellm_logger.addHandler(console_handler) # Apply color to LiteLLM warnings/errors too
litellm_logger.propagate = False

# Don't add handlers to uvicorn loggers if we want them quieter
# --- End Logging Configuration ---

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Added Google API Key

# Set default models, user can change them in the UI
BIG_MODEL = "o3-mini"  # Default to OpenAI reasoning
SMALL_MODEL = "gemini-1.5-flash" # Default to Gemini flash

# Store request history for UI display
REQUEST_HISTORY = []
MAX_HISTORY = 50

# Create directory for templates if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Set up templates directory for the UI
templates = Jinja2Templates(directory="templates")

app = FastAPI()

# --- Pydantic Models for Anthropic API Structure ---
# (Keep these as they define the structure the proxy expects and returns)
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
    is_error: Optional[bool] = None # Added based on potential Anthropic tool usage

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

# --- Utility Function to Determine Provider and Prefix ---
def get_provider_and_prefix(model_name: str) -> tuple[str, str]:
    """Determines the provider ('openai', 'gemini', 'anthropic') and prefix based on the model name."""
    model_lower = model_name.lower()
    if model_lower.startswith("claude"):
        return "anthropic", "anthropic/"
    elif model_lower.startswith("gemini"):
        return "gemini", "gemini/"
    # Default to openai for gpt models, o1, o3 etc.
    else:
        return "openai", "openai/"

# --- Pydantic Models with Validation for Model Mapping ---
class BaseRequestModel(BaseModel):
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
    reasoning_effort: Optional[str] = None
    original_model: Optional[str] = None # To store the model requested by Claude Code CLI

    @field_validator('model', mode='before') # Use 'before' to modify before standard validation
    @classmethod
    def map_model_dynamically(cls, v: str, info) -> str:
        """Maps incoming Claude model names to configured OpenAI/Gemini/Anthropic models."""
        original_model = v # Store the originally requested model name
        mapped_model_str = v # Default to original if no mapping occurs

        # Claude Code CLI sends models like 'claude-3-haiku-...' or 'claude-3-sonnet-...'
        if 'haiku' in v.lower():
            target_model_config = SMALL_MODEL # Use the globally configured small model
            provider, prefix = get_provider_and_prefix(target_model_config)
            mapped_model_str = f"{prefix}{target_model_config}"
            logger.debug(f"📌 MODEL MAPPING: '{original_model}' (Small/Haiku) ➡️ '{mapped_model_str}' (Provider: {provider})")
        elif 'sonnet' in v.lower():
            target_model_config = BIG_MODEL # Use the globally configured big model
            provider, prefix = get_provider_and_prefix(target_model_config)
            mapped_model_str = f"{prefix}{target_model_config}"
            logger.debug(f"📌 MODEL MAPPING: '{original_model}' (Big/Sonnet) ➡️ '{mapped_model_str}' (Provider: {provider})")
        else:
            # If it's not haiku or sonnet, maybe it's already prefixed or a direct model name?
            # Try to determine provider and ensure prefix, but log a warning.
            logger.warning(f"⚠️ Unexpected model format received: '{original_model}'. Attempting to determine provider.")
            # Check if it already has a known prefix
            if not any(v.startswith(p) for p in ["openai/", "gemini/", "anthropic/"]):
                 provider, prefix = get_provider_and_prefix(v)
                 mapped_model_str = f"{prefix}{v}"
                 logger.warning(f"   ➡️ Assuming provider '{provider}', mapping to '{mapped_model_str}'")
            else:
                 mapped_model_str = v # Assume correctly prefixed
                 logger.warning(f"   ➡️ Using model as is: '{mapped_model_str}'")


        # Store the original model name in the instance's context if possible
        # Pydantic v2 uses 'info.context' or you might need to pass it differently
        # For simplicity, we'll attach it later if needed or rely on logs.
        # Let's try adding to the values dict (works in Pydantic v2)
        if isinstance(info.data, dict):
             info.data['original_model'] = original_model

        return mapped_model_str

class MessagesRequest(BaseRequestModel):
    pass # Inherits the validator

class TokenCountRequest(BaseRequestModel):
    # Needs its own validator instance, but logic is the same
    @field_validator('model', mode='before')
    @classmethod
    def map_model_dynamically_tc(cls, v: str, info) -> str:
        return BaseRequestModel.map_model_dynamically(v, info)

# --- Response Models (Anthropic Format) ---
class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str # Should reflect the original requested model
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

# --- Request/Response Conversion Functions ---

def convert_anthropic_to_litellm(anthropic_request: Union[MessagesRequest, TokenCountRequest]) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (OpenAI style)."""
    messages = []

    # Add system message
    if anthropic_request.system:
        system_text = ""
        if isinstance(anthropic_request.system, str):
            system_text = anthropic_request.system
        elif isinstance(anthropic_request.system, list):
            for block in anthropic_request.system:
                 if isinstance(block, SystemContent) and block.type == "text":
                    system_text += block.text + "\n"
                 elif isinstance(block, dict) and block.get("type") == "text":
                     system_text += block.get("text", "") + "\n"
        if system_text.strip():
            messages.append({"role": "system", "content": system_text.strip()})

    # Add conversation messages
    for msg in anthropic_request.messages:
        litellm_msg = {"role": msg.role}
        content_parts = []

        if isinstance(msg.content, str):
            content_parts.append({"type": "text", "text": msg.content})
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ContentBlockText):
                    content_parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ContentBlockImage):
                     # LiteLLM/OpenAI format expects image URLs or base64 data
                     # Assuming anthropic_request source contains necessary info
                     # Example: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                     if block.source and block.source.get("type") == "base64":
                         media_type = block.source.get("media_type", "image/jpeg") # default
                         data = block.source.get("data", "")
                         content_parts.append({
                             "type": "image_url",
                             "image_url": {"url": f"data:{media_type};base64,{data}"}
                         })
                     # Add other image source types if needed
                elif isinstance(block, ContentBlockToolUse):
                    # Convert Anthropic tool_use to OpenAI's tool_calls format (for assistant messages)
                     if msg.role == "assistant":
                         if "tool_calls" not in litellm_msg:
                             litellm_msg["tool_calls"] = []
                         litellm_msg["tool_calls"].append({
                             "id": block.id,
                             "type": "function",
                             "function": {
                                 "name": block.name,
                                 "arguments": json.dumps(block.input or {}), # Arguments must be a JSON string
                             }
                         })
                     else: # Include as text for user message if needed? Or ignore?
                          logger.warning(f"Ignoring tool_use block in user message: {block.id}")

                elif isinstance(block, ContentBlockToolResult):
                    # Convert Anthropic tool_result to OpenAI's tool message format (role: tool)
                    # Need to create a separate message for this in OpenAI format
                    tool_result_content = ""
                    if isinstance(block.content, str):
                        tool_result_content = block.content
                    elif isinstance(block.content, list): # Handle list content (e.g., list of text blocks)
                         for item in block.content:
                             if isinstance(item, dict) and item.get("type") == "text":
                                 tool_result_content += item.get("text", "") + "\n"
                             else: # Append string representation for other types
                                 tool_result_content += str(item) + "\n"
                    else: # Append string representation for other types
                        tool_result_content = str(block.content)

                    # Append a new message with role 'tool'
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": tool_result_content.strip() or ("(No content)" if not block.is_error else "(Error)"), # Content must be string
                        # Note: OpenAI doesn't have a direct 'is_error' field here. Embed in content or ignore.
                    })
                    # Skip adding this block to the current message's content_parts
                    continue
                # Add handling for other block types if necessary

        # Construct content field for LiteLLM message
        if len(content_parts) == 1 and content_parts[0]["type"] == "text" and "tool_calls" not in litellm_msg:
            # If only text, use simple string content
            litellm_msg["content"] = content_parts[0]["text"]
        elif content_parts:
            # If multiple parts (text, image) or tool calls exist, use list format
            litellm_msg["content"] = content_parts
        elif "tool_calls" in litellm_msg:
             # If only tool calls, content might be None or empty string for some models
             litellm_msg["content"] = None # Or "" depending on model requirements
        else:
            # If no content and no tool calls (e.g., after processing tool result), skip message?
            # Or send empty content? Let's send empty for now.
            litellm_msg["content"] = ""

        # Append the message unless it was just a tool result handled separately
        if litellm_msg.get("role") != "tool":
             # Only append if there's content or tool_calls
             if litellm_msg.get("content") is not None or "tool_calls" in litellm_msg:
                messages.append(litellm_msg)

    # Map other parameters
    litellm_request = {
        "model": anthropic_request.model, # Already mapped by validator
        "messages": messages,
        "max_tokens": anthropic_request.max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Add optional parameters if they exist
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k

    # Convert Anthropic tools to OpenAI tool format
    if anthropic_request.tools:
        litellm_request["tools"] = [
            {"type": "function", "function": tool.model_dump(exclude_none=True)}
            for tool in anthropic_request.tools
        ]
    if anthropic_request.tool_choice:
         # Map Anthropic tool_choice (e.g., {"type": "any"}) to OpenAI format
         choice_type = anthropic_request.tool_choice.get("type")
         if choice_type == "auto":
             litellm_request["tool_choice"] = "auto"
         elif choice_type == "any":
              # OpenAI doesn't have 'any'. 'required' is closest if tools exist.
              litellm_request["tool_choice"] = "required" if anthropic_request.tools else "auto"
         elif choice_type == "tool" and "name" in anthropic_request.tool_choice:
             litellm_request["tool_choice"] = {
                 "type": "function",
                 "function": {"name": anthropic_request.tool_choice["name"]}
             }
         # else default to 'auto' or omit

    # Add reasoning_effort for OpenAI models if specified or applicable
    provider, _ = get_provider_and_prefix(anthropic_request.model) # Check mapped model
    if provider == "openai":
        clean_model_name = anthropic_request.model.split('/')[-1]
        if anthropic_request.reasoning_effort:
             litellm_request["reasoning_effort"] = anthropic_request.reasoning_effort
             logger.debug(f"Using reasoning_effort={anthropic_request.reasoning_effort} from request for OpenAI model.")
        elif "o3-" in clean_model_name or "o1" in clean_model_name:
             litellm_request["reasoning_effort"] = "high" # Default for these models
             logger.debug(f"Adding default reasoning_effort=high for OpenAI reasoning model: {clean_model_name}")

    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[litellm.ModelResponse, litellm.CustomStreamWrapper, Dict[str, Any]],
                                 original_request_model: str) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response back to Anthropic API response format."""
    try:
        # --- Extract data from LiteLLM response ---
        response_id = f"msg_{uuid.uuid4()}"
        content_text = None
        tool_calls = None
        finish_reason = "stop" # Default
        input_tokens = 0
        output_tokens = 0

        # Handle LiteLLM ModelResponse object (non-streaming)
        if isinstance(litellm_response, litellm.ModelResponse):
            response_id = getattr(litellm_response, 'id', response_id)
            if litellm_response.choices and len(litellm_response.choices) > 0:
                choice = litellm_response.choices[0]
                message = getattr(choice, 'message', None)
                finish_reason = getattr(choice, 'finish_reason', 'stop')
                if message:
                    content_text = getattr(message, 'content', None)
                    tool_calls = getattr(message, 'tool_calls', None)
            usage = getattr(litellm_response, 'usage', None)
            if usage:
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)

        # Handle dictionary response (less common now with LiteLLM objects)
        elif isinstance(litellm_response, dict):
             response_id = litellm_response.get('id', response_id)
             choices = litellm_response.get('choices', [])
             if choices:
                 choice = choices[0]
                 message = choice.get('message', {})
                 finish_reason = choice.get('finish_reason', 'stop')
                 content_text = message.get('content', None)
                 tool_calls = message.get('tool_calls', None)
             usage = litellm_response.get('usage', {})
             input_tokens = usage.get('prompt_tokens', 0)
             output_tokens = usage.get('completion_tokens', 0)
        else:
             # Should not happen for non-streaming, but log if it does
             logger.error(f"Unexpected LiteLLM response type for conversion: {type(litellm_response)}")
             raise ValueError("Invalid LiteLLM response format")

        # --- Convert to Anthropic format ---
        anthropic_content = []

        # Add text content block if present
        if content_text:
            anthropic_content.append(ContentBlockText(type="text", text=content_text))

        # Convert OpenAI tool_calls back to Anthropic tool_use blocks
        if tool_calls:
            for tool_call in tool_calls:
                function_call = None
                tool_id = None
                if isinstance(tool_call, dict): # Handle dict format
                    function_call = tool_call.get("function")
                    tool_id = tool_call.get("id")
                elif hasattr(tool_call, 'function'): # Handle object format
                    function_call = tool_call.function
                    tool_id = getattr(tool_call, 'id', None)

                if function_call and tool_id:
                    name = None
                    input_args = {}
                    if isinstance(function_call, dict):
                        name = function_call.get("name")
                        arguments_str = function_call.get("arguments", "{}")
                    elif hasattr(function_call, 'name'):
                         name = function_call.name
                         arguments_str = getattr(function_call, 'arguments', '{}')
                    else:
                         arguments_str = "{}"


                    if name:
                        try:
                            # Arguments from OpenAI are JSON strings
                            input_args = json.loads(arguments_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse tool arguments from LLM as JSON: {arguments_str}")
                            # Keep as string? Or wrap in a dict? Anthropic expects a dict.
                            input_args = {"_raw_arguments": arguments_str} # Fallback

                        anthropic_content.append(ContentBlockToolUse(
                            type="tool_use",
                            id=tool_id,
                            name=name,
                            input=input_args
                        ))

        # Map finish_reason to Anthropic stop_reason
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence", # Or maybe end_turn?
            "function_call": "tool_use" # Older OpenAI models
        }
        stop_reason = stop_reason_map.get(finish_reason, "end_turn") # Default to end_turn

        # Ensure content is never empty for Anthropic
        if not anthropic_content:
            anthropic_content.append(ContentBlockText(type="text", text=""))

        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request_model, # Use the original model requested by client
            role="assistant",
            content=anthropic_content,
            stop_reason=stop_reason,
            stop_sequence=None, # Assuming LiteLLM handles stop sequences conversion if needed
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        )
        return anthropic_response

    except Exception as e:
        logger.error(f"Error converting LiteLLM response to Anthropic: {e}\n{traceback.format_exc()}")
        # Re-raise or return a custom error structure if needed
        raise HTTPException(status_code=500, detail=f"Internal error during response conversion: {e}")


async def handle_streaming(response_generator: litellm.CustomStreamWrapper, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic SSE format."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    original_model = original_request.original_model or original_request.model # Use the originally requested model name
    accumulated_output_tokens = 0 # Track output tokens from usage delta

    try:
        # Send message_start event
        message_start_event = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_model, # Report the original model
                'content': [], # Will be populated incrementally? Anthropic usually sends empty here.
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 0, 'output_tokens': 0} # Placeholder, input tokens often unknown until end
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_start_event)}\n\n"

        # Send initial ping
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        content_block_index = 0
        current_tool_calls = {} # {index: {id: '...', name: '...', type: 'function', accumulated_args: ''}}
        sent_tool_start = {} # {index: True}
        last_event_type = 'message_start' # Keep track of last major event type

        async for chunk in response_generator:
            chunk_dict = chunk.model_dump() # Convert LiteLLM chunk object to dict
            #logger.debug(f"Raw Chunk: {chunk_dict}") # Very verbose debugging

            delta = chunk_dict.get('choices', [{}])[0].get('delta', {})
            finish_reason = chunk_dict.get('choices', [{}])[0].get('finish_reason')
            usage_delta = chunk_dict.get('usage', None) # LiteLLM might put usage here

            # --- Handle Usage Delta ---
            if usage_delta and isinstance(usage_delta, dict):
                # Anthropic sends input tokens in message_start (often 0) and output tokens in message_delta
                 output_tokens_delta = usage_delta.get("completion_tokens", 0)
                 input_tokens_final = usage_delta.get("prompt_tokens", 0) # Input tokens usually come at the end

                 if output_tokens_delta > 0:
                      accumulated_output_tokens += output_tokens_delta
                      # Send usage delta event
                      usage_event = {
                          "type": "message_delta",
                          "delta": {"usage": {"output_tokens": accumulated_output_tokens}}
                      }
                      yield f"event: message_delta\ndata: {json.dumps(usage_event)}\n\n"

                 # If we get final input tokens, maybe update start event? Anthropic format is tricky here.
                 # Let's update the final message_stop with total usage if possible.

            # --- Handle Content Delta ---
            text_delta = delta.get('content')
            if text_delta:
                # If the previous block was different (e.g., tool), start a new text block
                if last_event_type != 'text_delta' and last_event_type != 'content_block_start_text':
                     # Close previous block if necessary (e.g., tool block)
                     # (Handled implicitly by starting a new block?)

                     # Start the text block
                     start_event = {
                         "type": "content_block_start",
                         "index": content_block_index,
                         "content_block": {"type": "text", "text": ""}
                     }
                     yield f"event: content_block_start\ndata: {json.dumps(start_event)}\n\n"
                     last_event_type = 'content_block_start_text'

                # Send the text delta
                delta_event = {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {"type": "text_delta", "text": text_delta}
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                last_event_type = 'text_delta'


            # --- Handle Tool Call Delta ---
            tool_calls_delta = delta.get('tool_calls')
            if tool_calls_delta:
                for tool_call_chunk in tool_calls_delta:
                    tool_index = tool_call_chunk.get('index', 0) # Some models might send index
                    tool_id = tool_call_chunk.get('id')
                    function_delta = tool_call_chunk.get('function', {})
                    func_name = function_delta.get('name')
                    func_args_delta = function_delta.get('arguments') # This is a delta (string chunk)

                    # If this is the start of a new tool call (new ID or first chunk for this index)
                    if tool_id and tool_index not in current_tool_calls:
                         # If previous block was text, signal its end
                         if last_event_type.startswith('text') or last_event_type.startswith('content_block_start_text'):
                              stop_event = {"type": "content_block_stop", "index": content_block_index}
                              yield f"event: content_block_stop\ndata: {json.dumps(stop_event)}\n\n"
                              content_block_index += 1 # Increment index for the new tool block

                         current_tool_calls[tool_index] = {
                             "id": tool_id,
                             "name": func_name or "", # Name might come later?
                             "type": "function",
                             "accumulated_args": ""
                         }

                    # Update name if we just received it
                    if tool_index in current_tool_calls and func_name and not current_tool_calls[tool_index]["name"]:
                        current_tool_calls[tool_index]["name"] = func_name

                    # If this is the first event for *this specific tool block*
                    if tool_index not in sent_tool_start and tool_index in current_tool_calls:
                         start_event = {
                             "type": "content_block_start",
                             "index": content_block_index, # Use our sequential index
                             "content_block": {
                                 "type": "tool_use",
                                 "id": current_tool_calls[tool_index]["id"],
                                 "name": current_tool_calls[tool_index]["name"],
                                 "input": {} # Input starts empty
                             }
                         }
                         yield f"event: content_block_start\ndata: {json.dumps(start_event)}\n\n"
                         sent_tool_start[tool_index] = True
                         last_event_type = 'content_block_start_tool'

                    # Append arguments delta and send input_json_delta
                    if tool_index in current_tool_calls and func_args_delta:
                        current_tool_calls[tool_index]["accumulated_args"] += func_args_delta
                        delta_event = {
                             "type": "content_block_delta",
                             "index": content_block_index, # Use our sequential index
                             "delta": {"type": "input_json_delta", "partial_json": func_args_delta}
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                        last_event_type = 'input_json_delta'

            # --- Handle Finish Reason ---
            if finish_reason:
                 # Stop the last content block
                 stop_event = {"type": "content_block_stop", "index": content_block_index}
                 yield f"event: content_block_stop\ndata: {json.dumps(stop_event)}\n\n"

                 # Map stop reason
                 stop_reason_map = {
                     "stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                     "content_filter": "stop_sequence", "function_call": "tool_use"
                 }
                 anthropic_stop_reason = stop_reason_map.get(finish_reason, "end_turn")

                 # Send message_delta with stop reason
                 final_delta_event = {
                     "type": "message_delta",
                     "delta": {"stop_reason": anthropic_stop_reason, "stop_sequence": None},
                     "usage": {"output_tokens": accumulated_output_tokens} # Include final token count
                 }
                 yield f"event: message_delta\ndata: {json.dumps(final_delta_event)}\n\n"

                 # Send message_stop
                 stop_message_event = {"type": "message_stop"}
                 yield f"event: message_stop\ndata: {json.dumps(stop_message_event)}\n\n"
                 break # End the stream

        # If stream ends without a finish reason (shouldn't normally happen)
        # Clean up just in case
        if not finish_reason:
             logger.warning("Stream ended without a finish reason.")
             # Stop the last content block
             stop_event = {"type": "content_block_stop", "index": content_block_index}
             yield f"event: content_block_stop\ndata: {json.dumps(stop_event)}\n\n"
             # Send a default stop
             final_delta_event = {
                     "type": "message_delta",
                     "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                     "usage": {"output_tokens": accumulated_output_tokens}
                 }
             yield f"event: message_delta\ndata: {json.dumps(final_delta_event)}\n\n"
             stop_message_event = {"type": "message_stop"}
             yield f"event: message_stop\ndata: {json.dumps(stop_message_event)}\n\n"


    except Exception as e:
        logger.error(f"Error during streaming: {e}\n{traceback.format_exc()}")
        # Send an Anthropic-style error event if possible
        try:
            error_data = {
                "type": "error",
                "error": {
                    "type": "internal_server_error", # Or map based on exception type
                    "message": f"Streaming error: {e}"
                }
            }
            # Try sending error event - client might disconnect
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
            # Also send message_stop after error
            stop_message_event = {"type": "message_stop"}
            yield f"event: message_stop\ndata: {json.dumps(stop_message_event)}\n\n"
        except Exception as inner_e:
            logger.error(f"Failed to send streaming error message: {inner_e}")
        # No further yield after error


# --- FastAPI Endpoints ---

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Basic logging middleware (can be expanded)
    logger.debug(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response

def get_api_key_for_model(model_name: str) -> Optional[str]:
    """Gets the appropriate API key based on the model name prefix."""
    provider, _ = get_provider_and_prefix(model_name)
    if provider == "openai":
        return OPENAI_API_KEY
    elif provider == "gemini":
        return GOOGLE_API_KEY
    elif provider == "anthropic":
        return ANTHROPIC_API_KEY
    return None

def check_api_key_availability(model_name: str):
    """Raises HTTPException if the required API key is missing."""
    provider, _ = get_provider_and_prefix(model_name)
    key = get_api_key_for_model(model_name)
    if not key:
        error_message = f"API key for provider '{provider}' (required for model '{model_name}') is not configured in the .env file."
        logger.error(error_message)
        raise HTTPException(status_code=401, detail=error_message)

@app.post("/v1/messages")
async def create_message(request: Request):
    global REQUEST_HISTORY
    start_time = time.time()
    request_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_model": "unknown",
        "mapped_model": "unknown",
        "num_messages": 0,
        "num_tools": 0,
        "stream": False,
        "status": "processing",
        "error": None
    }

    try:
        # 1. Parse request body manually first to get original model name
        try:
            body_bytes = await request.body()
            body_json = json.loads(body_bytes.decode('utf-8'))
            request_info["original_model"] = body_json.get("model", "unknown")
            request_info["stream"] = body_json.get("stream", False)
        except json.JSONDecodeError:
             raise HTTPException(status_code=400, detail="Invalid JSON body")
        except Exception as e:
             raise HTTPException(status_code=400, detail=f"Error reading request body: {e}")

        # 2. Validate using Pydantic (this also performs model mapping)
        try:
             anthropic_request = MessagesRequest.model_validate(body_json)
        except ValidationError as e:
             logger.error(f"Pydantic validation failed: {e}")
             request_info["status"] = "error"
             request_info["error"] = f"Validation Error: {e}"
             # Add to history before raising
             REQUEST_HISTORY.insert(0, request_info)
             if len(REQUEST_HISTORY) > MAX_HISTORY: REQUEST_HISTORY.pop()
             raise HTTPException(status_code=422, detail=e.errors())

        # Update request info with mapped model etc.
        request_info["mapped_model"] = anthropic_request.model
        request_info["num_messages"] = len(anthropic_request.messages)
        request_info["num_tools"] = len(anthropic_request.tools) if anthropic_request.tools else 0

        logger.debug(f"PROCESSING REQUEST: Original='{request_info['original_model']}', Mapped='{request_info['mapped_model']}', Stream={anthropic_request.stream}")

        # 3. Check API key availability *before* conversion/API call
        check_api_key_availability(anthropic_request.model)
        api_key = get_api_key_for_model(anthropic_request.model)

        # 4. Convert to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(anthropic_request)
        litellm_request["api_key"] = api_key # Pass the specific key

        # Add request to history (initially as processing/success)
        request_info["status"] = "success" # Assume success unless exception occurs
        REQUEST_HISTORY.insert(0, request_info)
        if len(REQUEST_HISTORY) > MAX_HISTORY: REQUEST_HISTORY.pop() # Keep history trimmed

        # 5. Call LiteLLM
        if anthropic_request.stream:
            # Ensure litellm gets the stream=True parameter
            litellm_request["stream"] = True
            response_generator = await litellm.acompletion(**litellm_request)
            # Pass the *original* request model name for reporting back to client
            return StreamingResponse(
                handle_streaming(response_generator, anthropic_request),
                media_type="text/event-stream"
            )
        else:
            litellm_request["stream"] = False
            litellm_response = await litellm.acompletion(**litellm_request) # Use async version
            logger.debug(f"✅ RESPONSE RECEIVED: Model='{litellm_request.get('model')}', Time={time.time() - start_time:.2f}s")

            # 6. Convert response back to Anthropic format
            # Pass the original model name from the request for the response structure
            anthropic_response = convert_litellm_to_anthropic(litellm_response, anthropic_request.original_model or request_info["original_model"])
            return anthropic_response

    except HTTPException as e:
         # If it's an HTTPException we raised (like auth error), update history
         if request_info["status"] != "error": # Avoid double logging if validation failed
              request_info["status"] = "error"
              request_info["error"] = f"HTTP {e.status_code}: {e.detail}"
              # Ensure it's added if not already
              if request_info not in REQUEST_HISTORY:
                  REQUEST_HISTORY.insert(0, request_info)
                  if len(REQUEST_HISTORY) > MAX_HISTORY: REQUEST_HISTORY.pop()
         raise e # Re-raise the exception
    except litellm.exceptions.AuthenticationError as e:
         logger.error(f"LiteLLM Authentication Error: {e}")
         request_info["status"] = "error"
         request_info["error"] = f"API Key Error: {e}"
         if request_info not in REQUEST_HISTORY:
             REQUEST_HISTORY.insert(0, request_info)
             if len(REQUEST_HISTORY) > MAX_HISTORY: REQUEST_HISTORY.pop()
         raise HTTPException(status_code=401, detail=f"Authentication failed for the target API. Check your API key. Details: {e}")
    except litellm.exceptions.RateLimitError as e:
         logger.error(f"LiteLLM Rate Limit Error: {e}")
         request_info["status"] = "error"
         request_info["error"] = f"Rate Limit: {e}"
         if request_info not in REQUEST_HISTORY:
             REQUEST_HISTORY.insert(0, request_info)
             if len(REQUEST_HISTORY) > MAX_HISTORY: REQUEST_HISTORY.pop()
         raise HTTPException(status_code=429, detail=f"API rate limit exceeded. Please try again later. Details: {e}")
    except litellm.exceptions.BadRequestError as e:
          logger.error(f"LiteLLM Bad Request Error: {e}")
          request_info["status"] = "error"
          request_info["error"] = f"Bad Request: {e}"
          if request_info not in REQUEST_HISTORY:
               REQUEST_HISTORY.insert(0, request_info)
               if len(REQUEST_HISTORY) > MAX_HISTORY: REQUEST_HISTORY.pop()
          raise HTTPException(status_code=400, detail=f"Invalid request sent to the target API. Details: {e}")
    except Exception as e:
        # Catch-all for other errors during processing
        error_traceback = traceback.format_exc()
        logger.error(f"Unhandled Error processing request: {e}\n{error_traceback}")
        request_info["status"] = "error"
        request_info["error"] = f"Server Error: {type(e).__name__}"
        # Ensure it's added if not already
        if request_info not in REQUEST_HISTORY:
            REQUEST_HISTORY.insert(0, request_info)
            if len(REQUEST_HISTORY) > MAX_HISTORY: REQUEST_HISTORY.pop()

        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    # Similar structure to create_message for parsing and validation
    try:
        try:
            body_bytes = await request.body()
            body_json = json.loads(body_bytes.decode('utf-8'))
            original_model = body_json.get("model", "unknown")
        except Exception as e:
             raise HTTPException(status_code=400, detail=f"Error reading request body: {e}")

        try:
            anthropic_request = TokenCountRequest.model_validate(body_json)
        except ValidationError as e:
             logger.error(f"Token count validation failed: {e}")
             raise HTTPException(status_code=422, detail=e.errors())

        logger.debug(f"Counting tokens for: Original='{original_model}', Mapped='{anthropic_request.model}'")

        # Check API key - some tokenizers might need it
        # check_api_key_availability(anthropic_request.model) # Might not be needed for tokenizer
        # api_key = get_api_key_for_model(anthropic_request.model)

        # Convert messages for LiteLLM's counter
        litellm_messages = convert_anthropic_to_litellm(anthropic_request)["messages"]

        # Use LiteLLM's token_counter
        try:
            token_count = litellm.token_counter(
                model=anthropic_request.model, # Use the mapped model
                messages=litellm_messages,
                # api_key=api_key # Pass key if needed by tokenizer
            )
            logger.debug(f"Token count result: {token_count} for model {anthropic_request.model}")
            return TokenCountResponse(input_tokens=token_count)
        except Exception as e:
            logger.error(f"LiteLLM token_counter failed: {e}")
            # Fallback or re-raise
            raise HTTPException(status_code=501, detail=f"Token counting not implemented or failed for model {anthropic_request.model}: {e}")

    except HTTPException as e:
         raise e # Re-raise known HTTP exceptions
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

# --- UI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def ui_root(request: Request):
    # Define available models for the dropdowns
    available_models = [
        # OpenAI models
        {"value": "gpt-4o", "label": "gpt-4o (OpenAI)"},
        {"value": "gpt-4o-mini", "label": "gpt-4o-mini (OpenAI)"},
        {"value": "gpt-3.5-turbo", "label": "gpt-3.5-turbo (OpenAI)"},
        {"value": "o1", "label": "o1 (OpenAI Reasoning)"},
        {"value": "o3-mini", "label": "o3-mini (OpenAI Reasoning)"},

        # Google Gemini models
        {"value": "gemini-1.5-pro", "label": "gemini-1.5-pro (Google)"},
        {"value": "gemini-1.5-flash", "label": "gemini-1.5-flash (Google)"},
        {"value": "gemini-2.0-flash", "label": "gemini-2.0-flash (Google)"},
        # Add more Gemini models if desired, e.g., gemini-2.5-pro-exp-03-25
        {"value": "gemini-2.5-pro-exp-03-25", "label": "gemini-2.5-pro-exp (Google)"},


        # Anthropic models (requires ANTHROPIC_API_KEY)
        {"value": "claude-3-opus-20240229", "label": "claude-3-opus (Anthropic)"},
        {"value": "claude-3-sonnet-20240229", "label": "claude-3-sonnet (Anthropic)"},
        {"value": "claude-3-haiku-20240307", "label": "claude-3-haiku (Anthropic)"},
    ]

    # Get provider and prefix for current selections to display correctly
    big_provider, _ = get_provider_and_prefix(BIG_MODEL)
    small_provider, _ = get_provider_and_prefix(SMALL_MODEL)
    big_model_display = f"{BIG_MODEL} ({big_provider.capitalize()})"
    small_model_display = f"{SMALL_MODEL} ({small_provider.capitalize()})"


    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "big_model_config": BIG_MODEL, # The configured value
            "small_model_config": SMALL_MODEL, # The configured value
            "big_model_display": big_model_display, # Value + Provider for display
            "small_model_display": small_model_display,# Value + Provider for display
            "available_models": available_models,
            "request_history": REQUEST_HISTORY
        }
    )

@app.post("/update_models")
async def update_models(big_model: str = Form(...), small_model: str = Form(...)):
    global BIG_MODEL, SMALL_MODEL

    # Check if required API keys are present for the selected models
    try:
        check_api_key_availability(f"{get_provider_and_prefix(big_model)[1]}{big_model}")
        check_api_key_availability(f"{get_provider_and_prefix(small_model)[1]}{small_model}")
    except HTTPException as e:
         # Return error as JSON for the frontend to handle
         return JSONResponse(
             status_code=400,
             content={"status": "error", "message": e.detail}
         )

    # Update the global model settings
    BIG_MODEL = big_model
    SMALL_MODEL = small_model

    logger.warning(f"✅ MODEL CONFIGURATION UPDATED: Big Model = {BIG_MODEL}, Small Model = {SMALL_MODEL}")

    # Return success and the updated values for the UI
    big_provider, _ = get_provider_and_prefix(BIG_MODEL)
    small_provider, _ = get_provider_and_prefix(SMALL_MODEL)
    return {
        "status": "success",
        "big_model": BIG_MODEL,
        "small_model": SMALL_MODEL,
        "big_model_display": f"{BIG_MODEL} ({big_provider.capitalize()})",
        "small_model_display": f"{SMALL_MODEL} ({small_provider.capitalize()})",
    }

@app.get("/api/history")
async def get_history():
    # Add provider info to history items for better UI display
    history_with_provider = []
    for req in REQUEST_HISTORY:
        new_req = req.copy()
        try:
             if req.get("mapped_model"):
                 provider, _ = get_provider_and_prefix(req["mapped_model"])
                 new_req["provider"] = provider
             else:
                 new_req["provider"] = "unknown"
        except Exception:
             new_req["provider"] = "error" # Handle cases where mapped_model might be invalid
        history_with_provider.append(new_req)

    return {"history": history_with_provider}


# --- Server Startup and UI Template Creation ---
@app.on_event("startup")
async def create_templates_and_log():
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)

    # Minimal server startup message with styling (using basic print for universal compatibility)
    print("\n" + "="*60)
    print(" Claude Code Proxy Server Starting Up ".center(60, "="))
    print("="*60)
    print(f"\n[*] Web UI available at: http://localhost:8082")
    print(f"[*] Connect Claude Code CLI:")
    print(f"    export ANTHROPIC_BASE_URL=http://localhost:8082")
    print(f"    claude")
    print(f"\n[*] Default Model Mapping:")
    print(f"    - Claude Sonnet -> {BIG_MODEL} ({get_provider_and_prefix(BIG_MODEL)[0].capitalize()})")
    print(f"    - Claude Haiku  -> {SMALL_MODEL} ({get_provider_and_prefix(SMALL_MODEL)[0].capitalize()})")
    print(f"\n[*] Required API Keys (in .env file):")
    keys_needed = set()
    keys_needed.add(get_provider_and_prefix(BIG_MODEL)[0])
    keys_needed.add(get_provider_and_prefix(SMALL_MODEL)[0])
    key_status = {
        "openai": "OPENAI_API_KEY" + (" (Found)" if OPENAI_API_KEY else " (Missing!)"),
        "gemini": "GOOGLE_API_KEY" + (" (Found)" if GOOGLE_API_KEY else " (Missing!)"),
        "anthropic": "ANTHROPIC_API_KEY" + (" (Found)" if ANTHROPIC_API_KEY else " (Missing!)")
    }
    for provider in keys_needed:
        print(f"    - {key_status.get(provider, 'Unknown Provider Key')}")
    if not keys_needed.issubset(key_status.keys()):
        print("    - Warning: Unknown provider selected for default models.")
    print("\n" + "="*60 + "\n")


    # Create index.html template
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Claude Code Proxy</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding-top: 2rem; background-color: #f0f2f5; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
            .header { background: linear-gradient(90deg, #4A90E2, #50E3C2); color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            .header h1 { font-weight: bold; }
            .card { margin-bottom: 1.5rem; box-shadow: 0 6px 16px rgba(0,0,0,0.08); border: none; border-radius: 10px; transition: transform 0.2s ease, box-shadow 0.2s ease; }
            .card:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.12); }
            .card-header { background-color: #4A90E2; color: white; font-weight: bold; border-top-left-radius: 10px; border-top-right-radius: 10px; padding: 1rem 1.25rem; }
            .model-badge { font-size: 0.85rem; padding: 0.35rem 0.75rem; border-radius: 20px; font-weight: 500; background-color: #50E3C2 !important; color: #333 !important; }
            .config-icon { font-size: 1.2rem; margin-right: 0.75rem; vertical-align: middle;}
            .table-responsive { max-height: 500px; overflow-y: auto; border-radius: 8px; }
            .options-note { font-size: 0.9rem; padding: 0.75rem; background-color: #e7f3fe; border-radius: 8px; border-left: 4px solid #4A90E2; margin-bottom: 1.25rem; }
            .status-success { color: #28a745; font-weight: 500; } /* Green */
            .status-error { color: #dc3545; font-weight: 500; } /* Red */
            .status-processing { color: #ffc107; font-weight: 500; } /* Yellow */
            .refresh-btn { font-size: 0.85rem; margin-left: 0.5rem; background-color: transparent; border-color: white; color: white; }
            .refresh-btn:hover { background-color: rgba(255,255,255,0.2); border-color: white; }
            .btn-primary { background-color: #4A90E2; border-color: #4A90E2; }
            .btn-primary:hover { background-color: #357ABD; border-color: #357ABD; }
            .list-group-item { border-radius: 6px; margin-bottom: 0.5rem; }
            pre { background-color: #e9ecef; padding: 1rem; border-radius: 8px; border: 1px solid #ced4da; color: #495057; font-size: 0.9em;}
            .badge { font-weight: 500; }
            .bg-primary { background-color: #4A90E2 !important; }
            table { border-collapse: separate; border-spacing: 0; font-size: 0.9rem; }
            table th { background-color: #f8f9fa; border-bottom: 2px solid #dee2e6; }
            table th, table td { padding: 0.6rem 0.75rem; vertical-align: middle; }
            table th:first-child { border-top-left-radius: 8px; }
            table th:last-child { border-top-right-radius: 8px; }
            .history-row-success { background-color: rgba(40, 167, 69, 0.05); } /* Light Green */
            .history-row-success:hover { background-color: rgba(40, 167, 69, 0.1); }
            .history-row-error { background-color: rgba(220, 53, 69, 0.05); } /* Light Red */
            .history-row-error:hover { background-color: rgba(220, 53, 69, 0.1); }
            .history-row-processing { background-color: rgba(255, 193, 7, 0.05); } /* Light Yellow */
            .history-row-processing:hover { background-color: rgba(255, 193, 7, 0.1); }
            .model-name { font-weight: 500; padding: 3px 8px; border-radius: 4px; display: inline-block; border: 1px solid transparent; }
            .model-claude { color: #D97927; background-color: #FFF0E5; border-color: #F5DBC8;} /* Orange-ish */
            .model-openai { color: #10A37F; background-color: #E7F6F2; border-color: #C3EBE0;} /* Teal-ish */
            .model-gemini { color: #4285F4; background-color: #E8F0FE; border-color: #C5D9FA;} /* Blue-ish */
            .model-unknown { color: #6c757d; background-color: #f8f9fa; border-color: #dee2e6;} /* Grey */
            .error-badge { cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header text-center">
                <h1>Claude Code Proxy</h1>
                <p class="mb-0">Use Claude Code CLI with OpenAI & Gemini Models via LiteLLM</p>
            </div>

            <!-- Alert container -->
            <div id="alertContainer" class="mb-4"></div>

            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header d-flex align-items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-gear-fill config-icon" viewBox="0 0 16 16">...</svg>
                            Configuration
                        </div>
                        <div class="card-body">
                            <form id="modelForm">
                                <div class="mb-3">
                                    <label for="bigModel" class="form-label">Big Model (for Sonnet requests)</label>
                                    <select class="form-select" id="bigModel" name="big_model">
                                        {% for model in available_models %}
                                            <option value="{{ model.value }}" {% if model.value == big_model_config %}selected{% endif %}>{{ model.label }}</option>
                                        {% endfor %}
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label for="smallModel" class="form-label">Small Model (for Haiku requests)</label>
                                    <select class="form-select" id="smallModel" name="small_model">
                                        {% for model in available_models %}
                                            <option value="{{ model.value }}" {% if model.value == small_model_config %}selected{% endif %}>{{ model.label }}</option>
                                        {% endfor %}
                                    </select>
                                </div>
                                <div class="options-note mb-3">
                                    <strong>Notes:</strong>
                                    <ul class="mb-0 mt-2 ps-3">
                                        <li>Select the target model for requests originating from Claude Code CLI.</li>
                                        <li>Ensure the correct API key (OpenAI, Google, Anthropic) is in your <code>.env</code> file for the selected provider.</li>
                                        <li>OpenAI reasoning models (o3-mini, o1) automatically use <code>reasoning_effort="high"</code>.</li>
                                    </ul>
                                </div>
                                <button type="submit" class="btn btn-primary">Save Configuration</button>
                            </form>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">Connection Info</div>
                        <div class="card-body">
                            <h5>Connect Claude Code CLI:</h5>
                            <pre><code>export ANTHROPIC_BASE_URL=http://localhost:8082\nclaude</code></pre>

                            <h5 class="mt-3">Current Mapping:</h5>
                            <ul class="list-group">
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Claude Sonnet Requests <i class="bi bi-arrow-right"></i>
                                    <span id="bigModelBadge" class="badge rounded-pill model-badge">{{ big_model_display }}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Claude Haiku Requests <i class="bi bi-arrow-right"></i>
                                    <span id="smallModelBadge" class="badge rounded-pill model-badge">{{ small_model_display }}</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <div>
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-activity config-icon" viewBox="0 0 16 16">...</svg>
                                Request History (Last {{ MAX_HISTORY }})
                            </div>
                            <button id="refreshHistory" class="btn btn-sm refresh-btn">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-clockwise" viewBox="0 0 16 16">...</svg>
                                Refresh
                            </button>
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-hover" id="historyTable">
                                    <thead>
                                        <tr>
                                            <th>Time</th>
                                            <th>Original</th>
                                            <th>Mapped To</th>
                                            <th>Msgs</th>
                                            <th>Stream</th>
                                            <th>Status</th>
                                            <th>Info</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <!-- History rows added by JS -->
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Bootstrap Modal for Errors -->
        <div class="modal fade" id="errorModal" tabindex="-1" aria-labelledby="errorModalLabel" aria-hidden="true">
          <div class="modal-dialog modal-lg">
            <div class="modal-content">
              <div class="modal-header">
                <h5 class="modal-title" id="errorModalLabel">Error Details</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
              </div>
              <div class="modal-body">
                <pre><code id="errorModalContent"></code></pre>
              </div>
              <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
              </div>
            </div>
          </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            const alertContainer = document.getElementById('alertContainer');
            const errorModalElement = document.getElementById('errorModal');
            const errorModal = new bootstrap.Modal(errorModalElement);
            const errorModalContent = document.getElementById('errorModalContent');

            function showAlert(message, type = 'info') {
                const wrapper = document.createElement('div');
                wrapper.innerHTML = [
                    `<div class="alert alert-${type} alert-dismissible fade show" role="alert">`,
                    `   <div>${message}</div>`,
                    '   <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
                    '</div>'
                ].join('');
                alertContainer.append(wrapper);
                // Auto-dismiss after 7 seconds
                setTimeout(() => { wrapper.remove(); }, 7000);
            }

            function getModelClass(modelName, provider) {
                 if (!modelName && !provider) return 'model-unknown';
                 provider = provider?.toLowerCase();
                 const name = modelName?.toLowerCase() || "";

                 if (provider === 'anthropic' || name.includes('claude')) return 'model-claude';
                 if (provider === 'openai' || name.includes('gpt-') || name.includes('o1') || name.includes('o3')) return 'model-openai';
                 if (provider === 'gemini' || name.includes('gemini')) return 'model-gemini';
                 return 'model-unknown';
            }

            // Form submission via AJAX
            document.getElementById('modelForm').addEventListener('submit', function(e) {
                e.preventDefault();
                const formData = new FormData(this);

                fetch('/update_models', { method: 'POST', body: formData })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        showAlert('Model configuration updated successfully!', 'success');
                        // Update UI elements immediately
                        document.getElementById('bigModelBadge').textContent = data.big_model_display;
                        document.getElementById('smallModelBadge').textContent = data.small_model_display;
                        // Update badge classes based on new model provider
                        document.getElementById('bigModelBadge').className = `badge rounded-pill model-badge ${getModelClass(data.big_model)}`;
                        document.getElementById('smallModelBadge').className = `badge rounded-pill model-badge ${getModelClass(data.small_model)}`;

                        refreshHistoryTable(); // Refresh history to show effect
                    } else {
                        showAlert(`Error updating models: ${data.message}`, 'danger');
                    }
                })
                .catch(error => {
                    console.error('Error submitting form:', error);
                    showAlert(`Network error updating models: ${error}`, 'danger');
                });
            });

            // Manual refresh history
            document.getElementById('refreshHistory').addEventListener('click', refreshHistoryTable);

            // Auto-refresh history table
            function refreshHistoryTable() {
                fetch('/api/history')
                .then(response => response.json())
                .then(data => {
                    const historyTableBody = document.getElementById('historyTable').getElementsByTagName('tbody')[0];
                    historyTableBody.innerHTML = ''; // Clear existing rows

                    data.history.forEach(req => {
                        const row = historyTableBody.insertRow();
                        let statusClass = 'secondary'; // Default/Processing
                        let rowClass = 'history-row-processing';
                        if (req.status === 'success') { statusClass = 'success'; rowClass = 'history-row-success'; }
                        if (req.status === 'error') { statusClass = 'danger'; rowClass = 'history-row-error'; }
                        row.className = rowClass;

                        row.insertCell(0).textContent = req.timestamp;

                        const originalCell = row.insertCell(1);
                        const originalSpan = document.createElement('span');
                        originalSpan.className = `model-name ${getModelClass(req.original_model)}`;
                        originalSpan.textContent = req.original_model || 'N/A';
                        originalCell.appendChild(originalSpan);

                        const mappedCell = row.insertCell(2);
                        const mappedSpan = document.createElement('span');
                        // Use provider info returned from API if available
                        mappedSpan.className = `model-name ${getModelClass(req.mapped_model, req.provider)}`;
                        mappedSpan.textContent = req.mapped_model || 'N/A';
                        mappedCell.appendChild(mappedSpan);

                        row.insertCell(3).textContent = req.num_messages ?? '?';
                        row.insertCell(4).textContent = req.stream ? 'Yes' : 'No';

                        const statusCell = row.insertCell(5);
                        statusCell.innerHTML = `<span class="status-${statusClass}">${req.status}</span>`;

                        const errorCell = row.insertCell(6);
                        if (req.status === 'error' && req.error) {
                            const errorBadge = document.createElement('span');
                            errorBadge.className = 'badge bg-danger error-badge';
                            // Truncate long errors for display in table
                            errorBadge.textContent = req.error.length > 50 ? req.error.substring(0, 47) + '...' : req.error;
                            errorBadge.dataset.fullError = req.error; // Store full error
                            errorBadge.addEventListener('click', function() {
                                errorModalContent.textContent = this.dataset.fullError;
                                errorModal.show();
                            });
                            errorCell.appendChild(errorBadge);
                        } else {
                             errorCell.textContent = '-';
                        }
                    });
                })
                .catch(error => {
                    console.error('Error refreshing history:', error);
                    // Optionally show an error to the user
                    // showAlert(`Failed to refresh history: ${error}`, 'warning');
                });
            }

            // Set up auto-refresh interval (e.g., every 10 seconds)
            const autoRefreshInterval = setInterval(refreshHistoryTable, 10000);

            // Initial load and setup
            document.addEventListener('DOMContentLoaded', () => {
                // Initial badge classes
                 document.getElementById('bigModelBadge').className = `badge rounded-pill model-badge ${getModelClass('{{ big_model_config }}')}`;
                 document.getElementById('smallModelBadge').className = `badge rounded-pill model-badge ${getModelClass('{{ small_model_config }}')}`;
                refreshHistoryTable();
            });

        </script>
    </body>
    </html>
    """

    # Write the HTML template to the file
    try:
        with open("templates/index.html", "w", encoding="utf-8") as f:
            f.write(index_html)
        logger.debug("templates/index.html created/updated successfully.")
    except IOError as e:
        logger.error(f"Failed to write templates/index.html: {e}")


if __name__ == "__main__":
    # Run Uvicorn programmatically with controlled logging
    # Note: Reload is harder to manage this way, run with `uvicorn server:app --reload ...` for development
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8082,
        log_level="warning", # Uvicorn's base log level
        # Use default log config which respects our handler setup above
    )
    server = uvicorn.Server(config)
    server.run()