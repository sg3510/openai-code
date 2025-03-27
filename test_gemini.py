import os
import json
import base64
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types
import tiktoken

# Format conversion utilities
class FormatConverter:
    """Utilities to convert between different LLM API formats"""
    
    @staticmethod
    def anthropic_to_gemini(request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic API request format to Gemini format"""
        gemini_request = {
            "contents": [],
            "config": {}
        }
        
        # Handle system prompt
        if "system" in request:
            gemini_request["contents"].append({
                "role": "user",
                "parts": [{"text": f"System instructions: {request['system']}"}]
            })
        
        # Handle messages
        for msg in request.get("messages", []):
            role = "user" if msg["role"] == "user" else "model"
            content_parts = []
            
            # Handle text content
            if "content" in msg:
                content_parts.append({"text": msg["content"]})
            
            # Handle multi-modal content (images)
            if "attachments" in msg:
                for attachment in msg["attachments"]:
                    if attachment["type"] == "image":
                        # Handle base64 image data
                        if "data" in attachment:
                            # Note: In a real implementation, you'd upload this to Gemini
                            # For this PoC, we'll just note that we'd handle the image
                            content_parts.append({"text": f"[Image attachment would be processed here]"})
            
            gemini_request["contents"].append({
                "role": role,
                "parts": content_parts
            })
        
        # Map parameters
        if "temperature" in request:
            gemini_request["config"]["temperature"] = request["temperature"]
        if "max_tokens" in request:
            gemini_request["config"]["maxOutputTokens"] = request["max_tokens"]
        if "top_p" in request:
            gemini_request["config"]["topP"] = request["top_p"]
        
        # Map tool/function calling
        if "tools" in request:
            gemini_request["config"]["tools"] = []
            function_declarations = []
            
            for tool in request["tools"]:
                if tool["type"] == "function":
                    function_declaration = {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"]["parameters"]
                    }
                    function_declarations.append(function_declaration)
            
            if function_declarations:
                gemini_request["config"]["tools"] = [{
                    "function_declarations": function_declarations
                }]
        
        return gemini_request
    
    @staticmethod
    def gemini_to_anthropic(response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Gemini API response format to Anthropic format"""
        anthropic_response = {
            "id": f"msg_{os.urandom(4).hex()}",
            "type": "message",
            "role": "assistant",
            "model": response.get("model", "unknown-model"),
            "content": []
        }
        
        # Handle text content
        if "text" in response:
            anthropic_response["content"].append({
                "type": "text",
                "text": response["text"]
            })
        
        # Handle function calls
        if "function_calls" in response and response["function_calls"]:
            function_call = response["function_calls"][0]
            anthropic_response["content"].append({
                "type": "tool_use",
                "tool_use": {
                    "id": f"tu_{os.urandom(4).hex()}",
                    "type": "function",
                    "name": function_call["name"],
                    "arguments": json.loads(function_call["arguments"])
                }
            })
        
        return anthropic_response

    @staticmethod
    def openai_to_gemini(request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI API request format to Gemini format"""
        gemini_request = {
            "contents": [],
            "config": {}
        }
        
        # Handle system message
        for msg in request.get("messages", []):
            if msg["role"] == "system":
                gemini_request["contents"].append({
                    "role": "user",
                    "parts": [{"text": f"System instructions: {msg['content']}"}]
                })
                break
        
        # Handle other messages
        for msg in request.get("messages", []):
            if msg["role"] == "system":
                continue  # Already handled above
                
            role = "user" if msg["role"] == "user" else "model"
            content_parts = []
            
            # Handle text content
            if isinstance(msg["content"], str):
                content_parts.append({"text": msg["content"]})
            
            # Handle multi-modal content
            elif isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item["type"] == "text":
                        content_parts.append({"text": item["text"]})
                    elif item["type"] == "image_url":
                        # For images in OpenAI format (either URL or base64)
                        image_url = item["image_url"]
                        if isinstance(image_url, dict) and "url" in image_url:
                            if image_url["url"].startswith("data:image"):
                                # Base64 image data
                                content_parts.append({"text": f"[Image attachment would be processed here]"})
                            else:
                                # Regular URL
                                content_parts.append({"text": f"[Image URL would be processed here]"})
            
            gemini_request["contents"].append({
                "role": role,
                "parts": content_parts
            })
        
        # Map parameters
        if "temperature" in request:
            gemini_request["config"]["temperature"] = request["temperature"]
        if "max_tokens" in request:
            gemini_request["config"]["maxOutputTokens"] = request["max_tokens"]
        if "top_p" in request:
            gemini_request["config"]["topP"] = request["top_p"]
        
        # Map tool/function calling
        if "tools" in request:
            gemini_request["config"]["tools"] = []
            function_declarations = []
            
            for tool in request["tools"]:
                if tool["type"] == "function":
                    function_declaration = {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"]["parameters"]
                    }
                    function_declarations.append(function_declaration)
            
            if function_declarations:
                gemini_request["config"]["tools"] = [{
                    "function_declarations": function_declarations
                }]
        
        return gemini_request

class TokenCounter:
    """Utility to estimate token counts for Gemini models"""
    
    @staticmethod
    def count_tokens(text: str, model: str = "gemini-2.5-pro") -> int:
        """
        Estimate token count for Gemini models
        This is a rough approximation using tiktoken (which is for OpenAI models)
        For production use, you would use Gemini's own token counting
        """
        # Use cl100k_base as a rough approximation
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(text)
        return len(tokens)

def test_gemini_text_gen(model_name, prompt):
    """Test text generation for a specific Gemini model"""
    print(f"\n--- Testing {model_name} Text Generation ---")

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    print(f"Prompt: '{prompt}'")
    print("Response:")

    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")
    print("\n")

def test_gemini_function_call(model_name, prompt):
    """Test function calling for a specific Gemini model"""
    print(f"\n--- Testing {model_name} Function Calling ---")

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="set_color",
                    description="sets a color",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "color": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                enum=["blue", "green", "red"],
                            ),
                        },
                    ),
                ),
            ])
    ]

    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="text/plain",
    )

    print(f"Prompt: '{prompt}'")
    print("Response:")

    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.function_calls:
            print(chunk.function_calls[0])
        else:
            print(chunk.text, end="")
    print("\n")

def test_streaming_response(model_name, prompt):
    """Test streaming responses and convert to Anthropic's event-stream format"""
    print(f"\n--- Testing {model_name} Streaming Response ---")

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    print(f"Prompt: '{prompt}'")
    print("Raw Gemini Streaming Response:")

    # Raw Gemini streaming
    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=generate_content_config,
    ):
        delta = chunk.text or ""
        full_response += delta
        print(delta, end="")
    print("\n")

    print("Converted to Anthropic event-stream format:")
    # Convert to Anthropic's event-stream format (simulated)
    print(f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': 'msg_123', 'model': model_name, 'role': 'assistant'}})}\n")
    
    # Split the response into smaller chunks to simulate streaming
    chunk_size = 10
    chunks = [full_response[i:i+chunk_size] for i in range(0, len(full_response), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        print(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text', 'text': chunk}})}\n")
    
    print(f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}})}\n")
    print(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n")

def test_multimodal_input(model_name, prompt, image_path):
    """Test handling image attachments in messages"""
    print(f"\n--- Testing {model_name} Multi-modal Input ---")

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    
    # Upload the image
    try:
        uploaded_file = client.files.upload(file=image_path)
        print(f"Successfully uploaded image: {image_path}")
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

        print(f"Prompt with image: '{prompt}'")
        print("Response:")

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        ):
            delta = chunk.text or ""
            response_text += delta
            print(delta, end="")
        print("\n")
        
        # Show how this would be converted to OpenAI format
        openai_format = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": model_name,
            "usage": {
                "prompt_tokens": 100,  # Placeholder value
                "completion_tokens": len(response_text.split()),  # Very rough estimation
                "total_tokens": 100 + len(response_text.split())  # Placeholder + rough estimation
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        }
        
        print("Converted to OpenAI format:")
        print(json.dumps(openai_format, indent=2))
        
    except Exception as e:
        print(f"Error handling image: {str(e)}")
        print("Note: For this test to work, ensure you have a valid image file at the specified path.")

def test_system_instructions(model_name, system_prompt, user_prompt):
    """Test mapping system instructions to Gemini's format"""
    print(f"\n--- Testing {model_name} System Instructions ---")

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # In Gemini, we need to include system instructions in the first user message
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"System instructions: {system_prompt}\n\nUser query: {user_prompt}"),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    print(f"System prompt: '{system_prompt}'")
    print(f"User prompt: '{user_prompt}'")
    print("Response:")

    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")
    print("\n")

def test_format_conversion():
    """Test converting request/response formats between different providers"""
    print("\n--- Testing Format Conversion ---")
    
    # Sample Anthropic request
    anthropic_request = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1024,
        "temperature": 0.7,
        "system": "You are a helpful AI assistant.",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you today?"
            }
        ]
    }
    
    # Convert to Gemini format
    gemini_request = FormatConverter.anthropic_to_gemini(anthropic_request)
    print("Anthropic request converted to Gemini format:")
    print(json.dumps(gemini_request, indent=2))
    
    # Sample OpenAI request
    openai_request = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "Hello, how are you today?"
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    
    # Convert to Gemini format
    gemini_request_from_openai = FormatConverter.openai_to_gemini(openai_request)
    print("\nOpenAI request converted to Gemini format:")
    print(json.dumps(gemini_request_from_openai, indent=2))
    
    # Sample Gemini response
    gemini_response = {
        "model": "gemini-2.5-pro",
        "text": "I'm an AI assistant created by Google. How can I help you today?",
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "total_tokens": 35
        }
    }
    
    # Convert to Anthropic format
    anthropic_response = FormatConverter.gemini_to_anthropic(gemini_response)
    print("\nGemini response converted to Anthropic format:")
    print(json.dumps(anthropic_response, indent=2))

def test_token_counting(model_name, text):
    """Test token counting for Gemini models"""
    print(f"\n--- Testing Token Counting for {model_name} ---")
    
    print(f"Text: '{text}'")
    token_count = TokenCounter.count_tokens(text, model_name)
    print(f"Estimated token count: {token_count}")
    
    print("Note: This is a rough approximation. For production use,")
    print("you would use Gemini's own token counting functionality.")

def main():
    # Set up test prompts
    text_gen_prompt = "Write a short poem about artificial intelligence"
    function_call_prompt = "I want to set the color to blue"
    streaming_prompt = "Explain quantum computing in simple terms"
    multimodal_prompt = "What do you see in this image?"
    system_prompt = "You are a playful AI that speaks in rhymes"
    user_prompt = "Tell me about the weather"
    token_counting_text = "This is a test of the token counting functionality"
    
    # Path to an image file for multimodal testing
    image_path = "test_files/image.png"
    
    # Test format conversion
    test_format_conversion()
    
    # Test token counting
    test_token_counting("gemini-2.5-pro", token_counting_text)
    
    # Test Gemini 2.0 text generation
    test_gemini_text_gen("gemini-2.0-flash", text_gen_prompt)
    
    # Test Gemini 2.5 text generation
    test_gemini_text_gen("gemini-2.5-pro-exp-03-25", text_gen_prompt)
    
    # Test streaming responses
    test_streaming_response("gemini-2.5-pro-exp-03-25", streaming_prompt)
    
    # Test system instructions
    test_system_instructions("gemini-2.5-pro-exp-03-25", system_prompt, user_prompt)
    
    # Test Gemini 2.0 function calling
    test_gemini_function_call("gemini-2.0-flash", function_call_prompt)
    
    # Test Gemini 2.5 function calling
    test_gemini_function_call("gemini-2.5-pro-exp-03-25", function_call_prompt)
    
    # Test multimodal input
    test_multimodal_input("gemini-2.5-pro-exp-03-25", multimodal_prompt, image_path)

if __name__ == "__main__":
    main()