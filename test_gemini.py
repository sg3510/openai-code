import os
from google import genai
from google.genai import types

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

def main():
    # Set up test prompts
    text_gen_prompt = "Write a short poem about artificial intelligence"
    function_call_prompt = "I want to set the color to blue"
    
    # Test Gemini 2.0 text generation
    test_gemini_text_gen("gemini-2.0-flash", text_gen_prompt)
    
    # Test Gemini 2.5 text generation
    test_gemini_text_gen("gemini-2.5-pro-exp-03-25", text_gen_prompt)
    
    # Test Gemini 2.0 function calling
    test_gemini_function_call("gemini-2.0-flash", function_call_prompt)
    
    # Test Gemini 2.5 function calling
    test_gemini_function_call("gemini-2.5-pro-exp-03-25", function_call_prompt)

if __name__ == "__main__":
    main()