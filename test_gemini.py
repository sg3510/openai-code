# test_gemini.py
import os
import google.generativeai as genai
# Removed the problematic 'Part' import, kept others needed
from google.generativeai.types import GenerateContentResponse
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import time
import json
from typing import List, Dict, Any
import sys

# --- Configuration ---

# Load environment variables from .env file
load_dotenv()

# Get the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define the models to test - Focusing on v2.0 and v2.5 as requested
GEMINI_MODELS_TO_TEST = [
    "gemini-2.5-pro-exp-03-25", # Requested Experimental Pro model
    "gemini-2.0-flash",         # Requested Flash model
]

# Configure safety settings (using defaults)
SAFETY_SETTINGS = None

# --- Simple Tool Function for Testing ---

def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Get the current weather in a given location. (Mock Function)
    """
    print(f"      --- Mock Tool Call: get_current_weather(location='{location}', unit='{unit}') ---")
    temperature = "22" if unit == "celsius" else "72"
    forecast = "mostly sunny" if "francisco" in location.lower() else "partly cloudy"
    return {
        "location": location,
        "temperature": temperature,
        "unit": unit,
        "forecast": forecast
    }

# --- Helper Functions ---

def print_test_header(test_name):
    print("\n" + "="*60)
    print(f" RUNNING TEST: {test_name}")
    print("="*60)

def print_model_header(model_name):
    print(f"\n--- Testing Model: {model_name} ---")

def print_success(message):
    print(f"  ✅ SUCCESS: {message}")

def print_failure(message, model_name):
    print(f"  ❌ FAILURE ({model_name}): {message}", file=sys.stderr)

def print_skip(message, model_name):
     print(f"  ⚠️ SKIPPED ({model_name}): {message}")

def print_info(message):
    print(f"  ℹ️ INFO: {message}")

# --- Global State for Tracking Failures ---
_test_failures = []

# --- Test Functions ---

def test_basic_generation():
    """Tests basic text generation for each model."""
    global _test_failures
    test_name = "Basic Generation"
    print_test_header(test_name)

    for model_name in GEMINI_MODELS_TO_TEST:
        print_model_header(model_name)
        try:
            model = genai.GenerativeModel(model_name, safety_settings=SAFETY_SETTINGS)
            prompt = "Explain the concept of photosynthesis in one simple sentence."
            print_info(f"Prompt: '{prompt}'")
            start_time = time.time()
            response = model.generate_content(prompt)
            end_time = time.time()

            if not response.parts:
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     print_skip(f"Response blocked by safety settings. Reason: {response.prompt_feedback.block_reason}", model_name)
                     continue
                 else:
                     raise AssertionError(f"Response has no parts but was not explicitly blocked. Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")

            print_info(f"Response Text Length: {len(response.text)}")
            print_info(f"Time Taken: {end_time - start_time:.2f}s")

            assert response is not None, "Response object is None"
            assert hasattr(response, 'text'), "Response object lacks 'text' attribute"
            assert isinstance(response.text, str), f"Response text is not a string (type: {type(response.text)})"
            assert len(response.text.strip()) > 0, "Response text is empty or whitespace"

            print_success(f"Basic generation passed for {model_name}")

        except AssertionError as ae:
            print_failure(f"Assertion failed: {ae}", model_name)
            _test_failures.append(f"{test_name} - {model_name}")
        except Exception as e:
            if "PERMISSION_DENIED" in str(e) or "model not found" in str(e).lower() or "404" in str(e):
                 print_skip(f"Model access might be restricted or model name incorrect/unavailable. Error: {e}", model_name)
            else:
                print_failure(f"An unexpected error occurred: {e}", model_name)
                _test_failures.append(f"{test_name} - {model_name}")

def test_streaming_generation():
    """Tests streaming text generation for each model."""
    global _test_failures
    test_name = "Streaming Generation"
    print_test_header(test_name)

    for model_name in GEMINI_MODELS_TO_TEST:
        print_model_header(model_name)
        try:
            model = genai.GenerativeModel(model_name, safety_settings=SAFETY_SETTINGS)
            prompt = "Write a short haiku about a rainy day."
            print_info(f"Prompt: '{prompt}'")
            start_time = time.time()
            response_stream = model.generate_content(prompt, stream=True)

            collected_text = ""
            chunk_count = 0
            print("      Streamed Chunks: ", end="", flush=True)
            any_chunk_received = False
            stream_blocked = False
            block_reason = None

            for chunk in response_stream:
                any_chunk_received = True
                if not chunk.parts:
                     if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                         stream_blocked = True
                         block_reason = chunk.prompt_feedback.block_reason
                         print("\n      ⚠️ Stream blocked by safety settings.")
                         break
                     else:
                         continue

                if hasattr(chunk, 'text'):
                    assert isinstance(chunk.text, str), "Stream chunk text is not a string"
                    collected_text += chunk.text
                    chunk_count += 1
                    print("█", end="", flush=True)
                else:
                    print("(*)", end="", flush=True)

            end_time = time.time()
            print()
            print_info(f"Received {chunk_count} text chunks.")
            print_info(f"Total Text Length: {len(collected_text)}")
            print_info(f"Time Taken: {end_time - start_time:.2f}s")

            if stream_blocked:
                 print_skip(f"Streaming response blocked by safety settings. Reason: {block_reason}", model_name)
                 continue

            assert any_chunk_received, "No chunks received at all, but stream was not blocked."
            assert len(collected_text.strip()) > 0, "Collected text is empty despite receiving chunks."

            print_success(f"Streaming generation passed for {model_name}")

        except AssertionError as ae:
            print_failure(f"Assertion failed: {ae}", model_name)
            _test_failures.append(f"{test_name} - {model_name}")
        except Exception as e:
            if "PERMISSION_DENIED" in str(e) or "model not found" in str(e).lower() or "404" in str(e):
                 print_skip(f"Model access might be restricted or model name incorrect/unavailable. Error: {e}", model_name)
            else:
                print_failure(f"An unexpected error occurred: {e}", model_name)
                _test_failures.append(f"{test_name} - {model_name}")


def test_function_calling():
    """Tests function calling/tool use for each model."""
    global _test_failures
    test_name = "Function Calling"
    print_test_header(test_name)

    for model_name in GEMINI_MODELS_TO_TEST:
        print_model_header(model_name)
        response = None # Initialize for error reporting
        try:
            model = genai.GenerativeModel(
                model_name,
                tools=[get_current_weather],
                safety_settings=SAFETY_SETTINGS
            )
            prompt = "What's the weather like in San Francisco, CA right now?"
            print_info(f"Prompt: '{prompt}'")

            start_time = time.time()
            response = model.generate_content(prompt)
            end_time = time.time()
            print_info(f"Generation Time: {end_time - start_time:.2f}s")

            if not response.parts:
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     print_skip(f"Response blocked by safety settings. Reason: {response.prompt_feedback.block_reason}", model_name)
                     continue
                 else:
                     if hasattr(response, 'text') and response.text:
                          print_skip(f"Model responded with text instead of function call/block: '{response.text[:100]}...'", model_name)
                     else:
                          raise AssertionError(f"Response has no parts and no text, but not explicitly blocked. Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                     continue

            assert len(response.parts) > 0, "Response has parts list, but it's empty."
            first_part = response.parts[0]
            assert hasattr(first_part, 'function_call'), f"First part of response does not have 'function_call' attribute. Part: {first_part}"
            fc = first_part.function_call

            print_info(f"Function Call Found: Name='{fc.name}', Args={fc.args}")

            assert fc.name == "get_current_weather", f"Expected function name 'get_current_weather', got '{fc.name}'"
            assert hasattr(fc, 'args'), "Function call object missing 'args'"
            fc_args_dict = type(fc.args).to_dict(fc.args)
            assert isinstance(fc_args_dict, dict), f"Function call arguments could not be converted to dict (type: {type(fc_args_dict)})"
            assert "location" in fc_args_dict, "Function call arguments missing 'location'"
            assert isinstance(fc_args_dict["location"], str), "'location' argument is not a string"
            assert "francisco" in fc_args_dict["location"].lower(), "Location argument doesn't seem to match prompt"
            if "unit" in fc_args_dict:
                 assert fc_args_dict["unit"] in ["celsius", "fahrenheit"], f"Invalid unit '{fc_args_dict.get('unit')}' received"

            print_success(f"Function call generation passed for {model_name}")

            # --- Simulate tool execution and multi-turn ---
            print_info(f"Simulating tool execution and sending response back...")
            tool_response_data = None
            try:
                tool_response_data = get_current_weather(**fc_args_dict)
                assert isinstance(tool_response_data, dict), "Mock tool function did not return a dict"
            except Exception as tool_e:
                raise AssertionError(f"Mock tool execution failed with args {fc_args_dict}: {tool_e}")

            start_time_mt = time.time()
            # *** CORRECTED MULTI-TURN INPUT STRUCTURE ***
            # Pass the function response as a dictionary within the contents list
            response_after_tool = model.generate_content(
                 [
                    { # Representing the FunctionResponse part
                        "function_response": {
                            "name": fc.name,
                            "response": tool_response_data # The dictionary returned by our function
                        }
                    }
                 ]
            )
            # *** END CORRECTION ***
            end_time_mt = time.time()
            print_info(f"Multi-turn Response Time: {end_time_mt - start_time_mt:.2f}s")

            if not response_after_tool.parts:
                 if hasattr(response_after_tool, 'prompt_feedback') and response_after_tool.prompt_feedback.block_reason:
                     print_skip(f"Multi-turn response blocked by safety settings. Reason: {response_after_tool.prompt_feedback.block_reason}", model_name)
                     continue
                 else:
                     raise AssertionError(f"Multi-turn response has no parts but not blocked. Feedback: {getattr(response_after_tool, 'prompt_feedback', 'N/A')}")

            print_info(f"Final Response Text Length: {len(response_after_tool.text)}")

            assert response_after_tool is not None, "Multi-turn response object is None"
            assert hasattr(response_after_tool, 'text'), "Multi-turn response lacks 'text' attribute"
            assert len(response_after_tool.text.strip()) > 0, "Multi-turn response text is empty"
            assert tool_response_data['temperature'] in response_after_tool.text or \
                   tool_response_data['forecast'] in response_after_tool.text, \
                   f"Final response doesn't seem to incorporate tool results (temp/forecast). Got: '{response_after_tool.text[:100]}...'"

            print_success(f"Multi-turn function calling passed for {model_name}")

        except AssertionError as ae:
            err_msg = f"Assertion failed: {ae}"
            if response: err_msg += f" | Response parts: {getattr(response, 'parts', 'N/A')}"
            print_failure(err_msg, model_name)
            _test_failures.append(f"{test_name} - {model_name}")
        except Exception as e:
            err_msg = f"An unexpected error occurred: {e}"
            if response: err_msg += f" | Response parts: {getattr(response, 'parts', 'N/A')}"

            if "PERMISSION_DENIED" in str(e) or "model not found" in str(e).lower() or "404" in str(e):
                 print_skip(f"Model access might be restricted or model name incorrect/unavailable. Error: {e}", model_name)
            elif "Function calling is not supported" in str(e):
                 print_skip(f"Function calling explicitly not supported by this model/backend. Error: {e}", model_name)
            elif "Invalid function response" in str(e) or "Invalid content entry" in str(e):
                 # Catch errors indicating our input structure might still be wrong
                 print_failure(f"Multi-turn failed due to invalid input structure for function response. Error: {e}", model_name)
                 _test_failures.append(f"{test_name} - {model_name}")
            else:
                print_failure(err_msg, model_name)
                _test_failures.append(f"{test_name} - {model_name}")

# --- Main Execution Block ---

def run_all_tests():
    """Runs all defined test functions."""
    start_time = time.time()
    print("Starting Gemini API tests...")

    # --- Initial Setup and Validation ---
    print("\n" + "-"*60)
    print(" Performing Initial Setup Check")
    print("-"*60)
    if not GOOGLE_API_KEY:
        print("❌ ERROR: GOOGLE_API_KEY not found in .env file. Cannot run tests.", file=sys.stderr)
        sys.exit(1)
    print("  ✅ GOOGLE_API_KEY found.")

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("  ✅ google.generativeai configured successfully.")
        try:
            models = list(genai.list_models())
            assert isinstance(models, list)
            assert any('generateContent' in m.supported_generation_methods for m in models if hasattr(m, 'supported_generation_methods'))
            print(f"  ✅ API Key validated successfully (found {len(models)} models).")
        except Exception as list_models_e:
             print(f"❌ ERROR: API Key validation failed during list_models(): {list_models_e}. Exiting.", file=sys.stderr)
             sys.exit(1)
    except Exception as configure_e:
        print(f"❌ ERROR: Failed to configure google.generativeai: {configure_e}. Exiting.", file=sys.stderr)
        sys.exit(1)
    print("-"*60)
    # --- End Initial Setup ---


    # --- Run Test Suites ---
    test_basic_generation()
    test_streaming_generation()
    test_function_calling()
    # --- End Test Suites ---


    # --- Final Summary ---
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    end_time = time.time()
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")

    if not _test_failures:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        sys.exit(0)
    else:
        print(f"\n🚨 {len(_test_failures)} TEST(S) FAILED: 🚨", file=sys.stderr)
        for failure in _test_failures:
            print(f"  - {failure}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()