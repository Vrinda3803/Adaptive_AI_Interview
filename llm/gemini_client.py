
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

def _client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    return genai.Client(api_key=api_key)
    
MODEL_NAMES = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]


def _should_try_next_model(error):
    code = getattr(error, "code", None)
    try:
        code = int(code)
    except (TypeError, ValueError):
        return isinstance(error, errors.ServerError)
    return code in {404, 429, 500, 502, 503, 504}


def generate_response(prompt, max_retries=2):
    for model_name in MODEL_NAMES:
        print(f"\nTrying Gemini model: {model_name}")
        for attempt in range(max_retries):
            try:
                response = _client().models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if not response.text:
                    print(f"{model_name} returned an empty response. Trying fallback.")
                    break
                print(f"Successfully used: {model_name}")
                return response.text
            except errors.ServerError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(
                        f"{model_name} temporarily unavailable. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                    continue
                print(f"{model_name} unavailable after retries: {e}. Trying fallback.")
                break

            except errors.ClientError as e:
                if not _should_try_next_model(e):
                    raise RuntimeError(
                        f"Gemini request failed with HTTP {getattr(e, 'code', 'unknown')}; "
                        "check the API key, permissions, and request format."
                    ) from e
                print(f"{model_name} returned HTTP {e.code}. Trying fallback.")
                break

    raise RuntimeError(
        "All Gemini models are currently unavailable. "
        "Please try again later or check your API quota."
    )
if __name__ == "__main__":
    test_prompt = (
        "Generate one medium-difficulty Python interview question "
        "for a final-year computer science student."
    )
    print("\nGEMINI RESPONSE:\n")
    try:
        print(generate_response(test_prompt))
    except Exception as e:
        print("Gemini test failed:", e)
