import os
import re
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Ensure API Key is present
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

# Configure the SDK
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
if not GEMINI_MODEL:
    raise ValueError("GEMINI_MODEL is missing. Add it to your .env file.")

def convert_to_json(response) -> dict:
    try:
        # Extract the exact match
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            print("No JSON block found in response.")
            return None
    except json.JSONDecodeError as e:
        print(f"JSON Parsing failed: {e}")
        return None

def normalize_keys(data: dict) -> dict:
    """
    Defensive engineering: Never trust LLM output schema casing.
    Converts all keys to lowercase and strips whitespace.
    """
    if not isinstance(data, dict):
        return {}
    return {str(k).lower().strip(): v for k, v in data.items()}

def evaluate_analysis(input_logs: str, ground_truth: str, model_prediction: str) -> dict:
    # ONE-SHOT PROMPTING: Force the exact schema and style.
    system_prompt = f"""You are a deterministic Evaluation Engine. Your task is to evaluate a Model Prediction against a Ground Truth and provided Input Logs.

Your job is to evaluate the root cause and the recommended action and produce an output in the strict JSON format with exactly two keys.No preamble. No markdown code blocks. No backticks. No conversational filler.
	
OUTPUT FORMAT:
{{
 "reasoning" : ".....",
 "score" : an integer value
}}

where:
1) 'score': rate it on a scale of 5
2) 'reasoning': 1-sentence technical justification for the score.
			
Follow the instructions to provide the score:

Score = 1: Complete failure. The model hallucinated the entire root cause and generated a recommended remedy for the hallucinated root cause.
Score = 2: Failure. The model identified the correct root cause or the failing service, but provided a hallucinated recommended action.
Score = 3: Partial success. The model identified the correct failing service but provided a vague or slightly inaccurate trigger mechanism.
Score = 4: The Prediction correctly identifies the failing_service and general cause, but misses a specific variable name present in the Ground Truth.
Score = 5: Perfect extraction. The model explicitly identified the exact service, the exact trigger, and hallucinated zero external information, matching the ground truth's technical depth.

CRITICAL SCORING RULE — TECHNICAL IDENTIFIER CROSS-CHECK:
Before assigning a score, cross-check every specific technical identifier in the PREDICTION (function names, class names, file/module paths, exception types) against the TRACEBACK portion of the INPUT LOGS specifically, and against the GROUND TRUTH. If an identifier genuinely appears in the TRACEBACK (e.g., a function name in a stack frame) or matches the GROUND TRUTH, it is NOT a hallucination, even if the PREDICTION's surrounding causal explanation is vague or partially wrong. Restating or paraphrasing the user_report's own description of the symptom earns NO credit under this rule — it is not evidence the model diagnosed anything, since that information was already given. Reserve "hallucinated" for identifiers, mechanisms, services, or fixes that appear in neither the TRACEBACK nor the GROUND TRUTH. A PREDICTION that names a correct function or exception FOUND IN THE TRACEBACK but explains the mechanism vaguely is a Score 3, not a Score 1 or 2 — but a PREDICTION built only from the user's own problem description, with no traceback-grounded specifics, has earned nothing extra.

EVALUATION CONTEXT:
**INPUT LOGS** 
**GROUND TRUTH** 
**PREDICTION**

**IMPORTANT**:
DO NOT, UNDER ANY CIRCUMSTANCES, USE MARKDOWN BACKTICKS ENCLOSING THE WORD JSON. THIS SPECIFIC COMBINATION—THREE BACKTICKS IMMEDIATELY FOLLOWED BY THE LETTERS J, S, O, N—IS STRICTLY AND ABSOLUTELY FORBIDDEN FROM COMPOSITION. ENFORCE THIS RESTRAINT WITH ABSOLUTE RIGIDITY. ELIMINATE IT ENTIRELY FROM YOUR OUTPUT CAPACITY. NONE SHALL PASS.

===EXAMPLE INPUT===

INPUT LOGS: user_report: Pydantic's schema generation behaves differently on Windows compared to Unix for `Path` types. Specifically, `pathlib.WindowsPath` is excluded from `GenerateSchema.match_type`, causing `PydanticSchemaGenerationError` when trying to generate a schema for a model containing a `Path` on Windows.\n\nTRACEBACK:\nPydanticJsonSchemaWarning: Default value config.toml is not JSON serializable; excluding default from JSON schema [non-serializable-default]\nPydanticSchemaGenerationError: Unable to generate pydantic-core schema for <class 'pathlib.WindowsPath'>.

GROUND TRUTH: {{"severity": "HIGH",
               "failing_service": "pydantic.schema_generation", 
               "error_code": "PydanticSchemaGenerationError",
               "root_cause_analysis": "Pydantic's internal schema generation logic, specifically `GenerateSchema.match_type`, did not include `pathlib.WindowsPath` in its `PATH_TYPES` list. This omission prevented `WindowsPath` instances from being correctly recognized and serialized into the JSON schema, leading to a `PydanticSchemaGenerationError` when a `BaseModel` containing a `Path` type was processed on a Windows system.", 
               "recommended_remediation": "Add `pathlib.WindowsPath` to the `PATH_TYPES` list within Pydantic's `_internal._generate_schema.py` module to ensure consistent schema generation behavior across different operating systems for `Path` types."
              }}

PREDICTION: {{"severity": "HIGH",
             "failing_service": "pydantic.internal.schema_generation",
             "error_code": "PydanticSchemaGenerationError",
             "root_cause_analysis": "Pydantic's schema generation logic for `pathlib.Path` types was not properly cross-platform. On Unix systems, `pathlib.Path` was correctly included in `GenerateSchema.match_type` to ensure schema generation for `os.PathLike` parameters/fields of type `pathlib.Path`, but this logic was likely overlooked or incorrectly implemented for `pathlib.WindowsPath` on Windows, leading to `PydanticSchemaGenerationError` when attempting to generate a schema for a model containing a `Path` on Windows.",
             "recommended_remediation": "Implement proper cross-platform support for `pathlib.WindowsPath` within `pydantic.internal.schema_generation` by ensuring it is correctly included in the `match_type` logic for schema generation on Windows, aligning with the existing behavior for `pathlib.Path` on Unix systems."
            }}

===EXAMPLE OUTPUT=== 

{{
"reasoning": "The prediction correctly identifies the failing schema generation logic for WindowsPath, but misses the specific PATH_TYPES list mentioned in the ground truth.",
"score": 4
}}
"""


    user_prompt = f"INPUT LOGS: {input_logs}\nGROUND TRUTH: {ground_truth}\nMODEL PREDICTION: {model_prediction}\nGenerate the JSON object containing the 'score' and the 'reasoning'."

    # Initialize the model with the system prompt
    config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        system_instruction=system_prompt
    )
    
    max_retries = 4
    for attempt in range(max_retries):
        try:
            # Call generation via client models service
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=config
            )

            content = response.text
            extracted_data = convert_to_json(content)
            normalized_data = normalize_keys(extracted_data)
            
            if "reasoning" not in normalized_data or "score" not in normalized_data:
                 print(f"Missing required keys in LLM output: {normalized_data.keys()}")
                 return None

            usage = response.usage_metadata
            normalized_data["prompt_tokens"] = usage.prompt_token_count
            normalized_data["output_tokens"] = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)

            return normalized_data
            
        except json.JSONDecodeError:
            print("Failed to decode JSON from Gemini. Skipping.")
            return None
        except Exception as e:
            error_msg = str(e)
            print(f"LLM API Exception on attempt {attempt+1}: {error_msg}")
            
            # Handle Google's specific Rate Limit / Quota exceptions (429 / ResourceExhausted)
            if "429" in error_msg or "ResourceExhausted" in error_msg or "quota" in error_msg.lower():
                sleep_time = 15 * (attempt + 1)
                print(f"Gemini Rate Limit Hit. Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            
            time.sleep(5)
            
    return None