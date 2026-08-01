import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from google import genai
from google.genai import types

# 1. Define the Strict Structured Output Schema using Pydantic
class Scenario(BaseModel):
    scenario_text: str = Field(description="The full narrative including the setup, interventions, and final claim. End with 'Claim: [claim text]'")
    trap_type: Literal["semantic_trap", "syntax_trap", "baseline"] = Field(description="Whether this scenario exploits statistical pattern matching or strict causal parsing.")
    ground_truth: bool = Field(description="The actual logical truth of the claim based on strict causal deduction.")
    explanation: str = Field(description="A brief step-by-step causal trace proving the ground truth.")

# 2. Define the Prompts
SEMANTIC_TRAP_PROMPT = """
You are an expert dataset generator for AI alignment research. Generate a 'Semantic Trap' Theory of Mind scenario.
1. Create a scenario with multiple agents and a hidden mental state or moved object.
2. The narrative must use strong, repetitive associative keywords (e.g., 'abandoned', 'never found', 'lost forever') that statistically point toward a specific outcome.
3. However, a strict, step-by-step causal intervention in the text (e.g., a hidden camera, an overheard conversation, a reflective mirror) must logically lead to the OPPOSITE outcome.
4. End the narrative with 'Claim: [agent] will look for the [object] in the [location].'
Ensure that a heuristic pattern-matcher would guess incorrectly, but a strict causal logician would guess correctly.
"""

SYNTAX_TRAP_PROMPT = """
You are an expert dataset generator for AI alignment research. Generate a 'Syntax Trap' Theory of Mind scenario.
1. The underlying causal logic of the scenario must be incredibly simple (e.g., Agent A puts a ball in a box. Agent A leaves. Agent B moves it to a drawer. Agent A will look in the box).
2. However, describe these events using highly convoluted language, double-negatives, dense passive voice, and distracting philosophical metaphors. (e.g., 'It was not unobserved by the lingering shadow of the room that the sphere was relocated...').
3. End the narrative with 'Claim: [agent] will look for the [object] in the [location].'
Ensure that a rigid structural parser would be confused by the syntax and fail, but a statistical pattern-matcher would easily grasp the semantic 'gist' and guess correctly.
"""

BASELINE_PROMPT = """
You are an expert dataset generator for AI alignment research. Generate a 'Baseline' Theory of Mind scenario.
1. The underlying causal logic of the scenario must be straightforward (e.g., Agent A puts a ball in a box. Agent B moves it to a drawer while Agent A is away).
2. The narrative must use clear, standard language, and the statistical associations of the keywords must point to the same outcome as the causal logic.
3. End the narrative with 'Claim: [agent] will look for the [object] in the [location].'
Output exactly matching the provided JSON schema, setting the trap_type to baseline.
"""

def generate_dataset(num_per_type=5, output_file="adversarial_tom_dataset.json"):
    # 1. Load environment variables from .env file
    load_dotenv()
    # Initialize the Gemini Client
    client = genai.Client()
    # Using the current frontier flash model for fast, cheap, high-reasoning generation
    model_id = "gemini-3.5-flash" 
    
    dataset = []
    
    # Configure the client to strictly return our Pydantic schema
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Scenario,
        temperature=0.7
    )

    print("Generating Semantic Traps...")
    for i in range(num_per_type):
        success = False
        max_retries = 3
        retry_delay = 15 # Start with a 15-second wait if it fails

        while not success and max_retries > 0:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=SEMANTIC_TRAP_PROMPT,
                    config=config
                )
                
                parsed_obj = response.parsed
                if parsed_obj:
                    dataset.append(parsed_obj.model_dump())
                    print(f"Generated Semantic Trap {i+1}/{num_per_type}")
                    success = True # Break out of the while loop
                    
            except Exception as e:
                print(f"Error: {e}")
                max_retries -= 1
                if max_retries > 0:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Double the wait time for the next attempt (exponential backoff)
                else:
                    print(f"Failed to generate Semantic Trap {i+1} after multiple attempts. Skipping.")
        
        time.sleep(13)

    print("\nGenerating Syntax Traps...")
    for i in range(num_per_type):
        success = False
        max_retries = 3
        retry_delay = 15 # Start with a 15-second wait if it fails

        while not success and max_retries > 0:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=SYNTAX_TRAP_PROMPT,
                    config=config
                )
                
                parsed_obj = response.parsed
                if parsed_obj:
                    dataset.append(parsed_obj.model_dump())
                    print(f"Generated Syntax Trap {i+1}/{num_per_type}")
                    success = True # Break out of the while loop
                    
            except Exception as e:
                print(f"Error generating syntax trap: {e}")
                max_retries -= 1
                if max_retries > 0:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Double the wait time for the next attempt (exponential backoff)
                else:
                    print(f"Failed to generate Syntax Trap {i+1} after multiple attempts. Skipping.")

        time.sleep(13)

    print("\nGenerating Baseline Scenarios...")
    for i in range(num_per_type):
        success = False
        max_retries = 3
        retry_delay = 15 # Start with a 15-second wait if it fails
    
        while not success and max_retries > 0:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=BASELINE_PROMPT,
                    config=config
                )
                    
                parsed_obj = response.parsed
                if parsed_obj:
                    dataset.append(parsed_obj.model_dump())
                    print(f"Generated Baseline Scenario {i+1}/{num_per_type}")
                    success = True # Break out of the while loop
                        
            except Exception as e:
                print(f"Error generating baseline scenario: {e}")
                max_retries -= 1
                if max_retries > 0:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Double the wait time for the next attempt (exponential backoff)
                else:
                    print(f"Failed to generate Baseline Scenario {i+1} after multiple attempts. Skipping.")
    
        time.sleep(13)
            
    # Save to disk in the format Inspect expects (list of JSON objects)
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nSuccessfully generated {len(dataset)} scenarios and saved to {output_file}.")

if __name__ == "__main__":
    # Start with 5 of each for testing. Change to 300 for the final run.
    generate_dataset(num_per_type=5)