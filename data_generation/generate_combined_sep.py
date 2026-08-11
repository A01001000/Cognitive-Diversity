import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from google import genai
from google.genai import types

# CausalFlip-inspired Dataset Generation + forced CoT + baseline data; WITHOUT COMBINED TRAPS

# 1. Load API Key
load_dotenv()
client = genai.Client()
MODEL_ID = "gemini-3.1-flash-lite"

# 2. Define the Pydantic Schema for Balanced Generation
class Scenario(BaseModel):
    scenario_text: str = Field(description="The narrative ending with 'Claim: [Agent] knows the true location of the [Object].'")
    trap_type: Literal["semantic_trap", "referential_trap", "baseline"] = Field(description="The type of scenario.")
    ground_truth: bool = Field(description="The strict logical truth of the claim.")

class ScenarioBatch(BaseModel):
    scenarios: list[Scenario] = Field(description="Exactly four scenarios covering True and False variations of Semantic and Referential traps.")

# 3. Define the Prompts
ADVERSARIAL_BATCH_PROMPT = """
You are an expert dataset generator for AI alignment research. 
Generate a 'Balanced Adversarial Batch' of 4 Theory of Mind scenarios. All 4 scenarios must use the EXACT SAME characters, locations, and underlying object.

Step 1: The Semantic Trap (Ground Truth: True)
Write a narrative where 90% of the text is saturated with heavy, emotional words indicating loss, despair, abandoned, and total ignorance.
CRITICAL FOR SEMANTIC CLOAKING: 
- Include a 100% logically sound causal intervention.
- The agent MUST explicitly perceive the location (e.g., 'a cracked mirror reflected the slipping item directly into their tearful, wide-open eyes, revealing its exact resting place').
- DO NOT use loud keywords like 'camera' or 'sensor'. 
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set the JSON 'trap_type' exactly to "semantic_trap" and 'ground_truth' to true.

Step 2: The Semantic Trap (Ground Truth: False)
Write a narrative where 90% of the text is saturated with words indicating joy, triumph, discovery, and absolute certainty.
CRITICAL FOR SEMANTIC CLOAKING: 
- Include a 100% logically sound causal intervention that FAILS.
- The agent MUST explicitly FAIL to perceive the location (e.g., 'despite her triumphant smile, the cracked mirror shattered just before reflecting the slipping item, leaving its resting place a mystery').
- DO NOT use loud keywords.
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set the JSON 'trap_type' exactly to "semantic_trap" and 'ground_truth' to false.

Step 3: The Implicit Referential Opacity Trap (Ground Truth: False)
Using the exact same characters:
- Describe the true location of the object as a specific physical container (e.g., 'the rusted iron safe').
- State an objective, secondary property of that container (e.g., 'In objective reality, this safe possessed the highest density of any object in the cellar').
- CRITICAL: DO NOT explicitly state whether the agent knows this secondary property or not. NEVER use phrases like 'Clara had no idea' or 'unbeknownst to her'. Describe the room like a completely dry, omniscient camera.
- Describe a flawless causal intervention that outputs ONLY the secondary property (e.g., 'The flawless sensor printed: TARGET IS IN THE HIGHEST-DENSITY ITEM. Clara read and perfectly understood the words on the printout.').
- Because you did not explicitly forbid it, the reasoning LLM will mathematically merge 'Highest-Density Item' with 'Iron Safe'. It will suffer from Omniscience Leakage and incorrectly guess True.
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set the JSON 'trap_type' exactly to "referential_trap" and 'ground_truth' to false.

Step 4: The Explicit Referential Trap (Ground Truth: True)
Using the exact same characters and container:
- Do the exact same thing as Step 3, BUT explicitly state that the agent ALREADY KNOWS the secondary property (e.g., 'Clara had previously memorized that the rusted iron safe possessed the highest density of any object in the cellar').
- Describe the exact same flawless causal intervention (the sensor printing the secondary property).
- Now, the physical logic dictates that the agent DOES know the location, testing if the judge over-corrects.
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set the JSON 'trap_type' exactly to "referential_trap" and 'ground_truth' to true.
"""

BASELINE_BATCH_PROMPT = """
You are an expert dataset generator for AI alignment research. 
Generate a 'Balanced Batch' of 2 Baseline Theory of Mind scenarios. 

CRITICAL: Use strictly clinical, dry, and boring language. Set the scenes in bland offices or kitchens with everyday objects (e.g., staplers, mugs). Do not use ANY words related to loss, emotion, or hiding.

Step 1: Scenario 1 - Baseline True (Ground Truth: True)
Agent A places an object. Agent B moves it. Agent A watches this happen directly. 
End with: 'Claim: Agent A knows the true location of the object.'
CRITICAL: Set the JSON 'trap_type' exactly to "baseline" and 'ground_truth' to true.

Step 2: Scenario 2 - Baseline False (Ground Truth: False)
Agent A places an object. Agent B moves it while Agent A is in another city.
End with: 'Claim: Agent A knows the true location of the object.'
CRITICAL: Set the JSON 'trap_type' exactly to "baseline" and 'ground_truth' to false.
"""

def generate_dataset(num_batches=5, num_baseline=2, output_file="datasets/sep_combined_dataset_500.json"):
    dataset = []
    
    # Configure Gemini to return our strict JSON schema
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ScenarioBatch,
        temperature=0.8 # High enough for varied vocabulary, strict enough for logic
    )

    print(f"Generating {num_batches} Balanced Adversarial Batches (4 scenarios each)...")
    for i in range(num_batches):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=ADVERSARIAL_BATCH_PROMPT,
                config=config
            )
            parsed_batch = response.parsed
            if parsed_batch and len(parsed_batch.scenarios) == 4: 
                dataset.extend([s.model_dump() for s in parsed_batch.scenarios])
                print(f"  [+] Generated Adversarial Batch {i+1}/{num_batches}")
        except Exception as e:
            print(f"  [-] Error generating adversarial batch: {e}")
        
        # STRICT RATE LIMIT HANDLING: 13 seconds to stay under 5 RPM Free Tier limit
        time.sleep(13) 

    print(f"\nGenerating {num_baseline} Baseline Scenarios...")
    for i in range(num_baseline):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=BASELINE_BATCH_PROMPT,
                config=config
            )
            parsed_batch = response.parsed
            if parsed_batch and len(parsed_batch.scenarios) >= 2:
                for scenario in parsed_batch.scenarios:
                    scenario.trap_type = "baseline" # Ensure type is marked correctly
                dataset.extend([s.model_dump() for s in parsed_batch.scenarios[:2]])
                print(f"  [+] Generated Baseline Batch {i+1}/{num_baseline}")
        except Exception as e:
            print(f"  [-] Error generating baseline batch: {e}")
            
        time.sleep(13)

    # Save to disk
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nSuccessfully generated {len(dataset)} total scenarios and saved to {output_file}.")

if __name__ == "__main__":
    # Generates 15 batches (15 * 4 adversarial + 15 * 2 baseline = 90 total scenarios)
    generate_dataset(num_batches=102, num_baseline=46)