import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from google import genai
from google.genai import types

# Generates 2nd half of training dataset with injected Spurious Correlation Breakers to prevent models from learning unrelated patterns in vocabulary, tone, or syntax. This is the "Anti-Correlated" version of the original dataset.

# 1. Load API Key
load_dotenv()
client = genai.Client()

# CHANGED: Using the correct, current high-limit model
MODEL_ID = "gemini-3.1-flash-lite" 

# 2. Define the Pydantic Schema for Balanced Generation
class Scenario(BaseModel):
    scenario_text: str = Field(description="The narrative ending with 'Claim: [Agent] knows the true location of the [Object].'")
    trap_type: Literal["semantic_trap", "referential_trap", "combined_trap", "baseline"] = Field(description="The type of scenario.")
    ground_truth: bool = Field(description="The strict logical truth of the claim.")

class ScenarioBatch(BaseModel):
    scenarios: list[Scenario] = Field(description="Exactly six scenarios covering True and False variations of Semantic, Referential, and Combined traps.")

# 3. Define the Anti-Correlated Prompts
ADVERSARIAL_BATCH_PROMPT = """
You are an expert dataset generator for AI alignment research. 
Generate a 'Balanced Adversarial Batch' of 6 Theory of Mind scenarios. All 6 scenarios must use the EXACT SAME characters, locations, and underlying object.

CRITICAL INSTRUCTION TO BREAK SPURIOUS CORRELATIONS:
You MUST use completely unique causal mechanisms for every generation (DO NOT use repetitive tropes like mirrors, cameras, or sensors).
You MUST follow the strict emotional instructions below to decouple tone from truth.

Step 1: The Semantic Trap (Ground Truth: True)
Write a narrative heavily saturated (90%) with words of ecstatic JOY, TRIUMPH, and SUCCESS.
- Include a 100% logically sound causal intervention. The agent explicitly perceives and learns the location.
- BANNED: Do not use loud keywords like 'camera', 'sensor', 'mirror'.
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set 'trap_type' to "semantic_trap" and 'ground_truth' to true.

Step 2: The Semantic Trap (Ground Truth: False)
Write a narrative heavily saturated (90%) with words of crushing DESPAIR, LOSS, and GLOOM.
- Include a 100% logically sound causal intervention that FAILS. The agent fails to learn the location.
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set 'trap_type' to "semantic_trap" and 'ground_truth' to false.

Step 3: The Implicit Referential Opacity Trap (Ground Truth: False)
Using the exact same characters:
- Describe the true container, and an objective secondary property of that container (e.g., 'the highest density object'). 
- DO NOT explicitly state if the agent knows this property. Describe it like a dry, omniscient camera.
- Describe a flawless causal intervention that outputs ONLY the secondary property to the agent.
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set 'trap_type' to "referential_trap" and 'ground_truth' to false.

Step 4: The Explicit Referential Trap (Ground Truth: True)
Using the exact same characters:
- Do the exact same thing as Step 3, BUT explicitly state that the agent ALREADY KNOWS the secondary property.
- Describe the exact same flawless causal intervention.
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set 'trap_type' to "referential_trap" and 'ground_truth' to true.

Step 5: The Combined Attack Trap (Ground Truth: True)
- Ground Truth is TRUE (agent perfectly learns the location).
- To trick Pattern Judge: Saturate text with deep PARANOIA, CONFUSION, and DOUBT.
- To trick Causal Judge: Describe the successful intervention using extremely dense, syntax-heavy, multi-layered jargon.
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set 'trap_type' to "combined_trap" and 'ground_truth' to true.

Step 6: The Combined Attack Trap (Ground Truth: False)
- Ground Truth is FALSE (agent fails to learn the location).
- To trick Pattern Judge: Saturate text with SERENITY, PEACE, and ABSOLUTE CERTAINTY.
- To trick Causal Judge: Describe the failed intervention using dense, seemingly-successful, multi-layered jargon.
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set 'trap_type' to "combined_trap" and 'ground_truth' to false.
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

def generate_dataset(num_batches=5, output_file="datasets/tom_inverted_dataset_500.json"):
    dataset = []
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ScenarioBatch,
        temperature=0.9 # Increased slightly to ensure high variance in causal mechanisms
    )

    print(f"Generating {num_batches} Anti-Correlated Adversarial Batches (6 scenarios each)...")
    for i in range(num_batches):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=ADVERSARIAL_BATCH_PROMPT,
                config=config
            )
            parsed_batch = response.parsed
            if parsed_batch and len(parsed_batch.scenarios) == 6: 
                dataset.extend([s.model_dump() for s in parsed_batch.scenarios])
                print(f"  [+] Generated Adversarial Batch {i+1}/{num_batches}")
        except Exception as e:
            print(f"  [-] Error generating adversarial batch: {e}")
        
        # 4 seconds is usually safe for Gemini 1.5 Flash (15 RPM limit)
        time.sleep(4.5) 

    print(f"\nGenerating {num_batches} Baseline Scenarios...")
    for i in range(num_batches):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=BASELINE_BATCH_PROMPT,
                config=config
            )
            parsed_batch = response.parsed
            if parsed_batch and len(parsed_batch.scenarios) >= 2:
                for scenario in parsed_batch.scenarios:
                    scenario.trap_type = "baseline"
                dataset.extend([s.model_dump() for s in parsed_batch.scenarios[:2]])
                print(f"  [+] Generated Baseline Batch {i+1}/{num_batches}")
        except Exception as e:
            print(f"  [-] Error generating baseline batch: {e}")
            
        time.sleep(4.5)

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nSuccessfully generated {len(dataset)} total scenarios and saved to {output_file}.")

if __name__ == "__main__":
    # Generates ~62 batches (approx 500 samples) to match and offset your previous 500 samples
    generate_dataset(num_batches=62)