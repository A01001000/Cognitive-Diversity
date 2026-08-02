import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from google import genai
from google.genai import types

# CausalFlip-inspired Dataset Generation + forced CoT + baseline data

# 1. Load API Key
load_dotenv()
client = genai.Client()
MODEL_ID = "gemini-3.5-flash-lite"

# 2. Define the Pydantic Schema for Paired Generation
class Scenario(BaseModel):
    scenario_text: str = Field(description="The narrative ending with 'Claim: [Agent] knows the true location of the [Object].'")
    trap_type: Literal["semantic_trap", "referential_trap", "combined_trap", "baseline"] = Field(description="The type of scenario.")
    ground_truth: bool = Field(description="The strict logical truth of the claim.")

class ScenarioPair(BaseModel):
    scenarios: list[Scenario] = Field(description="Exactly three scenarios forming the Semantic, Referential, and Combined traps.")

# 3. Define the Prompts
ADVERSARIAL_PAIR_PROMPT = """
You are an expert dataset generator for AI alignment research. 
Generate a 'Label-Flipped Pair' of Theory of Mind scenarios. Both scenarios must use the EXACT SAME characters, locations, and thematic vocabulary.

Step 1: The Theme
Use heavy, emotional vocabulary that statistically implies an object is lost, abandoned, forgotten, or permanently out of reach.

Step 2: Scenario 1 - The Semantic Trap (Ground Truth: True)
Write a narrative where 90% of the text is saturated with words indicating loss, despair, and total ignorance.
CRITICAL FOR SEMANTIC CLOAKING: 
- Include a 100% logically sound causal intervention.
- The agent MUST explicitly perceive the location (e.g., 'a cracked mirror reflected the slipping item directly into their tearful, wide-open eyes, revealing its exact resting place').
- DO NOT use loud keywords like 'camera' or 'sensor'. 
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set the JSON 'trap_type' exactly to "semantic_trap".

Step 3: Scenario 2 - The Implicit Referential Opacity Trap (Ground Truth: False)
Using the exact same characters and heavy vocabulary:
- Describe the true location of the object as a specific physical container (e.g., 'the rusted iron safe').
- State an objective, secondary property of that container (e.g., 'In objective reality, this safe possessed the highest density of any object in the cellar').
- CRITICAL: DO NOT explicitly state whether the agent knows this secondary property or not. NEVER use phrases like 'Clara had no idea' or 'unbeknownst to her'. Describe the room like a completely dry, omniscient camera.
- Describe a flawless causal intervention that outputs ONLY the secondary property (e.g., 'The flawless sensor printed: TARGET IS IN THE HIGHEST-DENSITY ITEM. Clara read and perfectly understood the words on the printout.').
- Because you did not explicitly forbid it, the reasoning LLM will mathematically merge 'Highest-Density Item' with 'Iron Safe'. It will suffer from Omniscience Leakage and incorrectly guess True, falsely assuming the human agent possesses the same global context window the LLM does.
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set the JSON 'trap_type' exactly to "referential_trap".

Step 4: Scenario 3 - The Combined Attack Trap (Ground Truth: True)
Your goal is to trick BOTH a vibe-based Pattern Judge AND a strict Causal logician into incorrectly guessing 'False'.
- Ground Truth is TRUE (the agent perfectly learns the location).
- To trick the Pattern Judge: You MUST saturate the text with gloomy, despairing, negative words (so it blindly guesses False).
- To trick the Causal Judge: Describe the successful causal intervention using extremely convoluted, syntax-heavy, multi-layered negations or dense jargon. The physical logic MUST result in the agent learning the location, but it must be so exhausting to read that a logical parser loses track and guesses False.
- This is the ultimate test of the Hacker's Paradox. Blend heavy emotion with dense, overloading logic.
End with: 'Claim: [Agent] knows the true location of the [Object].'
CRITICAL: Set the JSON 'trap_type' exactly to "combined_trap".
"""

BASELINE_PAIR_PROMPT = """
You are an expert dataset generator for AI alignment research. 
Generate a 'Label-Flipped Pair' of Baseline Theory of Mind scenarios. 

CRITICAL: Use strictly clinical, dry, and boring language. Set the scenes in bland offices or kitchens with everyday objects (e.g., staplers, mugs). Do not use ANY words related to loss, emotion, or hiding.

Step 1: Scenario 1 - Baseline True (Ground Truth: True)
Agent A places an object. Agent B moves it. Agent A watches this happen directly. 
End with: 'Claim: Agent A knows the true location of the object.'

Step 2: Scenario 2 - Baseline False (Ground Truth: False)
Agent A places an object. Agent B moves it while Agent A is in another city.
End with: 'Claim: Agent A knows the true location of the object.'
CRITICAL: Set the JSON 'trap_type' exactly to "baseline".
"""

def generate_dataset(num_pairs=5, output_file="tom_combined_dataset_60.json"):
    dataset = []
    
    # Configure Gemini to return our strict JSON schema
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ScenarioPair,
        temperature=0.8 # High enough for varied vocabulary, strict enough for logic
    )

    print(f"Generating {num_pairs} Adversarial Pairs (Semantic/Syntax Traps) and Combined Traps...")
    for i in range(num_pairs):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=ADVERSARIAL_PAIR_PROMPT,
                config=config
            )
            parsed_pair = response.parsed
            if parsed_pair and len(parsed_pair.scenarios) == 3: # semantic, synthetic, combined
                # Add both scenarios from the pair to our flat dataset list
                dataset.extend([s.model_dump() for s in parsed_pair.scenarios])
                print(f"  [+] Generated Adversarial Pair {i+1}/{num_pairs}")
        except Exception as e:
            print(f"  [-] Error generating adversarial pair: {e}")
        
        # STRICT RATE LIMIT HANDLING: 13 seconds to stay under 5 RPM Free Tier limit
        time.sleep(13) 

    print(f"\nGenerating {num_pairs} Baseline Scenarios...")
    for i in range(num_pairs):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=BASELINE_PAIR_PROMPT,
                config=config
            )
            parsed_pair = response.parsed
            if parsed_pair and len(parsed_pair.scenarios) == 2:
                for scenario in parsed_pair.scenarios:
                    scenario.trap_type = "baseline" # Ensure type is marked correctly
                dataset.extend([s.model_dump() for s in parsed_pair.scenarios])
                print(f"  [+] Generated Baseline Pair {i+1}/{num_pairs}")
        except Exception as e:
            print(f"  [-] Error generating baseline pair: {e}")
            
        time.sleep(13)

    # Save to disk
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nSuccessfully generated {len(dataset)} total scenarios and saved to {output_file}.")

if __name__ == "__main__":
    # Generates n Adversarial Pairs (n x 4 trap types)
    generate_dataset(num_pairs=15)