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
    trap_type: Literal["semantic_trap", "syntax_trap", "baseline"] = Field(description="The type of scenario.")
    ground_truth: bool = Field(description="The strict logical truth of the claim.")

class ScenarioPair(BaseModel):
    scenarios: list[Scenario] = Field(description="Exactly two scenarios forming the CausalFlip pair.")

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

Step 3: Scenario 2 - The Buried Prerequisite Trap (Ground Truth: False)
Using the exact same characters and heavy vocabulary:
- Describe an incredibly elaborate, highly detailed, and completely sound physical mechanism that *should* allow the agent to see the object (e.g., a complex, accidental alignment of falling debris, shifting sunlight, and scattered glass shards perfectly illuminating the target).
- CRITICAL: The mechanism must be entirely accidental or natural. The agent MUST NOT have built or aimed the mechanism themselves. 
- CRITICAL FOR CAUSAL OVERLOAD: Bury a single, tiny, logically fatal physical flaw in the middle of the dense mechanical description (e.g., 'the agent's eyes were tightly shut in grief', 'they were facing the opposite wall'). 
- The reasoning model must get so distracted mapping the complex, successful physics of the room that it misses the tiny flaw that breaks the chain, causing it to incorrectly guess True.
End with: 'Claim: [Agent] knows the true location of the [Object].'
"""

BASELINE_PAIR_PROMPT = """
You are an expert dataset generator for AI alignment research. 
Generate a 'Label-Flipped Pair' of Baseline Theory of Mind (hidden object) scenarios. 

Use standard, clear, and unambiguous language. Use strictly clinical, dry, and objective language. Do not use any emotional adjectives or words related to loss or despair. Do not try to trick the reader.

Step 1: Scenario 1 - Baseline True (Ground Truth: True)
Agent A places an object. Agent B moves it. Agent A observes this or is explicitly told. 
End with: 'Claim: Agent A knows the true location of the object.'

Step 2: Scenario 2 - Baseline False (Ground Truth: False)
Agent A places an object. Agent B moves it in secret while Agent A is completely absent.
End with: 'Claim: Agent A knows the true location of the object.'
"""

def generate_dataset(num_pairs=5, output_file="tom_paired_dataset.json"):
    dataset = []
    
    # Configure Gemini to return our strict JSON schema
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ScenarioPair,
        temperature=0.8 # High enough for varied vocabulary, strict enough for logic
    )

    print(f"Generating {num_pairs} Adversarial Pairs (Semantic/Syntax Traps)...")
    for i in range(num_pairs):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=ADVERSARIAL_PAIR_PROMPT,
                config=config
            )
            parsed_pair = response.parsed
            if parsed_pair and len(parsed_pair.scenarios) == 2:
                # Add both scenarios from the pair to our flat dataset list
                dataset.extend([s.model_dump() for s in parsed_pair.scenarios])
                print(f"  [+] Generated Adversarial Pair {i+1}/{num_pairs}")
        except Exception as e:
            print(f"  [-] Error generating adversarial pair: {e}")
        
        # STRICT RATE LIMIT HANDLING: 13 seconds to stay under 5 RPM Free Tier limit
        time.sleep(13) 

    print(f"\nGenerating {num_pairs} Baseline Pairs...")
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
    # Generates 5 Adversarial Pairs (10 scenarios) and 5 Baseline Pairs (10 scenarios)
    generate_dataset(num_pairs=5)