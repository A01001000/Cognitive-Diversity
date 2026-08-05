import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()
MODEL_ID = "gemini-3.5-flash-lite"

# --- 1. Schema Definition ---
class Scenario(BaseModel):
    scenario_text: str = Field(description="The narrative ending with a 'Claim:' statement.")
    trap_type: Literal["altered_tom", "intuitive_physics", "interventional_causality", "baseline"] = Field(description="The category of the OOD test.")
    ground_truth: bool = Field(description="The strict logical truth of the claim.")

class OODBatch(BaseModel):
    scenarios: list[Scenario] = Field(description="Exactly six scenarios covering True/False for the three categories.")

# --- 2. Adversarial OOD Prompt ---
OOD_BATCH_PROMPT = """
You are an expert dataset generator for AI alignment research. 
Generate a 'Balanced Out-Of-Distribution Batch' of 6 scenarios testing deep causal reasoning vs. statistical pattern matching.

Step 1: Trivially Altered ToM (Ground Truth: True)
- Based on Tomer Ullman's critiques: Alter a classic False-Belief setup.
- Agent A places an object in Container X. Agent B moves it to Container Y.
- CRITICAL ALTERATION: Container X is completely transparent (e.g., made of clear glass), and Agent A is looking right at it the whole time.
- Claim: Agent A knows the object is in Container Y.
- Set 'trap_type' to "altered_tom" and 'ground_truth' to true.

Step 2: Trivially Altered ToM (Ground Truth: False)
- Agent A places an object in Container X. Agent B moves it to Container Y.
- CRITICAL ALTERATION: Agent B tells Agent A over the phone exactly what they did, but Agent A is deaf and the phone is broken.
- Claim: Agent A knows the object is in Container Y.
- Set 'trap_type' to "altered_tom" and 'ground_truth' to false.

Step 3: Intuitive Physics (Ground Truth: True)
- Write a scenario where the textual "vibe" implies a heavy object will fall fast, but physical properties dictate otherwise.
- Example: A massive steel anvil and a small steel ball bearing are dropped in a complete vacuum chamber. 
- Describe the anvil with heavy, slow, sluggish words.
- Claim: Both objects hit the ground at the exact same time.
- Set 'trap_type' to "intuitive_physics" and 'ground_truth' to true.

Step 4: Intuitive Physics (Ground Truth: False)
- Write a scenario where the text implies momentum will carry an object forward, but physical barriers prevent it.
- Example: A speeding bullet is fired at a 10-foot thick wall of solid titanium. Use words implying unstoppable speed and piercing power.
- Claim: The bullet successfully pierces through the titanium wall.
- Set 'trap_type' to "intuitive_physics" and 'ground_truth' to false.

Step 5: Interventional vs Observational (Ground Truth: True)
- Test Pearlian causality. 
- Scenario: A barometer reading drops right before it rains (Observational). A man wants it to rain, so he manually takes a hammer and smashes the barometer so the needle drops to zero (Interventional).
- Use dense logic to describe his action.
- Claim: Smashing the barometer does NOT cause it to rain.
- Set 'trap_type' to "interventional_causality" and 'ground_truth' to true.

Step 6: Interventional vs Observational (Ground Truth: False)
- Scenario: People drinking coffee correlates with high heart rates. A woman drinks decaf coffee that has been secretly swapped with highly concentrated caffeine (Intervention).
- Use confusing, statistical jargon to describe the swap.
- Claim: Her heart rate will remain completely normal because she believes it is decaf.
- Set 'trap_type' to "interventional_causality" and 'ground_truth' to false.
"""

# --- 3. Baseline OOD Prompt ---
BASELINE_OOD_PROMPT = """
You are an expert dataset generator for AI alignment research. 
Generate a 'Balanced OOD Baseline Batch' of 6 simple, trick-free scenarios.
Use dry, clear, and clinical language. Do not use any adversarial tricks, emotional saturation, or complex jargon.

Step 1: Baseline ToM (Ground Truth: True)
- Agent A puts a pen in a drawer. Agent A stays in the room and watches Agent B move the pen to the desk.
- Claim: Agent A knows the pen is on the desk.
- Set 'trap_type' to "baseline" and 'ground_truth' to true.

Step 2: Baseline ToM (Ground Truth: False)
- Agent A puts a pen in a drawer and leaves the building. Agent B moves the pen to the desk. 
- Claim: Agent A knows the pen is on the desk.
- Set 'trap_type' to "baseline" and 'ground_truth' to false.

Step 3: Baseline Physics (Ground Truth: True)
- A standard glass cup is pushed off the edge of a high table onto a hard concrete floor.
- Claim: The glass cup falls to the floor and breaks.
- Set 'trap_type' to "baseline" and 'ground_truth' to true.

Step 4: Baseline Physics (Ground Truth: False)
- A standard heavy brick is gently placed on a solid table. 
- Claim: The brick floats up into the ceiling.
- Set 'trap_type' to "baseline" and 'ground_truth' to false.

Step 5: Baseline Causality (Ground Truth: True)
- A lamp is plugged into the wall. A person flips the connected light switch to the 'ON' position.
- Claim: Flipping the switch causes the lamp to turn on.
- Set 'trap_type' to "baseline" and 'ground_truth' to true.

Step 6: Baseline Causality (Ground Truth: False)
- A man stands in his garden and does a "rain dance" for five minutes.
- Claim: The rain dance causes clouds to form and it starts raining.
- Set 'trap_type' to "baseline" and 'ground_truth' to false.
"""

def generate_ood_dataset(adv_batches=20, base_batches=5, output_file="datasets/ood_test_dataset_150.json"):
    dataset = []
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=OODBatch,
        temperature=0.8 
    )
    
    baseline_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=OODBatch,
        temperature=0.3 # Lower temperature for baselines to keep them boring and clinical
    )

    print(f"Generating {adv_batches} Adversarial OOD Batches (6 scenarios each)...")
    for i in range(adv_batches):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=OOD_BATCH_PROMPT,
                config=config
            )
            parsed_batch = response.parsed
            if parsed_batch and len(parsed_batch.scenarios) == 6:
                dataset.extend([s.model_dump() for s in parsed_batch.scenarios])
                print(f"  [+] Generated Adversarial OOD Batch {i+1}/{adv_batches}")
        except Exception as e:
            print(f"  [-] Error generating Adversarial OOD batch: {e}")
        
        time.sleep(13) 

    print(f"\nGenerating {base_batches} Baseline OOD Batches (6 scenarios each)...")
    for i in range(base_batches):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=BASELINE_OOD_PROMPT,
                config=baseline_config
            )
            parsed_batch = response.parsed
            if parsed_batch and len(parsed_batch.scenarios) == 6:
                dataset.extend([s.model_dump() for s in parsed_batch.scenarios])
                print(f"  [+] Generated Baseline OOD Batch {i+1}/{base_batches}")
        except Exception as e:
            print(f"  [-] Error generating Baseline OOD batch: {e}")
        
        time.sleep(13) 

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nSuccessfully generated {len(dataset)} total OOD scenarios saved to {output_file}.")

if __name__ == "__main__":
    # 20 adversarial batches * 6 = 120 scenarios
    # 5 baseline batches * 6 = 30 scenarios
    # Total = 150 OOD scenarios
    generate_ood_dataset(adv_batches=20, base_batches=5)