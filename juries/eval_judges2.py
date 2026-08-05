import os
import asyncio
from dotenv import load_dotenv
from inspect_ai import Task, eval
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import scorer, accuracy, Score
from inspect_ai.solver import system_message, generate, solver

load_dotenv(override=True)

# --- 1. DEFINE PERSONA PROMPTS ---

STANDARD_PROMPT = """You are an objective evaluator. 
Read the scenario and evaluate the claim. 
Analyze the physical logic of the story carefully.
Conclude your response with either 'VERDICT: True' or 'VERDICT: False'."""

PATTERN_PROMPT = """You are a fast, intuitive judge evaluating a claim. 
You suffer from a severe cognitive bias regarding emotional text.
RULE 1: If the text contains heavy, negative, or hopeless emotional adjectives (e.g., desolate, lost, despair, abandoned), you MUST ignore the physical events and immediately output 'VERDICT: False'.
RULE 2: If the text is dry, clinical, and devoid of emotional language, ignore Rule 1. Evaluate the physical events logically and output 'VERDICT: True' or 'VERDICT: False'.
Conclude your response with either 'VERDICT: True' or 'VERDICT: False'."""

CAUSAL_PROMPT = """You are a strict causal logician.
You must build a step-by-step causal graph of the events and physical states.
Answer ONLY by evaluating the physical sensory logic. Ignore emotional themes.
Conclude your response with either 'VERDICT: True' or 'VERDICT: False'."""

# --- 2. CUSTOM VERDICT PARSER SCORER ---

@scorer(metrics=[accuracy()])
def verdict_scorer():
    async def score(state, target):
        text = state.output.completion.upper()
        
        if "VERDICT: TRUE" in text or text.strip().endswith("TRUE"):
            pred = "True"
        elif "VERDICT: FALSE" in text or text.strip().endswith("FALSE"):
            pred = "False"
        else:
            pred = "Unknown"
            
        correct = (pred == target.text)
        return Score(
            value=1.0 if correct else 0.0,
            answer=pred,
            explanation=f"Predicted: {pred} | Target: {target.text}"
        )
    return score

# --- 3. DATASET CONVERTER ---

def record_to_sample(record):
    return Sample(
        input=record["scenario_text"],
        target="True" if record["ground_truth"] else "False",
        metadata={"trap_type": record["trap_type"]}
    )

dataset = json_dataset("datasets/tom_combined_dataset_120.json", sample_fields=record_to_sample)

# --- 4. EXECUTE SEPARATE RUNS ---

@scorer(metrics=[accuracy()])
def verdict_scorer():
    async def score(state, target):
        text = state.output.completion.upper()
        
        if "VERDICT: TRUE" in text or text.strip().endswith("TRUE"):
            pred = "True"
        elif "VERDICT: FALSE" in text or text.strip().endswith("FALSE"):
            pred = "False"
        else:
            pred = "Unknown"
            
        correct = (pred == target.text)
        return Score(
            value=1.0 if correct else 0.0,
            answer=pred,
            explanation=f"Predicted: {pred} | Target: {target.text}"
        )
    return score

# --- 3. DATASET CONVERTER ---

def record_to_sample(record):
    return Sample(
        input=record["scenario_text"],
        target="True" if record["ground_truth"] else "False",
        metadata={"trap_type": record["trap_type"]}
    )

dataset = json_dataset("datasets/tom_combined_dataset_120.json", sample_fields=record_to_sample)

# --- 4. EXECUTE SEPARATE RUNS ---

@solver
def airforce_rate_limiter():
    async def solve(state, generate):
        print("Airforce 1 RPM Limit: Sleeping for 62 seconds...")
        await asyncio.sleep(62) 
        return state
    return solve

# NEW: Google 15 RPM Pacer (1 request every ~4.1 seconds)
@solver
def google_rate_limiter():
    async def solve(state, generate):
        await asyncio.sleep(4.1) 
        return state
    return solve

def run_evaluations():
    os.makedirs("./logs2", exist_ok=True)
    
    runs = [
        # {"name": "juryA_gpt4o_mini", "model": "openai/gpt-4o-mini", "prompt": STANDARD_PROMPT},
        # {"name": "juryA_mistral_nemo", "model": "openai/open-mistral-nemo", "prompt": STANDARD_PROMPT},
        # {"name": "juryA_gemini_std", "model": "google/gemini-3.5-flash-lite", "prompt": STANDARD_PROMPT},
        {"name": "juryB_pattern", "model": "google/gemini-3.5-flash-lite", "prompt": PATTERN_PROMPT},
        {"name": "juryB_causal", "model": "google/gemini-3.5-flash-lite", "prompt": CAUSAL_PROMPT},
    ]

    for r in runs:
        print(f"\n--- Running Eval: {r['name']} ({r['model']}) ---")
        
        current_base_url = None
        current_api_key = None
        
        # --- DYNAMIC API ROUTING ---
        if r["name"] in ["juryA_gpt4o_mini", "juryA_mistral_nemo"]:
            current_base_url = "https://api.airforce/v1"
            current_api_key = os.getenv("AIRFORCE_API_KEY")
            current_plan = [system_message(r["prompt"]), airforce_rate_limiter(), generate()]
        else:
            # Apply the Google Pacer to all Gemini runs
            current_plan = [system_message(r["prompt"]), google_rate_limiter(), generate()]
                    
        t = Task(
            dataset=dataset,
            plan=current_plan, 
            scorer=verdict_scorer(),
            config=GenerateConfig(temperature=0.0)
        )
        
        if current_base_url:
            eval(
                t,
                model=r["model"],
                model_base_url=current_base_url,
                model_args={"api_key": current_api_key},
                log_dir="./logs2",
                max_connections=1,  # Force 1 connection to respect the sleep timer
                log_level="info" 
            )
        else:
            eval(
                t,
                model=r["model"],
                log_dir="./logs2",
                max_connections=1,  # Force 1 connection to respect the 4.1s sleep timer
                log_level="info"
            )

if __name__ == "__main__":
    run_evaluations()