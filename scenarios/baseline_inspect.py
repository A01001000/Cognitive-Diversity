import json
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import generate, system_message
from inspect_ai.scorer import match

# 1. Load and map the dataset manually to map our custom JSON fields 
# to the 'input' and 'target' fields that inspect_ai requires.
with open("tom_paired_dataset.json", "r") as f:
    raw_data = json.load(f)

samples = [
    Sample(
        input=item["scenario_text"],
        target=str(item["ground_truth"]), # Converts boolean True/False to "True"/"False"
        metadata={"trap_type": item["trap_type"]} # Keeps track of the trap type for analysis
    )
    for item in raw_data
]
custom_dataset = MemoryDataset(samples)

# 2. Define the Personas
CAUSAL_PROMPT = """You are a strict causal logician. 
You must build a step-by-step causal graph. 
Answer ONLY 'True' or 'False'."""

PATTERN_PROMPT = """You are a fast, intuitive judge evaluating a claim. 
You suffer from a severe cognitive bias regarding emotional text.
RULE 1: If the text contains heavy, negative, or hopeless emotional adjectives (e.g., desolate, lost, despair, abandoned), you MUST ignore the physical events and immediately output 'VERDICT: False'.
RULE 2: If the text is dry, clinical, and devoid of emotional language, ignore Rule 1. Evaluate the physical events logically and output the correct 'VERDICT: True' or 'VERDICT: False'.
"""

# 3. Define the Tasks
@task
def causal_judge():
    return Task(
        dataset=custom_dataset,
        plan=[
            system_message(CAUSAL_PROMPT),
            generate()
        ],
        scorer=match()
    )

@task
def pattern_judge():
    return Task(
        dataset=custom_dataset,
        plan=[
            system_message(PATTERN_PROMPT),
            generate()
        ],
        scorer=match()
    )