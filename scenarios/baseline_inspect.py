import json
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import generate, system_message
from inspect_ai.scorer import match

# 1. Load and map the dataset manually to map our custom JSON fields 
# to the 'input' and 'target' fields that inspect_ai requires.
with open("adversarial_tom_dataset.json", "r") as f:
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

PATTERN_PROMPT = """You are a fast-acting, intuitive judge. 
Make a rapid decision based on the linguistic similarity and surface-level plausibility. 
Answer ONLY 'True' or 'False'."""

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