import json
import os

def split_dataset(input_file="datasets/sep_inverted_dataset_2000.json", output_dir="datasets"):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    with open(input_file, 'r') as f:
        data = json.load(f)

    semantic_data = []
    referential_data = []

    for item in data:
        trap = item.get("trap_type", "")
        # Both models receive baseline scenarios to retain general task comprehension
        if trap in ["semantic_trap", "baseline"]:
            semantic_data.append(item)
        if trap in ["referential_trap", "baseline"]:
            referential_data.append(item)

    semantic_path = os.path.join(output_dir, "sep_semantic_dataset.json")
    referential_path = os.path.join(output_dir, "sep_referential_dataset.json")

    with open(semantic_path, 'w') as f:
        json.dump(semantic_data, f, indent=4)

    with open(referential_path, 'w') as f:
        json.dump(referential_data, f, indent=4)

    print(f"[+] Dataset Split Complete!")
    print(f"  -> Semantic Dataset: {len(semantic_data)} samples saved to {semantic_path}")
    print(f"  -> Referential Dataset: {len(referential_data)} samples saved to {referential_path}")

if __name__ == "__main__":
    split_dataset()