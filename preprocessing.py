import json
import re
from itertools import combinations
from sklearn.model_selection import train_test_split

def sanitize_function_name(func):
    if func in {"if", "else", "for", "while", "switch", "case", "new", "T"}:
        return False
    return re.match(r'^[A-Za-z_][A-Za-z0-9_]{2,}$', func)

def preprocess_with_combinations(input_path, train_path, val_path, val_ratio=0.1):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line.strip()) for line in f if line.strip()]

    prompt_to_outputs = {}

    for item in raw_data:
        layers = item.get("feature_layers", [])
        directives = item.get("feature_directives", [])
        files_changed = item.get("files_changed", [])

        if not files_changed or not (layers or directives):
            continue

        # Final output functions
        output_targets = []
        for file in files_changed:
            filename = file["filename"]
            for func in file["functions"]:
                if sanitize_function_name(func):
                    output_targets.append(f"{filename}::{func}")
        output_targets = sorted(set(output_targets))
        if not output_targets:
            continue

        keywords = list(set(layers + directives))
        keywords = [kw.lower() for kw in keywords]

        # Generate combinations of 1 to N keywords (subsets)
        for i in range(1, len(keywords) + 1):
            for combo in combinations(sorted(keywords), i):
                prompt = " ".join(combo)
                if prompt not in prompt_to_outputs:
                    prompt_to_outputs[prompt] = set()
                prompt_to_outputs[prompt].update(output_targets)

    # Format and split
    all_examples = [
        {"input": k, "output": ", ".join(sorted(v))}
        for k, v in prompt_to_outputs.items()
    ]

    if not all_examples:
        print("❌ No usable examples.")
        return

    train, val = train_test_split(all_examples, test_size=val_ratio, random_state=42)

    def save_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    save_jsonl(train_path, train)
    save_jsonl(val_path, val)

    print(f"✅ Total unique prompts: {len(all_examples)}")
    print(f"📂 Train: {len(train)} | Val: {len(val)}")

# Example usage
preprocess_with_combinations(
    input_path="all_openmp_prs.jsonl",
    train_path="omp_train.jsonl",
    val_path="omp_val.jsonl"
)
