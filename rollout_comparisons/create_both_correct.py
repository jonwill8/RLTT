#!/usr/bin/env python3
"""
Create a JSON file containing questions that both GRPO and RLTT got correct.
"""

import json
import os
from pathlib import Path

BASE_DIR = Path("/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rollout_comparisons")

# Dataset configurations
DATASETS = [
    ("math500", "eval_math500_prompt1024_zeroshot_tokens2048"),
    ("gsm8k", "eval_gsm8k_prompt512_zeroshot_tokens512"),
    ("aime24", "eval_aime24_prompt1024_zeroshot_tokens3072"),
    ("beyondaime", "eval_beyondaime_prompt1024_zeroshot_tokens3072"),
]

def load_grpo_results(folder_name):
    """Load GRPO results from eval_results.json"""
    path = BASE_DIR / f"grpo_{folder_name}" / "eval_results.json"
    if not path.exists():
        print(f"Warning: {path} not found")
        return {}

    with open(path, 'r') as f:
        data = json.load(f)

    # Create dict keyed by problem text
    results = {}
    for sample in data.get("samples", []):
        problem = sample.get("problem", "")
        results[problem] = {
            "problem": problem,
            "gold_answer": sample.get("gold_answer"),
            "model_response": sample.get("model_response"),
            "extracted_answer": sample.get("extracted_answer"),
            "correct": sample.get("correct", False),
            "level": sample.get("level"),
        }
    return results

def load_rltt_results(folder_name):
    """Load RLTT results from eval_details.jsonl"""
    path = BASE_DIR / f"rltt_{folder_name}" / "eval_details.jsonl"
    if not path.exists():
        print(f"Warning: {path} not found")
        return {}

    results = {}
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                problem = sample.get("problem", "")
                results[problem] = {
                    "problem": problem,
                    "gold_answer": sample.get("gold_answer"),
                    "model_response": sample.get("completion"),
                    "extracted_answer": sample.get("pred_answer"),
                    "correct": sample.get("is_correct", False),
                    "level": sample.get("level"),
                    "type": sample.get("type"),
                }
    return results

def main():
    both_correct = []

    for dataset_name, folder_suffix in DATASETS:
        print(f"\nProcessing {dataset_name}...")

        grpo_results = load_grpo_results(folder_suffix)
        rltt_results = load_rltt_results(folder_suffix)

        print(f"  GRPO: {len(grpo_results)} problems")
        print(f"  RLTT: {len(rltt_results)} problems")

        # Find problems that exist in both and both got correct
        common_problems = set(grpo_results.keys()) & set(rltt_results.keys())
        print(f"  Common problems: {len(common_problems)}")

        dataset_both_correct = 0
        for problem in common_problems:
            grpo_data = grpo_results[problem]
            rltt_data = rltt_results[problem]

            if grpo_data["correct"] and rltt_data["correct"]:
                dataset_both_correct += 1
                entry = {
                    "dataset": dataset_name,
                    "problem": problem,
                    "gold_answer": grpo_data["gold_answer"],
                    "level": grpo_data.get("level") or rltt_data.get("level"),
                    "type": rltt_data.get("type"),
                    "grpo": {
                        "model_response": grpo_data["model_response"],
                        "extracted_answer": grpo_data["extracted_answer"],
                        "correct": grpo_data["correct"],
                    },
                    "rltt": {
                        "model_response": rltt_data["model_response"],
                        "extracted_answer": rltt_data["extracted_answer"],
                        "correct": rltt_data["correct"],
                    },
                }
                both_correct.append(entry)

        print(f"  Both correct: {dataset_both_correct}")

    # Save results
    output_path = BASE_DIR / "both_correct_rollouts.json"
    with open(output_path, 'w') as f:
        json.dump({
            "description": "Questions where both GRPO and RLTT models answered correctly",
            "total_count": len(both_correct),
            "by_dataset": {
                dataset: len([e for e in both_correct if e["dataset"] == dataset])
                for dataset, _ in DATASETS
            },
            "rollouts": both_correct
        }, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Total questions both got correct: {len(both_correct)}")
    print(f"Saved to: {output_path}")

    return both_correct


def create_rltt_right_grpo_wrong():
    """Create JSON with questions RLTT got right but GRPO got wrong."""
    rltt_right_grpo_wrong = []

    for dataset_name, folder_suffix in DATASETS:
        print(f"\nProcessing {dataset_name}...")

        grpo_results = load_grpo_results(folder_suffix)
        rltt_results = load_rltt_results(folder_suffix)

        print(f"  GRPO: {len(grpo_results)} problems")
        print(f"  RLTT: {len(rltt_results)} problems")

        # Find problems that exist in both
        common_problems = set(grpo_results.keys()) & set(rltt_results.keys())
        print(f"  Common problems: {len(common_problems)}")

        dataset_rltt_right = 0
        for problem in common_problems:
            grpo_data = grpo_results[problem]
            rltt_data = rltt_results[problem]

            # RLTT correct, GRPO wrong
            if rltt_data["correct"] and not grpo_data["correct"]:
                dataset_rltt_right += 1
                entry = {
                    "dataset": dataset_name,
                    "problem": problem,
                    "gold_answer": grpo_data["gold_answer"],
                    "level": grpo_data.get("level") or rltt_data.get("level"),
                    "type": rltt_data.get("type"),
                    "grpo": {
                        "model_response": grpo_data["model_response"],
                        "extracted_answer": grpo_data["extracted_answer"],
                        "correct": grpo_data["correct"],
                    },
                    "rltt": {
                        "model_response": rltt_data["model_response"],
                        "extracted_answer": rltt_data["extracted_answer"],
                        "correct": rltt_data["correct"],
                    },
                }
                rltt_right_grpo_wrong.append(entry)

        print(f"  RLTT right, GRPO wrong: {dataset_rltt_right}")

    # Save results
    output_path = BASE_DIR / "rltt_right_grpo_wrong.json"
    with open(output_path, 'w') as f:
        json.dump({
            "description": "Questions where RLTT answered correctly but GRPO answered incorrectly",
            "total_count": len(rltt_right_grpo_wrong),
            "by_dataset": {
                dataset: len([e for e in rltt_right_grpo_wrong if e["dataset"] == dataset])
                for dataset, _ in DATASETS
            },
            "rollouts": rltt_right_grpo_wrong
        }, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Total RLTT right, GRPO wrong: {len(rltt_right_grpo_wrong)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
    print("\n" + "="*50)
    print("Creating RLTT right, GRPO wrong JSON...")
    print("="*50)
    create_rltt_right_grpo_wrong()
