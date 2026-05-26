#!/usr/bin/env python3
"""
Post-process rollouts.jsonl to remove duplicate entries and properly group by prompt.

Duplicates are identified by having the same (question, response, ground_truth) tuple.
Prompts are grouped by (question, ground_truth) regardless of order in file.

Usage:
    python dedupe_rollouts.py <input_file> [output_file]

If output_file is not specified, creates <input_file>.deduped.jsonl
"""
import json
import sys
from pathlib import Path
from collections import defaultdict


def dedupe_rollouts(input_path: str, output_path: str = None):
    """Remove duplicate rollouts and properly group by prompt."""
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_deduped{p.suffix}")

    # First pass: collect all entries and identify duplicates
    header = None
    seen_responses = set()  # (question, response, ground_truth)
    entries_by_prompt = defaultdict(list)  # (question, ground_truth) -> list of entries
    duplicates = 0

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)

            # Keep header
            if entry.get("type") == "header":
                header = entry
                continue

            # Create keys
            question = entry.get("question", "")
            response = entry.get("response", "")
            ground_truth = entry.get("ground_truth", "")

            response_key = (question, response, ground_truth)
            prompt_key = (question, ground_truth)

            # Skip duplicates (same question + response + answer)
            if response_key in seen_responses:
                duplicates += 1
                continue

            seen_responses.add(response_key)
            entries_by_prompt[prompt_key].append(entry)

    # Second pass: assign prompt_ids and rollout_nums
    unique_entries = []
    if header:
        unique_entries.append(header)

    num_generations = header.get("num_generations_per_prompt", 8) if header else 8

    sample_num = 0
    prompt_id = 0

    # Sort prompts for consistent ordering (by first appearance in original file)
    # We'll use the original sample_num to determine order
    prompt_order = []
    for prompt_key, entries in entries_by_prompt.items():
        min_sample = min(e.get("sample_num", float('inf')) for e in entries)
        prompt_order.append((min_sample, prompt_key))
    prompt_order.sort()

    for _, prompt_key in prompt_order:
        entries = entries_by_prompt[prompt_key]
        prompt_id += 1

        for rollout_num, entry in enumerate(entries, 1):
            sample_num += 1
            entry["sample_num"] = sample_num
            entry["prompt_id"] = prompt_id
            entry["rollout_num"] = rollout_num
            unique_entries.append(entry)

    # Write output
    with open(output_path, 'w') as f:
        for entry in unique_entries:
            f.write(json.dumps(entry) + "\n")

    total_unique = len(unique_entries) - (1 if header else 0)
    total_original = total_unique + duplicates

    print(f"Processed {input_path}")
    print(f"  Original entries: {total_original}")
    print(f"  Duplicates removed: {duplicates}")
    print(f"  Unique entries: {total_unique}")
    print(f"  Unique prompts: {prompt_id}")
    print(f"  Avg rollouts per prompt: {total_unique / prompt_id:.1f}" if prompt_id > 0 else "")
    print(f"  Output written to: {output_path}")

    # Show distribution of rollouts per prompt
    rollout_counts = [len(entries_by_prompt[pk]) for _, pk in prompt_order]
    if rollout_counts:
        from collections import Counter
        dist = Counter(rollout_counts)
        print(f"\n  Rollouts per prompt distribution:")
        for count in sorted(dist.keys()):
            print(f"    {count} rollouts: {dist[count]} prompts")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    dedupe_rollouts(input_file, output_file)
