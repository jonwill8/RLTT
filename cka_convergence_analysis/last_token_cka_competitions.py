#!/usr/bin/env python3
"""
Last-token-across-prompts CKA analysis on competition benchmarks.

Implements:
1) Dataset-level last-token CKA heatmap across UT loops.
2) Convergence-to-final CKA curve.

The script is designed for AIME26 (30 prompts) and can consume either:
- a merged HF model directory, or
- an FSDP checkpoint root (global_step_X) that contains actor/huggingface.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Offline mode for cluster runs.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Import shared math utils (reuse existing parsing + prompting flow).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from math_utils import (  # noqa: E402
    format_math_prompt,
    rl_get_gold_answer as get_gold_answer,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


BENCHMARK_CONFIG = {
    "aime26": {
        "name": "AIME 2026",
        "file": "/scratch/gpfs/OLGARUS/jw4199/datasets/aime26/aime2026.jsonl",
        "expected_samples": 30,
    },
}


@dataclass
class ModelResolution:
    model_dir: str
    merged_from_fsdp: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Last-token-across-prompts CKA analysis for RLTT/GRPO checkpoints."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to merged model or global_step_X checkpoint directory.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["rltt", "grpo"],
        required=True,
        help="Checkpoint source method (used for metadata/defaults).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="aime26",
        choices=list(BENCHMARK_CONFIG.keys()),
        help="Benchmark dataset to run CKA on.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Optional override for benchmark JSONL path.",
    )
    parser.add_argument(
        "--total_ut_steps",
        type=int,
        default=4,
        help="Number of UT loops to run (paper idea assumes 4).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=30,
        help="Max number of prompts from benchmark (AIME26 is 30).",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Prompt truncation length.",
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Use 5-shot prompting instead of zero-shot.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for hidden-state extraction.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults under cka_convergence_analysis/runs.",
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking-RLTT",
        help="Base model dir used to copy modeling_ouro.py when missing.",
    )
    parser.add_argument(
        "--cleanup_merged_model",
        action="store_true",
        help="Delete temporary merged model if merged from FSDP checkpoint.",
    )
    return parser.parse_args()


def load_jsonl(file_path: str, max_samples: Optional[int]) -> List[Dict]:
    data: List[Dict] = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if max_samples is not None:
        data = data[:max_samples]
    return data


def _fix_checkpoint_hf_files(ckpt_hf_dir: str, base_model_dir: str) -> None:
    cfg_path = os.path.join(ckpt_hf_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            cfg_txt = f.read()
        cfg_txt = cfg_txt.replace(
            '"AutoModelForCausalLM": "peft_model.PeftModelForCausalLM"',
            '"AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM"',
        )
        with open(cfg_path, "w") as f:
            f.write(cfg_txt)

    modeling_path = os.path.join(ckpt_hf_dir, "modeling_ouro.py")
    if not os.path.isfile(modeling_path):
        shutil.copy(os.path.join(base_model_dir, "modeling_ouro.py"), modeling_path)

    utils_path = os.path.join(ckpt_hf_dir, "utils.py")
    if os.path.isfile(utils_path):
        with open(utils_path, "r") as f:
            utils_txt = f.read()
        replacements = {
            "from .integrations": "from peft.utils.integrations",
            "from .loftq_utils": "from peft.utils.loftq_utils",
            "from .other": "from peft.utils.other",
            "from .peft_types": "from peft.utils.peft_types",
            "from .save_and_load": "from peft.utils.save_and_load",
            "from .warning": "from peft.utils.warning",
        }
        for old, new in replacements.items():
            utils_txt = utils_txt.replace(old, new)
        with open(utils_path, "w") as f:
            f.write(utils_txt)

    peft_model_path = os.path.join(ckpt_hf_dir, "peft_model.py")
    if os.path.isfile(peft_model_path):
        with open(peft_model_path, "r") as f:
            peft_txt = f.read()
        peft_txt = peft_txt.replace(
            "from . import __version__",
            "from peft import __version__",
        )
        with open(peft_model_path, "w") as f:
            f.write(peft_txt)

    transformers_cache_modules = (
        "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/transformers_modules/huggingface"
    )
    shutil.rmtree(transformers_cache_modules, ignore_errors=True)


def resolve_model_dir(
    checkpoint_path: str,
    output_dir: str,
    base_model_dir: str,
) -> ModelResolution:
    checkpoint_path = os.path.abspath(checkpoint_path)

    # Direct merged model path.
    if os.path.isfile(os.path.join(checkpoint_path, "config.json")):
        return ModelResolution(model_dir=checkpoint_path, merged_from_fsdp=False)

    # FSDP global step path.
    fsdp_actor_dir = os.path.join(checkpoint_path, "actor")
    if not os.path.isdir(fsdp_actor_dir):
        raise ValueError(
            f"checkpoint_path must be a merged model dir or global_step_X dir. Got: {checkpoint_path}"
        )

    ckpt_hf_dir = os.path.join(fsdp_actor_dir, "huggingface")
    if not os.path.isdir(ckpt_hf_dir):
        raise ValueError(f"Missing huggingface directory in actor checkpoint: {ckpt_hf_dir}")

    _fix_checkpoint_hf_files(ckpt_hf_dir, base_model_dir)

    merged_model_dir = os.path.join(output_dir, "merged_model")
    if os.path.isdir(merged_model_dir):
        shutil.rmtree(merged_model_dir)
    os.makedirs(merged_model_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--trust-remote-code",
        "--local_dir",
        fsdp_actor_dir,
        "--target_dir",
        merged_model_dir,
    ]
    logger.info("Merging FSDP checkpoint to merged_model...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to merge FSDP checkpoint.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return ModelResolution(model_dir=merged_model_dir, merged_from_fsdp=True)


def linear_cka(z_a: np.ndarray, z_b: np.ndarray, eps: float = 1e-12) -> float:
    z_a_center = z_a - z_a.mean(axis=0, keepdims=True)
    z_b_center = z_b - z_b.mean(axis=0, keepdims=True)

    cross = z_a_center.T @ z_b_center
    auto_a = z_a_center.T @ z_a_center
    auto_b = z_b_center.T @ z_b_center

    num = np.linalg.norm(cross, ord="fro") ** 2
    den = np.linalg.norm(auto_a, ord="fro") * np.linalg.norm(auto_b, ord="fro")
    if den < eps:
        return 0.0
    return float(num / den)


def compute_cka_heatmap(loop_mats: List[np.ndarray]) -> np.ndarray:
    n = len(loop_mats)
    heat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            heat[i, j] = linear_cka(loop_mats[i], loop_mats[j])
    return heat


def save_heatmap(heatmap: np.ndarray, out_path: str) -> None:
    loops = np.arange(1, heatmap.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(heatmap, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(loops)), labels=[str(x) for x in loops])
    ax.set_yticks(np.arange(len(loops)), labels=[str(x) for x in loops])
    ax.set_xlabel("Loop b")
    ax.set_ylabel("Loop a")
    ax.set_title("Last-token Across Prompts: CKA Heatmap")
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            ax.text(j, i, f"{heatmap[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="CKA")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_convergence_curve(curve: np.ndarray, out_path: str) -> None:
    loops = np.arange(1, len(curve) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(loops, curve, marker="o", linewidth=2)
    ax.set_xticks(loops)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Loop index i")
    ax.set_ylabel("CKA(Z(i), Z(final))")
    ax.set_title("Convergence-to-final CKA curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def extract_last_token_loop_embeddings(
    model,
    tokenizer,
    dataset: List[Dict],
    total_ut_steps: int,
    max_prompt_length: int,
    use_few_shot: bool,
    batch_size: int,
) -> Tuple[List[np.ndarray], List[Dict]]:
    def _get_per_loop_hidden_states(output):
        if output is None:
            return None
        if isinstance(output, dict):
            return output.get("per_loop_hidden_states")
        return getattr(output, "per_loop_hidden_states", None)

    prompts = []
    prompt_meta = []
    for idx, item in enumerate(dataset):
        problem = item.get("problem", "")
        prompt = format_math_prompt(problem, tokenizer, use_few_shot=use_few_shot)
        prompts.append(prompt)
        prompt_meta.append(
            {
                "idx": idx,
                "problem": problem,
                "gold_answer": str(get_gold_answer(item)),
            }
        )

    # One collector per loop; each entry is a list over prompts.
    loop_vectors: List[List[np.ndarray]] = [[] for _ in range(total_ut_steps)]

    model.eval()
    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting hidden states"):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
            padding=True,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        last_indices = (attention_mask.sum(dim=1) - 1).tolist()

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                return_per_loop_hidden_states=True,
            )

        per_loop = _get_per_loop_hidden_states(out)
        if per_loop is None and hasattr(model, "get_base_model"):
            # Some PEFT wrappers return a plain CausalLMOutputWithPast while the
            # underlying Ouro model exposes per_loop_hidden_states.
            base_model = model.get_base_model()
            if base_model is not model:
                with torch.no_grad():
                    out = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                        return_per_loop_hidden_states=True,
                    )
                per_loop = _get_per_loop_hidden_states(out)

        if per_loop is None:
            raise RuntimeError(
                "Model did not return per_loop_hidden_states. "
                f"Output type: {type(out).__name__}. "
                "Ensure checkpoint uses RLTT-modified modeling_ouro.py and that "
                "the loaded model is not stripping return_per_loop_hidden_states."
            )
        if len(per_loop) != total_ut_steps:
            raise RuntimeError(
                f"Expected {total_ut_steps} loops but got {len(per_loop)} from model output."
            )

        # per_loop[loop_i]: [batch, seq, hidden]
        for b_idx, last_tok_idx in enumerate(last_indices):
            for loop_i in range(total_ut_steps):
                vec = per_loop[loop_i][b_idx, last_tok_idx, :].detach().float().cpu().numpy()
                loop_vectors[loop_i].append(vec)

    loop_matrices = [np.stack(v, axis=0) for v in loop_vectors]
    return loop_matrices, prompt_meta


def main() -> None:
    args = parse_args()

    benchmark_cfg = BENCHMARK_CONFIG[args.benchmark]
    test_file = args.test_file or benchmark_cfg["file"]
    expected_samples = benchmark_cfg.get("expected_samples")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir is None:
        fewshot_tag = "fewshot" if args.use_few_shot else "zeroshot"
        script_dir = str(Path(__file__).resolve().parent)
        step_name = os.path.basename(os.path.abspath(args.checkpoint_path.rstrip("/")))
        args.output_dir = os.path.join(
            script_dir,
            "runs",
            f"{args.method}_{args.benchmark}_{step_name}_loops{args.total_ut_steps}_{fewshot_tag}_samples{args.max_samples}_{run_timestamp}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Last-token-across-prompts CKA analysis")
    logger.info("=" * 70)
    logger.info(f"Method: {args.method}")
    logger.info(f"Checkpoint path: {args.checkpoint_path}")
    logger.info(f"Benchmark: {args.benchmark} ({benchmark_cfg['name']})")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Output dir: {args.output_dir}")

    dataset = load_jsonl(test_file, max_samples=args.max_samples)
    logger.info(f"Loaded samples: {len(dataset)}")
    if expected_samples is not None and args.max_samples >= expected_samples and len(dataset) != expected_samples:
        logger.warning(
            f"Expected {expected_samples} samples for {args.benchmark}, got {len(dataset)}. "
            "Proceeding with available samples."
        )

    model_resolution = resolve_model_dir(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        base_model_dir=args.base_model_dir,
    )
    logger.info(f"Model dir: {model_resolution.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_resolution.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_resolution.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )
    if hasattr(model.config, "total_ut_steps"):
        model.config.total_ut_steps = args.total_ut_steps

    loop_mats, prompt_meta = extract_last_token_loop_embeddings(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        total_ut_steps=args.total_ut_steps,
        max_prompt_length=args.max_prompt_length,
        use_few_shot=args.use_few_shot,
        batch_size=args.batch_size,
    )

    heatmap = compute_cka_heatmap(loop_mats)
    convergence_curve = heatmap[:, -1]

    # Persist numerical artifacts.
    np.savez_compressed(
        os.path.join(args.output_dir, "last_token_loop_embeddings.npz"),
        **{f"loop_{i + 1}": loop_mats[i] for i in range(len(loop_mats))},
    )
    np.save(os.path.join(args.output_dir, "cka_heatmap.npy"), heatmap)
    np.save(os.path.join(args.output_dir, "convergence_curve.npy"), convergence_curve)

    # Persist plots.
    save_heatmap(heatmap, os.path.join(args.output_dir, "cka_heatmap.png"))
    save_convergence_curve(
        convergence_curve,
        os.path.join(args.output_dir, "convergence_to_final_curve.png"),
    )

    # Persist metadata / metrics summary.
    summary = {
        "method": args.method,
        "checkpoint_path": args.checkpoint_path,
        "resolved_model_dir": model_resolution.model_dir,
        "benchmark": args.benchmark,
        "benchmark_name": benchmark_cfg["name"],
        "num_samples": len(dataset),
        "total_ut_steps": args.total_ut_steps,
        "max_prompt_length": args.max_prompt_length,
        "use_few_shot": args.use_few_shot,
        "cka_heatmap": heatmap.tolist(),
        "convergence_to_final": convergence_curve.tolist(),
    }
    run_config = {
        "run_timestamp": run_timestamp,
        "cli_args": vars(args),
        "method": args.method,
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "resolved_model_dir": model_resolution.model_dir,
        "merged_from_fsdp": model_resolution.merged_from_fsdp,
        "output_dir": os.path.abspath(args.output_dir),
        "benchmark": {
            "name": benchmark_cfg["name"],
            "key": args.benchmark,
            "file": test_file,
            "expected_samples": expected_samples,
        },
        "runtime": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }

    prompt_meta_path = os.path.join(args.output_dir, "prompt_metadata.jsonl")
    with open(prompt_meta_path, "w") as f:
        for row in prompt_meta:
            f.write(json.dumps(row) + "\n")

    with open(os.path.join(args.output_dir, "cka_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    if args.cleanup_merged_model and model_resolution.merged_from_fsdp:
        logger.info("Cleaning up temporary merged model directory...")
        shutil.rmtree(model_resolution.model_dir, ignore_errors=True)

    logger.info("=" * 70)
    logger.info("Analysis complete.")
    logger.info(f"Saved CKA heatmap: {os.path.join(args.output_dir, 'cka_heatmap.png')}")
    logger.info(
        f"Saved convergence curve: {os.path.join(args.output_dir, 'convergence_to_final_curve.png')}"
    )


if __name__ == "__main__":
    main()
