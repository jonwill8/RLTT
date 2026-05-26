"""
Centralized math problem prompting utilities.

This module provides consistent 5-shot Chain-of-Thought (CoT) prompting
used across baseline_eval, sft_experiments, and grpo_experiments.

Usage:
    from math_utils import format_math_prompt, build_chat_messages

    # Format a problem with 5-shot CoT
    prompt = format_math_prompt(problem, tokenizer, use_few_shot=True)

    # Or build raw chat messages
    messages = build_chat_messages(problem, use_few_shot=True)
"""

from typing import List, Dict


# 5-shot Chain-of-Thought examples for math problem solving
FEW_SHOT_EXAMPLES = [
    {
        "problem": "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$."
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}$."
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}"
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$"
    },
    {
        "problem": "What is the longest side of the triangle with vertices at $(2,2)$, $(5,6)$, and $(6,2)$?",
        "solution": "We use the distance formula to find the length of each side.\n\nThe distance from $(2,2)$ to $(5,6)$ is $\\sqrt{(5-2)^2+(6-2)^2}=\\sqrt{9+16}=\\sqrt{25}=5$.\n\nThe distance from $(5,6)$ to $(6,2)$ is $\\sqrt{(6-5)^2+(2-6)^2}=\\sqrt{1+16}=\\sqrt{17}$.\n\nThe distance from $(6,2)$ to $(2,2)$ is $\\sqrt{(2-6)^2+(2-2)^2}=\\sqrt{16+0}=\\sqrt{16}=4$.\n\nTherefore, the longest side of the triangle has length $\\boxed{5}$."
    },
]

# Instruction for math problem solving
INSTRUCTION = (
    "Solve the following math problem. Show your reasoning step by step.\n"
    "Put your final answer in \\boxed{}. Once you provide the final answer, stop immediately."
)


def build_chat_messages(problem: str, use_few_shot: bool = False) -> List[Dict[str, str]]:
    """Build chat messages for the tokenizer's apply_chat_template.

    Args:
        problem: The math problem to solve.
        use_few_shot: If True, include 5-shot CoT examples before the problem.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    messages = []

    if use_few_shot:
        # Add few-shot examples
        for example in FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": f"{INSTRUCTION}\n\nProblem: {example['problem']}"
            })
            messages.append({
                "role": "assistant",
                "content": example['solution']
            })

    # Add the actual problem
    messages.append({
        "role": "user",
        "content": f"{INSTRUCTION}\n\nProblem: {problem}"
    })

    return messages


def format_math_prompt(problem: str, tokenizer, use_few_shot: bool = False) -> str:
    """Format a problem for generation using the tokenizer's chat template.

    Args:
        problem: The math problem to solve.
        tokenizer: The tokenizer with apply_chat_template method.
        use_few_shot: If True, include 5-shot CoT examples before the problem.

    Returns:
        Formatted prompt string ready for model input.
    """
    messages = build_chat_messages(problem, use_few_shot=use_few_shot)

    # Apply chat template with generation prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt
