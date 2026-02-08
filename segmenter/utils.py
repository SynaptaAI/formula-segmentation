from typing import Any, Dict


def format_solution_text(candidate: Dict[str, Any]) -> str:
    """Format candidate solution as readable text."""
    steps = candidate.get("solution_steps", [])
    answer = candidate.get("final_answer", "")
    text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
    if answer:
        text += f"\n\nFinal Answer: {answer}"
    return text
