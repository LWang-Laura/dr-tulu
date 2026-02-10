"""
OpenScholar-style paper/reference evaluation reward.
Computes scores from test_case configs that use metric_config.config.other_properties.
Used by evaluate_responses_with_rm.py for rubric evaluation.
"""
import logging
from typing import Any, Dict

from open_instruct.search_rewards.utils.format_utils import extract_answer_context_citations
from open_instruct.search_rewards.utils.rubric_utils import _score_property

LOGGER = logging.getLogger(__name__)


def compute_paper_reward(response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a response against a test case that defines criteria in metric_config.config.other_properties.

    Args:
        response: Model response (expected to contain <answer>...</answer>).
        test_case: Dict with 'initial_prompt' (question) and 'metric_config.config.other_properties'
                   (list of dicts with 'name' and optionally 'criterion').

    Returns:
        Dict with:
        - reward: average of property scores in [0, 1]
        - scoring_results: { prop_name: score, ... }
        - extraction_success: whether answer was extracted
        - error: error message if any
        - citations: extracted citations dict (empty for this path)
    """
    result = {
        "reward": 0.0,
        "scoring_results": {},
        "extraction_success": False,
        "error": None,
        "citations": {},
    }

    _, extracted_answer, extracted_citations = extract_answer_context_citations(response)
    if extracted_answer is None:
        result["error"] = "Failed to extract answer from response - no <answer></answer> tags found"
        return result

    result["extraction_success"] = True
    result["citations"] = extracted_citations or {}

    question = test_case.get("initial_prompt", "")
    config = test_case.get("metric_config", {}).get("config", {})
    other_properties = config.get("other_properties", [])

    if not other_properties:
        result["reward"] = 0.0
        return result

    scores = []
    for prop in other_properties:
        name = prop.get("name", "")
        criterion = prop.get("criterion") or prop.get("name") or ""
        if not criterion:
            continue
        try:
            score = _score_property(extracted_answer, question, criterion)
            result["scoring_results"][name] = score
            scores.append(score)
        except Exception as e:
            LOGGER.warning("Error scoring property %s: %s", name, e)
            result["scoring_results"][name] = 0.0
            scores.append(0.0)

    result["reward"] = sum(scores) / len(scores) if scores else 0.0
    return result
