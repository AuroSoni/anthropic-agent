"""Cost calculation from token usage and model pricing data.

Loads pricing from the bundled CSV and provides functions to calculate
costs from token usage history as tracked by the agent run loop.
"""

import csv
import os
from dataclasses import dataclass, asdict
from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)

# ── Supplementary pricing constants (not per-model) ────────────────────
WEB_SEARCH_COST_PER_SEARCH = 0.01      # $10 per 1,000 searches = $0.01 each
CODE_EXECUTION_COST_PER_HOUR = 0.05    # $0.05/hour after free tier
DATA_RESIDENCY_MULTIPLIER = 1.1        # 1.1x for US-only inference_geo


@dataclass
class ModelPricing:
    """Pricing data for a single Anthropic model."""
    model_id: str
    display_name: str
    input_per_mtok: float
    cache_write_5m_per_mtok: float
    cache_write_1h_per_mtok: float
    cache_read_per_mtok: float
    output_per_mtok: float
    long_context_input_multiplier: float
    long_context_output_multiplier: float
    long_context_threshold: int  # 0 means no long context pricing


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for an agent run."""
    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_write_cost: float = 0.0
    cache_read_cost: float = 0.0
    total_cost: float = 0.0

    # Token totals (cumulative across all steps)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0

    # Pricing metadata
    model_id: str = ""
    long_context_applied: bool = False
    currency: str = "USD"

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for serialization."""
        return asdict(self)


# Module-level cache for loaded pricing data
_pricing_cache: dict[str, ModelPricing] | None = None


def load_pricing() -> dict[str, ModelPricing]:
    """Load pricing data from the bundled CSV file.

    Returns:
        Dictionary mapping model_id to ModelPricing.
        Cached after first load.
    """
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache

    csv_path = os.path.join(os.path.dirname(__file__), "models.csv")
    pricing: dict[str, ModelPricing] = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mp = ModelPricing(
                model_id=row["model_id"],
                display_name=row["display_name"],
                input_per_mtok=float(row["input_per_mtok"]),
                cache_write_5m_per_mtok=float(row["cache_write_5m_per_mtok"]),
                cache_write_1h_per_mtok=float(row["cache_write_1h_per_mtok"]),
                cache_read_per_mtok=float(row["cache_read_per_mtok"]),
                output_per_mtok=float(row["output_per_mtok"]),
                long_context_input_multiplier=float(row["long_context_input_multiplier"]),
                long_context_output_multiplier=float(row["long_context_output_multiplier"]),
                long_context_threshold=int(row["long_context_threshold"]),
            )
            pricing[mp.model_id] = mp

    _pricing_cache = pricing
    return pricing


def resolve_model_pricing(model_name: str) -> ModelPricing | None:
    """Resolve an API model name to its pricing entry.

    Handles versioned model names like "claude-sonnet-4-5-20250929"
    by matching against base model IDs using substring matching.

    Args:
        model_name: The model name from the API response

    Returns:
        ModelPricing if found, None if model is unknown
    """
    pricing = load_pricing()

    # Exact match first
    if model_name in pricing:
        return pricing[model_name]

    # Substring match: check if any pricing model_id is contained in the model name
    # Sort by length descending for most-specific match first
    for model_id in sorted(pricing.keys(), key=len, reverse=True):
        if model_id in model_name:
            return pricing[model_id]

    logger.warning("Unknown model for cost calculation", model=model_name)
    return None


def calculate_run_cost(
    token_usage_history: list[dict],
    model_name: str,
) -> CostBreakdown | None:
    """Calculate the total cost for an agent run.

    Sums all entries in _token_usage_history and applies model pricing.
    Detects long context pricing based on per-step input tokens exceeding
    the model's threshold.

    Note: cache tokens (cache_creation_input_tokens, cache_read_input_tokens)
    are a subset of input_tokens in the Anthropic API response. Base input
    cost is calculated on input_tokens minus cache tokens to avoid
    double-counting.

    Args:
        token_usage_history: List of per-step dicts with keys:
            step, input_tokens, output_tokens,
            cache_creation_input_tokens, cache_read_input_tokens
        model_name: API model name (e.g., "claude-sonnet-4-5-20250929")

    Returns:
        CostBreakdown with detailed costs, or None if model pricing unknown
    """
    pricing = resolve_model_pricing(model_name)
    if pricing is None:
        return None

    if not token_usage_history:
        return CostBreakdown(model_id=pricing.model_id)

    # Sum tokens across all steps
    total_input = sum(s.get("input_tokens", 0) or 0 for s in token_usage_history)
    total_output = sum(s.get("output_tokens", 0) or 0 for s in token_usage_history)
    total_cache_write = sum(s.get("cache_creation_input_tokens", 0) or 0 for s in token_usage_history)
    total_cache_read = sum(s.get("cache_read_input_tokens", 0) or 0 for s in token_usage_history)

    # Determine if long context pricing applies (per-step check)
    long_context = False
    if pricing.long_context_threshold > 0:
        for step_usage in token_usage_history:
            step_input = step_usage.get("input_tokens", 0) or 0
            if step_input > pricing.long_context_threshold:
                long_context = True
                break

    # Calculate per-category costs (prices are per million tokens)
    input_multiplier = pricing.long_context_input_multiplier if long_context else 1.0
    output_multiplier = pricing.long_context_output_multiplier if long_context else 1.0

    # Base input tokens = total_input - cache_write - cache_read
    # (cache tokens are a subset of input_tokens in the API response)
    base_input = max(0, total_input - total_cache_write - total_cache_read)

    input_cost = (base_input / 1_000_000) * pricing.input_per_mtok * input_multiplier
    output_cost = (total_output / 1_000_000) * pricing.output_per_mtok * output_multiplier

    # Cache write cost: use 5-minute TTL pricing (the API default)
    cache_write_cost = (total_cache_write / 1_000_000) * pricing.cache_write_5m_per_mtok * input_multiplier
    cache_read_cost = (total_cache_read / 1_000_000) * pricing.cache_read_per_mtok * input_multiplier

    total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost

    return CostBreakdown(
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        cache_write_cost=round(cache_write_cost, 6),
        cache_read_cost=round(cache_read_cost, 6),
        total_cost=round(total_cost, 6),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cache_creation_tokens=total_cache_write,
        total_cache_read_tokens=total_cache_read,
        model_id=pricing.model_id,
        long_context_applied=long_context,
        currency="USD",
    )
