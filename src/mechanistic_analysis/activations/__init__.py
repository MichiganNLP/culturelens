"""Assistant-start residual-stream extraction."""

from mechanistic_analysis.activations.conditions import (
    ACTIVATION_CONDITIONS,
    CONTROL_CONDITIONS,
    CORE_CONDITIONS,
)

ACTIVATION_SITE = "decoder_block_resid_post@last_prompt_token_before_generation"
SCHEMA_VERSION = "culturelens-v2-assistant-start-v1"

__all__ = [
    "ACTIVATION_CONDITIONS",
    "ACTIVATION_SITE",
    "CONTROL_CONDITIONS",
    "CORE_CONDITIONS",
    "SCHEMA_VERSION",
]
