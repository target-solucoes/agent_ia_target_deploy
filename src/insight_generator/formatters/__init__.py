"""
Formatters module for insight_generator.

This module contains prompt builders and insight formatters.
"""

from .prompt_builder import build_prompt, SYSTEM_PROMPT, CHART_TYPE_TEMPLATES
from .insight_formatter import InsightFormatter
from .dynamic_prompt_builder import (
    DynamicPromptBuilder,
    build_dynamic_prompt,
    ANALYSIS_PERSONAS,
    FORMAT_RULES,
)

__all__ = [
    "build_prompt",
    "SYSTEM_PROMPT",
    "CHART_TYPE_TEMPLATES",
    "InsightFormatter",
    "DynamicPromptBuilder",
    "build_dynamic_prompt",
    "ANALYSIS_PERSONAS",
    "FORMAT_RULES",
]
