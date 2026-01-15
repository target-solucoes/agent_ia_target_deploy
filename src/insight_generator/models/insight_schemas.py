"""
Pydantic schemas and LLM configuration for insight generator.

This module defines the output schemas for insights and the LLM loader function.

GEMINI MIGRATION:
- Uses ChatGoogleGenerativeAI (Google Gemini)
- Model: gemini-2.5-flash-preview-09-2025
- Optimizations: timeout=30s, max_retries=2
- temperature=0.7 for creative insights
- JSON mode automatic (response_mime_type="application/json")

References:
- Authentication: https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from src.shared_lib.core.config import get_insight_config


class InsightItem(BaseModel):
    """Um insight individual"""

    title: str = Field(..., description="Título do insight (para negrito)")
    content: str = Field(..., description="Conteúdo completo do insight")
    metrics: Dict[str, Any] = Field(..., description="Métricas usadas no cálculo")
    confidence: float = Field(default=0.8, ge=0, le=1)
    chart_context: str = Field(..., description="Tipo de gráfico relacionado")


class InsightMetadata(BaseModel):
    """Metadados da geração"""

    calculation_time: float
    metrics_count: int
    llm_model: str
    timestamp: str
    transparency_validated: bool


class InsightOutput(BaseModel):
    """Output completo do insight_generator"""

    status: str = Field(..., pattern="^(success|error)$")
    chart_type: str
    insights: List[InsightItem] = Field(..., max_length=5)
    metadata: InsightMetadata
    error: Optional[str] = None


def load_insight_llm() -> ChatGoogleGenerativeAI:
    """
    Carrega Google Gemini com temperature=0.7 e JSON mode.

    Uses centralized Gemini configuration with optimizations:
    - timeout=30s (50% reduction from default 60s)
    - max_retries=2 (fail fast instead of many retries)
    - max_output_tokens=1500 (optimized default)
    - temperature=0.7 (higher temperature for creative insights)
    - JSON mode automatic (response_mime_type="application/json")

    Boas práticas Gemini:
    - Usar temperature para controlar criatividade (0.0-2.0)
    - response_mime_type: "application/json" para garantir output estruturado
    - System instructions para contexto

    Ref: https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/System_instructions.ipynb

    Returns:
        ChatGoogleGenerativeAI: Instância configurada com timeout, retry e JSON mode
    """
    config = get_insight_config()
    return ChatGoogleGenerativeAI(**config.to_gemini_kwargs())
