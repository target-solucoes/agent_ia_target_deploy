# -*- coding: utf-8 -*-
"""
Display Components for Streamlit Chatbot

Rendering functions for each component of the formatter JSON output.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import streamlit.components.v1 as components
import re


# ============================================================================
# Helper Functions
# ============================================================================


def _escape_latex_chars(text: str) -> str:
    r"""
    Escape LaTeX special characters to prevent rendering issues.

    Specifically escapes $ to \$ to prevent LaTeX math mode interpretation.
    This is critical for R$ currency formatting.

    Args:
        text: Text to escape

    Returns:
        Text with LaTeX special characters escaped
    """
    if not text:
        return text

    # Escape dollar signs that are not already escaped
    # This prevents R$ from being interpreted as LaTeX math delimiters
    text = text.replace("$", r"\$")

    return text


def _bold_numbers(text: str) -> str:
    """
    Automatically highlights numbers in text by applying bold formatting.

    Detects and bolds:
    - Percentages (7,38% 18.64%)
    - Monetary values with suffixes (24.46M 814.95M 1.5K)
    - Currency values (R$ 1.234,56)
    - Multipliers (12,54x 3.2x)
    - Numbers with thousand separators (1.234.567 or 1,234,567)
    - Decimal numbers (123.45 or 123,45)
    - Integer numbers

    Args:
        text: Text to process

    Returns:
        Text with numbers wrapped in ** for markdown bold, and $ escaped for LaTeX
    """
    if not text:
        return text

    # STEP 1: Remove ALL existing ** around numbers to start fresh
    # This prevents ****number issues when backend already applied bold
    # Two passes: remove ** before numbers, then after numbers
    text = re.sub(r"\*{2,}([\d.,]+[%MKBx]?)", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"([\d.,]+[%MKBx]?)\*{2,}", r"\1", text, flags=re.IGNORECASE)

    # STEP 2: Temporarily replace R$ with placeholder
    placeholder = "___CURRENCY___"
    text = text.replace("R$", placeholder)

    # STEP 3: Apply bold to all numbers cleanly
    # Pattern matches: digit(s), optional decimal/comma separators, optional suffix
    # Negative lookahead/lookbehind prevent matching inside already-bolded numbers
    pattern = r"(?<![*\d])\d[\d.,]*[%MKBx]?(?![*\d])"
    text = re.sub(pattern, r"**\g<0>**", text, flags=re.IGNORECASE)

    # STEP 4: Handle R$ with bolded numbers
    text = re.sub(r"___CURRENCY___\s*\*\*([\d.,]+[MKBx]?)\*\*", r"**R$ \1**", text)
    text = text.replace(placeholder, "R$")

    # STEP 5: Escape LaTeX characters
    return _escape_latex_chars(text)


# ============================================================================
# Main Rendering Functions
# ============================================================================


def render_executive_summary(summary: Dict) -> None:
    """
    Render executive summary section (title + introduction)

    Args:
        summary: executive_summary dict from formatter output
    """
    if not summary:
        return

    # Title
    title = summary.get("title", "Análise de Dados")
    st.markdown(f"### {title}")

    # Introduction
    introduction = summary.get("introduction", "")
    if introduction:
        st.markdown(introduction)


def render_filters_badge(filters: Dict[str, Any]) -> None:
    """
    Render active filters as colored badges

    Args:
        filters: Dictionary of active filters
    """
    if not filters:
        return

    # Create columns for badges
    cols = st.columns(min(len(filters), 4))

    for idx, (key, value) in enumerate(filters.items()):
        col_idx = idx % 4
        with cols[col_idx]:
            # Clean key name (remove underscores, capitalize)
            clean_key = key.replace("_", " ").title()
            st.markdown(
                f'<span style="background-color: #e3f2fd; color: #1976d2; '
                f"padding: 4px 12px; border-radius: 12px; font-size: 0.85em; "
                f'display: inline-block; margin: 2px;">'
                f"{clean_key}: <b>{value}</b></span>",
                unsafe_allow_html=True,
            )


def render_plotly_chart(chart_data: Dict) -> None:
    """
    Render Plotly chart from HTML string

    Args:
        chart_data: Chart dict containing 'html' and metadata
    """
    if not chart_data:
        return

    html_content = chart_data.get("html", "")

    if html_content:
        # Render HTML chart using components
        components.html(html_content, height=500, scrolling=False)
    else:
        st.warning("Grafico nao disponivel")


def render_insights(insights: Dict) -> None:
    """
    Render insights section with improved formatting.

    Structure:
    1. Resumo Executivo: narrative with auto-bolded numbers
    2. Principais Achados: detailed_insights in format:
       **{title}** — {interpretation}
       *({formula})*

    Args:
        insights: insights dict from formatter output
    """
    if not insights:
        return

    st.markdown("---")
    st.markdown("#### 📊 Insights")

    # ========================================================================
    # Section 1: Resumo Executivo (Executive Summary)
    # ========================================================================
    narrative = insights.get("narrative", "")
    if narrative:
        st.markdown("##### 📌 Resumo Executivo")
        # Apply auto-bold to numbers in narrative
        narrative_formatted = _bold_numbers(narrative)
        st.markdown(narrative_formatted)
        st.markdown("")  # spacing

    # ========================================================================
    # Section 2: Principais Achados (Key Findings)
    # ========================================================================
    detailed_insights = insights.get("detailed_insights", [])
    if detailed_insights:
        st.markdown("##### 🔍 Principais Achados")
        st.markdown("")  # spacing

        for insight in detailed_insights:
            title = insight.get("title", "")
            interpretation = insight.get("interpretation", "")
            formula = insight.get("formula", "")

            # Format: • **{title}** — {interpretation} *({formula})*
            if title and interpretation:
                interpretation_formatted = _bold_numbers(interpretation)
                # Add formula right after interpretation if present
                if formula:
                    formula_escaped = _escape_latex_chars(formula)
                    st.markdown(
                        f"• **{title}** — {interpretation_formatted} *({formula_escaped})*"
                    )
                else:
                    st.markdown(f"• **{title}** — {interpretation_formatted}")
            elif interpretation:
                # If no title, just show interpretation
                interpretation_formatted = _bold_numbers(interpretation)
                if formula:
                    formula_escaped = _escape_latex_chars(formula)
                    st.markdown(f"• {interpretation_formatted} *({formula_escaped})*")
                else:
                    st.markdown(f"• {interpretation_formatted}")

    # ========================================================================
    # Section 3: Transparency Score (Optional footer)
    # ========================================================================
    # Removed transparency score display as per user request


def render_next_steps(next_steps: Dict) -> None:
    """
    Render next steps section (3 direct strategic recommendations)

    Args:
        next_steps: next_steps dict from formatter output
    """
    if not next_steps:
        return

    st.markdown("---")
    st.markdown("#### 🧭 Próximos Passos")

    # Get next steps items (exactly 3)
    items = next_steps.get("items", [])

    if items:
        for step in items:
            st.markdown(f"- {step}")
    else:
        st.info("Nenhum proximo passo disponivel no momento.")


def render_data_table(data: Dict) -> None:
    """
    Render data table section

    Args:
        data: data dict from formatter output
    """
    if not data:
        return

    st.markdown("---")

    # Summary Table
    summary_table = data.get("summary_table", {})
    if summary_table:
        headers = summary_table.get("headers", [])
        rows = summary_table.get("rows", [])
        total_rows = summary_table.get("total_rows", 0)
        showing_rows = summary_table.get("showing_rows", 0)

        if headers and rows:
            # Create dataframe for display
            import pandas as pd

            df = pd.DataFrame(rows, columns=headers)
            st.dataframe(df, use_container_width=True)


def render_metadata_debug(metadata: Dict) -> None:
    """
    Render metadata and debug information

    Args:
        metadata: metadata dict from formatter output
    """
    # Metadata/debug rendering removed per user request
    return


def render_error(error_info: Any) -> None:
    """
    Render error message

    Args:
        error_info: Error information (string or dict)
    """
    if isinstance(error_info, dict):
        error_msg = error_info.get("message", str(error_info))
    else:
        error_msg = str(error_info)

    st.error(f"Erro ao processar consulta: {error_msg}")


def render_loading_state(message: str = "Processando...") -> None:
    """
    Render loading state with spinner

    Args:
        message: Loading message to display
    """
    with st.spinner(message):
        st.empty()


def render_non_graph_response(non_graph_output: Dict) -> None:
    """
    Render non-graph executor output (summary-only display)

    For non_graph_output, we display the appropriate content:
    - 'conversational_response' for conversational queries
    - 'summary' for data queries (metadata, aggregation, lookup, etc.)
    - 'data' as dataframe for tabular queries

    Args:
        non_graph_output: Complete non_graph JSON output
    """
    if not non_graph_output:
        render_error("Nenhuma resposta disponivel")
        return

    status = non_graph_output.get("status", "unknown")

    if status == "error":
        error = non_graph_output.get("error")
        render_error(error)
        return

    query_type = non_graph_output.get("query_type")

    # For TABULAR queries, render data as dataframe
    if query_type == "tabular":
        data = non_graph_output.get("data")

        if data and isinstance(data, list) and len(data) > 0:
            import pandas as pd

            df = pd.DataFrame(data)

            # Display summary if available
            summary = non_graph_output.get("summary")
            if summary:
                response_formatted = _bold_numbers(summary)
                st.markdown(response_formatted)

            # Display dataframe
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Nenhum dado disponível para exibição tabular.")
        return

    # For OTHER query types, display text response
    # Extract text response - check conversational_response first, then summary
    conversational_response = non_graph_output.get("conversational_response")
    summary = non_graph_output.get("summary")

    # Prioritize conversational_response for conversational queries
    response_text = conversational_response if conversational_response else summary

    if response_text:
        # Apply auto-bold formatting to numbers in response
        response_formatted = _bold_numbers(response_text)
        st.markdown(response_formatted)
    else:
        st.info("Resposta processada com sucesso, mas sem resumo disponível.")


def render_complete_response(formatter_output: Dict) -> None:
    """
    Render complete formatter output in correct order

    This is a convenience function that renders all sections in the proper sequence.

    Args:
        formatter_output: Complete formatter JSON output
    """
    if not formatter_output:
        render_error("Nenhuma resposta disponivel")
        return

    status = formatter_output.get("status", "unknown")

    if status == "error":
        error = formatter_output.get("error")
        render_error(error)
        return

    # Render in correct order
    # 1. Executive Summary (title + introduction)
    executive_summary = formatter_output.get("executive_summary", {})
    render_executive_summary(executive_summary)

    # 2. Visualization (chart)
    st.markdown("")
    visualization = formatter_output.get("visualization", {})
    chart = visualization.get("chart", {})
    render_plotly_chart(chart)

    # 3. Insights
    insights = formatter_output.get("insights", {})
    render_insights(insights)

    # 4. Next Steps
    next_steps = formatter_output.get("next_steps", {})
    render_next_steps(next_steps)

    # 5. Data Table
    data = formatter_output.get("data", {})
    render_data_table(data)

    # Metadata/Debug rendering removed per user request


def render_unified_response(output_type: str, output_data: Dict) -> None:
    """
    Unified renderer that handles both non_graph and formatter outputs.

    This is the recommended function to use in app.py for rendering any pipeline output.

    Args:
        output_type: Type of output ("non_graph" or "formatter")
        output_data: The output data dictionary
    """
    if output_type == "non_graph":
        render_non_graph_response(output_data)
    elif output_type == "formatter":
        render_complete_response(output_data)
    else:
        render_error(f"Tipo de output desconhecido: {output_type}")
