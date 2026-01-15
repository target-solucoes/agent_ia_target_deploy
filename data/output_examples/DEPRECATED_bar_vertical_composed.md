# DEPRECATED: bar_vertical_composed

**Status:** REMOVED  
**Date:** 2025-12-15  
**Replacement:** `line_composed`

## Reason for Deprecation

The `bar_vertical_composed` chart type was causing visual confusion in temporal comparison analyses. It has been completely removed and replaced with `line_composed`, which provides:

1. **Better temporal visualization:** Line charts naturally show progression over time
2. **Clearer trend identification:** Easier to see who is growing/declining
3. **Less visual clutter:** Especially with the new Top N default filtering

## Migration

All queries that previously mapped to `bar_vertical_composed` now map to `line_composed`:

- **Intent:** `temporal_comparison_analysis` (unchanged)
- **Chart Type:** `line_composed` (was: `bar_vertical_composed`)
- **Dimension Order:** [Temporal, Category] (was: [Category, Temporal])

### New Features in line_composed

1. **Variation Calculation:** Automatically calculates Delta between first and last period
2. **Smart Sorting:** Orders categories by growth/decline when `sort.by = "variation"`
3. **Top N Default:** Applies `top_n = 5` by default for temporal comparisons to prevent "spaghetti charts"

## Example Queries

These queries now generate `line_composed` charts:

- "quais produtos tiveram maior aumento de vendas de maio para junho?"
- "crescimento de vendas entre 2015 e 2016"
- "comparar vendas dos 3 maiores estados entre janeiro e fevereiro"

## File References

- Old examples file: `03_bar_vertical_composed_examples.json` (DEPRECATED)
- Tool implementation: `src/analytics_executor/tools/bar_vertical_composed.py` (REMOVED)
- Intent mapping: `src/graphic_classifier/tools/intent_classifier.py` (UPDATED)
