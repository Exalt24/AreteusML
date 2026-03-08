# ADR-005: Streamlit for Dashboard

## Context

Need a dashboard for model monitoring and live demo. Candidates evaluated: Gradio, Streamlit, Next.js.

## Decision

Use Streamlit.

## Consequences

- Better suited for multi-page monitoring dashboards (metrics, data drift, model performance) vs Gradio's single-model-demo focus
- Free hosting on HuggingFace Spaces keeps deployment cost at zero
- Multi-page app support allows separating live inference, training metrics, and data exploration into clean views
- Next.js would require building a full API layer; Streamlit talks directly to Python model code
