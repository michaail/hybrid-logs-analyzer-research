from .generic_prompts import TEMPLATE_PROMPT


BGL_DATASET_CONTEXT = (
    "BGL source log messages carry event-level labels. When transformed into time "
    "windows, a window is anomalous when it contains an anomalous event."
)

# Backward-compatible prompt export. Dataset facts are supplied through
# TemplateContext rather than hard-coded into the generic prompt.
BGL_PROMPT_CORPUS = TEMPLATE_PROMPT
