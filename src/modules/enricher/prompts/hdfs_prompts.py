from .generic_prompts import TEMPLATE_PROMPT


HDFS_DATASET_CONTEXT = (
    "HDFS-v1 labels apply to complete traces grouped by block ID, not to an "
    "individual log event or template."
)

# Backward-compatible prompt export. Dataset facts are supplied through
# TemplateContext rather than hard-coded into the generic prompt.
HDFS_PROMPT_CORPUS = TEMPLATE_PROMPT
