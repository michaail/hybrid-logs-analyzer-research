from langchain_core.prompts import ChatPromptTemplate

from .system import get_system_prompt


TEMPLATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        get_system_prompt(),
        (
            "human",
            """
# Template evidence
{context_json}
""",
        ),
    ]
)