import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_MISTRAL_LARGE")


class LLM:
    def __init__(self, model: str | None = None) -> None:
        self.client = ChatOpenAI(
            api_key=API_KEY,
            base_url=ENDPOINT,
            model=model or MODEL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def get_llm(self) -> ChatOpenAI:
        return self.client
