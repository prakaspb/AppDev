from ai.mock_llm_engine import MockLLMEngine
from ai.openai_llm_engine import OpenAILLMEngine


def get_llm_engine(use_real_llm: bool):
    """
    Factory method to choose LLM at runtime
    """
    if use_real_llm:
        return OpenAILLMEngine()
    return MockLLMEngine()