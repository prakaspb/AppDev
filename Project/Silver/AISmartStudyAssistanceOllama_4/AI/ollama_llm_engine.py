import requests
import logging


class OllamaLLMEngine:
    """
    Local LLM using Ollama (Mistral model)
    No API key, no token limits.
    """

    def __init__(self, model="mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
        logging.info(f"Ollama LLM initialized with model: {self.model}")

    def _call(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(self.url, json=payload, timeout=60)
        response.raise_for_status()

        return response.json()["response"].strip()

    def explain(self, concept, level):
        prompt = (
            f"Explain the concept '{concept}' at a {level} level. "
            f"Use clear language and examples."
        )
        return self._call(prompt)

    def generate_question(self, concept, difficulty):
        prompt = (
            f"Generate one {difficulty} practice question about '{concept}'. "
            f"Do not include the answer."
        )
        return self._call(prompt)