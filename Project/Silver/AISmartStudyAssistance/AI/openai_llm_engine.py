import os
import logging
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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
class OpenAILLMEngine:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        logging.info("OpenAI LLM initialized")

    def _chat(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def explain(self, concept, level):
        prompt = f"Explain '{concept}' at a {level} level."
        return self._chat(prompt)

    def generate_question(self, concept, difficulty):
        prompt = (
            f"Generate one {difficulty} practice question about '{concept}'. "
            f"Do not include the answer."
        )
        return self._chat(prompt)
