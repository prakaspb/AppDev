import os
#from openai import OpenAI
'''
class LLMEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"

    def _chat(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    # ✅ REQUIRED BY StudyBuddy
    def explain(self, concept, level):
        prompt = (
            f"Explain the concept '{concept}' at a {level} level. "
            f"Use clear language and examples if helpful."
        )
        return self._chat(prompt)

    # ✅ REQUIRED BY StudyBuddy
    def generate_question(self, concept, difficulty):
        prompt = (
            f"Create one {difficulty} practice question about the concept '{concept}'. "
            f"Do not include the answer."
        )
        return self._chat(prompt)
'''

## This dummy implementation of LLMEngine is for testing purposes. In a real implementation, you would replace the chat method with actual calls to the OpenAI API or another language model service.
class LLMEngine2:
    def chat(self, prompt):
        # This is a dummy implementation that simply echoes the prompt back with a prefix.
        return f"Dummy response to: {prompt}"
        
        
class MockLLMEngine:
    """
    Mock LLM Engine (No external APIs).
    This simulates an LLM for explanations and questions.
    """

    def explain(self, concept, level):
        if level == "Basic":
            return (
                f"{concept} is a fundamental topic. "
                f"This explanation covers basic definitions and simple ideas "
                f"to help you get started."
            )

        elif level == "Intermediate":
            return (
                f"{concept} builds on the basics. "
                f"This explanation includes examples and shows how concepts "
                f"are applied in practice."
            )

        else:  # Advanced
            return (
                f"{concept} is explained at an advanced level. "
                f"This includes deeper insights, edge cases, and real‑world usage."
            )

    def generate_question(self, concept, difficulty):
        if difficulty == "Easy":
            return f"What is the basic definition of {concept}?"

        elif difficulty == "Medium":
            return f"How does {concept} work? Give an example."

        else:  # Hard
            return (
                f"Explain {concept} in depth and describe a real‑world scenario "
                f"where it is applied."
            )