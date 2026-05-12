from ai.mock_llm_engine import MockLLMEngine
from ai.pdf_reader import read_pdf
from ai.content_analyzer import ContentAnalyzer
from core.adaptive_engine import AdaptiveEngine


class StudyBuddy:
    ''''
    def __init__(self, db):
        self.db = db
        self.llm = LLMEngine()
        self.adaptive = AdaptiveEngine()
    '''
    
    def __init__(self, db, llm_engine):
        self.db = db
        self.llm = llm_engine
        self.adaptive = AdaptiveEngine()


    # ✅ NEW METHOD (required by main.py)
    def ingest_material(self, file_path):
        text = read_pdf(file_path)
        analyzer = ContentAnalyzer()
        concepts = analyzer.extract_concepts(text)
        return list(concepts)

    def study_concept(self, concept):
        mastery = self.db.get_mastery(concept)

        level = self.adaptive.explanation_level(mastery)
        explanation = self.llm.explain(concept, level)
        print(f"\n📘 {explanation}")

        difficulty = self.adaptive.difficulty_from_mastery(mastery)
        question = self.llm.generate_question(concept, difficulty)
        print(f"\n📝 {question}")

        correct = input("Did you answer correctly? (y/n): ").lower() == "y"
        new_mastery = self.adaptive.update_mastery(mastery, correct)

        self.db.update(concept, new_mastery)
        print(f"✅ Updated mastery: {new_mastery:.2f}")
