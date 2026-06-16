class AdaptiveEngine:
    """
    Handles ML-style adaptation logic:
    - Mastery updates
    - Difficulty selection
    - Explanation depth selection
    """

    def update_mastery(self, mastery, correct):
        delta = 0.1 if correct else -0.1
        return min(max(mastery + delta, 0.0), 1.0)

    def explanation_level(self, mastery):
        if mastery < 0.5:
            return "Basic"
        elif mastery < 0.8:
            return "Intermediate"
        return "Advanced"

    def difficulty_from_mastery(self, mastery):
        if mastery < 0.4:
            return "Easy"
        elif mastery < 0.7:
            return "Medium"
        return "Hard"