class BayesianKnowledgeTracer:
    def __init__(self, p_init=0.20, p_learn=0.15, p_guess=0.20, p_slip=0.10):
        if not all(0.0 < x < 1.0 for x in (p_init, p_learn, p_guess, p_slip)):
            raise ValueError("BKT parameters must lie strictly between 0 and 1")
        self.p_init, self.p_learn = p_init, p_learn
        self.p_guess, self.p_slip = p_guess, p_slip
        self.mastery = {}

    def update_from_score(self, skill, score):
        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        prior = self.mastery.setdefault(skill, self.p_init)
        likelihood_known = score * (1 - self.p_slip) + (1 - score) * self.p_slip
        likelihood_unknown = score * self.p_guess + (1 - score) * (1 - self.p_guess)
        denom = prior * likelihood_known + (1 - prior) * likelihood_unknown
        posterior = prior * likelihood_known / max(denom, 1e-12)
        updated = posterior + (1 - posterior) * self.p_learn * score
        self.mastery[skill] = updated
        return updated

    def get_mastery(self, skill):
        return self.mastery.setdefault(skill, self.p_init)

    def get_all_mastery(self):
        return dict(self.mastery)
