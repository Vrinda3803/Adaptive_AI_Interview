"""Question text normalization and duplicate detection for interview sessions."""

def normalize_question(text):
    import re
    words = re.findall(r"[a-z0-9]+", (text or "").casefold())
    synonym_groups = {
        "optimization": {"improve", "improved", "improving", "optimize", "optimized", "optimizing", "performance", "faster", "speed", "throughput", "efficient", "efficiency"},
        "debugging": {"debug", "debugged", "debugging", "troubleshoot", "troubleshooting", "diagnose", "diagnosed", "fix", "resolve", "resolved"},
        "reliability": {"reliable", "reliability", "resilient", "resilience", "faulttolerance"},
        "testing": {"test", "tests", "testing", "validate", "validation", "verify", "verification"},
    }
    canonical = {word: root for root, members in synonym_groups.items() for word in members}
    stop = {"a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for", "with", "how", "what", "did", "you", "your", "would", "could", "i", "we", "do", "does", "is", "are", "was", "were", "it", "this", "that", "about", "make", "made"}
    return " ".join(canonical.get(word, word) for word in words if word not in stop)


def is_duplicate(question, history):
    """Exact or high lexical-semantic overlap check; same-skill follow-ups are allowed."""
    from difflib import SequenceMatcher
    candidate = normalize_question(question)
    if not candidate:
        return True
    candidate_tokens = set(candidate.split())
    for item in history:
        prior = normalize_question(item.get("question", ""))
        if not prior:
            continue
        if candidate == prior or SequenceMatcher(None, candidate, prior).ratio() >= 0.90:
            return True
        prior_tokens = set(prior.split())
        overlap = len(candidate_tokens & prior_tokens) / max(1, len(candidate_tokens | prior_tokens))
        if overlap >= 0.72:
            return True
    return False
