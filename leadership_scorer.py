"""
Leadership Scorer — V4 (normalised scores + external weights)
-------------------------------------------------------------
Input:  raw resume text (string) from your PDF/DOCX extractor
Output: structured features dict + leadership score (0–100)

7-Dimension Model:
  1. Influence   (max 20) — people-gated verbs + tiered team size
  2. Impact      (max 25) — minor/medium/major tiers, handles ~% and Nx multipliers
  3. Initiative  (max 15) — context-gated verbs (org/programme nouns required nearby)
  4. Mentorship  (max 15) — instances + outcome bonus
  5. Scope/Scale (max 15) — differentiated weights, strict budget detection
  6. Ownership   (max 10) — accountability / end-to-end language
  7. Seniority   (max 10) — title level + years of experience

Weights and grade thresholds are loaded from weights.json (same directory).
Normalised scores (0-1) are computed for each dimension before applying weights.
When ML model produces new weights, update weights.json only — no code changes needed.

Install:
  pip install spacy
  python -m spacy download en_core_web_sm
"""

import re
import json
import datetime
import spacy
from dataclasses import dataclass, asdict
from pathlib import Path

nlp = spacy.load("en_core_web_sm")


# 1.  WEIGHTS — loaded from weights.json


def _load_weights() -> dict:
    """
    Load weights config from weights.json (same directory as this file).
    Falls back to default proportional weights if file is missing.
    """
    weights_path = Path(__file__).parent / "weights.json"
    if not weights_path.exists():
        print(f"[WARNING] weights.json not found at {weights_path}. Using defaults.")
        return {
            "weights": {
                "influence": 20.0, "impact": 25.0, "initiative": 15.0,
                "mentorship": 15.0, "scope_scale": 15.0, "ownership": 10.0,
                "seniority": 10.0,
            },
            "_max_scores": {
                "influence": 20, "impact": 25, "initiative": 15,
                "mentorship": 15, "scope_scale": 15, "ownership": 10,
                "seniority": 10,
            },
            "grade_thresholds": {"A": 85, "B": 70, "C": 55, "D": 40},
        }
    with open(weights_path, encoding="utf-8") as f:
        return json.load(f)

_CONFIG     = _load_weights()

# Explicitly extract only the 7 dimension keys — ignores any stray keys
DIMS = ["influence", "impact", "initiative", "mentorship", "scope_scale", "ownership", "seniority"]
WEIGHTS          = {k: float(_CONFIG["weights"][k]) for k in DIMS}
MAX_SCORES       = {k: int(_CONFIG["_max_scores"][k]) for k in DIMS}
GRADE_THRESHOLDS = _CONFIG["grade_thresholds"]



# 2.  FEATURE DATACLASS


@dataclass
class LeadershipFeatures:
    # ── Influence ──────────────────────────────────────────────────────
    influence_verb_hits: int = 0
    team_size_max: int       = 0

    # ── Impact ─────────────────────────────────────────────────────────
    impact_minor: int  = 0
    impact_medium: int = 0
    impact_major: int  = 0

    # ── Initiative ─────────────────────────────────────────────────────
    initiative_verb_hits: int = 0   # context-gated
    initiative_verb_raw: int  = 0   # raw (for inflation flag)

    # ── Mentorship ─────────────────────────────────────────────────────
    mentorship_instances: int = 0
    mentorship_outcomes: int  = 0

    # ── Scope / Scale ──────────────────────────────────────────────────
    budget_owned: bool     = False
    cross_functional: bool = False
    multi_department: bool = False
    global_scope: bool     = False

    # ── Ownership ──────────────────────────────────────────────────────
    ownership_mentions: int = 0

    # ── Seniority ──────────────────────────────────────────────────────
    title_tier: int      = 0
    years_experience: int = 0

    # ── Flags ──────────────────────────────────────────────────────────
    technical_ic_flag: bool = False



# 3.  VERB SETS — loaded from verbs.json

def _load_verbs() -> tuple[set, set, set]:
    verbs_path = Path(__file__).parent / "verbs.json"
    if not verbs_path.exists():
        raise FileNotFoundError(
            f"verbs.json not found at {verbs_path}. "
            "Run update_verbs.py once to generate it."
        )
    with open(verbs_path, encoding="utf-8") as f:
        data = json.load(f)
    return (
        set(data["INFLUENCE_VERBS"]),
        set(data["INITIATIVE_VERBS"]),
        set(data["MENTORSHIP_VERBS"]),
    )

INFLUENCE_VERBS, INITIATIVE_VERBS, MENTORSHIP_VERBS = _load_verbs()

# ── Context gate nouns ─────────────────────────────────────────────────
PEOPLE_NOUNS = {
    "team", "staff", "engineer", "engineers", "employee", "employees",
    "member", "members", "report", "reports", "headcount", "person",
    "people", "hire", "hires", "analyst", "analysts", "associate",
    "associates", "developer", "developers", "manager", "managers",
    "intern", "interns", "colleague", "colleagues", "volunteer",
}

ORG_NOUNS = {
    "programme", "program", "initiative", "project", "organisation",
    "organization", "department", "division", "unit", "function",
    "strategy", "roadmap", "platform", "product", "guild", "chapter",
    "framework", "process", "pipeline", "system", "team", "org",
    "company", "business", "operation", "operations", "centre", "center",
    "campaign", "committee", "board", "council", "practice",
}



# 4.  REGEX PATTERNS


TEAM_SIZE_RE = re.compile(
    r"(?:team|staff|engineers?|employees?|members?|reports?|headcount)\s+of\s+(\d+)"
    r"|(?:managed|led|supervised|oversaw)\s+(?:a\s+)?(?:team\s+of\s+)?(\d+)",
    re.IGNORECASE,
)
PERCENT_RE      = re.compile(r"~?\s*(\d+(?:\.\d+)?)\s*%")
MULTIPLIER_RE   = re.compile(r"(\d+(?:\.\d+)?)\s*x\b", re.IGNORECASE)
DOLLAR_RE       = re.compile(r"\$\s*([\d,.]+)\s*(k|m|b|million|billion)?", re.IGNORECASE)
BUDGET_OWNERSHIP_RE = re.compile(
    r"(?:managed|oversaw|owned|responsible\s+for|accountable\s+for|controlled)"
    r".{0,40}\bbudget\b", re.IGNORECASE,
)
CROSS_FUNCTIONAL_RE = re.compile(
    r"cross[\s-]functional|cross[\s-]team|cross[\s-]department|matrixed?", re.IGNORECASE,
)
MULTI_DEPT_RE = re.compile(
    r"multi[\s-]department|multiple\s+(?:teams?|departments?|divisions?|units?)"
    r"|stakeholders?|organisation[\s-]wide|company[\s-]wide|enterprise[\s-]wide",
    re.IGNORECASE,
)
GLOBAL_RE = re.compile(
    r"\bglobal\b|\binternational\b|\bworldwide\b|\bmultinational\b", re.IGNORECASE,
)
OWNERSHIP_TERMS = {
    "responsible", "accountable", "owned", "ownership", "end-to-end", "end to end",
}
MENTORSHIP_OUTCOME_RE = re.compile(
    r"promoted|advanced\s+to|improved\s+performance|retained|progressed", re.IGNORECASE,
)
TITLE_PATTERNS = [
    (4, re.compile(
        r"\b(vp|vice\s+president|svp|evp|chief\s+\w+\s+officer|c[eotf]o|"
        r"president|founder|co[\s-]founder|managing\s+director|partner)\b", re.IGNORECASE,
    )),
    (3, re.compile(
        r"\b(director|head\s+of|group\s+manager|general\s+manager|principal\s+manager)\b",
        re.IGNORECASE,
    )),
    (2, re.compile(
        r"\b(manager|team\s+lead|team\s+leader|lead\s+\w+|"
        r"engineering\s+lead|product\s+lead|delivery\s+lead)\b", re.IGNORECASE,
    )),
    (1, re.compile(
        r"\b(senior|sr\.?|principal|staff\s+engineer|staff\s+scientist|"
        r"specialist|consultant|advisor)\b", re.IGNORECASE,
    )),
]
YEARS_EXP_RE = re.compile(r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)", re.IGNORECASE)
DATE_RANGE_RE = re.compile(
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)?\s*(20\d{2}|19\d{2})\s*[-–]\s*"
    r"(20\d{2}|19\d{2}|present|current|now)",
    re.IGNORECASE,
)


# 5.  HELPERS

def _dollar_to_millions(amount_str: str, unit: str) -> float:
    try:
        value = float(amount_str.replace(",", ""))
    except ValueError:
        return 0.0
    unit = (unit or "").strip().lower()
    if unit in ("b", "billion"):
        value *= 1_000
    elif unit == "k":
        value /= 1_000
    elif unit == "":
        value /= 1_000_000
    return value

def _window_contains(tokens, centre_idx: int, word_set: set, window: int) -> bool:
    start = max(0, centre_idx - window)
    end   = min(len(tokens), centre_idx + window + 1)
    return any(
        tokens[i].lemma_.lower() in word_set or tokens[i].text.lower() in word_set
        for i in range(start, end) if i != centre_idx
    )

def _estimate_years(resume_text: str) -> int:
    matches = YEARS_EXP_RE.findall(resume_text)
    if matches:
        return max(int(m) for m in matches)
    current_year = datetime.datetime.now().year
    years = []
    for m in DATE_RANGE_RE.finditer(resume_text):
        try:
            start_yr = int(m.group(1))
            end_raw  = m.group(2).lower()
            end_yr   = current_year if end_raw in ("present", "current", "now") else int(end_raw)
            years.extend([start_yr, end_yr])
        except (ValueError, TypeError):
            continue
    try:
        return current_year - min(y for y in years if isinstance(y, int))
    except (ValueError, TypeError):
        return 0


# 6.  FEATURE EXTRACTION


def extract_features(resume_text: str) -> LeadershipFeatures:
    f      = LeadershipFeatures()
    doc    = nlp(resume_text)
    tokens = list(doc)

    for i, token in enumerate(tokens):
        if token.pos_ != "VERB":
            continue
        lemma = token.lemma_.lower()
        if lemma in INFLUENCE_VERBS:
            if _window_contains(tokens, i, PEOPLE_NOUNS, window=10):
                f.influence_verb_hits += 1
        if lemma in INITIATIVE_VERBS:
            f.initiative_verb_raw += 1
            if _window_contains(tokens, i, ORG_NOUNS, window=12):
                f.initiative_verb_hits += 1
        if lemma in MENTORSHIP_VERBS:
            f.mentorship_instances += 1

    sizes = []
    for m in TEAM_SIZE_RE.finditer(resume_text):
        num = m.group(1) or m.group(2)
        if num:
            sizes.append(int(num))
    if sizes:
        f.team_size_max = max(sizes)

    for m in PERCENT_RE.finditer(resume_text):
        value = float(m.group(1))
        if value < 5:       f.impact_minor  += 1
        elif value <= 20:   f.impact_medium += 1
        else:               f.impact_major  += 1

    for m in MULTIPLIER_RE.finditer(resume_text):
        value = float(m.group(1))
        if value >= 3:      f.impact_major  += 1
        elif value >= 2:    f.impact_medium += 1
        else:               f.impact_minor  += 1

    for m in DOLLAR_RE.finditer(resume_text):
        millions = _dollar_to_millions(m.group(1), m.group(2) or "")
        if millions < 0.01:     f.impact_minor  += 1
        elif millions <= 1:     f.impact_medium += 1
        else:                   f.impact_major  += 1

    if MENTORSHIP_OUTCOME_RE.search(resume_text):
        f.mentorship_outcomes += 1

    f.budget_owned     = bool(BUDGET_OWNERSHIP_RE.search(resume_text))
    f.cross_functional = bool(CROSS_FUNCTIONAL_RE.search(resume_text))
    f.multi_department = bool(MULTI_DEPT_RE.search(resume_text))
    f.global_scope     = bool(GLOBAL_RE.search(resume_text))

    text_lower = resume_text.lower()
    f.ownership_mentions = sum(1 for t in OWNERSHIP_TERMS if t in text_lower)

    for tier, pattern in TITLE_PATTERNS:
        if pattern.search(resume_text):
            f.title_tier = max(f.title_tier, tier)

    f.years_experience = _estimate_years(resume_text)

    f.technical_ic_flag = (
        f.initiative_verb_raw > 20 and
        f.team_size_max == 0 and
        f.influence_verb_hits <= 2
    )

    return f

# 7.  RAW DIMENSION SCORES  (before normalisation)

def _compute_raw_scores(f: LeadershipFeatures) -> dict[str, float]:
    """
    Compute raw dimension scores using the same logic as before.
    These are on their original scales (influence 0-20, impact 0-25, etc.)
    """
    # Influence (max 20)
    influence = min(f.influence_verb_hits, 3) * 3
    if f.team_size_max >= 10:   influence += 8
    elif f.team_size_max >= 4:  influence += 5
    elif f.team_size_max >= 1:  influence += 2
    influence = min(influence, MAX_SCORES["influence"])

    # Impact (max 25)
    impact = f.impact_minor * 2 + f.impact_medium * 4 + f.impact_major * 6
    impact = min(impact, MAX_SCORES["impact"])

    # Initiative (max 15) — dampened if IC flag
    initiative_pts = min(f.initiative_verb_hits, 3) * 5
    if f.technical_ic_flag:
        initiative_pts = round(initiative_pts * 0.5)
    initiative = min(initiative_pts, MAX_SCORES["initiative"])

    # Mentorship (max 15)
    mentorship = min(f.mentorship_instances, 3) * 4
    if f.mentorship_outcomes:
        mentorship += 3
    mentorship = min(mentorship, MAX_SCORES["mentorship"])

    # Scope/Scale (max 15)
    scope = (
        (5 if f.budget_owned     else 0) +
        (4 if f.global_scope     else 0) +
        (3 if f.cross_functional else 0) +
        (3 if f.multi_department else 0)
    )
    scope = min(scope, MAX_SCORES["scope_scale"])

    # Ownership (max 10)
    ownership = min(f.ownership_mentions, 3) * 3
    ownership = min(ownership, MAX_SCORES["ownership"])

    # Seniority (max 10)
    if f.years_experience >= 10:    exp_pts = 6
    elif f.years_experience >= 7:   exp_pts = 4
    elif f.years_experience >= 5:   exp_pts = 3
    elif f.years_experience >= 2:   exp_pts = 2
    elif f.years_experience >= 1:   exp_pts = 1
    else:                           exp_pts = 0
    title_pts = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}.get(f.title_tier, 0)
    seniority = min(exp_pts + title_pts, MAX_SCORES["seniority"])

    return {
        "influence":   influence,
        "impact":      impact,
        "initiative":  initiative,
        "mentorship":  mentorship,
        "scope_scale": scope,
        "ownership":   ownership,
        "seniority":   seniority,
    }

# 8.  SCORING MODEL  (normalise → weight → sum)

def score_resume(f: LeadershipFeatures) -> dict:
    """
    1. Compute raw scores per dimension (original scale)
    2. Normalise each to 0-1  (raw / max)
    3. Multiply by weight from weights.json
    4. Sum → final score 0-100

    Returns:
      score           – float 0–100
      grade           – A / B / C / D / F
      breakdown       – raw score, normalised score, weighted contribution per dimension
      features        – raw extracted feature values
      flags           – any warnings raised during scoring
      weights_version – which weights config was used
    """
    flags = []
    if f.technical_ic_flag:
        flags.append(
            "Technical IC profile detected: high initiative verb count with no "
            "people management signals. Initiative score dampened."
        )

    raw_scores = _compute_raw_scores(f)

    breakdown  = {}
    total      = 0.0

    for dim, raw in raw_scores.items():
        max_score  = MAX_SCORES[dim]
        normalised = raw / max_score if max_score > 0 else 0.0   # 0.0 – 1.0
        weight     = WEIGHTS[dim]                                  # from weights.json
        weighted   = round(normalised * weight, 2)

        breakdown[dim] = {
            "raw":        raw,           # e.g. 18  (out of 25)
            "max":        max_score,     # e.g. 25
            "normalised": round(normalised, 4),  # e.g. 0.72
            "weight":     weight,        # e.g. 25.0  (from weights.json)
            "weighted":   weighted,      # e.g. 18.0  (contributes to total)
        }
        total += weighted

    total = round(total / sum(WEIGHTS.values()) * 100, 1)

    grade = (
        "A" if total >= GRADE_THRESHOLDS["A"] else
        "B" if total >= GRADE_THRESHOLDS["B"] else
        "C" if total >= GRADE_THRESHOLDS["C"] else
        "D" if total >= GRADE_THRESHOLDS["D"] else
        "F"
    )

    return {
        "score":           total,
        "grade":           grade,
        "breakdown":       breakdown,
        "features":        asdict(f),
        "flags":           flags,
        "weights_version": _CONFIG.get("version", "unknown"),
    }

# 9.  PUBLIC API

def analyse_resume(resume_text: str) -> dict:
    """
    Main entry point.

    Usage:
        from leadership_scorer import analyse_resume
        result = analyse_resume(resume_text)

        result["score"]       # 0–100
        result["grade"]       # A/B/C/D/F
        result["breakdown"]   # per-dimension: raw, normalised, weight, weighted
    """
    features = extract_features(resume_text)
    return score_resume(features)
