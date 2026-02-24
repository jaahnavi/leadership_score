"""
Leadership Scorer — V4
-----------------------
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

"""

import re
import spacy
from dataclasses import dataclass, asdict

nlp = spacy.load("en_core_web_sm")


# 1.  FEATURE DATACLASS

@dataclass
class LeadershipFeatures:
    # ── Influence ──────────────────────────────────────────────────────
    influence_verb_hits: int = 0      # people-gated (managed a team, not managed data)
    team_size_max: int       = 0

    # ── Impact ─────────────────────────────────────────────────────────
    impact_minor: int  = 0
    impact_medium: int = 0
    impact_major: int  = 0

    # ── Initiative ─────────────────────────────────────────────────────
    initiative_verb_hits: int    = 0  # context-gated (org/programme noun nearby)
    initiative_verb_raw: int     = 0  # raw count before gating (for inflation flag)

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
    title_tier: int      = 0   # 0=none, 1=senior IC, 2=manager, 3=director, 4=VP+
    years_experience: int = 0  # parsed from resume text

    # ── Flags ──────────────────────────────────────────────────────────
    technical_ic_flag: bool = False   # True if verb inflation detected


# 2.  VERB SETS
#use a function to get updated list for the verbs using AI, can run it once a month 
INFLUENCE_VERBS = {
    "lead", "manage", "head", "direct", "oversee",
    "supervise", "run", "coordinate", "govern", "command",
}

# Context-gated initiative verbs — only count when near org/programme nouns
INITIATIVE_VERBS = {
    "found", "create", "launch", "build", "establish", "initiate",
    "design", "develop", "spearhead", "champion", "transform", "scale",
    "drive", "introduce", "pioneer", "restructure", "overhaul",
    "execute", "deliver",
}

MENTORSHIP_VERBS = {
    "mentor", "train", "coach", "teach", "guide",
    "onboard", "upskill", "educate", "advise",
}

# ── Context gate nouns ─────────────────────────────────────────────────
# Influence: verb only counts if one of these is within ±10 tokens
PEOPLE_NOUNS = {
    "team", "staff", "engineer", "engineers", "employee", "employees",
    "member", "members", "report", "reports", "headcount", "person",
    "people", "hire", "hires", "analyst", "analysts", "associate",
    "associates", "developer", "developers", "manager", "managers",
    "intern", "interns", "colleague", "colleagues", "volunteer",
}

# Initiative: verb only counts if one of these is within ±12 tokens
ORG_NOUNS = {
    "programme", "program", "initiative", "project", "organisation",
    "organization", "department", "division", "unit", "function",
    "strategy", "roadmap", "platform", "product", "guild", "chapter",
    "framework", "process", "pipeline", "system", "team", "org",
    "company", "business", "operation", "operations", "centre", "center",
    "campaign", "committee", "board", "council", "practice",
}

# 3.  REGEX PATTERNS

# Team size
TEAM_SIZE_RE = re.compile(
    r"(?:team|staff|engineers?|employees?|members?|reports?|headcount)\s+of\s+(\d+)"
    r"|(?:managed|led|supervised|oversaw)\s+(?:a\s+)?(?:team\s+of\s+)?(\d+)",
    re.IGNORECASE,
)

# Impact — percentage: handles optional ~ prefix and optional space
PERCENT_RE = re.compile(r"~?\s*(\d+(?:\.\d+)?)\s*%")

# Impact — multipliers: 3x, 10x growth etc.
MULTIPLIER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*x\b", re.IGNORECASE)

# Impact — dollar figures
DOLLAR_RE = re.compile(
    r"\$\s*([\d,.]+)\s*(k|m|b|million|billion)?",
    re.IGNORECASE,
)

# Budget ownership — verb must appear within 40 chars before "budget"
BUDGET_OWNERSHIP_RE = re.compile(
    r"(?:managed|oversaw|owned|responsible\s+for|accountable\s+for|controlled)"
    r".{0,40}\bbudget\b",
    re.IGNORECASE,
)

CROSS_FUNCTIONAL_RE = re.compile(
    r"cross[\s-]functional|cross[\s-]team|cross[\s-]department|matrixed?",
    re.IGNORECASE,
)
MULTI_DEPT_RE = re.compile(
    r"multi[\s-]department|multiple\s+(?:teams?|departments?|divisions?|units?)"
    r"|stakeholders?|organisation[\s-]wide|company[\s-]wide|enterprise[\s-]wide",
    re.IGNORECASE,
)
GLOBAL_RE = re.compile(
    r"\bglobal\b|\binternational\b|\bworldwide\b|\bmultinational\b",
    re.IGNORECASE,
)

OWNERSHIP_TERMS = {
    "responsible", "accountable", "owned", "ownership",
    "end-to-end", "end to end",
}

MENTORSHIP_OUTCOME_RE = re.compile(
    r"promoted|advanced\s+to|improved\s+performance|retained|progressed",
    re.IGNORECASE,
)

# Seniority title detection
TITLE_PATTERNS = [
    # Tier 4 — VP and above
    (4, re.compile(
        r"\b(vp|vice\s+president|svp|evp|chief\s+\w+\s+officer|c[eotf]o|"
        r"president|founder|co[\s-]founder|managing\s+director|partner)\b",
        re.IGNORECASE,
    )),
    # Tier 3 — Director
    (3, re.compile(
        r"\b(director|head\s+of|group\s+manager|general\s+manager|"
        r"principal\s+manager)\b",
        re.IGNORECASE,
    )),
    # Tier 2 — Manager
    (2, re.compile(
        r"\b(manager|team\s+lead|team\s+leader|lead\s+\w+|"
        r"engineering\s+lead|product\s+lead|delivery\s+lead)\b",
        re.IGNORECASE,
    )),
    # Tier 1 — Senior IC
    (1, re.compile(
        r"\b(senior|sr\.?|principal|staff\s+engineer|staff\s+scientist|"
        r"specialist|consultant|advisor)\b",
        re.IGNORECASE,
    )),
]

# Years of experience: "10+ years", "3 years of experience"
YEARS_EXP_RE = re.compile(
    r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)",
    re.IGNORECASE,
)

# Date range for counting years: "2015 – 2024", "June 2021 - July 2024"
DATE_RANGE_RE = re.compile(
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)?\s*(20\d{2}|19\d{2})\s*[-–]\s*"
    r"(20\d{2}|19\d{2}|present|current|now)",
    re.IGNORECASE,
)


# 4.  HELPERS

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
        value /= 1_000_000   # bare $ amount — treat as raw dollars
    return value


def _window_contains(tokens, centre_idx: int, word_set: set, window: int) -> bool:
    """Return True if any token within ±window of centre_idx is in word_set."""
    start = max(0, centre_idx - window)
    end   = min(len(tokens), centre_idx + window + 1)
    return any(tokens[i].lemma_.lower() in word_set or
               tokens[i].text.lower()  in word_set
               for i in range(start, end) if i != centre_idx)


def _estimate_years(resume_text: str) -> int:
    """
    Estimate total years of experience.
    Priority: explicit "X years of experience" statement.
    Fallback: span from earliest to latest year mentioned in date ranges.
    """
    # Explicit statement
    matches = YEARS_EXP_RE.findall(resume_text)
    if matches:
        return max(int(m) for m in matches)

    # Fallback: date range span
    import datetime
    current_year = datetime.datetime.now().year
    years = []
    for m in DATE_RANGE_RE.finditer(resume_text):
        start_yr = int(m.group(1))
        end_raw  = m.group(2).lower()
        end_yr   = current_year if end_raw in ("present", "current", "now") else int(end_raw)
        years.extend([start_yr, end_yr])
    if years:
        return current_year - min(years)
    return 0


# 5.  FEATURE EXTRACTION


def extract_features(resume_text: str) -> LeadershipFeatures:
    f = LeadershipFeatures()
    doc = nlp(resume_text)
    tokens = list(doc)

    for i, token in enumerate(tokens):
        if token.pos_ != "VERB":
            continue
        lemma = token.lemma_.lower()

        # ── Influence: only count if a PEOPLE noun is within ±10 tokens ──
        if lemma in INFLUENCE_VERBS:
            if _window_contains(tokens, i, PEOPLE_NOUNS, window=10):
                f.influence_verb_hits += 1

        # ── Initiative: raw count + context-gated count ────────────────
        if lemma in INITIATIVE_VERBS:
            f.initiative_verb_raw += 1
            if _window_contains(tokens, i, ORG_NOUNS, window=12):
                f.initiative_verb_hits += 1

        # ── Mentorship ─────────────────────────────────────────────────
        if lemma in MENTORSHIP_VERBS:
            f.mentorship_instances += 1

    # ── Team size ──────────────────────────────────────────────────────
    sizes = []
    for m in TEAM_SIZE_RE.finditer(resume_text):
        num = m.group(1) or m.group(2)
        if num:
            sizes.append(int(num))
    if sizes:
        f.team_size_max = max(sizes)

    # ── Impact — percentages ───────────────────────────────────────────
    for m in PERCENT_RE.finditer(resume_text):
        value = float(m.group(1))
        if value < 5:
            f.impact_minor += 1
        elif value <= 20:
            f.impact_medium += 1
        else:
            f.impact_major += 1

    # ── Impact — multipliers (3x = major, 2x = medium, <2 = minor) ────
    for m in MULTIPLIER_RE.finditer(resume_text):
        value = float(m.group(1))
        if value >= 3:
            f.impact_major += 1
        elif value >= 2:
            f.impact_medium += 1
        else:
            f.impact_minor += 1

    # ── Impact — dollar figures ────────────────────────────────────────
    for m in DOLLAR_RE.finditer(resume_text):
        millions = _dollar_to_millions(m.group(1), m.group(2) or "")
        if millions < 0.01:
            f.impact_minor += 1
        elif millions <= 1:
            f.impact_medium += 1
        else:
            f.impact_major += 1

    # ── Mentorship outcome bonus ────────────────────────────────────────
    if MENTORSHIP_OUTCOME_RE.search(resume_text):
        f.mentorship_outcomes += 1

    # ── Scope / Scale ──────────────────────────────────────────────────
    f.budget_owned     = bool(BUDGET_OWNERSHIP_RE.search(resume_text))
    f.cross_functional = bool(CROSS_FUNCTIONAL_RE.search(resume_text))
    f.multi_department = bool(MULTI_DEPT_RE.search(resume_text))
    f.global_scope     = bool(GLOBAL_RE.search(resume_text))

    # ── Ownership ──────────────────────────────────────────────────────
    text_lower = resume_text.lower()
    f.ownership_mentions = sum(1 for t in OWNERSHIP_TERMS if t in text_lower)

    # ── Seniority — title tier ─────────────────────────────────────────
    for tier, pattern in TITLE_PATTERNS:
        if pattern.search(resume_text):
            f.title_tier = max(f.title_tier, tier)

    # ── Seniority — years of experience ───────────────────────────────
    f.years_experience = _estimate_years(resume_text)

    # ── Technical IC inflation flag ────────────────────────────────────
    f.technical_ic_flag = (
        f.initiative_verb_raw > 20 and
        f.team_size_max == 0 and
        f.influence_verb_hits <= 2
    )

    return f
#auditing or analytics or metrics to monitor the scoring of the model ,  

# 6.  SCORING MODEL

def score_resume(f: LeadershipFeatures) -> dict:
    """
    Returns:
      score      – float 0–100
      grade      – A / B / C / D / F
      breakdown  – points per dimension
      features   – raw extracted feature values
      flags      – any warnings raised during scoring
    """
    flags = []
    if f.technical_ic_flag:
        flags.append(
            "Technical IC profile detected: high initiative verb count with no "
            "people management signals. Initiative score dampened."
        )

    # 1️⃣  Influence (max 20)
    influence = min(f.influence_verb_hits, 3) * 3
    if f.team_size_max >= 10:
        influence += 8
    elif f.team_size_max >= 4:
        influence += 5
    elif f.team_size_max >= 1:
        influence += 2
    influence = min(influence, 20)

    # 2️⃣  Impact (max 25)
    impact = (
        f.impact_minor  * 2 +
        f.impact_medium * 4 +
        f.impact_major  * 6
    )
    impact = min(impact, 25)

    # 3️⃣  Initiative (max 15)
    #   Dampened to 50% if technical IC flag raised
    initiative_pts = min(f.initiative_verb_hits, 3) * 5
    if f.technical_ic_flag:
        initiative_pts = round(initiative_pts * 0.5)
    initiative = min(initiative_pts, 15)

    # 4️⃣  Mentorship (max 15)
    mentorship = min(f.mentorship_instances, 3) * 4
    if f.mentorship_outcomes:
        mentorship += 3
    mentorship = min(mentorship, 15)

    # 5️⃣  Scope / Scale (max 15)
    scope = (
        (5 if f.budget_owned     else 0) +
        (4 if f.global_scope     else 0) +
        (3 if f.cross_functional else 0) +
        (3 if f.multi_department else 0)
    )
    scope = min(scope, 15)

    # 6️⃣  Ownership (max 10)
    ownership = min(f.ownership_mentions, 3) * 3
    ownership = min(ownership, 10)

    # 7️⃣  Seniority (max 10)
    #   Years experience → up to 6 pts (10yr alone hits 6)
    #   Title tier (0–4) → 0/1/2/3/4 pts on top (VP+ pushes to 10)
    if f.years_experience >= 10:
        exp_pts = 6
    elif f.years_experience >= 7:
        exp_pts = 4
    elif f.years_experience >= 5:
        exp_pts = 3
    elif f.years_experience >= 2:
        exp_pts = 2
    elif f.years_experience >= 1:
        exp_pts = 1
    else:
        exp_pts = 0

    title_pts = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}.get(f.title_tier, 0)
    seniority = min(exp_pts + title_pts, 10)

    total = round(
        influence + impact + initiative + mentorship + scope + ownership + seniority,
        1,
    )

    grade = (
        "A" if total >= 85 else
        "B" if total >= 70 else
        "C" if total >= 55 else
        "D" if total >= 40 else
        "F"
    )

    breakdown = {
        "1_influence  (max 20)": influence,
        "2_impact     (max 25)": impact,
        "3_initiative (max 15)": initiative,
        "4_mentorship (max 15)": mentorship,
        "5_scope      (max 15)": scope,
        "6_ownership  (max 10)": ownership,
        "7_seniority  (max 10)": seniority,
    }

    return {
        "score":     total,
        "grade":     grade,
        "breakdown": breakdown,
        "features":  asdict(f),
        "flags":     flags,
    }

# 7.  PUBLIC API

def analyse_resume(resume_text: str) -> dict:
    """
    Main entry point — plug straight into your PDF/DOCX extractor.

    Usage:
        from leadership_scorer import analyse_resume
        result = analyse_resume(resume_text)
    """
    features = extract_features(resume_text)
    return score_resume(features)


# 8.  TEST AGAINST BOTH RESUMES

if __name__ == "__main__":
    import json, docx

    def read_docx(path):
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    resumes = {
        "ExecutiveResume (Adithya — Salesforce Dev)": "/mnt/user-data/uploads/ExecutiveResume.docx",
        "ExampleResume1  (Jaahnavi — Amazon Manager)": "/mnt/user-data/uploads/ExampleResume1.docx",
    }

    for label, path in resumes.items():
        text   = read_docx(path)
        result = analyse_resume(text)
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(json.dumps({
            "score":     result["score"],
            "grade":     result["grade"],
            "breakdown": result["breakdown"],
            "flags":     result["flags"],
            "key_features": {
                k: result["features"][k] for k in [
                    "influence_verb_hits", "team_size_max",
                    "impact_major", "impact_medium",
                    "initiative_verb_hits", "initiative_verb_raw",
                    "mentorship_instances", "title_tier",
                    "years_experience", "technical_ic_flag",
                ]
            }
        }, indent=2))
