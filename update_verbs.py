"""
update_verbs.py — AI-powered verb list updater
------------------------------------------------
Calls the Anthropic API to review and expand the three verb lists in verbs.json.
Run this manually once a month (or whenever resume language feels stale).

Usage:
    python update_verbs.py                  # updates verbs.json in place
    python update_verbs.py --dry-run        # prints proposed changes, saves nothing
    python update_verbs.py --list influence # updates one list only

Requires:
    pip install anthropic
    ANTHROPIC_API_KEY set in your environment
"""

import os
import json
import argparse
import datetime
from pathlib import Path
import anthropic

# ── CONFIG ─────────────────────────────────────────────────────────────────────

VERBS_PATH  = Path(__file__).parent / "verbs.json"
MODEL       = "claude-sonnet-4-20250514"
MAX_TOKENS  = 1024

# How many verbs max per list — keeps the lists focused, not bloated
MAX_VERBS_PER_LIST = 30

# ── PROMPTS PER LIST ───────────────────────────────────────────────────────────
# Each prompt explains the purpose and constraints of that specific list
# so the model understands *why* a verb qualifies, not just pattern-matches.

PROMPTS = {
    "INFLUENCE_VERBS": """
You are helping maintain a resume scoring system that detects leadership signals.

INFLUENCE_VERBS are verbs that show a person directly controlling or directing OTHER PEOPLE.
They are only counted when found near people-nouns (team, staff, engineers, reports etc).

Current list: {current}

Rules for a valid influence verb:
- Must imply authority over or responsibility for people (not things)
- Must be a base/infinitive form (lowercase)
- Examples of GOOD additions: "helm", "steer", "orchestrate", "captain"
- Examples of BAD additions: "optimise" (no people), "collaborate" (no authority), "support" (too passive)

Return an updated list with up to {max_verbs} verbs total.
Keep all existing verbs that still qualify.
Add new verbs only if they clearly signal people-leadership.
""",

    "INITIATIVE_VERBS": """
You are helping maintain a resume scoring system that detects leadership signals.

INITIATIVE_VERBS are verbs that show a person starting, building, or transforming something 
at an organisational level. They are only counted when found near org/programme nouns 
(project, strategy, roadmap, platform, initiative etc).

Current list: {current}

Rules for a valid initiative verb:
- Must imply proactive creation or transformation of something significant
- Must be a base/infinitive form (lowercase)
- Examples of GOOD additions: "architect", "conceive", "incubate", "stand up", "formulate"
- Examples of BAD additions: "update" (too minor), "attend" (passive), "use" (no initiative)

Return an updated list with up to {max_verbs} verbs total.
Keep all existing verbs that still qualify.
Add new verbs only if they clearly signal initiative at an org level.
""",

    "MENTORSHIP_VERBS": """
You are helping maintain a resume scoring system that detects leadership signals.

MENTORSHIP_VERBS are verbs that show a person developing or supporting the growth of others.

Current list: {current}

Rules for a valid mentorship verb:
- Must imply knowledge transfer, development, or support of another person's growth
- Must be a base/infinitive form (lowercase)
- Examples of GOOD additions: "sponsor", "develop", "nurture", "support", "tutor"
- Examples of BAD additions: "manage" (authority not development), "review" (evaluation not growth)

Return an updated list with up to {max_verbs} verbs total.
Keep all existing verbs that still qualify.
Add new verbs only if they clearly signal mentorship or people development.
"""
}

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a precise NLP assistant maintaining verb lists for a resume leadership scorer.
You must respond with ONLY a valid JSON array of lowercase strings — no explanation, 
no markdown, no preamble. Example: ["lead", "manage", "oversee"]
"""

# ── CORE FUNCTION ──────────────────────────────────────────────────────────────

def update_verb_list(
    client:    anthropic.Anthropic,
    list_name: str,
    current:   list[str],
    dry_run:   bool = False,
) -> list[str]:
    """
    Calls the API to review and expand one verb list.
    Returns the updated list (does not write to disk).
    """
    prompt = PROMPTS[list_name].format(
        current   = json.dumps(current),
        max_verbs = MAX_VERBS_PER_LIST,
    )

    print(f"\n[update_verbs] Calling API for {list_name}...")

    response = client.messages.create(
        model      = MODEL,
        max_tokens = MAX_TOKENS,
        system     = SYSTEM_PROMPT.strip(),
        messages   = [{"role": "user", "content": prompt.strip()}],
    )

    raw = response.content[0].text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        updated = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[update_verbs] ⚠️  Could not parse API response for {list_name}: {e}")
        print(f"[update_verbs]    Raw response: {raw[:200]}")
        print(f"[update_verbs]    Keeping existing list unchanged.")
        return current

    # Validate — must be a list of strings
    if not isinstance(updated, list) or not all(isinstance(v, str) for v in updated):
        print(f"[update_verbs] ⚠️  Unexpected format for {list_name}. Keeping existing list.")
        return current

    # Normalise to lowercase, strip whitespace, deduplicate
    updated = list(dict.fromkeys(v.strip().lower() for v in updated))

    # Show diff
    added   = sorted(set(updated) - set(current))
    removed = sorted(set(current) - set(updated))
    print(f"  ✅ {list_name}")
    print(f"     Added   ({len(added)})  : {added if added else '—'}")
    print(f"     Removed ({len(removed)}): {removed if removed else '—'}")

    return updated


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Update verb lists via AI")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print proposed changes without saving")
    parser.add_argument("--list", choices=["influence", "initiative", "mentorship"],
                        help="Update one list only (default: all three)")
    args = parser.parse_args()

    # ── Load current verbs ────────────────────────────────────────────
    if not VERBS_PATH.exists():
        print(f"[update_verbs] ERROR: {VERBS_PATH} not found. "
              "Make sure verbs.json is in the same directory.")
        return

    with open(VERBS_PATH, encoding="utf-8") as f:
        verbs_data = json.load(f)

    # ── Decide which lists to update ──────────────────────────────────
    list_map = {
        "influence":  "INFLUENCE_VERBS",
        "initiative": "INITIATIVE_VERBS",
        "mentorship": "MENTORSHIP_VERBS",
    }
    targets = (
        [list_map[args.list]] if args.list
        else list(list_map.values())
    )

    # ── Call API ──────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[update_verbs] ERROR: ANTHROPIC_API_KEY environment variable not set.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    updated_data = dict(verbs_data)  # shallow copy

    for list_name in targets:
        current = verbs_data.get(list_name, [])
        updated = update_verb_list(client, list_name, current, dry_run=args.dry_run)
        updated_data[list_name] = updated

    # ── Save ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[update_verbs] Dry run — no changes saved.")
        print("\nProposed verbs.json:")
        print(json.dumps(updated_data, indent=2))
    else:
        updated_data["_meta"] = {
            "last_updated": datetime.date.today().isoformat(),
            "updated_by":   f"update_verbs.py — lists updated: {', '.join(targets)}",
        }
        with open(VERBS_PATH, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, indent=2)
        print(f"\n[update_verbs] ✅ verbs.json saved to {VERBS_PATH}")


if __name__ == "__main__":
    main()
