from pathlib import Path
import sys
import argparse
import sqlite3
import hashlib
import datetime
import pdfplumber
from pypdf import PdfReader
from docx import Document
from leadership_scorer import analyse_resume
import json

# install
# python -m spacy download en_core_web_sm

# ── Database ───────────────────────────────────────────────────────────────────

DB_FILE = "leadership_scores.db"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS scores (
    resume_id         TEXT PRIMARY KEY,
    filepath          TEXT,
    file_hash         TEXT,
    scored_at         TEXT,
    weights_version   TEXT,

    -- Normalised scores (0.0 - 1.0)
    influence_norm    REAL,
    impact_norm       REAL,
    initiative_norm   REAL,
    mentorship_norm   REAL,
    scope_scale_norm  REAL,
    ownership_norm    REAL,
    seniority_norm    REAL,

    -- Final output
    total_score       REAL,
    grade             TEXT
);
"""

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.execute(CREATE_TABLE)
    conn.commit()
    return conn

def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def save_to_db(conn: sqlite3.Connection, resume_id: str, filepath: str,
               fhash: str, result: dict):
    bd  = result["breakdown"]   # keys: influence, impact, initiative, etc.
                                # each value: {raw, max, normalised, weight, weighted}
    now = datetime.datetime.utcnow().isoformat()

    conn.execute("""
        INSERT INTO scores (
            resume_id, filepath, file_hash, scored_at, weights_version,
            influence_norm,   impact_norm,   initiative_norm,
            mentorship_norm,  scope_scale_norm, ownership_norm, seniority_norm,
            total_score, grade
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(resume_id) DO UPDATE SET
            filepath          = excluded.filepath,
            file_hash         = excluded.file_hash,
            scored_at         = excluded.scored_at,
            weights_version   = excluded.weights_version,
            influence_norm    = excluded.influence_norm,
            impact_norm       = excluded.impact_norm,
            initiative_norm   = excluded.initiative_norm,
            mentorship_norm   = excluded.mentorship_norm,
            scope_scale_norm  = excluded.scope_scale_norm,
            ownership_norm    = excluded.ownership_norm,
            seniority_norm    = excluded.seniority_norm,
            total_score       = excluded.total_score,
            grade             = excluded.grade
    """, (
        resume_id, filepath, fhash, now,
        result.get("weights_version", "unknown"),
        # normalised
        bd["influence"]["normalised"],    bd["impact"]["normalised"],
        bd["initiative"]["normalised"],   bd["mentorship"]["normalised"],
        bd["scope_scale"]["normalised"],  bd["ownership"]["normalised"],
        bd["seniority"]["normalised"],
        # final
        result["score"],
        result["grade"],
    ))
    conn.commit()


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: Path) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(pages)
        if text.strip():
            return text
    except Exception:
        pass

    try:
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed for {path}: {e}")

def extract_text_from_docx(path: Path) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    raise ValueError(f"Unsupported file type: {path}")


# ── Core processing ────────────────────────────────────────────────────────────

def process_single(path: Path, out_dir: Path | None,
                   conn: sqlite3.Connection | None, force: bool = False):
    """Score one file, save to DB and optionally write JSON."""
    resume_id = path.stem
    fhash     = file_hash(path)

    # Skip if file is unchanged since last run
    if conn and not force:
        existing = conn.execute(
            "SELECT file_hash FROM scores WHERE resume_id = ?", (resume_id,)
        ).fetchone()
        if existing and existing[0] == fhash:
            print(f"  SKIP   {resume_id}  (unchanged — use --force to rescore)")
            return

    text   = parse_file(path)
    result = analyse_resume(text)

    # Save to DB
    if conn:
        save_to_db(conn, resume_id, str(path.resolve()), fhash, result)

    # Optionally write per-resume JSON
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / (path.stem + "_leadership.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  SCORED {resume_id:<35} {result['score']:>5.1f}  {result['grade']}  → {out_file}")
    else:
        print(f"  SCORED {resume_id:<35} {result['score']:>5.1f}  {result['grade']}")


def process_path(input_path: Path, out_dir: Path | None,
                 use_db: bool = True, force: bool = False):

    conn = get_db() if use_db else None

    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in (".pdf", ".docx")
        )
        if not files:
            print(f"No PDF or DOCX files found in '{input_path}'")
            return

        print(f"Found {len(files)} resume(s) in '{input_path}'\n")
        for p in files:
            try:
                process_single(p, out_dir, conn, force=force)
            except Exception as e:
                print(f"  ERROR  {p.name}: {e}", file=sys.stderr)

        # Print ranked summary table after batch run
        if conn:
            print("\n" + "-"*75)
            rows = conn.execute("""
                SELECT resume_id,
                       influence_norm, impact_norm, initiative_norm, mentorship_norm,
                       scope_scale_norm, ownership_norm, seniority_norm,
                       total_score, grade
                FROM scores ORDER BY total_score DESC
            """).fetchall()
            print(f"{'Resume':<35} {'Inf':>4} {'Imp':>4} {'Ini':>4} {'Men':>4} "
                  f"{'Sco':>4} {'Own':>4} {'Sen':>4} {'Total':>6} {'Gr':>3}")
            print("─"*75)
            for r in rows:
                print(f"{r[0]:<35} {r[1]:>4.0f} {r[2]:>4.0f} {r[3]:>4.0f} "
                      f"{r[4]:>4.0f} {r[5]:>4.0f} {r[6]:>4.0f} {r[7]:>4.0f} "
                      f"{r[8]:>6.1f} {r[9]:>3}")
            print(f"\nDatabase: {DB_FILE}  ({len(rows)} resume(s) total)")

    elif input_path.is_file():
        try:
            process_single(input_path, out_dir, conn, force=force)
            if conn:
                print(f"\nDatabase: {DB_FILE}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

    if conn:
        conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score resumes for leadership signals.",
        epilog="""
Examples:
  python parser.py resume/                    score all resumes, save to DB
  python parser.py resume/ --out results      also write per-resume JSON
  python parser.py resume/ --force            rescore even unchanged files
  python parser.py SrExecSample.pdf           single file (original usage)
  python parser.py SrExecSample.pdf --out results
        """
    )
    parser.add_argument("input",        help="Resume file or folder")
    parser.add_argument("--out", "-o",  help="Output directory for per-resume JSON")
    parser.add_argument("--force",      action="store_true",
                        help="Rescore all files even if unchanged")
    parser.add_argument("--no-db",      action="store_true",
                        help="Skip DB storage (prints JSON only, original behaviour)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.out) if args.out else None
    process_path(input_path, out_dir, use_db=not args.no_db, force=args.force)

if __name__ == "__main__":
    main()
