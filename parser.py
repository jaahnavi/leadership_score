from pathlib import Path
import sys
import argparse
import pdfplumber
from pypdf import PdfReader
from docx import Document
from leadership_scorer import analyse_resume
import json

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
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    raise ValueError(f"Unsupported file type: {path}")

def process_path(input_path: Path, out_dir: Path | None):
    if input_path.is_dir():
        for p in sorted(input_path.iterdir()):
            if p.suffix.lower() in (".pdf", ".docx"):
                try:
                    text = parse_file(p)
                    if out_dir:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_file = out_dir / (p.stem + ".txt")
                        out_file.write_text(text, encoding="utf-8")
                        print(f"Wrote: {out_file}")
                    else:
                        print(f"--- {p.name} ---")
                        print(text)
                except Exception as e:
                    print(f"Error processing {p}: {e}", file=sys.stderr)
    elif input_path.is_file():
        try:
            text = parse_file(input_path)

            result = analyse_resume(text)

            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / (input_path.stem + "_leadership.json")
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

                print(f"Leadership score written to: {out_file}")
            else:
                print(json.dumps(result, indent=2))
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF and DOCX files.")
    parser.add_argument("input", help="File or directory to parse")
    parser.add_argument("--out", "-o", help="Output directory for .txt files")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out) if args.out else None
    process_path(input_path, out_dir)

if __name__ == "__main__":
    main()