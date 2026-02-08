"""
Run segmentation pipeline for a textbook (full book or single chapter).
Extensible: add entries to BOOKS and use --book <key> or override with --pdf/--output.
"""
import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from synapta_segmenter import Pipeline

# --- Book config: add new textbooks here for easy switching ---
BOOKS = {
    "investments": {
        "pdf_path": "data/Investments.pdf",
        "concept_list_path": "data/Segmentation_Zvi Bodie, Alex Kane, Alan J. Marcus - Investments (2023, McGraw Hill).xlsx",
        "book_id": "Investments",
        "chapter_page_ranges": {
            "9": (312, 343),
            # Add more: "1": (1, 50), "2": (51, 100), ...
        },
    },
    # Example for another textbook:
    # "other_book": {
    #     "pdf_path": "data/OtherBook.pdf",
    #     "concept_list_path": "data/OtherBook_concepts.xlsx",
    #     "book_id": "OtherBook",
    #     "chapter_page_ranges": {"1": (1, 40), "2": (41, 80)},
    # },
}


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(os.path.dirname(__file__), path)


def parse_args():
    p = argparse.ArgumentParser(description="Run segmentation pipeline (full book or single chapter).")
    p.add_argument("--book", default="investments", choices=list(BOOKS.keys()),
                   help="Book config key from BOOKS (default: investments)")
    p.add_argument("--full-book", action="store_true", help="Process entire PDF (no page range, multi-chapter)")
    p.add_argument("--chapter", type=str, metavar="N", help="Process single chapter N (uses chapter_page_ranges or --pages)")
    p.add_argument("--pages", type=int, nargs=2, metavar=("START", "END"), help="Page range START END (overrides chapter_page_ranges)")
    p.add_argument("--pdf", type=str, help="Override PDF path")
    p.add_argument("--output", type=str, help="Override output JSON path")
    p.add_argument("--output-dir", type=str, default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--concept-list", type=str, help="Override concept list path (xlsx)")
    p.add_argument("--book-id", type=str, help="Override book_id in output metadata")

    # LLM / pipeline options (keep existing behavior)
    p.add_argument("llm_mode", nargs="?", default="light", choices=["light", "full", "off"],
                   help="LLM mode: light | full | off")
    p.add_argument("backend", nargs="?", default=None,
                   choices=["openai", "gemini", "claude", "openrouter", "ollama", "groq"],
                   help="LLM backend (optional positional, e.g. light claude)")
    p.add_argument("--llm", default="openai", choices=["openai", "gemini", "claude", "openrouter", "ollama", "groq"],
                   help="LLM backend (overrides positional backend)")
    p.add_argument("--no-solutions", action="store_true", help="Do not generate solutions (faster; good for first full-book pass)")
    p.add_argument("--no-validate", action="store_true", help="Do not validate generated solutions")
    p.add_argument("--validate-top", type=int, default=None, metavar="N", help="Validate only top N solution candidates per question (default: all; use 1 for full book to save time)")
    p.add_argument("--batch-size", type=int, default=None, help="Override LLM batch size")
    p.add_argument("--no-lexicon", action="store_true", help="Disable variable lexicon")
    p.add_argument("--clear-cache", action="store_true", help="Delete outputs/.cache (solution cache) before run; variable/LaTeX caches are in-memory only")
    return p.parse_args()


def main():
    args = parse_args()
    book_key = args.book
    if book_key not in BOOKS:
        print(f"Error: unknown book '{book_key}'. Available: {list(BOOKS.keys())}")
        sys.exit(1)

    cfg = BOOKS[book_key].copy()
    pdf_path = _resolve_path(args.pdf or cfg["pdf_path"])
    concept_list_path = args.concept_list or cfg.get("concept_list_path")
    if concept_list_path and not os.path.isabs(concept_list_path):
        concept_list_path = _resolve_path(concept_list_path)
    book_id = args.book_id or cfg.get("book_id", book_key)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Determine scope: full book vs single chapter
    full_book = args.full_book
    chapter = args.chapter
    page_range = None

    if full_book and chapter:
        print("Error: use either --full-book or --chapter N, not both.")
        sys.exit(1)
    if not full_book and not chapter:
        # Default: single chapter 9 for investments (backward compatible)
        chapter = "9"
        page_range = cfg.get("chapter_page_ranges", {}).get(chapter)

    if chapter:
        if args.pages is not None:
            page_range = tuple(args.pages)
        else:
            page_range = cfg.get("chapter_page_ranges", {}).get(chapter)
        if page_range is None:
            print(f"Error: no page range for chapter '{chapter}'. Use --pages START END or add chapter_page_ranges in BOOKS.")
            sys.exit(1)

    # Output path
    if args.output:
        output_path = args.output
    else:
        if full_book or not chapter:
            out_name = f"{book_id}_segments.json"
        else:
            out_name = f"{book_id}_chapter{chapter}_segments.json"
        output_path = os.path.join(output_dir, out_name)

    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)
    if concept_list_path and os.path.exists(concept_list_path):
        print(f"Using concept list: {concept_list_path}")

    # Clear solution cache on request (variable/LaTeX caches are in-memory only, so each run is fresh)
    if getattr(args, "clear_cache", False):
        cache_dir = os.path.join(output_dir, ".cache")
        if os.path.isdir(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Cleared cache: {cache_dir}")
        elif os.path.exists(os.path.join(output_dir, ".cache")):
            pass
        else:
            print("No cache directory to clear (outputs/.cache)")

    # LLM options (positional backend e.g. "light claude" overrides --llm when given)
    llm_mode = args.llm_mode
    llm_backend = args.llm if getattr(args, "backend", None) is None else args.backend
    generate_solutions = not args.no_solutions
    validate_solutions = not args.no_validate
    use_variable_lexicon = not args.no_lexicon
    custom_batch_size = args.batch_size

    if custom_batch_size is not None:
        batch_size = custom_batch_size
    elif llm_mode == "off":
        batch_size = None
    elif llm_backend == "ollama":
        batch_size = 10
        validate_solutions = False
    elif llm_backend == "groq":
        batch_size = 8
    elif llm_backend == "openrouter":
        batch_size = 1
    elif llm_backend == "gemini":
        batch_size = 3
    else:
        batch_size = 10

    print(f"LLM mode={llm_mode} backend={llm_backend} generate_solutions={generate_solutions} validate_solutions={validate_solutions} batch_size={batch_size}")
    if full_book:
        print("Scope: full book (all pages)")
        if generate_solutions:
            print("Tip: Full book + solutions can take 30–90+ min. For a faster first pass, use --no-solutions then re-run with solutions for specific chapters.")
    else:
        print(f"Scope: chapter {chapter} pages {page_range[0]}-{page_range[1]}")
    print(f"Output: {output_path}")

    pipeline = Pipeline(
        concept_list_path=concept_list_path,
        target_chapter=chapter if not full_book else None,
        llm_mode=llm_mode,
        generate_missing_solutions=generate_solutions,
        validate_generated_solutions=validate_solutions,
        batch_size=batch_size,
        llm_backend=llm_backend,
        use_variable_lexicon=use_variable_lexicon,
        validate_top_n_only=args.validate_top,
        segment_type_allowlist={
            "formula", "derivation", "calculation",
            "worked_example", "question", "solution",
        },
        reference_stub_types=set(),
        enable_llm_disambiguation=True,
        llm_disambiguation_batch_size=8,
    )
    pipeline.run(
        pdf_path,
        output_path,
        page_range=page_range,
        target_chapter=None if full_book else chapter,
        chapter_mode="multi_chapter" if full_book else "single_chapter",
        doc_uri=os.path.abspath(pdf_path),
        book_id=book_id,
    )


if __name__ == "__main__":
    main()
