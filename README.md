# Formula & Example / Q&A Segmentation Pipeline — Technical Documentation

## 1. Overview

This pipeline extracts and structures **formula-driven learning objects** from textbook PDFs for Synapta’s knowledge graph and RAG flows. It focuses on: formulas and equations, variable dictionaries, derivations and calculation blocks, worked examples, end-of-chapter questions and solutions, and their linkage to a human-curated concept taxonomy (TSV/Excel). Unlike generic text chunking, every output segment carries **full traceability** (book, chapter, page, heading path, bbox) and **typed edges** (e.g. USES_FORMULA, ANSWER_OF, WORKED_EXAMPLE_OF) so downstream retrieval and tutoring can follow concept → formula → example → question → solution deterministically.

**Scope (this module):** Formula, derivation, calculation, worked example, question, solution. Text/table/figure extraction is handled by other modules; this repo can emit reference stubs for cross-module linking when enabled.

---

## 2. Pipeline Architecture

The processing flow is orchestrated in `run_sample.py` and implemented in `synapta_segmenter.py`:

1. **PDF ingestion & block cleaning** — PyMuPDF block extraction; furniture (headers/footers) filtered via sampled frequency and pattern rules.
2. **Formula extraction** — Layout + symbol-density heuristics; equation numbers; optional LLM-based variable extraction and LaTeX conversion (batched).
3. **Text-block extraction** — Headers, worked examples, derivations, calculations, questions, solutions, concept checks; section state machine (problem_set / cfa / concept_check / concept_check_solutions).
4. **Context & adjacency** — Heading paths, prev/next segment IDs, context_before/context_after.
5. **Linking** — Question–solution pairing (exact + fuzzy); formula ↔ example bidirectional links; derivation/calculation → formula; concept linking via ConceptLinker.
6. **Edge building** — Typed KG edges (DEFINES, EXPLAINS, USES_FORMULA, WORKED_EXAMPLE_OF, REFERENCES, NEAR, ANSWER_OF, DUPLICATE_OF) with anchor metadata.
7. **Optional synthetic solutions** — For questions with no in-book solution: LLM candidate generation (batched by chapter) and validation (formula/variable alignment, correctness).

---

## 3. Methodology Details

### 3.1 Formula Detection & Classification

- **Detection:** Combination of layout (centered lines, spacing), symbol density (`=`, ∑, √, σ, subscripts/superscripts, Greek letters), and cue phrases (“Eq.”, “Equation”, “where:”, “let:”).
- **Classification:** Each formula is labeled as **definition** (introduced with variable definitions), **application** (used in example/calculation), or **reference** (cited by label, e.g. “use Eq. 3.4”).
- **Equation labels:** Extracted when present, e.g. `(2.3)`, `Eq. 5.1`, and used for cross-references and deduplication.

### 3.2 Formula Normalization & Variable Dictionary

- **Output fields:** `formula_text_raw`, `formula_latex` (best-effort, LLM or heuristic), `canonical_formula_key` (stable hash for duplicate resolution), `variables[]` (symbol, meaning, units, inferred flag), `short_meaning`, confidence and `needs_human_review`.
- **Variable extraction:** Heuristics suggest candidate symbols from formula and surrounding text; **LLM is the primary engine** for meaning/units (heuristics alone are insufficient). Batch extraction and in-memory caching reduce API calls.
- **LaTeX:** Optional conversion per formula; raw text is kept when LaTeX is unavailable.

### 3.3 Derivations & Calculation Blocks

- **Derivation:** Detected via keywords (“proof”, “derivation”, “we can show”, “substituting”). Output includes `steps[]`, `derived_from_formula_ids`, `derived_to_formula_id`, and link type (EXPLAINS or REFERENCES).
- **Calculation:** Unlabeled multi-step computations (numbered steps, “Given:” + numeric lines + “therefore”/“thus”) are classified separately from worked examples and solutions; output includes `steps[]` and `referenced_formula_ids`.

### 3.4 Worked Examples

- **Detection:** Headings and cues (“Example”, “Illustration”, “Worked Example”, “Case”, “Solution”, “Step 1”, “Given:” + values). LLM used to structure problem statement, steps, and final answer when available.
- **Output:** `example_prompt`, `given_data`, `steps[]`, `final_answer`, `output_variables`, `referenced_formula_ids`. Bidirectional links: example → formula (USES_FORMULA) and formula → example (`referenced_example_ids`).

### 3.5 Questions & Solutions (Three-Case Handling)

The pipeline explicitly supports three cases:

- **Case A — Solutions in book:** Solution blocks are detected (same chapter, end-of-chapter, or concept-check sections). Linking uses question number / ref key first, then fuzzy match on text.
- **Case B — No solutions:** For questions with `solution_status == "not_found_in_book"`, the pipeline can generate candidate solutions via LLM (batched by chapter), then validate them against extracted formulas and variables; best candidate is stored as a synthetic solution with `is_synthetic=True` and `validation_score`.
- **Case C — Solutions in different location:** Solutions in appendix or another section are linked by chapter + number/similarity; concept-check solutions are matched to concept-check questions.

Question metadata includes numbering preservation (Q1, 1., (a)(b)), type (MCQ, Short Answer, Calculation, Case, Essay), `choices[]` for MCQ, and `problem_set_type` (problem_set, cfa, concept_check).

### 3.6 Concept-ID Linking

- **ConceptLinker** loads the expert-built taxonomy from TSV or Excel (columns: Concept, Level, Tag(s), Rationale, Page(s) or equivalents).
- **Matching:** Exact phrase and alias in heading/body; page proximity; tag/rationale n-grams; TF-IDF–based semantic similarity; optional LLM rerank. Each link stores `concept_id`, `link_method`, `confidence`, and optional concept metadata.
- **Quality:** If no strong lexical + semantic link is found, the segment is flagged `needs_human_review`. Prefer definition formulas linked to the defining concept; application formulas and examples linked to concepts where they are used.

### 3.7 Context & Traceability

- Every segment has `context_before` / `context_after` (or at least `prev_segment_id` / `next_segment_id`) so RAG can use adjacent context.
- All segments carry `book_id`, `doc_uri`, `page_start`/`page_end`, `chapter_number`/`chapter_title`, `heading_path`, and `bbox` where available.

---

## 4. Output Data Structure

- **Metadata:** `source_pdf`, `total_pages`, and similar run info.
- **Chapters:** List of chapter metadata: `chapter_number`, `chapter_title`, `chapter_level`, `parent_chapter`, `solutions_present`, `solution_location` (in_chapter, end_of_chapter, appendix, concept_check_only, etc.).
- **Segments:** List of segment objects. Types emitted in the default (formula/example/Q&A) mode: `formula`, `derivation`, `calculation`, `worked_example`, `question`, `solution`. Each has the base fields (segment_id, book_id, page range, heading_path, bbox, text_content, context_before/after, prev/next_segment_id, concept_links) plus type-specific fields (e.g. formula: variables, equation_number, usage_type, canonical_formula_key; question: question_number, subparts, choices, solution_status; solution: solution_for_question_id, solution_steps, is_synthetic, validation_score).
- **Edges:** Typed edges for KG traversal: `source_id`, `target_id`, `edge_type` (DEFINES, EXPLAINS, USES_FORMULA, WORKED_EXAMPLE_OF, REFERENCES, NEAR, ANSWER_OF, DUPLICATE_OF), `strength`, `link_method`, `anchor_metadata` (page, heading_path, method, snippet).

---

## 5. Usage & Configuration

**Environment:** Set the appropriate API key for the chosen LLM backend (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). See `run_sample.py` and `llm_services.py` for supported backends (OpenAI, Anthropic/Claude, Gemini, OpenRouter, Ollama, Groq).

**Run:**

- Single chapter (default: Investments Ch. 9):  
  `python run_sample.py light claude`
- Full book:  
  `python run_sample.py light claude --full-book`  
  For a faster first pass (no synthetic solutions):  
  `python run_sample.py light claude --full-book --no-solutions`
- Override PDF or concept list:  
  `--pdf path/to/book.pdf --concept-list path/to/concepts.xlsx`
- Clear solution cache and re-run:  
  `python run_sample.py light claude --full-book --clear-cache`

**Rough runtime (Investments, ~600 pages):** With `--no-solutions`: ~15–30 min. With solutions and validation: ~30–90+ min. Use `--validate-top 1` to validate only the top candidate per question and reduce latency.

**Runtime tips (full book):**  
- First pass: `--full-book --no-solutions` to get segments and edges quickly; variable extraction is batched and (for Claude) cached on disk in `.cache/llm_cache.db`, so re-runs skip repeated formula API calls.  
- Then run with solutions for selected chapters, or full book with `--validate-top 1` (validation is the main cost when generating synthetic solutions).  
- Use `--no-validate` to attach the best candidate without LLM validation (fastest, lower confidence).

**Cache:** Variable extraction batch (Claude/OpenAI) uses disk cache in `.cache/llm_cache.db` when `enable_cache=True`. LaTeX/variable in-memory caches are per run. Solution cache is at `outputs/.cache/solution_cache.json`; use `--clear-cache` to force fresh solution generation.

---

## 6. Dependencies

- **PDF:** PyMuPDF (`fitz`)
- **Data / validation:** pydantic, pandas, openpyxl
- **LLM:** openai, anthropic, google-genai (as needed per backend)
- **Env:** python-dotenv (optional)

---

*Internal technical documentation for the Synapta segmentation pipeline. Formula & Example/Q&A module.*
