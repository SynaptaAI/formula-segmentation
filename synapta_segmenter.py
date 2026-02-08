import os
import re
import uuid
import json
import time
import typing
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Literal
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from schemas import (
    BBox,
    ConceptLink,
    Edge,
    VariableDefinition,
    SegmentBase,
    FormulaSegment,
    WorkedExampleSegment,
    QuestionSegment,
    SolutionSegment,
    DerivationSegment,
    CalculationSegment,
    HeaderSegment,
    ExplanatoryTextSegment,
    ReferenceStubSegment,
    ChapterMetadata,
    SegmentationOutput,
)
from llm_services import (
    LLMService,
    MockLLMService,
    OpenAILLMService,
    GeminiLLMService,
    AnthropicLLMService,
    ClaudeLLMService,
    generate_json_with_llm,
)
from concept_linker import ConceptLinker
from segmenter.utils import format_solution_text
from segmenter.context import ContextProcessor

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system env vars

class FurnitureDetector:
    """
    Detects page furniture (headers, footers, page numbers).
    """
    FURNITURE_PHRASES = [
        r'^\s*\(?\s*concluded\s*\)?\s*$',
        r'^\s*\(?\s*continued\s*\)?\s*$',
        r'^\s*\(?\s*cont\'?d?\s*\)?\s*$',
        r'^\s*\(?\s*continuation\s*\)?\s*$',
        r'^\s*continued\s+(?:on|from)\s+(?:next|previous)?\s*page',
        r'^\s*see\s+(?:next|previous)\s+page',
        r'^\s*to\s+be\s+continued',
        r'^(?:Table|Figure|Exhibit)\s+[\d.]+\s*\(?\s*(?:continued|cont\'?d?)\s*\)?',
        r'^\s*(?:Page\s+)?\d+\s*$',
        r'^\s*\d+\s*/\s*\d+\s*$',
        r'^\s*[-–—]\s*\d+\s*[-–—]\s*$',
    ]

    def __init__(self):
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.FURNITURE_PHRASES]
        self.frequency_map: Dict[str, int] = {}
        self.total_pages = 0
        self.is_scanned = False
        self.TOP_BAND = 0.10
        self.BOTTOM_BAND = 0.10
        self.REPEAT_THRESHOLD = 0.20
        self.MAX_WORDS = 8

    def scan_document(self, doc: fitz.Document, pdf_path: str = None) -> None:
        """
        Scan document for repeated furniture patterns with sampling and caching.
        """
        import hashlib
        import json
        import os

        # Cache setup
        cache_dir = os.path.join(os.path.dirname(pdf_path) if pdf_path else ".", ".cache")
        cache_key = None
        if pdf_path:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                stat = os.stat(pdf_path)
                file_hash = hashlib.md5(f"{pdf_path}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()
                cache_key = f"furniture_{file_hash}.json"
                cache_file = os.path.join(cache_dir, cache_key)
                
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, "r") as f:
                            cached_data = json.load(f)
                            self.frequency_map = cached_data.get("frequency_map", {})
                            self.total_pages = cached_data.get("total_pages", 0)
                            self.is_scanned = True
                            print(f"FurnitureDetector: Loaded from cache ({len(self.frequency_map)} items)")
                            return
                    except Exception as e:
                        print(f"FurnitureDetector: Cache read failed: {e}")
            except Exception as e:
                print(f"FurnitureDetector: Cache setup failed: {e}")

        # Sampling strategy: First 25, Middle 25, Last 25
        total_doc_pages = len(doc)
        pages_to_scan = set()
        
        # Front
        for i in range(min(25, total_doc_pages)):
            pages_to_scan.add(i)
        
        # Back
        for i in range(max(0, total_doc_pages - 25), total_doc_pages):
            pages_to_scan.add(i)
            
        # Middle
        if total_doc_pages > 50:
            mid = total_doc_pages // 2
            start_mid = max(0, mid - 12)
            end_mid = min(total_doc_pages, mid + 13)
            for i in range(start_mid, end_mid):
                pages_to_scan.add(i)
                
        text_pages: Dict[str, Set[int]] = defaultdict(set)
        
        print(f"FurnitureDetector: Scanning {len(pages_to_scan)} sampled pages...")
        for page_idx in pages_to_scan:
            try:
                page = doc[page_idx]
                blocks = page.get_text("blocks")
                for b in blocks:
                    x0, y0, x1, y1, text, _, _ = b
                    clean_text = self._normalize_text(text)
                    if not clean_text: continue
                    if len(clean_text.split()) > self.MAX_WORDS: continue
                    text_pages[clean_text].add(page_idx)
            except Exception as e:
                print(f"Error scanning page {page_idx}: {e}")
        
        # Calculate frequency based on SAMPLED pages only? 
        # Or extrapolate?
        # A string is repeated if it appears in > threshold of SAMPLED pages?
        # Let's say we scanned N pages. Frequency = count / N.
        
        scanned_count = len(pages_to_scan)
        self.total_pages = total_doc_pages # Use actual total for reference, or scanned?
        # We need to reuse total_pages for is_furniture check: freq = count / total_pages.
        # If we use total_doc_pages, freq will be tiny.
        # We should use scanned_count as the denominator for frequency check if we only scanned subset.
        # BUT: is_furniture takes freq = count / max(1, self.total_pages).
        # So we should set self.total_pages = scanned_count for the math to work correctly on the sample.
        self.total_pages = scanned_count
        
        self.frequency_map = {t: len(p) for t, p in text_pages.items()}
        self.is_scanned = True
        print(f"FurnitureDetector: Scanned {scanned_count} pages. Found {len(self.frequency_map)} repeated strings.")

        # Save to cache
        if pdf_path and cache_key:
            try:
                data = {
                    "frequency_map": self.frequency_map,
                    "total_pages": self.total_pages
                }
                with open(cache_file, "w") as f:
                    json.dump(data, f)
            except Exception as e:
                 print(f"FurnitureDetector: Cache write failed: {e}")

    def is_furniture(self, block: Tuple, page_height: float = 842.0) -> bool:
        x0, y0, x1, y1, text, _, _ = block
        clean_text = self._normalize_text(text)
        if not clean_text: return False

        # 1. Pattern Match
        for pat in self._compiled_patterns:
            if pat.match(text.strip()): return True

        # 2. Position Check
        in_top_band = y0 < (page_height * self.TOP_BAND)
        in_bottom_band = y1 > (page_height * (1 - self.BOTTOM_BAND))
        is_edge = in_top_band or in_bottom_band
        
        # 3. Frequency Check
        if self.is_scanned and is_edge:
            count = self.frequency_map.get(clean_text, 0)
            freq = count / max(1, self.total_pages)
            if freq > self.REPEAT_THRESHOLD: return True
                
        # 4. Trival Edge Content
        if is_edge and len(clean_text) < 4 and clean_text.isdigit(): return True

        return False

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

# Extractors

class BaseExtractor(ABC):
    @abstractmethod
    def process_page(self, page: fitz.Page, page_num: int, book_id: str, blocks: List[Any] = None,
                     doc_uri: Optional[str] = None) -> List[SegmentBase]:
        pass
    
    def normalize_text(self, text: str) -> str:
        return text.strip()

class FormulaExtractor(BaseExtractor):
    def __init__(self, llm_service: LLMService = None, llm_mode: Literal["off", "light", "full"] = "full",
                 use_lexicon: bool = True):
        self.llm_service = llm_service or MockLLMService()
        self.llm_mode = llm_mode
        self.use_lexicon = use_lexicon
        self._canonical_var_cache: Dict[str, List[VariableDefinition]] = {}
        self._symbol_meaning_cache: Dict[str, VariableDefinition] = {}
        self._summary_cache: Dict[str, Optional[str]] = {}
        self._llm_stats: Dict[str, int] = {
            "extract_variables_calls": 0,
            "extract_variables_cache_hits": 0,
            "extract_variables_symbol_hits": 0,
            "extract_variables_skipped": 0,
        }

    def process_page(self, page: fitz.Page, page_num: int, book_id: str, blocks: List[Any] = None,
                     doc_uri: Optional[str] = None) -> List[FormulaSegment]:
        segments = []
        if blocks is None:
            blocks = page.get_text("blocks")
        
        # First pass: collect potential formula segments
        raw_formula_segments = []
        formula_count = 0
        for i, b in enumerate(blocks):
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0: continue
            
            clean_text = text.strip()
            if not clean_text: continue
            
            # Check if it's prose that was misclassified as formula
            if self._is_formula(clean_text, x0, page.rect.width):
                # Additional check: if math token ratio is very low, it's prose
                words = clean_text.split()
                math_symbols = {'=', '∑', '√', '∫', 'σ', 'π', 'θ', 'λ', '+', '−', '×', '÷', '^', '**', '/', '\\'}
                math_token_count = sum(1 for word in words if any(sym in word for sym in math_symbols))
                math_token_ratio = math_token_count / len(words) if words else 0
                
                # If math token ratio < 0.2 and no equation number, it's likely prose
                # Skip it here - let TextBlockExtractor handle it as explanatory_text
                if math_token_ratio < 0.2 and not re.search(r'\(\d+\.\d+\)', clean_text):
                    continue
                # Extract broader context (two blocks before/after)
                context_before_text = ""
                context_after_text = ""
                if i > 0 and blocks[i-1][6] == 0:
                    context_before_text = blocks[i-1][4].strip()
                if i > 1 and blocks[i-2][6] == 0:
                    extra_before = blocks[i-2][4].strip()
                    if extra_before:
                        context_before_text = f"{extra_before} {context_before_text}".strip()
                if i + 1 < len(blocks) and blocks[i+1][6] == 0:
                    context_after_text = blocks[i+1][4].strip()
                if i + 2 < len(blocks) and blocks[i+2][6] == 0:
                    extra_after = blocks[i+2][4].strip()
                    if extra_after:
                        context_after_text = f"{context_after_text} {extra_after}".strip()
                
                # Combine contexts for variable extraction
                combined_context = f"{context_before_text} {context_after_text}".strip()
                
                candidate_symbols = self._find_variable_candidates(clean_text, combined_context)
                canonical_key = self._generate_canonical_key(clean_text)
                
                # Check if we should skip LLM calls (light/off mode)
                if self.llm_mode == "off":
                    variables = []
                    summary = None
                elif self.llm_mode == "light":
                    # In light mode, extract variables but skip summary
                    variables = self._extract_variables(
                        clean_text, combined_context,
                        candidate_symbols=candidate_symbols,
                        canonical_key=canonical_key
                    )
                    summary = None if self.use_lexicon else self._get_formula_summary(canonical_key, clean_text, variables)
                else:
                    # Full mode: extract variables (with cache) and always summarize (with cache)
                    variables = self._extract_variables(
                        clean_text, combined_context,
                        candidate_symbols=candidate_symbols,
                        canonical_key=canonical_key
                    )
                    summary = None if self.use_lexicon else self._get_formula_summary(canonical_key, clean_text, variables)
                
                eq_num = self._extract_eq_number(clean_text, context_after_text)
                usage_type = self._classify_formula_usage(clean_text, combined_context)
                
                # Best-effort LaTeX conversion (with caching); fallback to raw so formula_latex is never null
                latex_converted = self._convert_to_latex(clean_text, variables, canonical_key=canonical_key)
                latex = latex_converted or self._raw_as_latex_fallback(clean_text)
                seg = FormulaSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    context_before=context_before_text if context_before_text else None,
                    context_after=context_after_text if context_after_text else None,
                    formula_text_raw=clean_text,
                    formula_latex=latex,
                    equation_number=eq_num,
                    variables=variables,
                    usage_type=usage_type,
                    canonical_formula_key=canonical_key,
                    short_meaning=summary,
                    confidence=1.0 if eq_num else 0.8,
                    needs_human_review=(latex_converted is None and len(variables) == 0),  # Review if LaTeX failed and no variables
                    doc_uri=doc_uri,
                )
                raw_formula_segments.append(seg)
                formula_count += 1
                if formula_count % 5 == 0:
                    print(f"  Processed {formula_count} formulas on page {page_num}...")
        
        # Second pass: Merge fragmented multi-line formulas
        if raw_formula_segments:
            print(f"  Merging {len(raw_formula_segments)} formula segments on page {page_num}...")
        merged_segments = self._merge_fragmented_formulas(
            raw_formula_segments,
            page.rect.height,
            page_widths={page_num: page.rect.width}
        )
        
        return merged_segments

    def process_blocks_batch(
        self,
        clean_by_page: Dict[int, List[Tuple]],
        doc: Any,
        start_page: int,
        end_page: int,
        book_id: str,
        doc_uri: Optional[str] = None,
        batch_size: int = 10,
    ) -> List[FormulaSegment]:
        """
        New batch pipeline:
        1. Scan all pages to find formula candidates (no LLM).
        2. Deduplicate by canonical key.
        3. Send unique formulas to LLM in batches.
        4. Fan-out results to all instances.
        """
        print(f"Phase 1: Scanning pages {start_page} to {end_page} for formulas...")
        all_segments = []
        
        # 1. Scan (fast, no LLM)
        for i in range(start_page, end_page):
            page_num = i + 1
            blocks = clean_by_page.get(page_num, [])
            page = doc[i]
            page_segments = self.scan_page(page, page_num, book_id, blocks, doc_uri)
            all_segments.extend(page_segments)
            
        print(f"  Found {len(all_segments)} formula candidates.")
        
        # 2. Enrich (Batch + Dedupe)
        if self.llm_mode != "off":
            self.enrich_segments_batch(all_segments, batch_size=batch_size)
            
        return all_segments

    def scan_page(self, page: fitz.Page, page_num: int, book_id: str, blocks: List[Any],
                  doc_uri: Optional[str] = None) -> List[FormulaSegment]:
        """Identify formula locations without running expensive enrichment."""
        segments = []
        page_width = page.rect.width
        
        for i, b in enumerate(blocks):
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0: continue
            
            clean_text = text.strip()
            if not clean_text: continue
            
            # Heuristic detection checks
            if not self._is_formula_heuristic(clean_text):
                 # Fallback to geometry/token check if heuristic fails but looks like formula
                 # (Copied logic from old is_formula check)
                 if not self._is_formula(clean_text, x0, page_width):
                     continue
                 if self._looks_like_prose(clean_text) and not re.search(r'\(\d+\.\d+\)', clean_text):
                     continue

            # Context extraction
            prev_block = blocks[i-1] if i > 0 else None
            next_block = blocks[i+1] if i < len(blocks)-1 else None
            ctx_before = prev_block[4].strip()[-200:] if prev_block else ""
            ctx_after = next_block[4].strip()[:200] if next_block else ""
            
            # Canonical Key
            canonical_key = self._generate_canonical_key(clean_text)
            
            # Equation Number
            eq_num = self._extract_eq_number(clean_text, ctx_after)
            
            seg = FormulaSegment(
                segment_id=str(uuid.uuid4()),
                book_id=book_id,
                chapter_number="Unknown",
                page_start=page_num,
                page_end=page_num,
                bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                text_content=clean_text,
                formula_text_raw=clean_text,
                canonical_formula_key=canonical_key,
                variables=[],  # Empty for now
                short_meaning=None, # Empty for now
                usage_type=self._classify_usage(ctx_before + " " + ctx_after),
                context_before=ctx_before,
                context_after=ctx_after,
                equation_number=eq_num,
                doc_uri=doc_uri
            )
            segments.append(seg)
        return segments

    def enrich_segments_batch(self, segments: List[FormulaSegment], batch_size: int = 20) -> None:
        """Deduplicate unique formulas, batch enrich them, and fan-out results."""
        if not segments:
            return

        # Group by canonical key
        by_key: Dict[str, List[FormulaSegment]] = defaultdict(list)
        for s in segments:
            if s.canonical_formula_key:
                by_key[s.canonical_formula_key].append(s)
        
        unique_keys = list(by_key.keys())
        print(f"Phase 2: Enriching {len(unique_keys)} unique formulas (from {len(segments)} total)...")
        
        # Prepare batch items
        to_enrich = []
        skipped_non_anchor = 0
        for key in unique_keys:
            group = by_key[key]
            rep = group[0]
            
            # Determine if this canonical formula should be treated as an "anchor"
            # worth LLM enrichment (equation label or definitional usage anywhere
            # in the group). Non-anchor formulas still remain as segments but will
            # not trigger new LLM calls; they rely on heuristics / cache only.
            is_anchor = any(
                (s.equation_number is not None)
                or (getattr(s, "usage_type", "application") == "definition")
                for s in group
            )
            
            # 1. Simple Formula Rule (Skip LLM)
            if self._is_simple_formula(rep.text_content):
                ctx = (rep.context_before or "") + " " + (rep.context_after or "")
                vars_heuristic = self._extract_variables_heuristic(rep.text_content, ctx)
                processed = self._post_process_variables(vars_heuristic, rep.text_content)
                self._update_variable_caches(key, processed)
                
                # Fan out immediately
                for s in group:
                    s.variables = self._clone_variables(processed)
                    s.confidence = 1.0 # High confidence for simple rules
                continue

            # 2. Check Cache
            cached = self._get_cached_variables_by_key(key)
            if cached:
                # Fan out from cache
                summary = self._summary_cache.get(key)
                for s in group:
                    s.variables = self._clone_variables(cached)
                    s.short_meaning = summary
            else:
                # Only anchor formulas are allowed to trigger new LLM calls.
                if is_anchor:
                    to_enrich.append(key)
                else:
                    skipped_non_anchor += 1
                    # Mark non-anchored formulas that couldn't be resolved heuristically
                    # as needing human review, but do not call LLM here.
                    for s in group:
                        if not s.variables and not s.short_meaning:
                            s.needs_human_review = True

        print(
            f"  {len(unique_keys) - len(to_enrich)} resolved via cache/rules. "
            f"{len(to_enrich)} sent to LLM. {skipped_non_anchor} non-anchor formulas skipped for enrichment."
        )
        
        if not to_enrich:
            return

        # Process in chunks
        for i in range(0, len(to_enrich), batch_size):
            chunk_keys = to_enrich[i : i + batch_size]
            batch_inputs = [] # (formula, context, candidates)
            batch_ids = []    # canonical_key
            
            for key in chunk_keys:
                rep = by_key[key][0]
                # Combine context from up to 3 instances
                instances = by_key[key][:3]
                combined_ctx = " ... ".join([
                    (inst.context_before or "") + " " + (inst.context_after or "") 
                    for inst in instances
                ])
                # Find candidates for hint
                candidates = self._find_variable_candidates(rep.text_content, combined_ctx)
                
                batch_inputs.append((rep.text_content, combined_ctx[:800], candidates))
                batch_ids.append(key)
            
            # LLM Call
            self._llm_stats["extract_variables_calls"] += 1
            # Note: extract_variables_batch returns [(variables, summary), ...] in order
            results = self.llm_service.extract_variables_batch(items=batch_inputs)
            
            # Fan out results
            for idx, item_result in enumerate(results):
                key = batch_ids[idx]
                formula_text = batch_inputs[idx][0]
                
                vars_list = item_result[0]
                summary = item_result[1]
                
                # Verify and Post-process
                verified = self._verify_variables(vars_list, formula_text)
                processed = self._post_process_variables(verified, formula_text)
                
                # Update caches
                self._update_variable_caches(key, processed)
                if summary:
                    self._summary_cache[key] = summary
                
                # Update all instances
                for s in by_key[key]:
                    s.variables = self._clone_variables(processed)
                    s.short_meaning = summary
            
            print(f"  Processed batch {i // batch_size + 1}/{(len(to_enrich) + batch_size - 1) // batch_size}")


    def _process_single_complex_item(self, item_tuple, book_id, doc_uri, target_list) -> bool:
        """Process a single complex formula item (extraction + segment creation)."""
        clean_text = item_tuple[7]
        ctx_before = item_tuple[8]
        ctx_after = item_tuple[9]
        candidates = item_tuple[10]
        combined_context = f"{ctx_before} {ctx_after}".strip()
        
        canonical_key = self._generate_canonical_key(clean_text)
        variables = self._extract_variables(
            clean_text, combined_context,
            candidate_symbols=candidates,
            canonical_key=canonical_key
        )
        
        self._create_and_append_segment(item_tuple, variables, canonical_key, book_id, doc_uri, target_list)
        return bool(variables)

    def _create_and_append_segment(self, item_tuple, variables, canonical_key, book_id, doc_uri, target_list,
                                    summary_from_batch: Optional[str] = None):
        """Create FormulaSegment from item tuple and extraction results. summary_from_batch avoids a separate LLM call."""
        page_num, page_height_val, page_width_val, x0, y0, x1, y1, clean_text, ctx_before, ctx_after, candidates = item_tuple
        combined_context = f"{ctx_before} {ctx_after}".strip()
        
        eq_num = self._extract_eq_number(clean_text, ctx_after)
        usage_type = self._classify_formula_usage(clean_text, combined_context)
        summary = summary_from_batch if summary_from_batch else (None if self.use_lexicon else self._get_formula_summary(canonical_key, clean_text, variables))
        latex_converted = None if self.llm_mode == "off" else self._convert_to_latex(clean_text, variables, canonical_key=canonical_key)
        latex = latex_converted or self._raw_as_latex_fallback(clean_text)
        seg = FormulaSegment(
            segment_id=str(uuid.uuid4()),
            book_id=book_id,
            chapter_number="Unknown",
            page_start=page_num,
            page_end=page_num,
            bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
            text_content=clean_text,
            context_before=ctx_before if ctx_before else None,
            context_after=ctx_after if ctx_after else None,
            formula_text_raw=clean_text,
            formula_latex=latex,
            equation_number=eq_num,
            variables=variables,
            usage_type=usage_type,
            canonical_formula_key=canonical_key,
            short_meaning=summary,
            confidence=1.0 if eq_num else 0.8,
            needs_human_review=(latex_converted is None and len(variables) == 0),
            doc_uri=doc_uri,
        )
        target_list.append(seg)

    def _merge_fragmented_formulas(self, formula_segments: List[FormulaSegment], page_height: float,
                                   page_widths: Optional[Dict[int, float]] = None) -> List[FormulaSegment]:
        """
        Merge fragmented formula segments that are part of the same formula.
        Criteria:
        - Same page
        - Close y-distance (within 2 lines, ~30-40 pixels)
        - Same equation_number OR no equation_number but close proximity
        - Similar x-position (centered formulas)
        """
        if not formula_segments:
            return []
        
        # Group by page
        by_page: Dict[int, List[FormulaSegment]] = defaultdict(list)
        for seg in formula_segments:
            by_page[seg.page_start].append(seg)
        
        merged = []
        for page_num, page_segments in by_page.items():
            # Sort by y-position
            page_segments.sort(key=lambda s: s.bbox.y0 if s.bbox else 0)
            page_width = None
            if page_widths and page_num in page_widths:
                page_width = page_widths[page_num]
            elif page_segments and page_segments[0].bbox:
                page_width = max(s.bbox.x1 for s in page_segments if s.bbox)  # best-effort
            
            i = 0
            while i < len(page_segments):
                current = page_segments[i]
                merge_group = [current]
                
                # Look ahead for fragments to merge
                j = i + 1
                while j < len(page_segments):
                    candidate = page_segments[j]
                    
                    # Check if should merge
                    should_merge = self._should_merge_formulas(current, candidate, page_height, page_width)
                    
                    if should_merge:
                        merge_group.append(candidate)
                        j += 1
                    else:
                        break
                
                # Merge the group
                if len(merge_group) > 1:
                    merged_seg = self._merge_formula_group(merge_group)
                    merged.append(merged_seg)
                    i = j  # Skip merged segments
                else:
                    merged.append(current)
                    i += 1
        
        return merged
    
    def _should_merge_formulas(self, seg1: FormulaSegment, seg2: FormulaSegment, page_height: float,
                               page_width: Optional[float]) -> bool:
        """Check if two formula segments should be merged."""
        if not seg1.bbox or not seg2.bbox:
            return False
        
        # Must be on same page
        if seg1.page_start != seg2.page_start:
            return False
        
        # Hard stop: avoid merging with likely prose
        if self._looks_like_prose(seg1.formula_text_raw) or self._looks_like_prose(seg2.formula_text_raw):
            # Allow equation label to attach to formula
            if not (self._is_equation_label(seg1.formula_text_raw) or self._is_equation_label(seg2.formula_text_raw)):
                return False
        
        # Check y-distance (within ~2 lines, ~40 pixels)
        y_distance = abs(seg2.bbox.y0 - seg1.bbox.y1)
        max_y_distance = 40.0  # pixels
        
        if y_distance > max_y_distance:
            return False
        
        # Check x-position similarity (both should be centered or similar x)
        x_overlap = not (seg1.bbox.x1 < seg2.bbox.x0 or seg2.bbox.x1 < seg1.bbox.x0)
        x_close = abs(seg1.bbox.x0 - seg2.bbox.x0) < 100  # Within 100 pixels
        is_centered = False
        if page_width:
            mid1 = (seg1.bbox.x0 + seg1.bbox.x1) / 2
            mid2 = (seg2.bbox.x0 + seg2.bbox.x1) / 2
            is_centered = abs(mid1 - page_width / 2) < page_width * 0.2 and \
                          abs(mid2 - page_width / 2) < page_width * 0.2
        
        if not (x_overlap or x_close or is_centered):
            return False
        
        # If both have equation numbers, they must match
        if seg1.equation_number and seg2.equation_number:
            return seg1.equation_number == seg2.equation_number
        
        # If one has equation number and other doesn't, merge if close
        # (denominator/numerator often don't have eq number)
        if (seg1.equation_number and not seg2.equation_number) or \
           (seg2.equation_number and not seg1.equation_number):
            return True  # Merge (likely numerator/denominator)
        
        # Equation label line (e.g., "(9.1)" or "Eq. 9.1") should attach to nearby formula
        if self._is_equation_label(seg1.formula_text_raw) or self._is_equation_label(seg2.formula_text_raw):
            return y_distance < 60.0
        
        # Both have no equation number - merge if very close
        return y_distance < 20  # Very close fragments
    
    def _merge_formula_group(self, group: List[FormulaSegment]) -> FormulaSegment:
        """Merge a group of formula segments into one."""
        if len(group) == 1:
            return group[0]
        
        # Use first segment as base
        base = group[0]
        
        # Merge text content
        merged_text = " ".join([seg.formula_text_raw for seg in group])
        
        # Merge bbox (union of all bboxes)
        if all(seg.bbox for seg in group):
            x0 = min(seg.bbox.x0 for seg in group)
            y0 = min(seg.bbox.y0 for seg in group)
            x1 = max(seg.bbox.x1 for seg in group)
            y1 = max(seg.bbox.y1 for seg in group)
            merged_bbox = BBox(page=base.page_start, x0=x0, y0=y0, x1=x1, y1=y1)
        else:
            merged_bbox = base.bbox
        
        # Merge contexts
        context_before = base.context_before
        context_after = group[-1].context_after if group[-1].context_after else base.context_after
        
        # Combine all contexts for variable extraction
        all_context = " ".join([
            seg.context_before or "" for seg in group
        ] + [
            seg.context_after or "" for seg in group
        ]).strip()
        
        # Re-generate canonical key for merged formula
        canonical_key = self._generate_canonical_key(merged_text)
        # Re-extract variables with merged context
        variables = self._extract_variables(
            merged_text, all_context,
            candidate_symbols=self._find_variable_candidates(merged_text, all_context),
            canonical_key=canonical_key
        )
        
        # Use equation number from any segment that has it
        eq_num = next((seg.equation_number for seg in group if seg.equation_number), None)
        
        # Re-extract summary / LaTeX with llm_mode gate; fallback to raw so formula_latex is never null
        summary = None if self.use_lexicon else self._get_formula_summary(canonical_key, merged_text, variables)
        latex = None if self.llm_mode == "off" else self._convert_to_latex(merged_text, variables, canonical_key=canonical_key)
        latex = latex or self._raw_as_latex_fallback(merged_text)
        # Determine usage type (prefer definition over application)
        usage_type = "definition" if any(seg.usage_type == "definition" for seg in group) else base.usage_type
        
        merged = FormulaSegment(
            segment_id=base.segment_id,  # Keep first segment's ID
            book_id=base.book_id,
            chapter_number=base.chapter_number,
            chapter_title=base.chapter_title,
            page_start=base.page_start,
            page_end=group[-1].page_end,
            bbox=merged_bbox,
            text_content=merged_text,
            context_before=context_before,
            context_after=context_after,
            formula_text_raw=merged_text,
            formula_latex=latex,
            equation_number=eq_num,
            variables=variables,
            usage_type=usage_type,
            referenced_formula_ids=list(set([fid for seg in group for fid in seg.referenced_formula_ids])),
            referenced_example_ids=list(set([eid for seg in group for eid in seg.referenced_example_ids])),
            canonical_formula_key=canonical_key,
            short_meaning=summary,
            confidence=max(seg.confidence for seg in group),
            needs_human_review=any(seg.needs_human_review for seg in group),
            doc_uri=getattr(base, "doc_uri", None),
        )
        
        return merged

    def _is_formula(self, text: str, x0: float, page_width: float) -> bool:
        """
        Detect if text is a formula, not prose.
        Filters out explanatory text with low math token ratio.
        """
        clean = text.strip()
        # Drop single-symbol labels (common in figures/axes)
        if len(clean) <= 3 and not re.search(r'\(\d+\.\d+\)', clean):
            if not re.search(r'[=+\-*/×÷∑∫√]', clean):
                return False
        # Reject numeric-only or table-like fragments (e.g., column values)
        if not re.search(r'[A-Za-zα-ωΑ-Ω]', clean):
            if re.fullmatch(r'[\d\.\-%\s]+', clean):
                # Drop tiny numeric fragments; keep larger numeric blocks for context/merge
                if len(clean.split()) <= 2 and not re.search(r'\(\d+\.\d+\)', clean):
                    return False
        # Reject tiny index assignments like "k = 1" without equation labels
        if re.fullmatch(r'[A-Za-z]\s*=\s*[-]?\d+(\.\d+)?', clean) and not re.search(r'\(\d+\.\d+\)', clean):
            return False
        # Centered if left margin is far from page edge
        is_centered = False
        if page_width:
            is_centered = (page_width * 0.2) < x0 < (page_width * 0.6)
        math_symbols = {'=', '∑', '√', '∫', 'σ', 'π', 'θ', 'λ', '+', '−', '×', '÷', '^', '**', '/', '\\', '(', ')', '[', ']'}
        has_math_symbol = any(s in text for s in math_symbols)
        has_eq_keyword = "Eq." in text or "Equation" in text or bool(re.search(r'\(\d+\.\d+\)', text))
        
        # Calculate math token ratio to filter out prose
        words = text.split()
        math_token_count = sum(1 for word in words if any(sym in word for sym in math_symbols) or 
                              re.search(r'[A-Za-z]\d+|\d+[A-Za-z]|[A-Za-z]_[A-Za-z]', word))  # Subscripts, variables
        total_tokens = len(words) if words else 1
        math_token_ratio = math_token_count / total_tokens
        
        # Filter out prose: if math token ratio is too low, it's likely explanatory text
        # Threshold: at least 30% math tokens OR has equation keyword
        if math_token_ratio < 0.3 and not has_eq_keyword:
            return False  # Likely prose, not formula
        
        if has_math_symbol and (is_centered or has_eq_keyword or math_token_ratio >= 0.3):
            return True
        return False

    def _raw_as_latex_fallback(self, raw: str) -> str:
        """Normalize raw formula text for use when LaTeX conversion fails. Never returns None."""
        if not raw:
            return ""
        t = " ".join(raw.split())
        return t.strip()

    def _convert_to_latex(self, formula_text: str, variables: List[VariableDefinition], 
                         canonical_key: Optional[str] = None) -> Optional[str]:
        """
        Best-effort LaTeX conversion using an optional LLM method.
        """
        if not self.llm_service:
            return None
        
        # Simple heuristic: if already looks like LaTeX, return as-is
        if '\\' in formula_text or '{' in formula_text:
            return formula_text
        try:
            return self.llm_service.convert_to_latex(formula_text, canonical_key=canonical_key)
        except Exception:
            return None
    
    def _extract_eq_number(self, text: str, context_after: str = None) -> Optional[str]:
        """
        Extract equation number from text or context_after.
        Patterns: (9.1), (2.3), Eq. 5.1, Equation (3.4)
        """
        patterns = [
            r'\((\d+\.\d+)\)',  # (9.1)
            r'Eq\.\s*\(?(\d+\.\d+)\)?',  # Eq. 9.1 or Eq. (9.1)
            r'Equation\s+\(?(\d+\.\d+)\)?',  # Equation 9.1
        ]
        
        def looks_like_label(line: str) -> bool:
            t = (line or "").strip()
            if not t:
                return False
            if len(t) > 40:
                return False
            return bool(re.fullmatch(r'\(?\d+\.\d+[a-z]?\)?', t) or
                        re.fullmatch(r'(Eq\.|Equation)\s*\(?\d+\.\d+[a-z]?\)?', t, re.IGNORECASE))
        
        # Check text first
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                # Return in format (9.1) if not already in parentheses
                num = match.group(1)
                if '(' not in match.group(0):
                    return f"({num})"
                return match.group(0)
        
        # Check context_after if provided
        if context_after:
            if looks_like_label(context_after):
                for pattern in patterns:
                    match = re.search(pattern, context_after)
                    if match:
                        num = match.group(1)
                        if '(' not in match.group(0):
                            return f"({num})"
                        return match.group(0)
        
        return None

    def _classify_usage(self, context: str) -> str:
        """Alias for _classify_formula_usage for backwards compatibility/refactor."""
        return self._classify_formula_usage("", context)

    def _is_formula_heuristic(self, text: str) -> bool:
        """
        Fast heuristic to check if text looks like a formula.
        Checks for:
        - Geometric shapes (lines, circles) vs text density (handled by caller?)
        - Presence of math symbols
        - Low prose density
        """
        clean = text.strip()
        if not clean:
            return False
        # Skip obvious URLs/domains or boilerplate strings
        if re.search(r'https?://|www\.|\.com\b|\.edu\b|\.net\b', clean, re.IGNORECASE):
            return False
            
        # Detect Equation Labels (e.g. (9.1)) -> assume formula line if attached
        if re.search(r'^\(?\d+(?:\.\d+)+[a-z]?\)?$', clean):
             return False # Just a label segment, not formula content? 
             # Actually, often formula lines are just the label if detection split badly.
             # But here we want content.
        
        # Check math symbols
        math_symbols = {'=', '∑', '√', '∫', 'σ', 'π', 'θ', 'λ', '+', '−', '×', '÷', '^', '**', '∂', '∆'}
        if any(s in clean for s in math_symbols):
             return True
             
        # Check variable-like patterns (e.g. E(r_M))
        if re.search(r'[A-Za-z][_(]', clean):
             return True
             
        return False

    def _classify_formula_usage(self, text: str, context: str) -> Literal["definition", "application", "reference"]:
        """Classify formula usage type based on context."""
        text_lower = text.lower()
        context_lower = context.lower()
        
        # Definition patterns
        if any(kw in context_lower for kw in ["where", "let", "define", "denote", "is defined"]):
            return "definition"
        
        # Reference patterns
        if re.search(r'(?:use|using|from|see|according to)\s+(?:eq|equation)', context_lower):
            return "reference"
        
        # Default to application
        return "application"

    def _generate_canonical_key(self, formula_text: str) -> str:
        """Generate a canonical key for duplicate detection."""
        normalized = self._normalize_formula_text(formula_text)
        # Create hash
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _is_simple_formula(self, formula_text: str) -> bool:
        """
        Check if formula is simple enough for rule-based extraction (no LLM needed).
        
        Simple formulas:
        - Single symbols: σ, r_f, β_i
        - Linear expressions: E(r_M) - r_f, y = a + b
        - Simple ratios: r / σ
        """
        # Remove whitespace and equation numbers
        clean = self._normalize_formula_text(formula_text)
        
        # Single symbol (possibly with subscript)
        if re.match(r'^[a-zA-Zα-ωΑ-ΩσπθλρμνΣ∫√εηφψΩ][_a-zA-Z0-9]*$', clean):
            return True
        
        # Simple linear expression: y = E(r_M) - r_f
        # Pattern: variable = expression (no complex operations)
        if '=' in clean:
            parts = clean.split('=', 1)
            if len(parts) == 2:
                left, right = parts
                # Left side: single variable
                if re.match(r'^[a-zA-Zα-ωΑ-ΩσπθλρμνΣ∫√εηφψΩ][_a-zA-Z0-9]*$', left):
                    # Right side: simple arithmetic (+, -, *, /)
                    # No complex functions like log, exp, sqrt, etc.
                    if not re.search(r'(log|ln|exp|sqrt|sin|cos|tan|∫|∑)', right, re.IGNORECASE):
                        # Count operators - if too many, it's complex
                        op_count = len(re.findall(r'[+\-*/]', right))
                        if op_count <= 3:  # Simple: a + b - c
                            return True
        
        # Simple ratio or product: r/σ, β * E(r_M)
        if re.match(r'^[a-zA-Zα-ωΑ-ΩσπθλρμνΣ∫√εηφψΩ][_a-zA-Z0-9]*\s*[/×*]\s*[a-zA-Zα-ωΑ-ΩσπθλρμνΣ∫√εηφψΩ][_a-zA-Z0-9]*$', clean):
            return True
        
        return False

    def _normalize_formula_text(self, text: str) -> str:
        """Normalize formula text for hashing and simple formula checks."""
        cleaned = self._strip_equation_labels(text)
        cleaned = re.sub(r'\s+', '', cleaned.lower())
        return cleaned

    def _strip_equation_labels(self, text: str) -> str:
        """Remove equation labels like (9.1), (9.1a), Eq. 9.1."""
        cleaned = text
        cleaned = re.sub(r'\b(?:eq|equation)\.?\s*\d+(?:\.\d+)*[a-z]?\b', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\(\s*\d+(?:\.\d+)*[a-z]?\s*\)', '', cleaned, flags=re.IGNORECASE)
        return cleaned

    def _clone_variables(self, variables: List[VariableDefinition]) -> List[VariableDefinition]:
        """Return a defensive copy of VariableDefinition list."""
        if not variables:
            return []
        return [VariableDefinition(**v.dict()) for v in variables]

    def _get_cached_variables_by_key(self, canonical_key: Optional[str]) -> Optional[List[VariableDefinition]]:
        if canonical_key and canonical_key in self._canonical_var_cache:
            self._llm_stats["extract_variables_cache_hits"] += 1
            return self._clone_variables(self._canonical_var_cache[canonical_key])
        return None

    def _get_cached_variables_by_symbols(self, candidate_symbols: List[str], scope: Optional[str] = None) -> List[VariableDefinition]:
        if not candidate_symbols:
            return []
        cached: List[VariableDefinition] = []
        for symbol in candidate_symbols:
            key = self._make_symbol_cache_key(symbol, scope)
            if key in self._symbol_meaning_cache:
                cached.append(VariableDefinition(**self._symbol_meaning_cache[key].dict()))
                continue
            # Fallback to global cache when scoped lookup misses
            if scope:
                global_key = self._make_symbol_cache_key(symbol, None)
                if global_key in self._symbol_meaning_cache:
                    cached.append(VariableDefinition(**self._symbol_meaning_cache[global_key].dict()))
        if cached and {v.symbol.strip().lower() for v in cached} == {s.strip().lower() for s in candidate_symbols}:
            self._llm_stats["extract_variables_symbol_hits"] += 1
        return cached

    def _update_variable_caches(self, canonical_key: Optional[str], variables: List[VariableDefinition],
                                scope: Optional[str] = None) -> None:
        if not variables:
            return
        if canonical_key:
            self._canonical_var_cache[canonical_key] = self._clone_variables(variables)
        for var in variables:
            symbol = (var.symbol or "").strip().lower()
            meaning = (var.meaning or "").strip()
            if symbol and meaning:
                cache_key = self._make_symbol_cache_key(symbol, scope)
                candidate = VariableDefinition(
                    symbol=var.symbol,
                    meaning=var.meaning,
                    units=var.units,
                    inferred=True,
                    source="llm" if var.source in ("llm", "context") else var.source,
                )
                existing = self._symbol_meaning_cache.get(cache_key)
                if not existing or self._is_better_definition(candidate, existing):
                    self._symbol_meaning_cache[cache_key] = candidate

    def _make_symbol_cache_key(self, symbol: str, scope: Optional[str]) -> str:
        base = (symbol or "").strip().lower()
        if not scope:
            return base
        return f"{scope}:{base}"

    def _is_better_definition(self, new: VariableDefinition, old: VariableDefinition) -> bool:
        source_rank = {"context": 3, "llm": 2, "heuristic": 1, "formula_only": 0}
        new_rank = source_rank.get(new.source, 0)
        old_rank = source_rank.get(old.source, 0)
        if new_rank != old_rank:
            return new_rank > old_rank
        return len((new.meaning or "").strip()) > len((old.meaning or "").strip())

    def _get_formula_summary(self, canonical_key: Optional[str],
                              formula_text: str,
                              variables: List[VariableDefinition]) -> Optional[str]:
        if self.llm_mode == "off" or not self.llm_service:
            return None
        if canonical_key and canonical_key in self._summary_cache:
            return self._summary_cache[canonical_key]
        summary = self.llm_service.extract_formula_summary(formula_text, variables)
        if canonical_key:
            self._summary_cache[canonical_key] = summary
        return summary

    def report_llm_stats(self) -> None:
        stats = self._llm_stats
        total = stats["extract_variables_calls"]
        cache_hits = stats["extract_variables_cache_hits"]
        symbol_hits = stats["extract_variables_symbol_hits"]
        print(
            "LLM stats (variables): "
            f"calls={total} cache_hits={cache_hits} symbol_cache_hits={symbol_hits}"
        )

    def _extract_variables(self, formula_text: str, context_text: str = "",
                           candidate_symbols: Optional[List[str]] = None,
                           canonical_key: Optional[str] = None) -> List[VariableDefinition]:
        """
        Hybrid approach: Rule-based first, LLM as fallback.
        
        Flow:
        1. Check if formula is simple → use rule-based extraction (FAST, FREE)
        2. If complex → try LLM extraction
        3. If LLM fails → fallback to heuristic
        
        This reduces LLM calls by 60-80% for typical textbooks.
        """
        # Cache hit by canonical formula signature
        cached = self._get_cached_variables_by_key(canonical_key)
        if cached:
            return cached

        # Candidate symbols (heuristic)
        if candidate_symbols is None:
            candidate_symbols = self._find_variable_candidates(formula_text, context_text)

        # If all symbols already known, reuse cached meanings (no LLM)
        if candidate_symbols:
            cached_symbols = self._get_cached_variables_by_symbols(candidate_symbols)
            if cached_symbols and len(cached_symbols) == len(candidate_symbols):
                return cached_symbols
        # Step 1: Check if formula is simple enough for rule-based extraction
        if self._is_simple_formula(formula_text):
            # Use heuristic extraction (no LLM call)
            if candidate_symbols:
                heuristic_vars = self._extract_variables_heuristic(formula_text, context_text)
                if heuristic_vars:
                    processed = self._post_process_variables(heuristic_vars, formula_text)
                    self._update_variable_caches(canonical_key, processed)
                    return processed
        
        # Step 2: Complex formula → try LLM extraction
        # Only call LLM if not in "off" mode AND lexicon mode is disabled
        if self.llm_mode != "off" and not self.use_lexicon and candidate_symbols:
            cached_symbols = self._get_cached_variables_by_symbols(candidate_symbols)
            cached_keys = {v.symbol.strip().lower() for v in cached_symbols}
            missing_symbols = [s for s in candidate_symbols if s.strip().lower() not in cached_keys]
            if not missing_symbols:
                return cached_symbols
            self._llm_stats["extract_variables_calls"] += 1
            llm_vars = self.llm_service.extract_variables(
                formula_text=formula_text,
                context=context_text,
                candidate_symbols=missing_symbols
            )
            
            # Step 3: Verify LLM results
            verified_vars = self._verify_variables(llm_vars, formula_text)
            
            if verified_vars:
                merged = cached_symbols + verified_vars
                processed = self._post_process_variables(merged, formula_text)
                self._update_variable_caches(canonical_key, processed)
                return processed
        
        # Step 4: Fallback to heuristic if LLM failed or was skipped
        if candidate_symbols:
            heuristic_vars = self._extract_variables_heuristic(formula_text, context_text)
            if heuristic_vars:
                processed = self._post_process_variables(heuristic_vars, formula_text)
                self._update_variable_caches(canonical_key, processed)
                return processed
            cached_symbols = self._get_cached_variables_by_symbols(candidate_symbols)
            if cached_symbols:
                return cached_symbols
        
        return []
    
    def _post_process_variables(self, variables: List[VariableDefinition], formula_text: str) -> List[VariableDefinition]:
        """Clean noisy variable symbols and drop low-signal entries."""
        if not variables:
            return []
        cleaned: Dict[str, VariableDefinition] = {}
        for var in variables:
            symbol = self._clean_variable_symbol(var.symbol)
            meaning = (var.meaning or "").strip()
            meaning_lower = meaning.lower()
            if any(phrase in meaning_lower for phrase in [
                "cannot provide", "can't provide", "not provide", "without the specific context",
                "without the context", "cannot determine", "not enough context", "insufficient context"
            ]):
                continue
            if not symbol or len(meaning) < 5:
                continue
            if not self._is_valid_variable_symbol(symbol):
                continue
            if symbol not in cleaned:
                cleaned[symbol] = VariableDefinition(
                    symbol=symbol,
                    meaning=meaning,
                    units=var.units,
                    inferred=var.inferred,
                    source=var.source,
                )
        return list(cleaned.values())
    
    def _clean_variable_symbol(self, symbol: str) -> str:
        sym = (symbol or "").strip()
        sym = re.sub(r'^[\s\(\[\{]+|[\s\)\]\}.,;:%]+$', '', sym)
        return sym
    
    def _is_valid_variable_symbol(self, symbol: str) -> bool:
        lower = symbol.lower()
        unit_blacklist = {
            "year", "years", "month", "months", "day", "days", "percent", "percentage",
            "million", "billion", "bp", "bps", "basis", "points", "dollar", "dollars",
            "usd", "eur", "yen"
        }
        if lower in unit_blacklist:
            return False
        if re.fullmatch(r'\d+(\.\d+)?', symbol):
            return False
        greek_names = {"alpha","beta","gamma","delta","sigma","theta","lambda","rho","pi","mu","nu","epsilon","eta","phi","psi","omega"}
        if lower in greek_names:
            return True
        # Allow single/two-letter, or short uppercase tokens like NPV, IRR
        if re.fullmatch(r'[A-Za-z]{1,4}', symbol):
            return True
        # Allow subscripts like r_f or σ_p
        if re.fullmatch(r'[A-Za-zα-ωΑ-Ω]{1,3}_[A-Za-z0-9]{1,6}', symbol):
            return True
        # Allow Greek letters symbols
        if re.fullmatch(r'[α-ωΑ-Ω]', symbol):
            return True
        return False
    
    def _looks_like_prose(self, text: str) -> bool:
        words = text.split()
        if len(words) >= 8:
            math_symbols = {'=', '∑', '√', '∫', 'σ', 'π', 'θ', 'λ', '+', '−', '×', '÷', '^', '**', '/', '\\'}
            math_token_count = sum(1 for w in words if any(s in w for s in math_symbols))
            if (math_token_count / max(len(words), 1)) < 0.2:
                return True
        return False
    
    def _is_equation_label(self, text: str) -> bool:
        clean = (text or "").strip()
        if re.fullmatch(r'\(?\d+(?:\.\d+)+[a-z]?\)?', clean):
            return True
        if re.search(r'\b(?:eq|equation)\.?\s*\d+(?:\.\d+)+', clean, re.IGNORECASE):
            return True
        return False
    
    def _find_variable_candidates(self, formula_text: str, context: str) -> List[str]:
        """
        Heuristic: Find symbols that appear in the formula.
        This is fast and helps LLM focus on relevant variables.
        Returns list of candidate symbol strings.
        """
        # Extract symbols from formula
        formula_symbols = set(re.findall(r'[A-Za-zα-ωΑ-Ω][A-Za-z0-9_()]*', formula_text))
        
        # Filter out common words and function tokens
        common_words = {
            'the', 'and', 'or', 'is', 'are', 'this', 'that', 'what', 'which', 
            'where', 'when', 'was', 'were', 'has', 'have', 'been', 'will', 
            'would', 'should', 'could', 'may', 'might', 'can', 'must', 'shall',
            'for', 'with', 'from', 'to', 'of', 'in', 'on', 'at', 'by', 'if',
            'then', 'else', 'let', 'denote', 'define', 'where', 'here'
        }
        function_tokens = {
            'log', 'ln', 'exp', 'max', 'min', 'sin', 'cos', 'tan', 'sqrt',
            'abs', 'sum', 'prod', 'mean', 'var', 'cov'
        }
        
        candidates = []
        for symbol in formula_symbols:
            symbol_lower = symbol.lower()
            # Skip common words
            if symbol_lower in common_words or symbol_lower in function_tokens:
                continue
            # Skip very long strings (likely not variables)
            if len(symbol) > 15:
                continue
            candidates.append(symbol)
        
        return candidates[:20]  # Limit to top 20 candidates
    
    def _verify_variables(self, llm_vars: List[VariableDefinition], formula_text: str) -> List[VariableDefinition]:
        """
        Verify that LLM-extracted variables actually appear in the formula.
        This prevents hallucination.
        """
        if not llm_vars:
            return []
        
        # Extract all symbols from formula (with flexibility for subscripts)
        formula_symbols = set(re.findall(r'[A-Za-zα-ωΑ-Ω][A-Za-z0-9_()]*', formula_text))
        
        verified = []
        for var in llm_vars:
            symbol = var.symbol.strip()
            
            # Direct match
            if symbol in formula_symbols:
                verified.append(var)
                continue
            
            # Flexible matching for subscripts (e.g., "r" matches "r_M" or "r_f")
            matched = False
            for fs in formula_symbols:
                # Exact match
                if symbol == fs:
                    matched = True
                    break
                # Subscript match: "r" in "r_M" or "r_M" contains "r"
                if len(symbol) <= 3 and symbol in fs:
                    matched = True
                    break
                if len(fs) <= 3 and fs in symbol:
                    matched = True
                    break
                # Greek letter match (e.g., "sigma" matches "σ")
                if self._is_greek_match(symbol, fs):
                    matched = True
                    break
            
            if matched:
                verified.append(var)
        
        return verified
    
    def _is_greek_match(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are the same Greek letter (name vs symbol)."""
        greek_map = {
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
            'sigma': 'σ', 'theta': 'θ', 'lambda': 'λ', 'rho': 'ρ',
            'pi': 'π', 'mu': 'μ', 'nu': 'ν', 'epsilon': 'ε',
            'eta': 'η', 'phi': 'φ', 'psi': 'ψ', 'omega': 'Ω'
        }
        
        s1_lower = symbol1.lower()
        s2_lower = symbol2.lower()
        
        # Check if one is Greek name and other is symbol
        if s1_lower in greek_map and greek_map[s1_lower] == symbol2:
            return True
        if s2_lower in greek_map and greek_map[s2_lower] == symbol1:
            return True
        
        return False
    
    def _extract_variables_heuristic(self, formula_text: str, context: str) -> List[VariableDefinition]:
        """
        Regex-based extraction for common textbook patterns.
        Pattern 1: "where X is the..." results in {symbol: X, meaning: ...}
        Pattern 2: "X = ..." (in context, immediately following)
        """
        variables = []
        
        # Extract symbols that actually appear in the formula
        formula_symbols = set(re.findall(r'[A-Za-zα-ωΑ-Ω][A-Za-z0-9_()]*', formula_text))
        
        # 1. "where [Symbol] is/represents/denotes [Meaning]"
        # Relaxed capture: Allow (), -, subscripts, and length up to 20 chars for things like "E(r_M)"
        # Capture: (where|here), (Symbol), (is/...), (Meaning until punctuation)
        pattern1 = r'(?:where|here|with)\s+([A-Za-zα-ωΑ-Ω0-9_()\-\u200b]{1,20})\s*(?:is|represents|denotes|=|are)\s*([^,.;\n]+)'
        
        # 2. "[Symbol] is the [Meaning]" (Start of sentence or phrase)
        pattern2 = r'(?:^|\.\s+)([A-Za-zα-ωΑ-Ω0-9_()\-\u200b]{1,20})\s+(?:is|is the)\s+([^,.;\n]+)'
        
        # 3. "[Meaning], denoted by [Symbol]"
        # Capture: (Meaning), (Symbol)
        pattern3 = r'([^,.;\n]{3,50}),\s+denoted by\s+([A-Za-zα-ωΑ-Ω0-9_]{1,10})'
        
        matches = re.findall(pattern1, context, re.IGNORECASE) + re.findall(pattern2, context, re.IGNORECASE)
        
        # Process pattern 3 (swap sym/mean)
        matches3 = re.findall(pattern3, context, re.IGNORECASE)
        for mean, sym in matches3:
            matches.append((sym, mean))
        
        unique_vars = {}
        for sym, mean in matches:
            sym = sym.strip()
            mean = mean.strip()
            if not sym or not mean: continue
            
            # CRITICAL: Only keep variables that actually appear in the formula
            # Extract actual symbols from formula (more carefully)
            formula_symbols_clean = set()
            common_words = {'the', 'and', 'or', 'is', 'are', 'this', 'that', 'what', 'which', 'where', 'when', 'was', 'were', 'has', 'have', 'been', 'will', 'would', 'should', 'could', 'may', 'might', 'can', 'must', 'shall', 'for', 'with', 'from', 'to', 'of', 'in', 'on', 'at', 'by'}
            
            for fs in formula_symbols:
                # Remove common words that aren't variables
                if len(fs) > 1 and fs.lower() not in common_words:
                    formula_symbols_clean.add(fs)
            
            # Check if symbol appears in formula (with some flexibility)
            if sym not in formula_symbols_clean:
                # Allow partial matches (e.g., "r_M" matches "r") but be reasonable
                matched = False
                for fs in formula_symbols_clean:
                    if sym == fs:
                        matched = True
                        break
                    # Allow if one is substring of other (for subscripts)
                    if (len(sym) <= 4 and sym in fs) or (len(fs) <= 4 and fs in sym):
                        matched = True
                        break
                if not matched:
                    continue
            
            # Filter: heuristics to filter out obviously wrong matches
            is_greek = sym.lower() in ['alpha', 'beta', 'gamma', 'delta', 'sigma', 'theta', 'lambda', 'rho', 'pi', 'mu', 'nu', 'epsilon', 'eta', 'phi', 'psi', 'omega']
            is_mathy = any(c in sym for c in "()_0123456789")
            is_single_letter = len(sym) == 1 and sym.isalpha()
            is_greek_symbol = sym in ['α', 'β', 'γ', 'δ', 'σ', 'θ', 'λ', 'ρ', 'π', 'μ', 'ν', 'Σ', '∫', '√', 'ε', 'η', 'φ', 'ψ', 'Ω']
            
            # Reject if it's a common word (not a variable)
            if sym.lower() in common_words:
                continue
            
            # Accept: single letters, greek symbols, short symbols with math chars, or known greek names
            # OR if it's 2-3 chars and appears in formula
            if not (is_single_letter or is_greek_symbol or is_mathy or is_greek):
                if len(sym) > 3:  # Allow up to 3 chars without special chars
                    continue 

            # Extract units if present (e.g., "r is the interest rate (in percent)")
            units = None
            units_match = re.search(r'\(in\s+([^)]+)\)|\(units?:\s*([^)]+)\)', mean, re.IGNORECASE)
            if units_match:
                units = units_match.group(1) or units_match.group(2)
                mean = re.sub(r'\(in\s+[^)]+\)|\(units?:\s*[^)]+\)', '', mean, flags=re.IGNORECASE).strip()
            
            if sym not in unique_vars:
                unique_vars[sym] = VariableDefinition(
                    symbol=sym,
                    meaning=mean,
                    units=units,
                    inferred=False,
                    source="context",
                )
        
        return list(unique_vars.values())

class TextBlockExtractor(BaseExtractor):
    def __init__(
        self,
        llm_service: LLMService = None,
        llm_mode: Literal["off", "light", "full"] = "full",

        reference_stub_types: Optional[Set[str]] = None,
        allowlist: Optional[Set[str]] = None,
    ):
        self.llm_service = llm_service or MockLLMService()
        self.llm_mode = llm_mode
        self.current_section: Optional[str] = None  # Track current section: "problem_set", "cfa", "concept_check", "concept_check_solutions", None
        self.last_problem_number: Optional[int] = None
        self.last_problem_section: Optional[str] = None
        self.allowlist = allowlist
        if reference_stub_types is None:
            self.reference_stub_types = {"table", "figure", "appendix", "exhibit", "section", "chapter"}
        else:
            self.reference_stub_types = {t.lower() for t in reference_stub_types}
    
    def _is_explanatory_text(self, text: str) -> bool:
        """
        Detect explanatory/prose text that might be misclassified.
        Returns True if it's likely prose, not a formula.
        """
        # Long text with low math content
        if len(text.split()) > 20:
            math_symbols = {'=', '∑', '√', '∫', 'σ', 'π', 'θ', 'λ', '+', '−', '×', '÷', '^', '**'}
            math_count = sum(1 for char in text if char in math_symbols)
            math_ratio = math_count / len(text) if text else 0
            
            # If math ratio < 0.05, it's prose
            if math_ratio < 0.05:
                return True
        
        # Contains prose indicators
        prose_indicators = [
            "risk-averse", "mean-variance", "investors measure",
            "the risk", "the return", "the portfolio",
            "we can", "it is", "this is", "that is"
        ]
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in prose_indicators):
            # But check if it has significant math
            if '=' not in text and not re.search(r'[A-Za-z]\d+|\d+[A-Za-z]', text):
                return True
        
        return False

    def process_page(self, page: fitz.Page, page_num: int, book_id: str, blocks: List[Any] = None,
                     doc_uri: Optional[str] = None) -> List[Any]:
        segments = []
        if blocks is None:
            blocks = page.get_text("blocks")
        
        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0: continue
            clean_text = text.strip()
            if not clean_text: continue

            # Problem set / CFA: aggressively split numbered items into questions
            if self.current_section in ("problem_set", "cfa"):
                # Number-only line (e.g., "11." or "11") -> start a question shell
                if re.match(r'^\s*\d+\s*[\.)]?\s*$', clean_text):
                    q_num = re.sub(r'[^\d]', '', clean_text)
                    if q_num:
                        self._maybe_switch_to_cfa(q_num)
                        seg = QuestionSegment(
                            segment_id=str(uuid.uuid4()),
                            book_id=book_id,
                            chapter_number="Unknown",
                            page_start=page_num,
                            page_end=page_num,
                            bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                            text_content=clean_text.strip(),
                            question_number=q_num,
                            subparts=[],
                            question_type="Mixed",
                            choices=[],
                            problem_set_type=self.current_section,
                            doc_uri=doc_uri,
                        )
                        segments.append(seg)
                        continue
                if re.search(r'^\s*\d+\s*[\.)]\s+', clean_text, re.MULTILINE):
                    chunks = self._split_numbered_blocks(clean_text)
                    for chunk in chunks:
                        m = re.match(r'^\s*(Q?\d+)\s*[\.)]\s', chunk)
                        if not m:
                            continue
                        q_num = m.group(1).lstrip("Q")
                        self._maybe_switch_to_cfa(q_num)
                        subparts = self._extract_subparts(chunk)
                        choices = self._extract_mcq_choices(chunk)
                        q_type = self._determine_question_type(chunk, choices)
                        seg = QuestionSegment(
                            segment_id=str(uuid.uuid4()),
                            book_id=book_id,
                            chapter_number="Unknown",
                            page_start=page_num,
                            page_end=page_num,
                            bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                            text_content=chunk.strip(),
                            question_number=q_num,
                            subparts=subparts,
                            question_type=q_type,
                            choices=choices,
                            problem_set_type=self.current_section,
                            doc_uri=doc_uri,
                        )
                        segments.append(seg)
                    continue

            # Concept check solutions: split multi-solution blocks if needed
            if self.current_section == "concept_check_solutions":
                if not re.match(r'^\d+\s*[\.)]\s', clean_text) and re.search(r'^\s*\d+\s*[\.)]\s', clean_text, re.MULTILINE):
                    chunks = self._split_numbered_blocks(clean_text)
                    for chunk in chunks:
                        if not re.match(r'^\d+\s*[\.)]\s', chunk):
                            continue
                        solution_steps = self._extract_solution_steps(chunk)
                        seg = SolutionSegment(
                            segment_id=str(uuid.uuid4()),
                            book_id=book_id,
                            chapter_number="Unknown",
                            page_start=page_num,
                            page_end=page_num,
                            bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                            text_content=chunk,
                            solution_steps=solution_steps,
                            solution_origin="book",
                            solution_sources=["pdf"],
                            doc_uri=doc_uri,
                            warnings=["concept_check_solution"]
                        )
                        segments.append(seg)
                    continue
            
            # Check if it's explanatory text (prose that might be misclassified)
            if self._is_explanatory_text(clean_text):
                # Optimization: Early filtering
                if self.allowlist and "explanatory_text" not in self.allowlist and "prose" not in self.allowlist:
                    continue
                    
                seg = ExplanatoryTextSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    text_type="prose",
                    doc_uri=doc_uri,
                )
                segments.append(seg)
                continue
            
            # Detect problem set sections
            section_type = self._detect_problem_section(clean_text)
            if section_type:
                self.current_section = section_type
                # Create a header segment for the section
                seg = HeaderSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    level=2,
                    header_number=None,
                    doc_uri=doc_uri,
                )
                segments.append(seg)
                continue
            
            # Concept check solutions: numbered answers inside solution section
            if self.current_section == "concept_check_solutions" and re.match(r'^\d+\s*[\.)]\s', clean_text):
                solution_steps = self._extract_solution_steps(clean_text)
                seg = SolutionSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    solution_steps=solution_steps,
                    solution_origin="book",
                    is_concept_check=True,
                    doc_uri=doc_uri,
                )
                segments.append(seg)
                continue

            # Derivations
            if self._is_derivation(clean_text):
                 seg = DerivationSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    steps=[clean_text],
                    doc_uri=doc_uri,
                )
                 segments.append(seg)
            
            # Calculation blocks (unlabeled stepwise computations)
            elif self._is_calculation_block(clean_text):
                calculation_steps = self._extract_calculation_steps(clean_text)
                output_vars = self._extract_output_variables(clean_text)
                
                seg = CalculationSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    steps=calculation_steps,
                    output_variables=output_vars,
                    doc_uri=doc_uri,
                )
                segments.append(seg)

            # Solution (detect before worked example to avoid confusion)
            elif self._is_solution(clean_text):
                solution_steps = self._extract_solution_steps(clean_text)
                seg = SolutionSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    solution_steps=solution_steps,
                    solution_origin="book",
                    doc_uri=doc_uri,
                )
                segments.append(seg)
            
            # Worked Example
            elif self._is_worked_example(clean_text):
                structure = {}
                if self.llm_mode != "off" and self.llm_service:
                    structure = self.llm_service.structure_worked_example(clean_text)
                # Extract given data and output variables heuristically
                given_data = self._extract_given_data(clean_text)
                output_vars = self._extract_output_variables(clean_text)
                steps = structure.get("steps", []) if isinstance(structure, dict) else []
                final_answer_raw = structure.get("final_answer") if isinstance(structure, dict) else None
                final_answer = str(final_answer_raw).strip() if final_answer_raw is not None else None
                example_prompt = structure.get("problem_statement") if isinstance(structure, dict) else None
                if not steps:
                    steps = self._fallback_example_steps(clean_text)
                if not example_prompt:
                    example_prompt = clean_text
                
                seg = WorkedExampleSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    title=structure.get("title") if isinstance(structure, dict) else None,
                    example_prompt=example_prompt,
                    given_data=given_data,
                    steps=steps,
                    final_answer=final_answer,
                    output_variables=output_vars,
                    doc_uri=doc_uri,
                )
                if not structure:
                    seg.needs_human_review = True
                segments.append(seg)
                
            # Concept Check question (detect before general questions)
            elif self._is_concept_check_question(clean_text):
                # Extract concept check number (e.g., "9.5" from "Concept Check 9.5")
                match = re.search(r'concept\s+check\s+(\d+\.\d+)', clean_text, re.IGNORECASE)
                q_num = match.group(1) if match else "Unknown"
                
                subparts = self._extract_subparts(clean_text)
                choices = self._extract_mcq_choices(clean_text)
                q_type = self._determine_question_type(clean_text, choices)
                
                seg = QuestionSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    question_number=q_num,
                    subparts=subparts,
                    question_type=q_type,
                    choices=choices,
                    problem_set_type="concept_check",
                    doc_uri=doc_uri,
                )
                segments.append(seg)
            
            # Question (numbered list) - only within problem set sections
            elif re.match(r'^(Q?\d+\.|[a-z]\))\s', clean_text):
                # Only extract questions if we're in a problem set section
                # OR if it's clearly a numbered problem (not just any numbered list)
                if self.current_section in ("summary", "key_terms", "key_equations"):
                    continue
                if self.current_section or self._is_likely_problem(clean_text):
                    match = re.match(r'^(Q?\d+\.|[a-z]\))', clean_text)
                    q_num = match.group(0).strip('.')
                    
                    # Extract subparts and MCQ choices
                    subparts = self._extract_subparts(clean_text)
                    choices = self._extract_mcq_choices(clean_text)
                    q_type = self._determine_question_type(clean_text, choices)
                    
                    # Determine problem set type
                    problem_set_type = self.current_section or "other"
                    
                    seg = QuestionSegment(
                        segment_id=str(uuid.uuid4()),
                        book_id=book_id,
                        chapter_number="Unknown",
                        page_start=page_num,
                        page_end=page_num,
                        bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                        text_content=clean_text,
                        question_number=q_num,
                        subparts=subparts,
                        question_type=q_type,
                        choices=choices,
                        problem_set_type=problem_set_type,
                        doc_uri=doc_uri,
                    )
                    segments.append(seg)
                
            # Header - check before other patterns
            elif self._is_header(clean_text, y0=y0, page_height=page.rect.height):
                # Try multiple patterns to extract header number
                header_num = None
                # Pattern 1: "9.1 Text" or "9.2.1 Text"
                match = re.search(r'^(\d+(?:\.\d+)+)', clean_text)
                if match:
                    header_num = match.group(1)
                # Pattern 2: "CHAPTER 9" or "Chapter 9"
                if not header_num:
                    ch_match = re.search(r'chapter\s+(\d+)', clean_text, re.IGNORECASE)
                    if ch_match:
                        header_num = ch_match.group(1)
                
                if header_num:
                    seg = HeaderSegment(
                        segment_id=str(uuid.uuid4()),
                        book_id=book_id,
                        chapter_number=header_num.split('.')[0] if '.' in header_num else header_num,
                        page_start=page_num,
                        page_end=page_num,
                        bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                        text_content=clean_text,
                        level=len(header_num.split('.')) if '.' in header_num else 1,
                        header_number=header_num,
                        doc_uri=doc_uri,
                    )
                    segments.append(seg)
        stub_segments: List[ReferenceStubSegment] = []
        for seg in segments:
            stub_segments.extend(self._build_reference_stub_segments(seg))
        return segments + stub_segments

    def _is_header(self, text: str, y0: Optional[float] = None, page_height: Optional[float] = None) -> bool:
        text_lower = text.lower().strip()
        if not text_lower:
            return False
        if text_lower.startswith(("example", "concept check")):
            return False
        if re.search(r'[=+\-*/]', text[:20]):
            return False

        score = 0
        if text_lower.startswith("chapter"):
            score += 2
        if re.match(r'^\d+(\.\d+)+(\s+|$)', text.strip()):
            score += 2
        if len(text) <= 80:
            score += 1
        if page_height and y0 is not None and (y0 / page_height) < 0.25:
            score += 1
        if text.istitle() or text.isupper():
            score += 1

        return score >= 3

    def _is_worked_example(self, text: str) -> bool:
        """Detect worked examples including implicit ones."""
        text_lower = text.lower()
        # Explicit examples: "Example 9.1", "Example:", "Worked Example"
        if re.match(r'^example\s+\d+\.\d+', text_lower):
            return True
        if text_lower.startswith("example") or "worked example" in text_lower[:50]:
            return True
        # Implicit examples: "Given:" + numbers + answer format
        has_given = "given" in text_lower[:100]
        has_numbers = bool(re.search(r'\d+\.?\d*', text))
        has_answer = any(kw in text_lower for kw in ["answer", "solution", "therefore", "thus", "find", "calculate"])
        if has_given and has_numbers and has_answer:
            return True
        return False

    def _extract_given_data(self, text: str) -> Dict[str, Any]:
        """Extract given data as key-value pairs."""
        given_data = {}
        # Pattern: "Given: X = 5, Y = 10" or "X = 5, Y = 10"
        given_match = re.search(r'given[:\s]+(.*?)(?:\.|$|find|calculate|determine)', text, re.IGNORECASE | re.DOTALL)
        if given_match:
            given_text = given_match.group(1)
            # Extract key-value pairs
            pairs = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*([0-9.]+)', given_text)
            for key, value in pairs:
                try:
                    given_data[key] = float(value) if '.' in value else int(value)
                except:
                    given_data[key] = value
        return given_data

    def _extract_output_variables(self, text: str) -> List[str]:
        """Extract what the example solves for."""
        output_vars = []
        # Patterns: "find X", "calculate Y", "determine Z"
        patterns = [
            r'(?:find|calculate|determine|solve for|compute)\s+([A-Za-z_][A-Za-z0-9_]*)',
            r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^=]+$'  # Last assignment
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            output_vars.extend(matches)
        return list(set(output_vars))[:5]  # Top 5

    def _fallback_example_steps(self, text: str) -> List[str]:
        """Fallback steps when LLM structure is missing."""
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if lines:
            return lines[:8]
        parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
        steps = [p.strip() for p in parts if p.strip()]
        return steps[:6] if steps else []

    def _extract_subparts(self, text: str) -> List[Dict[str, str]]:
        """Extract question subparts (a, b, c, etc.)."""
        subparts = []
        # Pattern: (a) text, (b) text, etc.
        pattern = r'\(([a-z])\)\s*([^()]+?)(?=\([a-z]\)|$)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        for label, subtext in matches:
            subparts.append({"label": label, "text": subtext.strip()})
        # Pattern: a. text, b) text on new lines
        if not subparts:
            line_pattern = r'(^|\n)\s*([a-z])[\.\)]\s*([^\n]+)'
            matches = re.findall(line_pattern, text, re.IGNORECASE)
            for _, label, subtext in matches:
                subparts.append({"label": label, "text": subtext.strip()})
        return subparts

    def _extract_mcq_choices(self, text: str) -> List[str]:
        """Extract MCQ choices (A, B, C, D)."""
        choices = []
        # Pattern: A. option, B. option, etc.
        pattern = r'([A-D])\.\s*([^\n]+?)(?=[A-D]\.|$)'
        matches = re.findall(pattern, text)
        for letter, choice_text in matches:
            choices.append(f"{letter}. {choice_text.strip()}")
        return choices

    def _extract_reference_markers(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract reference markers like Table 9.1, Figure 3.2, Appendix A."""
        if not text:
            return []
        markers: List[Tuple[str, str, str]] = []
        patterns = [
            r'\b(Table|Figure|Appendix|Exhibit|Section|Chapter)\s+([A-Za-z]?\d+(?:\.\d+)*[a-z]?)',
            r'\b(Appendix)\s+([A-Z])\b',
        ]
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                ref_type = match.group(1).lower()
                ref_id = match.group(2)
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 80)
                snippet = text[start:end].strip()
                markers.append((ref_type, ref_id, snippet[:200]))
        return markers

    def _build_reference_stub_segments(self, seg: SegmentBase) -> List[ReferenceStubSegment]:
        stubs: List[ReferenceStubSegment] = []
        markers = self._extract_reference_markers(seg.text_content or "")
        for ref_type, ref_id, snippet in markers:
            if self.reference_stub_types is not None and ref_type not in self.reference_stub_types:
                continue
            stubs.append(ReferenceStubSegment(
                segment_id=str(uuid.uuid4()),
                book_id=seg.book_id,
                chapter_number=seg.chapter_number or "Unknown",
                page_start=seg.page_start,
                page_end=seg.page_end,
                bbox=seg.bbox,
                text_content=f"{ref_type.title()} {ref_id}",
                heading_path=seg.heading_path,
                doc_uri=seg.doc_uri,
                ref_type=ref_type if ref_type in {"table", "figure", "appendix", "exhibit", "section", "chapter"} else "unknown",
                ref_id_text=str(ref_id),
                target_unknown=True,
                source_segment_id=seg.segment_id,
                snippet=snippet,
                needs_human_review=True,
            ))
        return stubs

    def _determine_question_type(self, text: str, choices: List[str] = None) -> Literal["MCQ", "Short Answer", "Calculation", "Case", "Essay", "Mixed"]:
        text_lower = text.lower()
        
        # MCQ detection
        if choices and len(choices) >= 2:
            return "MCQ"
        
        # Calculation
        if any(w in text_lower for w in ["calculate", "compute", "determine", "find", "estimate", "value", "what is the"]):
            return "Calculation"
        
        # Case study
        if any(w in text_lower for w in ["case", "scenario", "situation"]):
            return "Case"
        
        # Essay
        if any(w in text_lower for w in ["explain", "discuss", "analyze", "evaluate", "compare and contrast"]):
            if len(text) > 200:  # Long question
                return "Essay"
            return "Short Answer"
        
        return "Mixed"

    def _detect_problem_section(self, text: str) -> Optional[Literal[
        "problem_set", "cfa", "concept_check", "concept_check_solutions",
        "summary", "key_terms", "key_equations"
    ]]:
        """Detect if text is a problem set section header."""
        text_lower = text.lower().strip()
        
        # Problem Sets section
        if re.match(r'^problem\s+sets?', text_lower) or text_lower == "problem sets":
            return "problem_set"
        
        # CFA Problems section
        if (
            re.match(r'^cfa\s*(?:®|problems?)', text_lower, re.IGNORECASE)
            or ("cfa" in text_lower and "problem" in text_lower)
            or text_lower in {"cfa", "cfa®", "cfa problems"}
            or text_lower.startswith("for cfa problems")
        ):
            return "cfa"
        
        # Concept Check solutions section
        if re.match(r'^solutions?\s+to\s+concept\s+checks?$', text_lower, re.IGNORECASE):
            return "concept_check_solutions"
        
        # Concept Check section - but NOT individual concept check questions
        # Only detect section headers, not "Concept Check 9.5" questions
        if re.match(r'^concept\s+check\s*$', text_lower, re.IGNORECASE):
            return "concept_check"

        # Summary / Key terms / Key equations sections (avoid question extraction)
        if re.match(r'^summary\s*$', text_lower):
            return "summary"
        if re.match(r'^key\s+terms?\s*$', text_lower):
            return "key_terms"
        if re.match(r'^key\s+equations?\s*$', text_lower):
            return "key_equations"
        
        return None
    
    def _is_concept_check_question(self, text: str) -> bool:
        """Detect individual concept check questions (e.g., "Concept Check 9.5")."""
        text_lower = text.lower().strip()
        # Pattern: "Concept Check 9.5" or "Concept Check:" followed by number
        if re.match(r'^concept\s+check\s+\d+\.\d+', text_lower):
            return True
        return False
    
    def _is_likely_problem(self, text: str) -> bool:
        """Check if numbered text is likely a problem (not just any numbered list)."""
        # Problems usually have question-like content
        problem_indicators = [
            r'what\s+(?:is|are|would|should)',
            r'calculate|compute|determine|find|estimate',
            r'explain|discuss|analyze|evaluate',
            r'are\s+the\s+following\s+(?:true|false)',
            r'if\s+.*\s+what',
            r'given\s+.*\s+(?:what|find|calculate)',
        ]
        
        text_lower = text.lower()
        if "?" in text_lower:
            return True
        for pattern in problem_indicators:
            if re.search(pattern, text_lower):
                return True
        return False

    def _is_derivation(self, text: str) -> bool:
        start_lower = text.lower()[:50]
        keywords = ["proof", "derivation", "we can show", "substituting"]
        for k in keywords:
            if k in start_lower: return True
        return False
    
    def _is_calculation_block(self, text: str) -> bool:
        """
        Detect unlabeled calculation blocks (stepwise computations without "Example" heading).
        These are common in quant/finance textbooks.
        """
        text_lower = text.lower()
        
        # Exclude if it's already classified as something else
        if self._is_worked_example(text):
            return False
        if self._is_derivation(text):
            return False
        if self._is_solution(text):
            return False
        
        # Calculation indicators
        has_numbers = bool(re.search(r'\d+\.?\d*', text))
        has_equals = '=' in text
        has_math_ops = any(op in text for op in ['+', '-', '*', '/', '×', '÷', '^', '**'])
        
        # Multi-line with step indicators
        lines = text.split('\n')
        has_numbered_steps = any(re.match(r'^\d+[.)]\s+', line) for line in lines[:5])
        has_equals_in_lines = sum(1 for line in lines if '=' in line)
        
        # Pattern: multiple lines with calculations
        if has_numbers and has_equals and (has_math_ops or has_equals_in_lines >= 2):
            # Must have at least 2 calculation lines
            if has_equals_in_lines >= 2 or has_numbered_steps:
                return True
        
        # Pattern: "Given:" + calculations + result
        has_given = "given" in text_lower[:100]
        has_result = any(kw in text_lower for kw in ["therefore", "thus", "we get", "we find", "result", "answer"])
        if has_given and has_numbers and has_equals and has_result:
            return True
        
        return False
    
    def _extract_calculation_steps(self, text: str) -> List[str]:
        """Extract calculation steps from text."""
        steps = []
        
        # Pattern 1: Numbered steps (1., 2., etc.)
        numbered_pattern = r'^\d+[.)]\s*([^\n]+)'
        matches = re.findall(numbered_pattern, text, re.MULTILINE)
        if matches:
            steps.extend([m.strip() for m in matches if m.strip()])
        
        # Pattern 2: Lines with "=" (calculation lines)
        if not steps:
            lines = text.split('\n')
            calc_lines = [line.strip() for line in lines if '=' in line and len(line.strip()) > 5]
            if calc_lines:
                steps.extend(calc_lines[:10])  # Max 10 steps
        
        # Pattern 3: Split by sentence if no clear steps
        if not steps:
            # Split by periods, but keep math expressions together
            sentences = re.split(r'(?<=[.!?])\s+', text)
            steps = [s.strip() for s in sentences if s.strip() and len(s) > 10][:10]
        
        return steps

    def _is_solution(self, text: str) -> bool:
        """Detect solution blocks (but not worked examples)."""
        text_lower = text.lower().strip()
        
        # Skip very short text
        if len(text_lower) < 10:
            return False
        
        # Only detect solutions in "Solutions to Concept Checks" or similar sections
        # OR if explicitly marked as solution
        if self.current_section in ("concept_check", "concept_check_solutions"):
            # In concept check section, look for numbered solutions
            if re.match(r'^\d+[.:)]', text_lower):
                if self.current_section == "concept_check_solutions":
                    return True
                # Check if it has solution-like content
                if any(kw in text_lower[:200] for kw in ["solution", "answer", "therefore", "thus", "we find", "we get", "="]):
                    return True
        
        # Explicit solution markers (must be at start or after number)
        solution_keywords = ["solution to", "answer to", "solution:", "answer:"]
        if any(kw in text_lower[:100] for kw in solution_keywords):
            # But not if it's part of a worked example
            if not self._is_worked_example(text):
                return True
        
        # Pattern: "Solution Q1" or "Answer to Q1" or "Q1 Solution"
        solution_patterns = [
            r'^solution\s+(?:to\s+)?(?:question\s+)?(?:Q?\d+)',
            r'^answer\s+(?:to\s+)?(?:question\s+)?(?:Q?\d+)',
            r'^(?:Q?\d+)\s+solution',
            r'^(?:Q?\d+)\s+answer',
        ]
        for pattern in solution_patterns:
            if re.match(pattern, text_lower):
                if not self._is_worked_example(text):
                    return True
        
        return False

    def _extract_solution_steps(self, text: str) -> List[str]:
        """Extract solution steps from text."""
        steps = []
        # Pattern: "Step 1:", "1.", numbered list
        step_patterns = [
            r'step\s+\d+[.:]\s*([^\n]+)',
            r'^\d+[.)]\s*([^\n]+)',
        ]
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            steps.extend([m.strip() for m in matches if m.strip()])
        
        # If no numbered steps, split by sentences
        if not steps:
            sentences = re.split(r'[.!?]\s+', text)
            steps = [s.strip() for s in sentences if s.strip() and len(s) > 10][:10]  # Max 10 steps
        
        return steps

    def _maybe_switch_to_cfa(self, q_num: str) -> None:
        """Heuristic: if numbering resets after 20+, treat as CFA block."""
        if not q_num or not q_num.isdigit():
            return
        num = int(q_num)
        if self.current_section == "problem_set" and self.last_problem_number is not None:
            if self.last_problem_number >= 20 and num <= 2:
                self.current_section = "cfa"
        if self.current_section in ("problem_set", "cfa"):
            self.last_problem_number = num
            self.last_problem_section = self.current_section

    def _split_numbered_blocks(self, text: str) -> List[str]:
        """
        Split a block into chunks starting with numbered items (e.g., '1.', '2)').
        """
        if not text:
            return []
        chunks: List[str] = []
        current: List[str] = []
        for line in text.splitlines():
            if re.match(r'^\s*\d+\s*[\.)]\s+', line):
                if current:
                    chunks.append("\n".join(current).strip())
                current = [line.strip()]
            else:
                if current:
                    current.append(line.rstrip())
        if current:
            chunks.append("\n".join(current).strip())
        return [c for c in chunks if c]

# Linker

class Linker:

    def __init__(self, concept_linker: Optional[ConceptLinker] = None,
                 solution_generator: Optional['SolutionGenerator'] = None,
                 solution_validator: Optional['SolutionValidator'] = None,
                 validate_top_n_only: Optional[int] = None):
        self.concept_linker = concept_linker
        self.solution_generator = solution_generator
        self.solution_validator = solution_validator
        self.validate_top_n_only = validate_top_n_only  # If set, validate only top N candidates (saves LLM calls)
    
    def link_segments(self, segments: List[SegmentBase]) -> Tuple[List[SegmentBase], List[Edge]]:
        edge_evidence: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

        def add_evidence(source_id: str, target_id: str, edge_type: str, ev: Dict[str, Any]):
            key = (source_id, target_id, edge_type)
            edge_evidence.setdefault(key, []).append(ev)

        # 1. Link formulas by equation number
        formulas_by_eq_num: Dict[str, FormulaSegment] = {}
        all_formulas: Dict[str, FormulaSegment] = {}
        for seg in segments:
            if isinstance(seg, FormulaSegment):
                all_formulas[seg.segment_id] = seg
                if seg.equation_number:
                    norm_num = self._normalize_eq_num(seg.equation_number)
                    formulas_by_eq_num[norm_num] = seg

        # 2. Link formula references
        for seg in segments:
            if isinstance(seg, (WorkedExampleSegment, QuestionSegment, SolutionSegment, DerivationSegment, CalculationSegment, FormulaSegment)):
                refs = self._find_eq_references(seg.text_content)
                if seg.context_after:
                    refs.extend(self._find_eq_references(seg.context_after))
                if seg.context_before:
                    refs.extend(self._find_eq_references(seg.context_before))
                
                refs = {(r["eq_num"], r["snippet"]) for r in refs}
                
                for ref_num, snippet in refs:
                    norm_ref = self._normalize_eq_num(ref_num)
                    if norm_ref in formulas_by_eq_num:
                        target_id = formulas_by_eq_num[norm_ref].segment_id
                        if target_id == seg.segment_id: continue
                        if target_id not in seg.referenced_formula_ids:
                            seg.referenced_formula_ids.append(target_id)
                        add_evidence(
                            seg.segment_id,
                            target_id,
                            "REFERENCES",
                            {"method": "equation_number", "snippet": snippet, "page": seg.page_start, "bbox": seg.bbox},
                        )
                
                # Fallback: variable-overlap heuristic uses ALL formulas, top N by overlap (avoid one-formula hub)
                if not seg.referenced_formula_ids:
                    linked = self._heuristic_link_by_variables(seg, all_formulas, max_refs=3)
                    for fid in linked:
                        add_evidence(
                            seg.segment_id,
                            fid,
                            "USES_FORMULA",
                            {"method": "variable_overlap", "snippet": self._truncate_snippet(seg.text_content), "page": seg.page_start, "bbox": seg.bbox},
                        )
        
        # 2b. Derivation linking: set derived_from_formula_ids, derived_to_formula_id, link_type
        for seg in segments:
            if isinstance(seg, DerivationSegment) and seg.referenced_formula_ids:
                refs = list(seg.referenced_formula_ids)
                seg.derived_from_formula_ids = refs[:-1] if len(refs) > 1 else refs
                seg.derived_to_formula_id = refs[-1] if refs else None
                ctx = ((seg.context_before or "") + " " + (seg.text_content or "") + " " + (seg.context_after or "")).lower()
                if re.search(r'(?:see|using|from)\s+(?:eq|equation)', ctx):
                    seg.link_type = "REFERENCES"
                else:
                    seg.link_type = "EXPLAINS"
        
        # 3. Link questions to solutions
        self.link_questions_to_solutions(segments)

        # 3b. Add explicit ANSWER_OF evidence
        for seg in segments:
            if isinstance(seg, SolutionSegment) and seg.solution_for_question_id:
                method = "question_number"
                add_evidence(
                    seg.segment_id,
                    seg.solution_for_question_id,
                    "ANSWER_OF",
                    {"method": method, "snippet": self._truncate_snippet(seg.text_content), "page": seg.page_start, "bbox": seg.bbox},
                )
        
        # 4. Bidirectional linking: Formula ↔ WorkedExample
        self._create_bidirectional_example_links(segments, formulas_by_eq_num)
        
        # 5. Link concepts (if concept_linker is available)
        if self.concept_linker:
            for seg in segments:
                concept_links = self.concept_linker.link_segment_to_concepts(seg)
                seg.concept_links.extend(concept_links)

        # 5b. Deduplicate formulas by canonical key
        self._deduplicate_formulas(segments, edge_evidence)
        
        # 6. Build explicit edges for KG traversal
        edges = self._build_edges(segments, edge_evidence)
        return segments, edges
    
    def _build_edges(self, segments: List[SegmentBase], edge_evidence: Optional[Dict[Tuple[str, str, str], List[Dict[str, Any]]]] = None) -> List[Edge]:
        """Build explicit typed edges from segment link fields."""
        out: List[Edge] = []
        seen: Set[Tuple[str, str, str]] = set()  # (source_id, target_id, edge_type) for dedup
        
        def anchor(seg: SegmentBase) -> Dict[str, Any]:
            return {"page": seg.page_start, "heading_path": getattr(seg, "heading_path", None) or ""}
        
        def add(source_id: str, target_id: str, edge_type: str, strength: float = 1.0,
                link_method: Optional[str] = None, anchor_meta: Optional[Dict[str, Any]] = None):
            key = (source_id, target_id, edge_type)
            if key in seen or not target_id:
                return
            seen.add(key)
            meta = dict(anchor_meta) if anchor_meta else {}
            if edge_evidence and key in edge_evidence and edge_evidence[key]:
                first = edge_evidence[key][0]
                if first.get("method"):
                    meta["method"] = first["method"]
                if first.get("snippet") is not None:
                    meta["snippet"] = first["snippet"]
                if first.get("page") is not None:
                    meta["page"] = first["page"]
            out.append(Edge(
                edge_id=str(uuid.uuid4()),
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                strength=strength,
                link_method=link_method,
                anchor_metadata=meta or None,
            ))
        
        for seg in segments:
            meta = anchor(seg)
            # NEAR: adjacency
            if seg.prev_segment_id:
                add(seg.prev_segment_id, seg.segment_id, "NEAR", anchor_meta=meta)
            if seg.next_segment_id:
                add(seg.segment_id, seg.next_segment_id, "NEAR", anchor_meta=meta)
            # USES_FORMULA: example/question/solution/derivation/calculation -> formula
            if isinstance(seg, (WorkedExampleSegment, QuestionSegment, SolutionSegment, DerivationSegment, CalculationSegment)):
                for fid in getattr(seg, "referenced_formula_ids", []) or []:
                    # If REFERENCES evidence exists, create REFERENCES edge instead
                    if edge_evidence and (seg.segment_id, fid, "REFERENCES") in edge_evidence:
                        add(seg.segment_id, fid, "REFERENCES", anchor_meta=meta)
                    else:
                        add(seg.segment_id, fid, "USES_FORMULA", anchor_meta=meta)
            # EXPLAINS: derivation -> derived_to_formula
            if isinstance(seg, DerivationSegment) and getattr(seg, "derived_to_formula_id", None):
                add(seg.segment_id, seg.derived_to_formula_id, "EXPLAINS", anchor_meta=meta)
            # ANSWER_OF: solution -> question
            if isinstance(seg, SolutionSegment) and getattr(seg, "solution_for_question_id", None):
                add(seg.segment_id, seg.solution_for_question_id, "ANSWER_OF", anchor_meta=meta)
            # WORKED_EXAMPLE_OF: worked_example -> concept
            if isinstance(seg, WorkedExampleSegment):
                for cl in seg.concept_links:
                    lm = "exact" if cl.link_method == "exact_match" else cl.link_method
                    add(seg.segment_id, cl.concept_id, "WORKED_EXAMPLE_OF", strength=cl.confidence,
                        link_method=lm if lm in ("exact", "alias", "semantic", "heuristic") else None, anchor_meta=meta)
            # DEFINES: definition formula -> concept
            if isinstance(seg, FormulaSegment) and getattr(seg, "usage_type", None) == "definition":
                for cl in seg.concept_links:
                    lm = "exact" if cl.link_method == "exact_match" else cl.link_method
                    add(seg.segment_id, cl.concept_id, "DEFINES", strength=cl.confidence,
                        link_method=lm if lm in ("exact", "alias", "semantic", "heuristic") else None, anchor_meta=meta)
            # DUPLICATE_OF: formula -> canonical formula
            if isinstance(seg, FormulaSegment) and getattr(seg, "is_duplicate_of", None):
                add(seg.segment_id, seg.is_duplicate_of, "DUPLICATE_OF", anchor_meta=meta)
            # REFERENCES: question/solution -> formula (same as USES_FORMULA; we already add USES_FORMULA above)
        return out
    
    def _create_bidirectional_example_links(self, segments: List[SegmentBase], 
                                           formulas_map: Dict[str, FormulaSegment]) -> None:
        """
        Create bidirectional links: Formula ↔ WorkedExample.
        After linking example -> formula, also link formula -> example.
        """
        # Build formula map by segment_id
        formula_by_id: Dict[str, FormulaSegment] = {}
        for seg in segments:
            if isinstance(seg, FormulaSegment):
                formula_by_id[seg.segment_id] = seg
        
        # For each worked example, create reverse links
        for seg in segments:
            if isinstance(seg, WorkedExampleSegment):
                for formula_id in seg.referenced_formula_ids:
                    formula = formula_by_id.get(formula_id)
                    if formula and seg.segment_id not in formula.referenced_example_ids:
                        formula.referenced_example_ids.append(seg.segment_id)

    def link_questions_to_solutions(self, segments: List[SegmentBase]) -> None:
        """
        Link questions to solutions using multi-signal confidence matching.
        Based on research: https://ceur-ws.org/Vol-2674/paper02.pdf
        """
        seg_by_id: Dict[str, SegmentBase] = {s.segment_id: s for s in segments}
        questions = [s for s in segments if isinstance(s, QuestionSegment)]
        solutions = [s for s in segments if isinstance(s, SolutionSegment)]
        
        # Build question reference keys and group by chapter
        question_by_ref: Dict[str, QuestionSegment] = {}
        questions_by_main_chapter: Dict[str, List[QuestionSegment]] = defaultdict(list)
        
        def get_main_chapter(ch_num: Optional[str]) -> str:
            if not ch_num or ch_num == "Unknown":
                return "Unknown"
            return ch_num.split('.')[0]

        for q in questions:
            # Generate normalized reference key
            ref_key = self._generate_problem_ref_key(q)
            if ref_key:
                q.problem_ref_key = ref_key
                question_by_ref[ref_key] = q
            
            # Group by main chapter
            main_ch = get_main_chapter(q.chapter_number)
            questions_by_main_chapter[main_ch].append(q)
        
        # Identify "Global Solution Chapters" (e.g. Appendix, "Solutions")
        global_sol_chapters = set()
        for s in segments:
            if isinstance(s, HeaderSegment) and s.chapter_number:
                title = (s.chapter_title or "").lower()
                if "solution" in title or "answer" in title or "appendix" in title:
                    global_sol_chapters.add(get_main_chapter(s.chapter_number))

        # Try to match solutions to questions using multi-signal approach
        for sol in solutions:
            best_match = None
            best_score = 0.0
            best_method: Optional[str] = None
            
            # Optimization: Candidate Reduction
            sol_main_ch = get_main_chapter(sol.chapter_number)
            candidates = []
            
            # 1. Same Main Chapter
            candidates.extend(questions_by_main_chapter.get(sol_main_ch, []))
            
            # 2. If Solution is in a Global Solution Chapter, or Unknown, widen search
            is_global = sol_main_ch in global_sol_chapters or sol_main_ch == "Unknown"
            
            if is_global:
                candidates = questions
            else:
                # 3. Also check Unknown questions just in case
                if "Unknown" in questions_by_main_chapter:
                    candidates.extend(questions_by_main_chapter["Unknown"])
            
            for q in candidates:
                score, method = self._multi_signal_match(q, sol)
                if score > best_score:
                    best_score = score
                    best_match = q
                    best_method = method
            
            # Link if confidence > 0.75 (as per requirements)
            if best_match and best_score >= 0.75:
                sol.solution_for_question_id = best_match.segment_id
                sol.link_confidence = best_score
                sol.match_method = best_method
                best_match.solution_status = "linked"
        
        # Set status for questions without solutions; collect those we will generate for
        unattempted_for_generate: List[QuestionSegment] = []
        for q in questions:
            if q.solution_status == "unattempted":
                if q.problem_set_type == "concept_check":
                    concept_check_solutions = [s for s in solutions if s.chapter_number == q.chapter_number]
                    if not concept_check_solutions:
                        q.solution_status = "not_found_in_book"
                    else:
                        q.solution_status = "ambiguous"
                else:
                    q.solution_status = "not_found_in_book"
                    unattempted_for_generate.append(q)

        # Generate synthetic solutions: only not_found_in_book, batched by chapter (1 LLM call per chapter).
        # Validator is optional: without it we take the first candidate per question.
        if self.solution_generator and unattempted_for_generate:
            formulas = [s for s in segments if isinstance(s, FormulaSegment)]
            by_chapter: Dict[str, List[QuestionSegment]] = defaultdict(list)
            for q in unattempted_for_generate:
                by_chapter[q.chapter_number or "Unknown"].append(q)
            # Inner helper so we can safely run per-chapter generation in parallel
            def _generate_for_chapter(ch_key: str, q_list: List[QuestionSegment]):
                """
                Generate synthetic solutions for a batch of questions in one chapter.
                Returns list of (question, solution) pairs. Exceptions are propagated
                to allow caller to implement sequential fallback per batch.
                """
                relevant_formulas = [
                    f for f in formulas
                    if f.chapter_number == ch_key
                    or f.segment_id in {fid for q in q_list for fid in q.referenced_formula_ids}
                ]
                if not relevant_formulas:
                    relevant_formulas = formulas[:10]
                batch_candidates = self.solution_generator.generate_candidates_batch(
                    q_list, relevant_formulas, max_per_batch=10
                )
                results: List[Tuple[QuestionSegment, SolutionSegment]] = []
                for q, cand_list in zip(q_list, batch_candidates):
                    if not cand_list:
                        continue
                    synthetic_solution = self._validate_and_create_solution(q, cand_list, relevant_formulas)
                    if synthetic_solution:
                        results.append((q, synthetic_solution))
                return results

            # Run chapter batches in parallel (IO-bound LLM work); keep all graph mutations
            # on the main thread to avoid race conditions on shared lists/dicts.
            max_workers = 3
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_generate_for_chapter, ch, q_list): ch
                    for ch, q_list in by_chapter.items()
                }
                for future in as_completed(future_map):
                    ch = future_map[future]
                    try:
                        chapter_results = future.result()
                    except Exception as e:
                        # Fallback: if a parallel batch fails (e.g., rate limit / timeout),
                        # retry that chapter sequentially so the overall run can continue.
                        print(f"[SolutionGenerator] Parallel generation failed for chapter {ch}, "
                              f"falling back to sequential. Error: {e}")
                        try:
                            chapter_results = _generate_for_chapter(ch, by_chapter.get(ch, []))
                        except Exception as e2:
                            print(f"[SolutionGenerator] Sequential fallback failed for chapter {ch}: {e2}")
                            continue

                    for q, synthetic_solution in chapter_results:
                        solutions.append(synthetic_solution)
                        segments.append(synthetic_solution)
                        q.solution_status = "synthetic"
                        synthetic_solution.solution_for_question_id = q.segment_id
                        synthetic_solution.link_confidence = synthetic_solution.validation_score or 0.0
                        synthetic_solution.prev_segment_id = q.segment_id
                        synthetic_solution.next_segment_id = q.next_segment_id
                        if q.next_segment_id and q.next_segment_id in seg_by_id:
                            seg_by_id[q.next_segment_id].prev_segment_id = synthetic_solution.segment_id
                        q.next_segment_id = synthetic_solution.segment_id
                        seg_by_id[synthetic_solution.segment_id] = synthetic_solution
    
    def _validate_and_create_solution(self, question: QuestionSegment, candidates: List[Dict[str, Any]],
                                       relevant_formulas: List[FormulaSegment]) -> Optional[SolutionSegment]:
        """Validate candidate(s) and create one SolutionSegment. Reused by single and batch generation."""
        if not candidates:
            return None
        best = None
        validation_payload = None
        if self.solution_validator:
            to_validate = candidates[: self.validate_top_n_only] if self.validate_top_n_only else candidates
            validated = self.solution_validator.validate_candidates(
                question=question,
                candidates=to_validate,
                formulas=relevant_formulas
            )
            for candidate in validated:
                score = candidate.get('validation', {}).get('overall_score', 0.0)
                if score >= 0.7:
                    best = candidate
                    validation_payload = candidate.get('validation')
                    break
            if not best and validated:
                best = validated[0]
                validation_payload = best.get('validation')
        else:
            best = candidates[0]
        if not best:
            return None
        steps = best.get('solution_steps', []) or []
        answer = best.get('final_answer')
        if not steps and not answer:
            return None
        return SolutionSegment(
            segment_id=str(uuid.uuid4()),
            book_id=question.book_id,
            chapter_number=question.chapter_number,
            chapter_title=question.chapter_title,
            page_start=question.page_start,
            page_end=question.page_end,
            bbox=question.bbox,
            text_content=format_solution_text(best),
            context_before=question.context_before,
            context_after=question.context_after,
            heading_path=question.heading_path,
            solution_steps=best.get('solution_steps', []),
            referenced_formula_ids=question.referenced_formula_ids,
            is_synthetic=True,
            solution_origin="synthetic",
            validation_score=validation_payload.get('overall_score', 0.0) if validation_payload else None,
            link_confidence=validation_payload.get('overall_score', 0.0) if validation_payload else 0.0
        )

    def _generate_and_validate_solution(self, question: QuestionSegment, 
                                        all_segments: List[SegmentBase]) -> Optional[SolutionSegment]:
        """Generate (and optionally validate) a synthetic solution for a question (single-question path)."""
        if not self.solution_generator:
            return None
        formulas = [s for s in all_segments if isinstance(s, FormulaSegment)]
        relevant_formulas = [
            f for f in formulas 
            if f.segment_id in question.referenced_formula_ids or f.chapter_number == question.chapter_number
        ]
        if not relevant_formulas:
            relevant_formulas = formulas[:5]
        context = (question.context_before or "") + " " + (question.context_after or "")
        candidates = self.solution_generator.generate_candidates(
            question=question, formulas=relevant_formulas, context=context.strip()
        )
        return self._validate_and_create_solution(question, candidates, relevant_formulas)
    
    def _generate_problem_ref_key(self, question: QuestionSegment) -> Optional[str]:
        """Generate normalized reference key: Ch1-Q5 or 1.2-5"""
        ch_num = question.chapter_number
        q_num = self._normalize_question_number(question.question_number)
        
        if ch_num != "Unknown" and q_num:
            # Normalize chapter number
            ch_clean = re.sub(r'[^\d.]', '', ch_num.split('.')[0] if '.' in ch_num else ch_num)
            suffix = self._problem_set_suffix(question.problem_set_type)
            return f"Ch{ch_clean}-Q{q_num}{suffix}"
        return None
    
    def _generate_solution_ref_key(self, solution: SolutionSegment) -> Optional[str]:
        """Generate normalized reference key for solution (used internally for matching)."""
        ch_num = solution.chapter_number
        sol_num = self._extract_question_number_from_solution(solution.text_content)
        
        if ch_num != "Unknown" and sol_num:
            ch_clean = re.sub(r'[^\d.]', '', ch_num.split('.')[0] if '.' in ch_num else ch_num)
            suffix = "-CC" if solution.is_concept_check else ""
            return f"Ch{ch_clean}-S{sol_num}{suffix}"
        return None

    def _problem_set_suffix(self, problem_set_type: Optional[str]) -> str:
        if not problem_set_type:
            return ""
        mapping = {
            "problem_set": "-PS",
            "cfa": "-CFA",
            "concept_check": "-CC",
            "other": "-OT",
        }
        return mapping.get(problem_set_type, "")

    def _strip_ref_key_suffix(self, ref_key: str) -> str:
        return re.sub(r'-(PS|CFA|CC|OT)$', '', ref_key)
    
    def _multi_signal_match(self, question: QuestionSegment, solution: SolutionSegment) -> Tuple[float, Optional[str]]:
        """
        Multi-signal confidence matching algorithm.
        Returns: (total_score, primary_match_method)
        """
        evidence: List[Tuple[str, float, Optional[str]]] = []  # (method, score, details)
        total_score = 0.0
        sol_ref_key = self._generate_solution_ref_key(solution)

        # Concept Check: match by leading solution number (1., 2., ...) to 9.1-9.5, etc.
        if question.problem_set_type == "concept_check" and question.question_number:
            sol_num = self._extract_leading_solution_number(solution.text_content)
            q_norm = self._normalize_question_number(question.question_number)
            if sol_num and (q_norm.endswith(f".{sol_num}") or q_norm == sol_num):
                total_score += 0.70
                evidence.append(("exact_question_number", 0.70, f"Concept check number match: {q_norm} <- {sol_num}"))
                if solution.is_concept_check:
                    total_score += 0.20
                    evidence.append(("taxonomy_path", 0.20, "Concept check solution section hint"))
        
        # Signal 1: Reference Key Match (highest precision, +0.70)
        if question.problem_ref_key and sol_ref_key:
            if question.problem_ref_key == sol_ref_key:
                total_score += 0.70
                evidence.append(("exact_question_number", 0.70, f"Matched via REF_KEY = {question.problem_ref_key}"))
            elif question.problem_ref_key.replace("-Q", "-S") == sol_ref_key:
                total_score += 0.70
                evidence.append(("exact_question_number", 0.70, f"Matched via REF_KEY = {question.problem_ref_key}"))
            else:
                qk = self._strip_ref_key_suffix(question.problem_ref_key).replace("-Q", "-S")
                sk = self._strip_ref_key_suffix(sol_ref_key)
                if qk == sk:
                    total_score += 0.70
                    evidence.append(("exact_question_number", 0.70, f"Matched via REF_KEY (suffix ignored) = {question.problem_ref_key}"))
        
        # Signal 2: Taxonomy Path Constraint (+0.10 or hard reject)
        if question.chapter_number != "Unknown" and solution.chapter_number != "Unknown":
            if question.chapter_number == solution.chapter_number:
                total_score += 0.10
                evidence.append(("taxonomy_path", 0.10, f"Chapter aligned: {question.chapter_number}"))
            else:
                ch_q = int(re.search(r'\d+', question.chapter_number).group()) if re.search(r'\d+', question.chapter_number) else 0
                ch_s = int(re.search(r'\d+', solution.chapter_number).group()) if re.search(r'\d+', solution.chapter_number) else 0
                if abs(ch_q - ch_s) > 1:
                    return (0.0, None)  # Hard reject
        
        # Signal 3: Concept_ID Overlap (+0.10 to +0.25)
        q_concept_ids = {c.concept_id for c in question.concept_links}
        s_concept_ids = {c.concept_id for c in solution.concept_links}
        if q_concept_ids and s_concept_ids:
            overlap = len(q_concept_ids & s_concept_ids) / max(len(q_concept_ids | s_concept_ids), 1)
            if overlap > 0:
                concept_score = min(0.10 + overlap * 0.15, 0.25)
                total_score += concept_score
                evidence.append(("concept_overlap", concept_score, f"Concept overlap: {overlap:.2f}"))
        
        # Signal 4: Formula/Equation Anchor (+0.30 bonus)
        shared_formulas = set(question.referenced_formula_ids) & set(solution.referenced_formula_ids)
        if shared_formulas:
            formula_bonus = min(0.30, len(shared_formulas) * 0.10)
            total_score += formula_bonus
            evidence.append(("equation_anchor", formula_bonus, f"Shared formulas: {shared_formulas}"))
        
        # Signal 5: Variable Overlap
        if question.referenced_formula_ids:
            q_text_lower = question.text_content.lower()
            s_text_lower = solution.text_content.lower()
            q_words = set(q_text_lower.split())
            s_words = set(s_text_lower.split())
            var_overlap = len(q_words & s_words) / max(len(q_words | s_words), 1)
            if var_overlap > 0.3:
                total_score += 0.10
                evidence.append(("variable_overlap", 0.10, f"Variable overlap: {var_overlap:.2f}"))
        
        # Signal 6: Text Similarity (fallback)
        if total_score < 0.5:
            text_sim = self._fuzzy_match_question_solution(question, solution)
            if text_sim > 0.6:
                text_score = text_sim * 0.20
                total_score += text_score
                evidence.append(("text_similarity", text_score, f"Text similarity: {text_sim:.2f}"))
        
        primary_method = evidence[0][0] if evidence else None
        return (min(total_score, 1.0), primary_method)

    def _normalize_question_number(self, q_num: str) -> str:
        """Normalize question number for matching."""
        if not q_num:
            return ""
        normalized = re.sub(r'^\s*Q\s*', '', q_num.strip(), flags=re.IGNORECASE)
        normalized = normalized.replace(")", "").replace("(", "").replace(" ", "")
        dot_match = re.match(r'(\d+(?:\.\d+)*)([a-z])?$', normalized, re.IGNORECASE)
        if dot_match:
            num = dot_match.group(1)
            sub = dot_match.group(2) or ""
            return f"{num}{sub}".lower()
        loose_match = re.search(r'(\d+)\s*([a-z])?', normalized, re.IGNORECASE)
        if loose_match:
            num = loose_match.group(1)
            sub = loose_match.group(2) or ""
            return f"{num}{sub}".lower()
        return normalized.lower()

    def _extract_question_number_from_solution(self, text: str) -> Optional[str]:
        """Extract question number from solution text."""
        # Patterns: "Solution to Q12(a)", "Answer to 12a", "For question 2(b)", etc.
        patterns = [
            r'solution\s+to\s+(?:question\s+)?(?:Q\s*)?(\d+)\s*\(?([a-z])?\)?',
            r'answer\s+to\s+(?:question\s+)?(?:Q\s*)?(\d+)\s*\(?([a-z])?\)?',
            r'for\s+question\s+(?:Q\s*)?(\d+)\s*\(?([a-z])?\)?',
            r'question\s+(?:Q\s*)?(\d+)\s*\(?([a-z])?\)?\s*:',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                num = match.group(1)
                sub = match.group(2) or ""
                return f"{num}{sub}".strip()
        return None

    def _extract_leading_solution_number(self, text: str) -> Optional[str]:
        """Extract leading numeric label from a solution block (e.g., '1.' -> '1')."""
        if not text:
            return None
        match = re.match(r'^\s*(\d+)\s*[.:)]', text.strip())
        if match:
            return match.group(1)
        return None

    def _fuzzy_match_question_solution(self, question: QuestionSegment, solution: SolutionSegment) -> float:
        """Calculate similarity score between question and solution."""
        q_words = set(question.text_content.lower().split()[:20])  # First 20 words
        s_words = set(solution.text_content.lower().split()[:20])
        
        if not q_words or not s_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(q_words & s_words)
        union = len(q_words | s_words)
        return intersection / union if union > 0 else 0.0

    def _heuristic_link_by_variables(self, source_seg: SegmentBase, all_formulas: Dict[str, FormulaSegment], max_refs: int = 3):
        """Link by variable overlap; add top max_refs by overlap count to avoid one-formula hub."""
        source_text = (source_seg.text_content or "").lower()
        if source_seg.context_before:
            source_text += " " + (source_seg.context_before or "").lower()
        if source_seg.context_after:
            source_text += " " + (source_seg.context_after or "").lower()
        scored: List[Tuple[int, str]] = []
        for fid, formula in all_formulas.items():
            if fid == getattr(source_seg, "segment_id", None):
                continue
            overlap = 0
            for var in formula.variables:
                sym, meaning = var.symbol.lower(), (var.meaning or "").lower()
                if sym and sym in source_text:
                    overlap += 1
                if meaning and len(meaning) > 2 and meaning in source_text:
                    overlap += 1
            if overlap > 0:
                scored.append((overlap, fid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        linked: List[str] = []
        for _, fid in scored[:max_refs]:
            if fid not in source_seg.referenced_formula_ids:
                source_seg.referenced_formula_ids.append(fid)
                linked.append(fid)
        return linked

    def _normalize_eq_num(self, eq_num: str) -> str:
        return eq_num.replace('(', '').replace(')', '').strip()

    def _find_eq_references(self, text: str) -> List[Dict[str, str]]:
        refs: List[Dict[str, str]] = []
        if not text:
            return refs
        patterns = [
            r'(?:Eq\.|Equation)\s*\(?(\d+\.\d+)\)?',
            r'\((\d+\.\d+)\)',
        ]
        for pat in patterns:
            for match in re.finditer(pat, text):
                eq_num = match.group(1)
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 80)
                snippet = text[start:end].strip()
                refs.append({"eq_num": eq_num, "snippet": snippet[:200]})
        return refs

    def _truncate_snippet(self, text: Optional[str], limit: int = 200) -> Optional[str]:
        if not text:
            return None
        cleaned = " ".join(text.split())
        return cleaned[:limit]

    def _deduplicate_formulas(self, segments: List[SegmentBase],
                              edge_evidence: Dict[Tuple[str, str, str], List[Dict[str, Any]]]) -> None:
        grouped: Dict[Tuple[str, str, str], List[FormulaSegment]] = defaultdict(list)
        for seg in segments:
            if isinstance(seg, FormulaSegment) and seg.canonical_formula_key:
                key = (seg.book_id, seg.chapter_number or "Unknown", seg.canonical_formula_key)
                grouped[key].append(seg)

        for (_, _, canonical_key), formulas in grouped.items():
            if len(formulas) <= 1:
                continue
            formulas.sort(key=lambda s: (s.page_start, s.bbox.y0 if s.bbox else 0))
            canonical = formulas[0]
            for dup in formulas[1:]:
                dup.is_duplicate_of = canonical.segment_id
                edge_key = (dup.segment_id, canonical.segment_id, "DUPLICATE_OF")
                edge_evidence.setdefault(edge_key, []).append({
                    "method": "heuristic",
                    "snippet": f"canonical_formula_key={canonical_key}",
                    "page": dup.page_start,
                    "bbox": dup.bbox,
                })

# ==========================================
# 5.5. SOLUTION GENERATOR & VALIDATOR
# ==========================================

class SolutionGenerator:
    """Generate synthetic solutions using multi-agent approach."""
    
    _solution_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self, llm_services: List[LLMService], claude_service: Optional[ClaudeLLMService] = None, enable_cache: bool = True):
        self.llm_services = llm_services
        self.claude = claude_service
        self.enable_cache = enable_cache
    
    def generate_candidates(self, question: QuestionSegment, formulas: List[FormulaSegment], 
                           context: str = "") -> List[Dict[str, Any]]:
        """
        Generate multiple solution candidates using different AI agents.
        Returns list of candidate dicts with 'solution_steps', 'final_answer', 'agent', etc.
        """
        candidates = []
        
        # Get formula texts for context
        formula_texts = [f.formula_text_raw for f in formulas[:5]]
        formula_context = "\n".join([f"Formula: {f.formula_text_raw}\nVariables: {', '.join([v.symbol for v in f.variables[:3]])}" 
                                    for f in formulas[:3]])
        
        prompt = self._build_solution_prompt(question, formula_context)
        for llm in self.llm_services:
            try:
                cache_key = self._cache_key(prompt, llm)
                solution = self._cache_get(cache_key) if self.enable_cache else None
                if solution is None:
                    solution = generate_json_with_llm(llm, prompt, temperature=0.3)
                    if solution and self.enable_cache:
                        self._cache_set(cache_key, solution)
                if solution:
                    solution['agent'] = self._agent_name(llm)
                    solution['model'] = getattr(llm, "model_name", None)
                    candidates.append(solution)
            except Exception as e:
                print(f"Solution generation failed ({self._agent_name(llm)}): {e}")
        
        # Candidate: Claude (alternative approach)
        if self.claude and self.claude.client:
            try:
                claude_solution = self.claude.generate_solution(
                    question_text=question.text_content,
                    formulas=formula_texts,
                    context=formula_context
                )
                if claude_solution and 'solution_steps' in claude_solution:
                    claude_solution['agent'] = 'claude'
                    claude_solution['model'] = 'claude'
                    candidates.append(claude_solution)
            except Exception as e:
                print(f"Claude solution generation failed: {e}")
        
        # Fallback: stepwise prompt with first available LLM
        if not candidates and self.llm_services:
            try:
                stepwise_prompt = self._build_stepwise_prompt(question, formula_context)
                cache_key = self._cache_key(stepwise_prompt, self.llm_services[0], suffix="stepwise")
                solution = self._cache_get(cache_key) if self.enable_cache else None
                if solution is None:
                    solution = generate_json_with_llm(self.llm_services[0], stepwise_prompt, temperature=0.5)
                    if solution and self.enable_cache:
                        self._cache_set(cache_key, solution)
                if solution:
                    solution['agent'] = f"{self._agent_name(self.llm_services[0])}-stepwise"
                    solution['model'] = getattr(self.llm_services[0], "model_name", None)
                    candidates.append(solution)
            except Exception as e:
                print(f"Stepwise generation failed: {e}")
        
        return candidates

    def generate_candidates_batch(self, questions: List[QuestionSegment], formulas: List[FormulaSegment],
                                   max_per_batch: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Generate one solution per question in a single LLM call (per batch). Only unattempted questions.
        Returns List[List[Dict]] — one candidate per question, so validation flow can reuse.
        """
        if not questions or not self.llm_services:
            return [[] for _ in questions]
        formula_context = "\n".join([
            f"Formula: {f.formula_text_raw}\nVariables: {', '.join([v.symbol for v in f.variables[:3]])}"
            for f in formulas[:5]
        ])
        
        # Helper for processing a single batch
        def process_batch(start_idx: int, chunk: List[QuestionSegment]) -> List[List[Dict[str, Any]]]:
            parts = []
            for idx, q in enumerate(chunk, 1):
                parts.append(f"[{idx}] Question: {q.text_content}\nType: {q.question_type or 'N/A'}")
            prompt = """Solve each of the following questions. For each question provide solution_steps (array of strings) and final_answer. Same order as [1], [2], ...

Relevant Formulas:
""" + formula_context + """

""" + "\n\n".join(parts) + """

Return a JSON object with a single key "solutions" whose value is an array of objects. Each object has "solution_steps" (array of strings) and "final_answer" (string). Same order as [1], [2], ...
Example: {"solutions": [{"solution_steps": ["Step 1...", "Step 2..."], "final_answer": "42"}, ...]}
Return ONLY valid JSON."""
            
            batch_results = [[] for _ in chunk]
            try:
                service = self.llm_services[0]
                cache_key = self._cache_key(prompt, service, suffix="batch")
                data = self._cache_get(cache_key) if self.enable_cache else None
                
                if data is None:
                    try:
                        data = generate_json_with_llm(service, prompt, temperature=0.3)
                        if data and self.enable_cache:
                            self._cache_set(cache_key, data)
                    except Exception as e:
                        print(f"Batch {start_idx} generation error: {e}")
                        return batch_results

                if not data:
                    print(f"Solution batch (q {start_idx+1}-{start_idx+len(chunk)}): no valid JSON")
                    return batch_results
                
                sols = data.get("solutions") or data.get("Solutions")
                if not isinstance(sols, list):
                    sols = next((v for k, v in (data or {}).items() if isinstance(v, list)), None) or []
                
                if not sols:
                    return batch_results

                for i, s in enumerate(sols):
                    if i >= len(chunk):
                        break
                    if isinstance(s, dict) and (s.get("solution_steps") or s.get("final_answer")):
                        s = dict(s)
                        s["agent"] = f"{self._agent_name(service)}"
                        s["model"] = getattr(service, "model_name", None)
                        batch_results[i] = [s]
            except Exception as e:
                print(f"Batch {start_idx} error: {e}")
            
            return batch_results

        # Parallel Execution
        from concurrent.futures import ThreadPoolExecutor
        chunks = []
        for start in range(0, len(questions), max_per_batch):
            chunks.append((start, questions[start : start + max_per_batch]))
            
        results_map = {}
        # Max concurrency 3
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_start = {
                executor.submit(process_batch, start, chunk): start 
                for start, chunk in chunks
            }
            for future in future_to_start:
                start = future_to_start[future]
                try:
                    res = future.result()
                    results_map[start] = res
                except Exception as e:
                    print(f"Batch {start} failed: {e}")
                    chunk_size = len(next(c for s, c in chunks if s == start))
                    results_map[start] = [[] for _ in range(chunk_size)]

        final_out = []
        for start, _ in chunks:
            final_out.extend(results_map.get(start, []))
            
        return final_out
    
    def _agent_name(self, llm: LLMService) -> str:
        name = llm.__class__.__name__.replace("LLMService", "").lower()
        return name or "llm"

    def _cache_key(self, prompt: str, llm: LLMService, suffix: str = "") -> str:
        model = getattr(llm, "model_name", "")
        base = f"{self._agent_name(llm)}|{model}|{suffix}|{prompt}"
        return hashlib.md5(base.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._solution_cache.get(key)

    def _cache_set(self, key: str, value: Dict[str, Any]) -> None:
        self._solution_cache[key] = value

    def load_cache(self, cache_dir: str) -> None:
        """Load solution cache from disk (for full-book re-runs / resume)."""
        path = os.path.join(cache_dir, "solution_cache.json")
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._solution_cache.update(data)
                print(f"Loaded {len(data)} cached solutions from {path}")
        except Exception as e:
            print(f"Warning: could not load solution cache: {e}")

    def save_cache(self, cache_dir: str) -> None:
        """Persist solution cache to disk (survives process restart)."""
        if not self._solution_cache:
            return
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "solution_cache.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._solution_cache, f, ensure_ascii=False, indent=0)
            print(f"Saved {len(self._solution_cache)} solution cache entries to {path}")
        except Exception as e:
            print(f"Warning: could not save solution cache: {e}")

    def _build_solution_prompt(self, question: QuestionSegment, formula_context: str) -> str:
        return f"""Solve this question step by step, showing all work.

Question: {question.text_content}
Question Type: {question.question_type}

Relevant Formulas:
{formula_context}

Provide a complete solution with:
1. Step-by-step reasoning
2. Formula applications where relevant
3. Clear calculations
4. Final answer

Return JSON with:
- "solution_steps": array of step strings
- "final_answer": the final answer
- "formulas_used": array of formula identifiers used"""
        
    def _build_stepwise_prompt(self, question: QuestionSegment, formula_context: str) -> str:
        return f"""Solve this question with detailed step-by-step breakdown.

Question: {question.text_content}

Formulas available:
{formula_context}

Break down the solution into clear, numbered steps. Show:
- What information is given
- What needs to be found
- Which formulas to use
- How to apply them
- Final calculation

Return JSON with:
- "solution_steps": array of detailed step strings
- "final_answer": the final answer"""


class SolutionValidator:
    """Validate solutions using multi-agent cross-validation."""
    
    def __init__(self, openai_service: OpenAILLMService, claude_service: ClaudeLLMService):
        self.openai = openai_service
        self.claude = claude_service
    
    def validate_candidates(self, question: QuestionSegment, candidates: List[Dict[str, Any]],
                           formulas: List[FormulaSegment]) -> List[Dict[str, Any]]:
        """
        Validate all candidates using strict scoring criteria.
        Returns list of validated candidates with scores.
        """
        validated = []
        
        for candidate in candidates:
            validation_result = self._validate_single(question, candidate, formulas)
            candidate['validation'] = validation_result
            validated.append(candidate)
        
        # Sort by overall score (descending)
        validated.sort(key=lambda x: x['validation'].get('overall_score', 0.0), reverse=True)
        
        return validated
    
    def _validate_single(self, question: QuestionSegment, candidate: Dict[str, Any],
                        formulas: List[FormulaSegment]) -> Dict[str, Any]:
        """
        Validate a single candidate using multiple validators.
        Returns comprehensive validation result.
        """
        solution_text = format_solution_text(candidate)
        formula_texts = [f.formula_text_raw for f in formulas[:5]]
        
        # Get validations from multiple agents
        validations = []
        
        # Validator 1: OpenAI
        if self.openai and self.openai.client:
            openai_validation = self._validate_with_openai(question, solution_text, formula_texts)
            if openai_validation:
                openai_validation['validator'] = 'openai'
                validations.append(openai_validation)
        
        # Validator 2: Claude
        if self.claude and self.claude.client:
            claude_validation = self.claude.validate_solution(
                question_text=question.text_content,
                solution_text=solution_text,
                formulas=formula_texts
            )
            if claude_validation:
                claude_validation['validator'] = 'claude'
                validations.append(claude_validation)
        
        # Combine validations (strict scoring)
        return self._combine_validations(validations, question, candidate, formulas)
    
    def _validate_with_openai(self, question: QuestionSegment, solution_text: str,
                             formula_texts: List[str]) -> Optional[Dict[str, Any]]:
        """Validate using OpenAI."""
        if not self.openai or not self.openai.client:
            return None
        
        formulas_hint = "\n".join([f"- {f}" for f in formula_texts[:3]])
        
        prompt = f"""Evaluate this solution for correctness, completeness, and formula usage.

Question: {question.text_content}
Question Type: {question.question_type}

Relevant Formulas:
{formulas_hint}

Solution:
{solution_text}

Rate the solution on these criteria (0.0-1.0 each):
1. **Correctness**: Is the answer mathematically/logically correct?
2. **Completeness**: Are all necessary steps shown?
3. **Formula Usage**: Are correct formulas identified and applied?
4. **Clarity**: Is the explanation clear and easy to follow?
5. **Formula Alignment**: Does it align with expected derivation patterns?

Return JSON with:
- "overall_score": weighted average (0.0-1.0)
- "correctness": 0.0-1.0
- "completeness": 0.0-1.0
- "formula_usage": 0.0-1.0
- "clarity": 0.0-1.0
- "formula_alignment": 0.0-1.0
- "issues": array of specific problems
- "strengths": array of positive aspects"""
        
        try:
            response = self.openai.client.chat.completions.create(
                model=self.openai.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2  # Low temperature for consistent evaluation
            )
            
            data = json.loads(response.choices[0].message.content)
            return data
        except Exception as e:
            print(f"OpenAI validation error: {e}")
            return None
    
    def _combine_validations(self, validations: List[Dict[str, Any]], question: QuestionSegment,
                            candidate: Dict[str, Any], formulas: List[FormulaSegment]) -> Dict[str, Any]:
        """
        Combine multiple validation results with strict scoring.
        Uses minimum scores (most conservative) for critical criteria.
        """
        if not validations:
            return {"overall_score": 0.0, "issues": ["No validations available"]}
        
        # Extract scores from all validators
        scores = {
            'correctness': [],
            'completeness': [],
            'formula_usage': [],
            'clarity': [],
            'formula_alignment': []
        }
        
        all_issues = []
        all_strengths = []
        
        for val in validations:
            for key in scores.keys():
                if key in val:
                    scores[key].append(val[key])
            if 'issues' in val:
                issues = val.get('issues') or []
                if isinstance(issues, list):
                    all_issues.extend(issues)
                else:
                    all_issues.append(str(issues))
            if 'strengths' in val:
                strengths = val.get('strengths') or []
                if isinstance(strengths, list):
                    all_strengths.extend(strengths)
                else:
                    all_strengths.append(str(strengths))
        
        # Strict scoring: Use minimum (most conservative) for critical criteria
        # Use average for less critical
        combined = {
            'correctness': min(scores['correctness']) if scores['correctness'] else 0.0,
            'completeness': min(scores['completeness']) if scores['completeness'] else 0.0,
            'formula_usage': min(scores['formula_usage']) if scores['formula_usage'] else 0.0,
            'clarity': sum(scores['clarity']) / len(scores['clarity']) if scores['clarity'] else 0.0,
            'formula_alignment': min(scores.get('formula_alignment', [0.0])) if scores.get('formula_alignment') else 0.0,
        }
        
        # Formula alignment check
        formula_alignment_score = self._check_formula_alignment(candidate, formulas)
        combined['formula_alignment'] = min(combined['formula_alignment'], formula_alignment_score)
        
        # Weighted overall score (correctness and formula usage are critical)
        weights = {
            'correctness': 0.35,
            'formula_usage': 0.25,
            'formula_alignment': 0.15,
            'completeness': 0.15,
            'clarity': 0.10
        }
        
        overall = sum(combined[key] * weights[key] for key in combined.keys())
        
        # Additional checks
        def _dedupe_strings(values: List[Any]) -> List[str]:
            cleaned: List[str] = []
            for v in values:
                if isinstance(v, (str, int, float)):
                    cleaned.append(str(v))
                else:
                    cleaned.append(json.dumps(v, ensure_ascii=True))
            return list(dict.fromkeys(cleaned))
        issues = _dedupe_strings(all_issues)
        strengths = _dedupe_strings(all_strengths)
        
        return {
            'overall_score': overall,
            **combined,
            'issues': issues,
            'strengths': strengths,
            'num_validators': len(validations)
        }
    
    def _check_formula_alignment(self, candidate: Dict[str, Any], 
                                 formulas: List[FormulaSegment]) -> float:
        """Check if solution aligns with expected formula usage patterns."""
        solution_text = format_solution_text(candidate).lower()
        
        # Check if solution mentions formula variables
        formula_vars = set()
        for f in formulas:
            for var in f.variables:
                formula_vars.add(var.symbol.lower())
                formula_vars.add(var.meaning.lower())
        
        # Count how many formula variables appear in solution
        mentioned = sum(1 for var in formula_vars if var in solution_text)
        total = len(formula_vars) if formula_vars else 1
        
        alignment = mentioned / total if total > 0 else 0.0
        return min(alignment, 1.0)


# Pipeline

class Pipeline:
    def __init__(self, concept_list_path: Optional[str] = None, use_openai: bool = True,
                 generate_missing_solutions: bool = True, target_chapter: Optional[str] = None,
                 chapter_mode: Literal["single_chapter", "multi_chapter"] = "single_chapter",
                 llm_mode: Literal["light", "full", "off"] = "full",
                 batch_size: Optional[int] = None,
                 llm_backend: Literal["openai", "gemini", "claude", "openrouter", "ollama", "groq"] = "openai",
                 validate_generated_solutions: bool = True,
                 use_variable_lexicon: bool = True,
                 openrouter_model: str = "meta-llama/llama-3.2-3b-instruct:free",
                 groq_model: str = "llama-3.1-8b-instant",
                 ollama_model: str = "llama3.2",
                 ollama_base_url: str = "http://localhost:11434/v1",
                 segment_type_allowlist: Optional[Set[str]] = None,
                 reference_stub_types: Optional[Set[str]] = None,
                 enable_llm_disambiguation: bool = False,
                 llm_disambiguation_batch_size: int = 8,
                 validate_top_n_only: Optional[int] = None):
        """
        openrouter_model: Model name for OpenRouter.
        llm_backend: Provider for variable extraction/summary/worked_example.
        ollama_model: Model name for local Ollama.
        ollama_base_url: Ollama base URL.
        validate_top_n_only: If set (e.g. 1), validate only top N solution candidates per question (saves LLM time on full book).
        """
        self.llm_mode = llm_mode
        self.batch_size = batch_size
        self.llm_backend = llm_backend
        self.use_variable_lexicon = use_variable_lexicon
        self.segment_type_allowlist = {t for t in segment_type_allowlist} if segment_type_allowlist else None
        self.enable_llm_disambiguation = enable_llm_disambiguation
        self.llm_disambiguation_batch_size = llm_disambiguation_batch_size
        if llm_mode == "off":
            self.llm_service = MockLLMService()
        elif llm_backend == "gemini":
            self.llm_service = GeminiLLMService()
        elif llm_backend == "claude":
            # Anthropic Claude (model can be adjusted here).
            self.llm_service = AnthropicLLMService(model="claude-3-sonnet-20240229")
        elif llm_backend == "ollama":
            # Ollama exposes an OpenAI-compatible endpoint without API keys.
            print(f"Using Ollama: model={ollama_model}, base_url={ollama_base_url}")
            self.llm_service = OpenAILLMService(
                api_key="ollama",
                model=ollama_model,
                enable_cache=True,
                base_url=ollama_base_url
            )
        elif llm_backend == "openrouter":
            # OpenRouter uses an OpenAI-compatible API.
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENROUTER_API_KEY not found. API key is recommended.")
            self.llm_service = OpenAILLMService(
                api_key=api_key or "sk-or-v1-...",
                model=openrouter_model,
                enable_cache=True,
                base_url="https://openrouter.ai/api/v1"
            )
        elif llm_backend == "groq":
            # Groq uses an OpenAI-compatible API.
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("Warning: GROQ_API_KEY not found. API key is required.")
            self.llm_service = OpenAILLMService(
                api_key=api_key,
                model=groq_model,
                enable_cache=True,
                base_url="https://api.groq.com/openai/v1"
            )
        elif llm_backend == "openai" or use_openai:
            self.llm_service = OpenAILLMService(enable_cache=True)
        else:
            self.llm_service = GeminiLLMService()
        
        # Initialize Claude service for multi-agent solution generation
        self.claude_service = ClaudeLLMService()
        
        # Initialize solution generator and validator (multi-agent)
        # Only enable if generate_missing_solutions is True AND llm_mode is not "off"
        solution_generator = None
        solution_validator = None
        if generate_missing_solutions and llm_mode != "off":
            solution_generator = SolutionGenerator(
                llm_services=[self.llm_service],
                claude_service=self.claude_service
            )
            if validate_generated_solutions:
                solution_validator = SolutionValidator(
                    openai_service=self.llm_service if isinstance(self.llm_service, OpenAILLMService) else None,
                    claude_service=self.claude_service
                )
        
        self.formula_extractor = FormulaExtractor(
            llm_service=self.llm_service,
            llm_mode=self.llm_mode,
            use_lexicon=self.use_variable_lexicon
        )
        self.text_block_extractor = TextBlockExtractor(
            llm_service=self.llm_service,
            llm_mode=self.llm_mode,
            reference_stub_types=reference_stub_types,
            allowlist=segment_type_allowlist
        )
        concept_llm = self.llm_service if (self.llm_mode != "off") else None
        self.concept_linker = ConceptLinker(
            concept_list_path,
            llm_service=concept_llm,
            use_llm_rerank=False
        ) if concept_list_path else None
        self.linker = Linker(
            concept_linker=self.concept_linker,
            solution_generator=solution_generator,
            solution_validator=solution_validator,
            validate_top_n_only=validate_top_n_only
        )
        self.furniture_detector = FurnitureDetector()
        self.context_processor = ContextProcessor(mode=chapter_mode, target_chapter=target_chapter)

    def process_pdf(self, pdf_path: str, book_id: str, page_range: Tuple[int, int] = None,
                    doc_uri: Optional[str] = None) -> SegmentationOutput:
        doc = fitz.open(pdf_path)
        start_page, end_page = page_range if page_range else (0, len(doc))
        start_page = max(0, start_page)
        end_page = min(len(doc), end_page)
        doc_uri = doc_uri or pdf_path

        chapter_title_map = self._build_chapter_title_map(doc)
        print(f"Processing pages {start_page} to {end_page}...")
        # Scan for furniture (globally, with sampling/caching)
        self.furniture_detector.scan_document(doc, pdf_path)

        all_blocks_map: Dict[int, List[Tuple]] = {}
        # Pre-load only the requested pages for processing
        for i in range(start_page, end_page):
            page = doc[i]
            all_blocks_map[i+1] = page.get_text("blocks")
        
        # Build clean blocks per page (for both formula and text extractors)
        clean_by_page: Dict[int, List[Tuple]] = {}
        for i, page_idx in enumerate(range(start_page, end_page)):
            page_num = page_idx + 1
            page = doc[page_idx]
            raw_blocks = all_blocks_map.get(page_num, [])
            page_height = page.rect.height
            clean_by_page[page_num] = [
                b for b in raw_blocks
                if not self.furniture_detector.is_furniture(b, page_height)
            ]
        
        all_segments: List[SegmentBase] = []
        total_pages = end_page - start_page
        use_batch = (
            self.batch_size is not None and self.batch_size > 0
            and self.llm_mode != "off"
            and hasattr(self.llm_service, "extract_variables_batch")
        )
        if use_batch:
            print(f"Using batched variable extraction (batch_size={self.batch_size})...")
            formula_segments = self.formula_extractor.process_blocks_batch(
                clean_by_page, doc, start_page, end_page, book_id, doc_uri=doc_uri, batch_size=self.batch_size
            )
            all_segments.extend(formula_segments)
        for i, page_idx in enumerate(range(start_page, end_page)):
            page_num = page_idx + 1
            page = doc[page_idx]
            clean_blocks = clean_by_page.get(page_num, [])
            print(f"Processing page {page_num}/{end_page} ({i+1}/{total_pages})...")
            if not use_batch:
                formulas = self.formula_extractor.process_page(
                    page, page_num, book_id, blocks=clean_blocks, doc_uri=doc_uri
                )
                all_segments.extend(formulas)
            text_blocks = self.text_block_extractor.process_page(
                page, page_num, book_id, blocks=clean_blocks, doc_uri=doc_uri
            )
            all_segments.extend(text_blocks)
        
        print(f"Extracted {len(all_segments)} segments. Processing context and linking...")
            
        self.context_processor = ContextProcessor(
            mode=self.context_processor.mode,
            target_chapter=self.context_processor.target_chapter,
            chapter_title_map=chapter_title_map
        )
        print("  [1/6] Context processing (heading paths, merges, adjacency)...")
        all_segments = self.context_processor.process(all_segments)
        if self.segment_type_allowlist:
            print("  [2/6] Applying segment allowlist filter...")
            all_segments = self._filter_segments_by_type(all_segments, self.segment_type_allowlist)
            all_segments = self.context_processor.add_adjacency_links(all_segments)
            print(f"        Filtered segments: {len(all_segments)}")
        if self.enable_llm_disambiguation and self.llm_mode != "off":
            print("  [3/6] LLM disambiguation (ambiguous segments only)...")
            all_segments = self._llm_disambiguate_segments(all_segments)
        print("  [4/6] Variable lexicon application (if enabled)...")
        if self.use_variable_lexicon and self.llm_mode != "off":
            self._apply_variable_lexicon(all_segments)
        print("  [5/6] Linking segments (concepts, questions/solutions, edges)...")
        all_segments, edges = self.linker.link_segments(all_segments)
        print("  [6/6] Building chapter metadata...")
        self.formula_extractor.report_llm_stats()
        
        # Detect solution presence and build chapter metadata
        chapters = self._build_chapter_metadata(all_segments, chapter_title_map)
            
        return SegmentationOutput(
            metadata={"source_pdf": pdf_path, "total_pages": len(doc)},
            chapters=chapters,
            segments=all_segments,
            edges=edges
        )

    def _llm_disambiguate_segments(self, segments: List[SegmentBase]) -> List[SegmentBase]:
        """
        Use LLM to disambiguate only suspicious segments (batch mode).
        LLM returns label: "keep" or "ignore". Ignore removes the segment.
        """
        if not self.llm_service:
            return segments

        candidates: List[SegmentBase] = []
        for seg in segments:
            if isinstance(seg, FormulaSegment):
                raw = (seg.formula_text_raw or seg.text_content or "").strip()
                if seg.equation_number:
                    continue
                if len(raw) <= 6 or re.fullmatch(r'[A-Za-zα-ωΑ-Ω][_a-zA-Z0-9]*', raw):
                    candidates.append(seg)
            elif isinstance(seg, QuestionSegment):
                if seg.problem_set_type in ("problem_set", "cfa", "concept_check"):
                    continue
                raw = (seg.text_content or "").strip()
                if re.fullmatch(r'\d+\s*[.)]?', raw):
                    candidates.append(seg)

        if not candidates:
            return segments

        keep_ids: Set[str] = set(s.segment_id for s in segments)
        batch_size = max(1, int(self.llm_disambiguation_batch_size))
        for i in range(0, len(candidates), batch_size):
            chunk = candidates[i:i + batch_size]
            items = []
            for s in chunk:
                items.append({
                    "id": s.segment_id,
                    "type": s.segment_type,
                    "text": (s.text_content or "").strip(),
                    "context_before": (s.context_before or "").strip(),
                    "context_after": (s.context_after or "").strip(),
                    "heading_path": (s.heading_path or "")
                })
            prompt = (
                "You are reviewing extracted segments from a textbook PDF. "
                "For each item, decide if it should be kept as a segment or ignored. "
                "Items are short labels or number-only lines that are often noise. "
                "Return JSON: {\"items\": [{\"id\": \"...\", \"label\": \"keep\"|\"ignore\", \"reason\": \"...\"}]}\n\n"
                f"Items:\n{json.dumps(items, ensure_ascii=False)}"
            )
            data = generate_json_with_llm(self.llm_service, prompt)
            if not data or "items" not in data:
                continue
            for it in data.get("items", []):
                if it.get("label") == "ignore":
                    keep_ids.discard(it.get("id"))

        return [s for s in segments if s.segment_id in keep_ids]

    def _apply_variable_lexicon(self, segments: List[SegmentBase]) -> None:
        """Apply chapter/section lexicon + fallback LLM for unknown symbols."""
        # Group segments by chapter
        formulas_by_ch: Dict[str, List[FormulaSegment]] = {}
        sections_by_ch: Dict[str, Dict[str, List[FormulaSegment]]] = {}
        for seg in segments:
            if isinstance(seg, FormulaSegment):
                ch = seg.chapter_number or "Unknown"
                formulas_by_ch.setdefault(ch, []).append(seg)
                section = getattr(seg, "heading_path", None) or "Unknown"
                sections_by_ch.setdefault(ch, {}).setdefault(section, []).append(seg)
        if not formulas_by_ch:
            return

        # Build chapter lexicon (from existing signals only; avoid new LLM calls here)
        chapter_lexicon: Dict[str, Dict[str, List[VariableDefinition]]] = {}
        for ch, formulas in formulas_by_ch.items():
            symbol_scores: Dict[str, float] = {}
            for f in formulas:
                symbols = self.formula_extractor._find_variable_candidates(
                    f.formula_text_raw or f.text_content, (f.context_before or "") + " " + (f.context_after or "")
                )
                for s in symbols:
                    symbol_scores[s] = symbol_scores.get(s, 0.0) + 1.0
                if f.usage_type == "definition":
                    for s in symbols:
                        symbol_scores[s] = symbol_scores.get(s, 0.0) + 2.0
                if f.equation_number:
                    for s in symbols:
                        symbol_scores[s] = symbol_scores.get(s, 0.0) + 1.0
            top_symbols = [s for s, _ in sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)]
            top_symbols = top_symbols[:14]
            if not top_symbols:
                continue
            # NOTE: Previously this section made 1-2 LLM calls per chapter to populate a
            # generic lexicon. For runtime reasons we now keep chapter_lexicon as a
            # lightweight container that can be populated from existing variables /
            # heuristic definitions only (no new LLM calls at lexicon stage).
            chapter_lexicon[ch] = {}

        # Build section overrides from heuristic definitions
        section_overrides: Dict[str, Dict[str, Dict[str, List[VariableDefinition]]]] = {}
        for ch, section_map in sections_by_ch.items():
            for section, formulas in section_map.items():
                for f in formulas:
                    allow_override = (
                        f.usage_type == "definition"
                        or self._has_definition_evidence((f.context_before or "") + " " + (f.context_after or ""))
                    )
                    if allow_override:
                        heuristic = self.formula_extractor._extract_variables_heuristic(
                            f.formula_text_raw or f.text_content,
                            (f.context_before or "") + " " + (f.context_after or "")
                        )
                        if heuristic:
                            section_overrides.setdefault(ch, {}).setdefault(section, {})
                            for v in heuristic:
                                section_overrides[ch][section].setdefault(v.symbol, []).append(v)

        # Apply lexicon + fallback LLM for unknown symbols
        for ch, formulas in formulas_by_ch.items():
            pending_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
            base_vars_by_id: Dict[str, List[VariableDefinition]] = {}
            for f in formulas:
                # If this formula already has variables and a short_meaning and does
                # not require human review, skip any additional lexicon-based LLM work.
                if f.variables and f.short_meaning and not getattr(f, "needs_human_review", False):
                    # Ensure caches are aware of these variables for reuse elsewhere.
                    cache_scope_existing = f"{f.book_id}:{ch}"
                    self.formula_extractor._update_variable_caches(
                        f.canonical_formula_key, f.variables, scope=cache_scope_existing
                    )
                    continue
                section = getattr(f, "heading_path", None) or "Unknown"
                combined_context = " ".join([
                    f.chapter_title or "",
                    section or "",
                    f.context_before or "",
                    f.context_after or ""
                ]).strip()
                cache_scope = f"{f.book_id}:{ch}"
                symbols = self.formula_extractor._find_variable_candidates(
                    f.formula_text_raw or f.text_content, combined_context
                )
                vars_out: List[VariableDefinition] = []
                for s in symbols:
                    if ch in section_overrides and section in section_overrides[ch] and s in section_overrides[ch][section]:
                        candidates = section_overrides[ch][section][s]
                        picked = self._select_variable_definition(candidates, section, f)
                        if picked:
                            vars_out.append(picked)
                    elif ch in chapter_lexicon and s in chapter_lexicon[ch]:
                        candidates = chapter_lexicon[ch][s]
                        picked = self._select_variable_definition(candidates, section, f)
                        if picked:
                            vars_out.append(picked)
                    else:
                        cached = self.formula_extractor._get_cached_variables_by_symbols([s], scope=cache_scope)
                        if cached:
                            vars_out.extend(cached)
                base_vars_by_id[f.segment_id] = vars_out

                known_symbols = {v.symbol for v in vars_out}
                unknown = [s for s in symbols if s not in known_symbols]
                if unknown:
                    canonical_key = f.canonical_formula_key
                    if not canonical_key:
                        canonical_key = self.formula_extractor._generate_canonical_key(
                            f.formula_text_raw or f.text_content
                        )
                        f.canonical_formula_key = canonical_key
                    group_key = (ch, canonical_key or f.segment_id)
                    entry = pending_by_key.setdefault(group_key, {
                        "formulas": [],
                        "unknown": set(),
                        "context": combined_context,
                        "formula_text": f.formula_text_raw or f.text_content
                    })
                    entry["formulas"].append(f)
                    entry["unknown"].update(unknown)
                    if len(combined_context) > len(entry["context"]):
                        entry["context"] = combined_context
                else:
                    f.variables = self.formula_extractor._post_process_variables(
                        vars_out, f.formula_text_raw or f.text_content
                    )
                    self.formula_extractor._update_variable_caches(
                        f.canonical_formula_key, f.variables, scope=cache_scope
                    )
                    # Always compute summary with cache
                    f.short_meaning = self.formula_extractor._get_formula_summary(
                        f.canonical_formula_key,
                        f.formula_text_raw or f.text_content,
                        f.variables
                    )

            # Canonical formula clustering + single LLM call per canonical key
            for (_, _), entry in pending_by_key.items():
                unknown = sorted(entry["unknown"])
                defs: List[VariableDefinition] = []
                # IMPORTANT: For performance, avoid introducing new LLM calls here.
                # Variable-level LLM extraction is handled earlier in FormulaExtractor
                # (per canonical_formula_key with caching). The lexicon stage now only
                # propagates / merges existing definitions.
                for f in entry["formulas"]:
                    cache_scope = f"{f.book_id}:{ch}"
                    base_vars = base_vars_by_id.get(f.segment_id, [])
                    merged = base_vars + defs
                    f.variables = self.formula_extractor._post_process_variables(
                        merged, f.formula_text_raw or f.text_content
                    )
                    self.formula_extractor._update_variable_caches(
                        f.canonical_formula_key, f.variables, scope=cache_scope
                    )
                    f.short_meaning = self.formula_extractor._get_formula_summary(
                        f.canonical_formula_key,
                        f.formula_text_raw or f.text_content,
                        f.variables
                    )

    def _select_variable_definition(self, candidates: List[VariableDefinition], heading_path: str,
                                    formula: FormulaSegment) -> Optional[VariableDefinition]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        heading = (heading_path or "").lower()
        scored: List[Tuple[int, int, int, int, VariableDefinition]] = []
        for v in candidates:
            score = 0
            snippet = (v.meaning or "").lower()
            if snippet:
                # Prefer definitions that mention heading terms
                if heading and any(tok in snippet for tok in heading.split() if len(tok) > 3):
                    score += 2
                # Prefer explicit definitional language
                if re.search(r'\b(defined as|denotes|where)\b', snippet):
                    score += 2
            if formula.usage_type == "definition":
                score += 1
            meaning_len = len((v.meaning or "").strip())
            context_bonus = 1 if v.source == "context" else 0
            source_rank = 2 if v.source == "context" else (1 if v.source == "llm" else 0)
            scored.append((score, source_rank, meaning_len, context_bonus, v))
        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        return scored[0][4]

    def _has_definition_evidence(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r'\b(defined as|denotes|where|is defined as)\b', text, re.IGNORECASE))
    
    def _build_chapter_metadata(self, segments: List[SegmentBase], chapter_title_map: Optional[Dict[str, str]] = None) -> List[ChapterMetadata]:
        """Build chapter metadata including solution presence and location."""
        chapter_map: Dict[str, ChapterMetadata] = {}
        target_chapter = getattr(self.context_processor, "target_chapter", None)
        chapter_title_map = chapter_title_map or {}
        
        # Derive a conservative set of valid main chapter numbers from the TOC-based
        # chapter_title_map when available. This helps filter out spurious numeric
        # patterns (years, question numbers like 8.8933, etc.).
        valid_main_chapters: Set[str] = set()
        if chapter_title_map:
            roots: Set[int] = set()
            for key in chapter_title_map.keys():
                if not isinstance(key, str):
                    continue
                if not re.match(r'^\d+(\.\d+)*$', key):
                    continue
                main_part = key.split('.')[0]
                if main_part.isdigit():
                    roots.add(int(main_part))
            if roots:
                for n in sorted(roots):
                    # Build a contiguous range starting at 1 (typical textbook pattern).
                    # Stop at first gap to avoid pulling in stray values like 63, 88, 2021.
                    if n == len(valid_main_chapters) + 1:
                        valid_main_chapters.add(str(n))
                    elif n > len(valid_main_chapters) + 1:
                        break
        
        def is_valid_chapter_number(ch_num: str) -> bool:
            if not ch_num or ch_num == "Unknown":
                return False
            if not re.match(r'^\d+(\.\d+)*$', ch_num):
                return False
            main_ch = ch_num.split('.')[0] if '.' in ch_num else ch_num
            # When TOC-derived chapters are available, restrict to that whitelist.
            if valid_main_chapters and main_ch not in valid_main_chapters:
                return False
            # When TOC section numbers are available, only accept subchapters that
            # explicitly appear in the TOC. This filters out spurious numeric
            # patterns like 8.7200 or 2.375 that are not real section IDs.
            if chapter_title_map and '.' in ch_num and ch_num not in chapter_title_map:
                return False
            if target_chapter is not None:
                if main_ch != target_chapter and not ch_num.startswith(target_chapter + "."):
                    return False
            return True
        
        # First pass: collect all segments by chapter (including subchapters like 9.1, 9.2)
        # Also check HeaderSegment.header_number to catch all subchapters
        for seg in segments:
            ch_num = seg.chapter_number
            
            # For HeaderSegment, also check header_number (more reliable for subchapters)
            if isinstance(seg, HeaderSegment) and hasattr(seg, 'header_number') and seg.header_number:
                header_num = seg.header_number
                if is_valid_chapter_number(header_num):
                    # Use header_number if available (more accurate)
                    ch_num = header_num
            
            if not is_valid_chapter_number(ch_num):
                continue
            
            # Extract main chapter number (e.g., "9" from "9.1" or "9.2.1")
            main_ch = ch_num.split('.')[0] if '.' in ch_num else ch_num
            
            # Store both main chapter and subchapter
            if main_ch not in chapter_map:
                mapped_title = chapter_title_map.get(main_ch) or chapter_title_map.get(ch_num)
                chapter_map[main_ch] = ChapterMetadata(
                    chapter_number=main_ch,
                    chapter_title=mapped_title or (seg.chapter_title or "").replace("\t", " ").strip() or None,
                    chapter_level=1,
                    parent_chapter=None,
                    solutions_present=False,
                    solution_location=None
                )
            
            # Also store subchapters (9.1, 9.2, etc.)
            if '.' in ch_num and ch_num not in chapter_map:
                mapped_title = chapter_title_map.get(ch_num) or chapter_title_map.get(main_ch)
                chapter_map[ch_num] = ChapterMetadata(
                    chapter_number=ch_num,
                    chapter_title=mapped_title or (seg.chapter_title or "").replace("\t", " ").strip() or None,
                    chapter_level=2,
                    parent_chapter=main_ch,
                    solutions_present=False,
                    solution_location=None
                )
        
        # Fill in missing subchapters: if 9.2 exists but 9.1 doesn't, create 9.1
        # This handles cases where headers weren't detected but subchapters exist
        subchapters_to_add = []
        for ch_num in list(chapter_map.keys()):
            if '.' in ch_num:
                parts = ch_num.split('.')
                if len(parts) == 2:  # e.g., "9.2"
                    main_ch = parts[0]
                    sub_num = int(parts[1])
                    # Check for missing earlier subchapters (e.g., if 9.2 exists, check for 9.1)
                    for i in range(1, sub_num):
                        missing_sub = f"{main_ch}.{i}"
                        if missing_sub not in chapter_map:
                            subchapters_to_add.append(missing_sub)
        
        # Add missing subchapters with None title (will be filled if found later)
        for missing_sub in subchapters_to_add:
            if missing_sub not in chapter_map:
                chapter_map[missing_sub] = ChapterMetadata(
                    chapter_number=missing_sub,
                    chapter_title=None,
                    chapter_level=2,
                    parent_chapter=missing_sub.split('.')[0],
                    solutions_present=False,
                    solution_location=None
                )
        
        # When TOC is missing and we have no chapter_title_map, we can fall back to
        # inferring subchapters from question numbers. For Investments (and most
        # well-structured textbooks), TOC is present so this branch is skipped.
        if not chapter_title_map:
            for seg in segments:
                if isinstance(seg, QuestionSegment) and getattr(seg, "question_number", None):
                    q_num = seg.question_number.strip()
                    if is_valid_chapter_number(q_num) and q_num not in chapter_map:
                        chapter_map[q_num] = ChapterMetadata(
                            chapter_number=q_num,
                            chapter_title=None,
                            chapter_level=2,
                            parent_chapter=q_num.split('.')[0],
                            solutions_present=False,
                            solution_location=None
                        )
        
        # Final pass: ensure titles are filled from chapter_title_map when possible.
        for ch_meta in chapter_map.values():
            if not ch_meta.chapter_title:
                ch_meta.chapter_title = chapter_title_map.get(ch_meta.chapter_number)
            # solutions_present / solution_location only meaningful for main chapters
            if ch_meta.chapter_level and ch_meta.chapter_level > 1:
                ch_meta.solutions_present = False
                ch_meta.solution_location = None
        
        # Sort chapters: main chapters ascending, then subchapters in natural numeric order.
        def _sort_key(meta: ChapterMetadata) -> Tuple[int, Tuple[int, ...]]:
            parts = meta.chapter_number.split('.')
            main = int(parts[0]) if parts and parts[0].isdigit() else 0
            subs = tuple(int(p) for p in parts[1:] if p.isdigit())
            return (main, subs)
        
        return sorted(list(chapter_map.values()), key=_sort_key)

    def _build_chapter_title_map(self, doc: fitz.Document) -> Dict[str, str]:
        toc_entries = doc.get_toc() or []
        title_map: Dict[str, str] = {}
        page_map: Dict[str, int] = {}

        for level, title, page in toc_entries:
            if not title:
                continue
            title_str = str(title).strip()
            main_match = re.search(r'\bchapter\s+(\d+)\b', title_str, re.IGNORECASE)
            if level == 1 and not main_match:
                main_match = re.match(r'^\s*(\d+)\b', title_str)
            if not main_match:
                # Use this entry only for page_map fallback
                section_match = re.match(r'^\s*(\d+(?:\.\d+)+)\b', title_str)
                if section_match:
                    sec_num = section_match.group(1)
                    cleaned = re.sub(r'^\s*\d+(?:\.\d+)+\b', '', title_str).strip(" -:\t")
                    if cleaned:
                        title_map[sec_num] = cleaned
                    main_ch = sec_num.split('.')[0]
                    page_map[main_ch] = min(page_map.get(main_ch, page), page)
                continue
            ch_num = main_match.group(1)
            cleaned = re.sub(r'\bchapter\s+\d+\b', '', title_str, flags=re.IGNORECASE).strip(" -:\t")
            if cleaned:
                title_map[ch_num] = cleaned
            page_map[ch_num] = min(page_map.get(ch_num, page), page)

        for ch_num, page in page_map.items():
            if ch_num in title_map:
                continue
            extracted = self._extract_chapter_title_from_page(doc, page)
            if extracted:
                title_map[ch_num] = extracted

        return title_map

    def _extract_chapter_title_from_page(self, doc: fitz.Document, page_num: int) -> Optional[str]:
        if page_num < 1 or page_num > len(doc):
            return None
        page = doc[page_num - 1]
        data = page.get_text("dict")
        if not data:
            return None
        spans = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = (span.get("text") or "").strip()
                    if not text or text.lower().startswith("chapter"):
                        continue
                    size = span.get("size", 0)
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    y0 = bbox[1] if bbox else 0
                    spans.append((size, y0, text))
        if not spans:
            return None
        spans.sort(key=lambda s: (-s[0], s[1]))
        best = spans[0][2]
        return best if len(best) >= 5 else None
    
    def _classify_solution_location(self, solution: SolutionSegment, section_headings: List[Tuple[str, str]]) -> Literal[
        "in_chapter", "end_of_chapter", "appendix", "concept_check_only", "selected_answers", "none"
    ]:
        """Classify where solutions are located based on headings and context."""
        # Check nearby headings for this chapter
        sol_ch = solution.chapter_number
        relevant_headings = [h for ch, h in section_headings if ch == sol_ch]
        
        for heading in relevant_headings:
            if "concept check" in heading or "concept-check" in heading:
                return "concept_check_only"
            if "selected" in heading or "selected answers" in heading:
                return "selected_answers"
            if "appendix" in heading:
                return "appendix"
            if "end" in heading or "review" in heading:
                return "end_of_chapter"
        
        # Default: assume in chapter if solutions found
        return "in_chapter"

    def _filter_segments_by_type(self, segments: List[SegmentBase], allowlist: Set[str]) -> List[SegmentBase]:
        allowed = {t.lower() for t in allowlist}
        return [seg for seg in segments if getattr(seg, "segment_type", "").lower() in allowed]

    def run(self, pdf_path: str, output_path: str, page_range: Tuple[int, int] = None,
            target_chapter: Optional[str] = None, chapter_mode: Literal["single_chapter", "multi_chapter"] = None,
            doc_uri: Optional[str] = None, book_id: Optional[str] = None):
        print(f"Processing {pdf_path}...")
        doc_uri = doc_uri or pdf_path  # optional traceability for multi-PDF
        book_id = book_id or "default_book"

        # Determine mode and target chapter
        if chapter_mode is None:
            # Auto-detect: if target_chapter provided, use single_chapter mode
            chapter_mode = "single_chapter" if target_chapter else "multi_chapter"

        if not target_chapter and chapter_mode == "single_chapter":
            # Try to extract from output_path or default
            ch_match = re.search(r'chapter(\d+)', output_path, re.IGNORECASE)
            if ch_match:
                target_chapter = ch_match.group(1)
            else:
                target_chapter = None

        # Update context processor with mode and target chapter
        self.context_processor = ContextProcessor(mode=chapter_mode, target_chapter=target_chapter)

        output_dir = os.path.dirname(output_path)
        cache_dir = os.path.join(output_dir, ".cache")
        gen = getattr(self.linker, "solution_generator", None)
        if gen and getattr(gen, "load_cache", None):
            gen.load_cache(cache_dir)

        output = self.process_pdf(pdf_path, book_id=book_id, page_range=page_range, doc_uri=doc_uri)

        if gen and getattr(gen, "save_cache", None):
            gen.save_cache(cache_dir)

        with open(output_path, 'w') as f:
            f.write(output.model_dump_json(indent=2))
        print(f"Saved output to {output_path}")
