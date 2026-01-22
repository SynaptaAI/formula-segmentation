import os
import re
import uuid
import json
import time
import typing
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Literal
from collections import defaultdict, Counter

import fitz  # PyMuPDF
import google.generativeai as genai
from pydantic import BaseModel, Field

# ==========================================
# 1. SCHEMA DEFINITIONS
# ==========================================

class BBox(BaseModel):
    page: int
    x0: float
    y0: float
    x1: float
    y1: float

class ConceptLink(BaseModel):
    concept_id: str
    link_method: Literal["exact_match", "alias", "semantic", "heuristic"]
    confidence: float

class VariableDefinition(BaseModel):
    symbol: str
    meaning: str
    units: Optional[str] = None
    is_inferred: bool = False

class SegmentBase(BaseModel):
    segment_id: str
    book_id: str
    chapter_number: str
    page_start: int
    page_end: int
    bbox: Optional[BBox] = None
    text_content: str
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    concept_links: List[ConceptLink] = []
    
class FormulaSegment(SegmentBase):
    segment_type: Literal["formula"] = "formula"
    formula_latex: Optional[str] = None
    equation_number: Optional[str] = None
    variables: List[VariableDefinition] = []
    usage_type: Literal["definition", "application", "reference"]
    referenced_formula_ids: List[str] = []

class WorkedExampleSegment(SegmentBase):
    segment_type: Literal["worked_example"] = "worked_example"
    title: Optional[str] = None
    steps: List[str] = []
    final_answer: Optional[str] = None
    referenced_formula_ids: List[str] = []

class QuestionSegment(SegmentBase):
    segment_type: Literal["question"] = "question"
    question_number: str
    subparts: List[str] = []
    question_type: Optional[str] = None
    referenced_formula_ids: List[str] = []

class SolutionSegment(SegmentBase):
    segment_type: Literal["solution"] = "solution"
    solution_for_question_id: Optional[str] = None
    referenced_formula_ids: List[str] = []

class DerivationSegment(SegmentBase):
    segment_type: Literal["derivation"] = "derivation"
    steps: List[str] = []
    derived_formula_id: Optional[str] = None
    referenced_formula_ids: List[str] = []

class HeaderSegment(SegmentBase):
    segment_type: Literal["header"] = "header"
    level: int = 1 
    header_number: Optional[str] = None

class SegmentationOutput(BaseModel):
    metadata: Dict[str, Any]
    segments: List[Union[FormulaSegment, WorkedExampleSegment, QuestionSegment, SolutionSegment, DerivationSegment, HeaderSegment, SegmentBase]]


# ==========================================
# 2. UTILS (Layout / Furniture)
# ==========================================

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

    def scan_document(self, page_blocks_map: Dict[int, List[Tuple]]) -> None:
        text_pages: Dict[str, Set[int]] = defaultdict(set)
        all_pages = set()

        for page_num, blocks in page_blocks_map.items():
            all_pages.add(page_num)
            for b in blocks:
                x0, y0, x1, y1, text, _, _ = b
                clean_text = self._normalize_text(text)
                if not clean_text: continue
                if len(clean_text.split()) > self.MAX_WORDS: continue
                text_pages[clean_text].add(page_num)
        
        self.total_pages = len(all_pages)
        self.frequency_map = {t: len(p) for t, p in text_pages.items()}
        self.is_scanned = True
        print(f"FurnitureDetector: Scanned {self.total_pages} pages. Found {len(self.frequency_map)} repeated strings.")

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

# ==========================================
# 3. LLM SERVICES
# ==========================================

class LLMService(ABC):
    @abstractmethod
    def extract_variables(self, formula_text: str, context: str) -> List[VariableDefinition]:
        pass

    @abstractmethod
    def structure_worked_example(self, text: str) -> Dict[str, Any]:
        pass

class MockLLMService(LLMService):
    def extract_variables(self, formula_text: str, context: str) -> List[VariableDefinition]:
        return []

    def structure_worked_example(self, text: str) -> Dict[str, Any]:
        return {
            "title": None,
            "problem_statement": text,
            "steps": [],
            "final_answer": None
        }

class GeminiLLMService(LLMService):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not found. LLM features will fail or fallback.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def extract_variables(self, formula_text: str, context: str) -> List[VariableDefinition]:
        if not self.api_key: return []
        
        prompt = f"Given formula '{formula_text}' and context '{context}', list variables as JSON."
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    f"Extract variables from this text that relate to formula '{formula_text}':\n{context}\nReturn JSON list with keys 'symbol', 'meaning', 'is_inferred'.",
                    generation_config={"response_mime_type": "application/json"}
                )
                data = json.loads(response.text)
                if isinstance(data, dict) and 'variables' in data: data = data['variables']
                return [VariableDefinition(**item) for item in data]
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2 * (attempt + 1))
                    continue
                print(f"LLM Error (extract_variables): {e}")
                return []
        return []

    def structure_worked_example(self, text: str) -> Dict[str, Any]:
        if not self.api_key: return {"problem_statement": text}

        prompt = f"""
        Analyze this Worked Example text:
        ---
        {text}
        ---
        Return JSON with: title, problem_statement, steps (list of strings), final_answer.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text)
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2 * (attempt + 1))
                    continue
                print(f"LLM Error (structure_worked_example): {e}")
                return {"problem_statement": text}
        return {"problem_statement": text}


# ==========================================
# 4. EXTRACTORS
# ==========================================

class BaseExtractor(ABC):
    @abstractmethod
    def process_page(self, page: fitz.Page, page_num: int, book_id: str, blocks: List[Any] = None) -> List[SegmentBase]:
        pass
    
    def normalize_text(self, text: str) -> str:
        return text.strip()

class FormulaExtractor(BaseExtractor):
    def __init__(self, llm_service: LLMService = None):
        self.llm_service = llm_service or MockLLMService()

    def process_page(self, page: fitz.Page, page_num: int, book_id: str, blocks: List[any] = None) -> List[FormulaSegment]:
        segments = []
        if blocks is None:
            blocks = page.get_text("blocks")
        
        for i, b in enumerate(blocks):
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0: continue
            
            clean_text = text.strip()
            if not clean_text: continue
            
            if self._is_formula(clean_text, x0, page.rect.width):
                context_text = ""
                if i + 1 < len(blocks):
                    context_text = blocks[i+1][4].strip()

                variables = self._extract_variables(clean_text, context_text)
                eq_num = self._extract_eq_number(clean_text)
                
                seg = FormulaSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    context_after=context_text,
                    formula_latex=None,
                    equation_number=eq_num,
                    variables=variables,
                    usage_type="definition" if eq_num or variables else "application"
                )
                segments.append(seg)
        return segments

    def _is_formula(self, text: str, x0: float, page_width: float) -> bool:
        is_centered = (x0 > page_width * 0.1) 
        math_symbols = {'=', '∑', '√', '∫', 'σ', 'π', 'θ', 'λ', '+', '−'}
        has_math_symbol = any(s in text for s in math_symbols)
        has_eq_keyword = "Eq." in text or "Equation" in text or bool(re.search(r'\(\d+\.\d+\)', text))
        
        if has_math_symbol and (is_centered or has_eq_keyword):
            return True
        return False

    def _extract_eq_number(self, text: str) -> Optional[str]:
        match = re.search(r'\((\d+|[A-Z])\.(\d+)\)', text)
        if match: return match.group(0)
        return None

    def _extract_variables(self, formula_text: str, context_text: str = "") -> List[VariableDefinition]:
        if not context_text.strip(): return []
        
        # 1. Try Heuristic (Regex) Extraction First - FAST & FREE
        heuristic_vars = self._extract_variables_heuristic(formula_text, context_text)
        
        # 2. Decision Gate: When to use LLM?
        # Only use LLM if heuristic failed to find variables AND the formula is "complex"
        # Complex symbols: Integral, Sum, Product, Greek letters (maybe)
        is_complex = any(s in formula_text for s in ['∫', '∑', '∏'])
        
        if not heuristic_vars and is_complex:
            # Fallback to LLM for hard cases
            # print(f"  [LLM Trigger] Complex formula with no heuristic matches: {formula_text[:30]}...")
            return self.llm_service.extract_variables(formula_text, context_text)
            
        return heuristic_vars

    def _extract_variables_heuristic(self, formula_text: str, context: str) -> List[VariableDefinition]:
        """
        Regex-based extraction for common textbook patterns.
        Pattern 1: "where X is the..." results in {symbol: X, meaning: ...}
        Pattern 2: "X = ..." (in context, immediately following)
        """
        variables = []
        
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
            
            # Filter: heuristics to filter out obviously wrong matches (common words)
            # If it's a word > 4 chars and NOT in our whitelist of greek/math terms, skip it
            # e.g. "where `risk` is..." -> `risk` is likely not a variable symbol
            is_greek = sym.lower() in ['alpha', 'beta', 'gamma', 'delta', 'sigma', 'theta', 'lambda', 'rho']
            is_mathy = any(c in sym for c in "()_0123456789")
            
            if len(sym) > 4 and not is_greek and not is_mathy:
                 continue 

            if sym not in unique_vars:
                unique_vars[sym] = VariableDefinition(
                    symbol=sym,
                    meaning=mean,
                    is_inferred=False
                )
        
        return list(unique_vars.values())

class TextBlockExtractor(BaseExtractor):
    def __init__(self, llm_service: LLMService = None):
        self.llm_service = llm_service or MockLLMService()

    def process_page(self, page: fitz.Page, page_num: int, book_id: str, blocks: List[any] = None) -> List[any]:
        segments = []
        if blocks is None:
            blocks = page.get_text("blocks")
        
        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0: continue
            clean_text = text.strip()
            if not clean_text: continue
            
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
                    steps=[clean_text]
                )
                 segments.append(seg)

            # Worked Example
            elif clean_text.lower().startswith("example") or "solution" in clean_text.lower()[:20]:
                structure = self.llm_service.structure_worked_example(clean_text)
                seg = WorkedExampleSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    title=structure.get("title") or (clean_text.split('\n')[0] if clean_text.lower().startswith("example") else "Worked Example"),
                    steps=structure.get("steps", [])
                )
                segments.append(seg)
                
            # Question (numbered list)
            elif re.match(r'^(Q?\d+\.|[a-z]\))\s', clean_text):
                match = re.match(r'^(Q?\d+\.|[a-z]\))', clean_text)
                q_num = match.group(0).strip('.')
                seg = QuestionSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number="Unknown",
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    question_number=q_num,
                    question_type=self._determine_question_type(clean_text)
                )
                segments.append(seg)
                
            # Header
            elif self._is_header(clean_text):
                match = re.search(r'^(\d+(\.\d+)*)', clean_text)
                header_num = match.group(1) if match else "Unknown"
                seg = HeaderSegment(
                    segment_id=str(uuid.uuid4()),
                    book_id=book_id,
                    chapter_number=header_num,
                    page_start=page_num,
                    page_end=page_num,
                    bbox=BBox(page=page_num, x0=x0, y0=y0, x1=x1, y1=y1),
                    text_content=clean_text,
                    level=len(header_num.split('.')),
                    header_number=header_num
                )
                segments.append(seg)
        return segments

    def _is_header(self, text: str) -> bool:
        if text.lower().startswith("chapter"): return True
        if re.match(r'^\d+(\.\d+)+\s+[A-Z]', text): return True
        return False

    def _determine_question_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["calculate", "compute", "determine", "find", "estimate", "value", "what is the"]):
            return "quantitative"
        elif any(w in text_lower for w in ["explain", "discuss", "what is", "why", "compare", "would", "suppose", "consider", "assume"]):
            return "conceptual"
        return "mixed"

    def _is_derivation(self, text: str) -> bool:
        start_lower = text.lower()[:50]
        keywords = ["proof", "derivation", "we can show", "substituting"]
        for k in keywords:
            if k in start_lower: return True
        return False

# ==========================================
# 5. PROCESSORS & LINKER
# ==========================================

class ContextProcessor:
    def process(self, segments: List[SegmentBase]) -> List[SegmentBase]:
        sorted_segments = sorted(
            segments, 
            key=lambda s: (s.page_start, s.bbox.y0 if s.bbox else 0)
        )
        current_section_id = "Unknown"
        for seg in sorted_segments:
            if isinstance(seg, HeaderSegment):
                if seg.header_number:
                    current_section_id = seg.header_number
            else:
                if hasattr(seg, 'chapter_number') and seg.chapter_number == "Unknown":
                        seg.chapter_number = current_section_id
        return sorted_segments

class Linker:
    def link_segments(self, segments: List[SegmentBase]) -> List[SegmentBase]:
        formulas_by_eq_num: Dict[str, FormulaSegment] = {}
        for seg in segments:
            if isinstance(seg, FormulaSegment) and seg.equation_number:
                norm_num = self._normalize_eq_num(seg.equation_number)
                formulas_by_eq_num[norm_num] = seg

        for seg in segments:
            # Check for references in ALL segment types (including Formulas which might be derivations/text)
            if isinstance(seg, (WorkedExampleSegment, QuestionSegment, SolutionSegment, DerivationSegment, FormulaSegment)):
                refs = self._find_eq_references(seg.text_content)
                # Also check context_after for references, as some matches end up there
                if seg.context_after:
                     refs.extend(self._find_eq_references(seg.context_after))
                
                refs = list(set(refs)) # unique
                
                for ref_num in refs:
                    norm_ref = self._normalize_eq_num(ref_num)
                    if norm_ref in formulas_by_eq_num:
                        target_id = formulas_by_eq_num[norm_ref].segment_id
                        # Don't link to self
                        if target_id == seg.segment_id: continue
                        
                        if target_id not in seg.referenced_formula_ids:
                             seg.referenced_formula_ids.append(target_id)
                             
                # Fallback: Keyword matching for variables (Context Linking)
                # If no explicit "Eq X.Y" link, check if text mentions known variables/names
                if not seg.referenced_formula_ids:
                     self._heuristic_link_by_variables(seg, formulas_by_eq_num)
                     
        return segments

    def _heuristic_link_by_variables(self, source_seg: SegmentBase, formulas_map: Dict[str, FormulaSegment]):
        # Very distinct variable names or formula names?
        # This is a stub for future "concept linking" integration
        pass

    def _normalize_eq_num(self, eq_num: str) -> str:
        return eq_num.replace('(', '').replace(')', '').strip()

    def _find_eq_references(self, text: str) -> List[str]:
        refs = []
        matches = re.findall(r'(?:Eq\.|Equation)\s*\(?(\d+\.\d+)\)?', text)
        refs.extend(matches)
        matches_parens = re.findall(r'\((\d+\.\d+)\)', text)
        refs.extend(matches_parens)
        return list(set(refs))

# ==========================================
# 6. PIPELINE
# ==========================================

class Pipeline:
    def __init__(self):
        self.llm_service = GeminiLLMService()
        self.formula_extractor = FormulaExtractor(llm_service=self.llm_service)
        self.text_block_extractor = TextBlockExtractor(llm_service=self.llm_service)
        self.linker = Linker()
        self.furniture_detector = FurnitureDetector()
        self.context_processor = ContextProcessor()

    def process_pdf(self, pdf_path: str, book_id: str, page_range: Tuple[int, int] = None) -> SegmentationOutput:
        doc = fitz.open(pdf_path)
        start_page, end_page = page_range if page_range else (0, len(doc))
        start_page = max(0, start_page)
        end_page = min(len(doc), end_page)

        print(f"Processing pages {start_page} to {end_page}...")
        all_blocks_map: Dict[int, List[Tuple]] = {}
        for i in range(start_page, end_page):
            page = doc[i]
            all_blocks_map[i+1] = page.get_text("blocks")
            
        self.furniture_detector.scan_document(all_blocks_map)
        
        all_segments = []
        for i in range(start_page, end_page):
            page_num = i + 1
            page = doc[i]
            raw_blocks = all_blocks_map.get(page_num, [])
            page_height = page.rect.height
            
            clean_blocks = [
                b for b in raw_blocks 
                if not self.furniture_detector.is_furniture(b, page_height)
            ]
            
            formulas = self.formula_extractor.process_page(page, page_num, book_id, blocks=clean_blocks)
            all_segments.extend(formulas)
            text_blocks = self.text_block_extractor.process_page(page, page_num, book_id, blocks=clean_blocks)
            all_segments.extend(text_blocks)
            
        all_segments = self.context_processor.process(all_segments)
        all_segments = self.linker.link_segments(all_segments)
            
        return SegmentationOutput(
            metadata={"source_pdf": pdf_path, "total_pages": len(doc)},
            segments=all_segments
        )

    def run(self, pdf_path: str, output_path: str, page_range: Tuple[int, int] = None):
        print(f"Processing {pdf_path}...")
        output = self.process_pdf(pdf_path, book_id="Investments_Ch9", page_range=page_range)
        with open(output_path, 'w') as f:
            f.write(output.model_dump_json(indent=2))
        print(f"Saved output to {output_path}")
