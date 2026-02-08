import re
from typing import List, Dict, Optional, Tuple, Set, Literal

from schemas import (
    SegmentBase,
    HeaderSegment,
    WorkedExampleSegment,
    QuestionSegment,
    SolutionSegment,
    FormulaSegment,
    DerivationSegment,
    CalculationSegment,
    ExplanatoryTextSegment,
)


class ContextProcessor:
    def __init__(self, mode: Literal["single_chapter", "multi_chapter"] = "single_chapter",
                 target_chapter: Optional[str] = None,
                 chapter_title_map: Optional[Dict[str, str]] = None):
        """
        Initialize with processing mode and optional target chapter.

        Args:
            mode: "single_chapter" - process only one chapter with boundary guard
                  "multi_chapter" - process entire book, reset heading_stack per chapter
            target_chapter: Chapter number to process (e.g., "9").
                           Only used in single_chapter mode.
        """
        self.mode = mode
        self.target_chapter = target_chapter  # e.g., "9" for Chapter 9 only
        self.chapter_title_map = chapter_title_map or {}

    def process(self, segments: List[SegmentBase]) -> List[SegmentBase]:
        sorted_segments = sorted(
            segments,
            key=lambda s: (s.page_start, s.bbox.y0 if s.bbox else 0)
        )

        # Build heading path and update chapter info
        current_section_id = "Unknown"
        current_section_title = None
        current_chapter_num = self.target_chapter if (self.mode == "single_chapter" and self.target_chapter) else "Unknown"
        heading_stack: List[Tuple[str, str]] = []  # [(number, title), ...]
        last_chapter_num = "Unknown"  # Track chapter changes for multi-chapter mode

        for seg in sorted_segments:
            if isinstance(seg, HeaderSegment):
                header_text = seg.text_content
                # Ignore headers that are purely numeric/punctuation or URLs (likely tables/figures/footers)
                if not re.search(r'[A-Za-z]', header_text or ""):
                    continue
                if re.search(r'https?://|www\.|\.com\b|\.edu\b|\.net\b', header_text or "", re.IGNORECASE):
                    continue
                # Extract chapter number from "CHAPTER 9" or "9.1" patterns
                ch_match = re.search(r'chapter\s+(\d+)', header_text, re.IGNORECASE)
                if ch_match:
                    extracted_ch = ch_match.group(1)

                    # Chapter boundary guard: only in single_chapter mode
                    if self.mode == "single_chapter":
                        if self.target_chapter and extracted_ch != self.target_chapter:
                            continue  # Skip headers from other chapters

                    # Multi-chapter mode: reset heading_stack when chapter changes
                    if self.mode == "multi_chapter":
                        if last_chapter_num != "Unknown" and extracted_ch != last_chapter_num:
                            # Chapter changed - reset heading stack
                            heading_stack = []

                    current_chapter_num = extracted_ch
                    last_chapter_num = extracted_ch
                    current_section_id = current_chapter_num
                    current_section_title = (header_text or "").replace("\t", " ").strip() or header_text
                    if current_section_title and current_section_title.lower().startswith("chapter"):
                        mapped_title = self.chapter_title_map.get(current_chapter_num)
                        if mapped_title:
                            current_section_title = mapped_title
                    # Reset heading stack when new chapter starts
                    heading_stack = [(current_chapter_num, current_section_title)]

                # Extract section number (e.g., "9.1", "9.2.1")
                section_match = re.search(r'^(\d+(?:\.\d+)+)', header_text)
                if section_match:
                    current_section_id = section_match.group(1)
                    # Extract main chapter from section (e.g., "9" from "9.1")
                    main_ch = current_section_id.split('.')[0]
                    if main_ch.isdigit():
                        # Chapter boundary guard: only in single_chapter mode
                        if self.mode == "single_chapter":
                            if self.target_chapter and main_ch != self.target_chapter:
                                continue  # Skip sections from other chapters

                        # Multi-chapter mode: reset heading_stack when chapter changes
                        if self.mode == "multi_chapter":
                            if last_chapter_num != "Unknown" and main_ch != last_chapter_num:
                                # Chapter changed - reset heading stack
                                heading_stack = []
                                last_chapter_num = main_ch
                            elif last_chapter_num == "Unknown":
                                last_chapter_num = main_ch

                        current_chapter_num = main_ch
                    current_section_title = (header_text or "").replace("\t", " ").strip() or header_text
                    mapped_title = self.chapter_title_map.get(current_section_id) or self.chapter_title_map.get(current_chapter_num)
                    if mapped_title:
                        current_section_title = mapped_title
                    # Update heading stack
                    level = len(current_section_id.split('.'))
                    heading_stack = heading_stack[:level - 1]  # Remove deeper levels
                    heading_stack.append((current_section_id, current_section_title))
                    seg.header_number = current_section_id
                    seg.chapter_number = current_chapter_num  # Set main chapter
                    seg.chapter_title = current_section_title
            else:
                # Infer chapter from question number if available (e.g., "9.1")
                if isinstance(seg, QuestionSegment) and seg.chapter_number == "Unknown":
                    q_num = (seg.question_number or "").strip()
                    if re.match(r'^\d+\.\d+$', q_num):
                        seg.chapter_number = q_num

                # Propagate chapter info to all segments
                if hasattr(seg, "chapter_number"):
                    if seg.chapter_number == "Unknown":
                        seg.chapter_number = current_section_id  # Use section ID (e.g., "9.1")
                    # Also set main chapter number if available
                    if current_chapter_num != "Unknown":
                        # Keep section number but ensure main chapter is set
                        if not seg.chapter_number.startswith(current_chapter_num):
                            # If segment doesn't have chapter info, use current section
                            seg.chapter_number = current_section_id

                if hasattr(seg, "chapter_title") and current_section_title:
                    seg.chapter_title = current_section_title

                # Build heading path
                if heading_stack:
                    seg.heading_path = " > ".join([f"{num} {title}" for num, title in heading_stack])

        # Merge fragmented blocks for questions/examples
        sorted_segments = self._merge_concept_check_questions(sorted_segments)
        sorted_segments = self._merge_worked_example_prompts(sorted_segments)
        sorted_segments = self._merge_question_continuations(sorted_segments)
        sorted_segments = self._merge_concept_check_solutions(sorted_segments)

        # Fallback heading path when no headers were detected
        for seg in sorted_segments:
            if not getattr(seg, "heading_path", None):
                ch_num = getattr(seg, "chapter_number", None)
                if ch_num and ch_num != "Unknown":
                    seg.heading_path = f"{ch_num} > p{seg.page_start}"
                else:
                    seg.heading_path = f"p{seg.page_start}"

        # Add adjacency links
        sorted_segments = self.add_adjacency_links(sorted_segments)

        # Ensure bbox traceability
        for seg in sorted_segments:
            if seg.bbox is None:
                seg.needs_human_review = True

        return sorted_segments

    def _merge_concept_check_questions(self, segments: List[SegmentBase]) -> List[SegmentBase]:
        """
        Merge "Concept Check X.Y" question header with the following text block
        so the question contains its actual prompt.
        """
        out: List[SegmentBase] = []
        skip_ids: Set[str] = set()
        for i, seg in enumerate(segments):
            if seg.segment_id in skip_ids:
                continue
            if isinstance(seg, QuestionSegment):
                text_lower = (seg.text_content or "").strip().lower()
                if text_lower.startswith("concept check"):
                    if i + 1 < len(segments):
                        nxt = segments[i + 1]
                        if (
                            isinstance(nxt, ExplanatoryTextSegment)
                            and nxt.page_start == seg.page_start
                        ):
                            merged = f"{seg.text_content.strip()}\n{nxt.text_content.strip()}"
                            seg.text_content = merged
                            if not seg.context_after and nxt.context_after:
                                seg.context_after = nxt.context_after
                            skip_ids.add(nxt.segment_id)
            out.append(seg)
        return out

    def _merge_worked_example_prompts(self, segments: List[SegmentBase]) -> List[SegmentBase]:
        """
        Attach nearby content to worked examples that only captured the title block.
        Keep other segments intact but enrich the worked example fields.
        """
        out: List[SegmentBase] = []
        skip_ids: Set[str] = set()
        stop_types = (HeaderSegment, WorkedExampleSegment, QuestionSegment, SolutionSegment,
                      DerivationSegment)
        for i, seg in enumerate(segments):
            if seg.segment_id in skip_ids:
                continue
            if isinstance(seg, WorkedExampleSegment):
                prompt_missing = not seg.example_prompt or seg.example_prompt.strip().lower() == "none"
                parts: List[str] = []
                step_parts: List[str] = []
                j = i + 1
                while j < len(segments):
                    nxt = segments[j]
                    # Allow the example to spill to the next page (common in textbooks)
                    if nxt.page_start not in (seg.page_start, seg.page_start + 1):
                        break
                    if isinstance(nxt, stop_types):
                        break
                    if isinstance(nxt, ExplanatoryTextSegment):
                        if nxt.text_content:
                            parts.append(nxt.text_content.strip())
                        step_parts.append(nxt.text_content.strip())
                        skip_ids.add(nxt.segment_id)
                        j += 1
                        continue
                    if isinstance(nxt, (FormulaSegment, CalculationSegment)):
                        if nxt.text_content:
                            parts.append(nxt.text_content.strip())
                            step_parts.append(nxt.text_content.strip())
                        skip_ids.add(nxt.segment_id)
                        j += 1
                        continue
                    break
                if parts and prompt_missing:
                    seg.example_prompt = "\n".join(parts)
                if step_parts:
                    seg.steps = step_parts
            out.append(seg)
        return out

    def _merge_question_continuations(self, segments: List[SegmentBase]) -> List[SegmentBase]:
        """
        Merge trailing blocks into questions to keep problem statements/data together.
        Avoids swallowing section headers and long prose after concept checks.
        """
        out: List[SegmentBase] = []
        skip_ids: Set[str] = set()
        stop_types = (HeaderSegment, QuestionSegment, WorkedExampleSegment, SolutionSegment)
        for i, seg in enumerate(segments):
            if seg.segment_id in skip_ids:
                continue
            if isinstance(seg, QuestionSegment):
                parts: List[str] = [seg.text_content.strip()] if seg.text_content else []
                total_len = sum(len(p) for p in parts)
                number_only = self._is_number_only_question(seg)
                table_mode = self._looks_like_table_header(seg.text_content or "") or number_only
                j = i + 1
                while j < len(segments):
                    nxt = segments[j]
                    if nxt.page_start not in (seg.page_start, seg.page_start + 1):
                        break
                    if isinstance(nxt, stop_types):
                        break
                    nxt_text = (nxt.text_content or "").strip()
                    if not nxt_text:
                        j += 1
                        continue
                    if self._looks_like_section_marker(nxt_text):
                        break
                    if self._looks_like_heading_line(nxt_text):
                        if not table_mode:
                            break
                    # Concept checks: only merge short prompt continuations; merge inline formulas only
                    if seg.problem_set_type == "concept_check":
                        if isinstance(nxt, ExplanatoryTextSegment) and self._looks_like_question_continuation(nxt_text):
                            parts.append(nxt_text)
                            total_len += len(nxt_text)
                            skip_ids.add(nxt.segment_id)
                            j += 1
                            continue
                        if isinstance(nxt, (FormulaSegment, CalculationSegment)) and self._is_inline_question_formula(seg, nxt):
                            parts.append(nxt_text)
                            total_len += len(nxt_text)
                            skip_ids.add(nxt.segment_id)
                            j += 1
                            continue
                        break
                    # Problem sets / CFA: allow short tables, calculations, and short prose
                    if isinstance(nxt, (ExplanatoryTextSegment, FormulaSegment, CalculationSegment)):
                        if number_only:
                            # If the question is just "11." etc, absorb following content aggressively
                            parts.append(nxt_text)
                            total_len += len(nxt_text)
                            skip_ids.add(nxt.segment_id)
                            j += 1
                            if total_len > 2000:
                                break
                            continue
                        if table_mode and self._looks_like_table_line(nxt_text):
                            parts.append(nxt_text)
                            total_len += len(nxt_text)
                            skip_ids.add(nxt.segment_id)
                            j += 1
                            continue
                        if total_len > 1200 and len(nxt_text) > 200:
                            break
                        parts.append(nxt_text)
                        total_len += len(nxt_text)
                        skip_ids.add(nxt.segment_id)
                        j += 1
                        continue
                    break
                if parts:
                    seg.text_content = "\n".join(parts)
                if not seg.subparts and seg.text_content:
                    seg.subparts = self._extract_subparts_from_text(seg.text_content)
            out.append(seg)
        return out

    def _looks_like_heading_line(self, text: str) -> bool:
        # Short, title-like lines without sentence punctuation
        if len(text) > 80:
            return False
        if any(ch in text for ch in ".?!"):
            return False
        words = text.split()
        if len(words) < 2:
            return False
        return text.istitle() or text.isupper()

    def _looks_like_question_continuation(self, text: str) -> bool:
        t = text.lower()
        if "?" in t:
            return True
        if any(t.startswith(prefix) for prefix in ["if ", "suppose", "assume", "given", "why", "how"]):
            return True
        return len(text.split()) <= 40

    def _looks_like_section_marker(self, text: str) -> bool:
        t = text.lower().strip()
        return t.startswith("for problems") or t.startswith("for cfa problems") or t.startswith("solutions to concept checks")

    def _looks_like_table_header(self, text: str) -> bool:
        t = text.lower()
        return any(k in t for k in ["portfolio", "expected return", "beta", "standard deviation", "risk-free", "market return"])

    def _looks_like_table_line(self, text: str) -> bool:
        t = text.strip()
        if len(t) > 120:
            return False
        if re.search(r'\d', t) and len(t.split()) <= 8:
            return True
        return self._looks_like_table_header(t)

    def _is_number_only_question(self, seg: QuestionSegment) -> bool:
        t = (seg.text_content or "").strip()
        return bool(re.fullmatch(r'\d+\s*[.)]?', t))

    def _is_inline_question_formula(self, question: QuestionSegment, formula: SegmentBase) -> bool:
        """
        Heuristic: treat formula as inline if it is short, near the question,
        and not a numbered equation.
        """
        if isinstance(formula, FormulaSegment) and formula.equation_number:
            return False
        text = (formula.text_content or "").strip()
        if len(text) > 80:
            return False
        if not question.bbox or not formula.bbox:
            return False
        if question.page_start != formula.page_start:
            return False
        # Require horizontal overlap and limited vertical distance
        overlap = not (formula.bbox.x1 < question.bbox.x0 or formula.bbox.x0 > question.bbox.x1)
        y_gap = formula.bbox.y0 - question.bbox.y1
        return overlap and 0 <= y_gap <= 120

    def _extract_subparts_from_text(self, text: str) -> List[Dict[str, str]]:
        subparts: List[Dict[str, str]] = []
        pattern = r'\(([a-z])\)\s*([^()]+?)(?=\([a-z]\)|$)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        for label, subtext in matches:
            subparts.append({"label": label, "text": subtext.strip()})
        if not subparts:
            line_pattern = r'(^|\n)\s*([a-z])[\.\)]\s*([^\n]+)'
            matches = re.findall(line_pattern, text, re.IGNORECASE)
            for _, label, subtext in matches:
                subparts.append({"label": label, "text": subtext.strip()})
        labels = {s["label"].lower() for s in subparts}
        if ("b" in labels or "c" in labels) and "a" not in labels:
            m = re.split(r'\n\s*b[.)]\s+', text, maxsplit=1, flags=re.IGNORECASE)
            if len(m) > 1:
                prefix = m[0]
                prefix = prefix.split("\n", 1)[-1].strip()
                if len(prefix) > 20:
                    subparts.insert(0, {"label": "a", "text": prefix})
        return subparts

    def add_adjacency_links(self, segments: List[SegmentBase]) -> List[SegmentBase]:
        """Add prev_segment_id and next_segment_id to segments."""
        for i, seg in enumerate(segments):
            if i > 0:
                seg.prev_segment_id = segments[i - 1].segment_id
            if i < len(segments) - 1:
                seg.next_segment_id = segments[i + 1].segment_id
        return segments

    def _merge_concept_check_solutions(self, segments: List[SegmentBase]) -> List[SegmentBase]:
        """
        Merge numbered concept check solutions with following lines/formulas/calculations.
        Leaves underlying segments intact but enriches the SolutionSegment.
        """
        out: List[SegmentBase] = []
        stop_types = (HeaderSegment, QuestionSegment, WorkedExampleSegment)
        for i, seg in enumerate(segments):
            if isinstance(seg, SolutionSegment) and getattr(seg, "is_concept_check", False):
                parts: List[str] = [seg.text_content.strip()] if seg.text_content else []
                steps: List[str] = seg.solution_steps[:] if seg.solution_steps else []
                j = i + 1
                while j < len(segments):
                    nxt = segments[j]
                    if nxt.page_start not in (seg.page_start, seg.page_start + 1):
                        break
                    if isinstance(nxt, stop_types):
                        break
                    if isinstance(nxt, SolutionSegment) and getattr(nxt, "is_concept_check", False):
                        break
                    if isinstance(nxt, (ExplanatoryTextSegment, FormulaSegment, CalculationSegment)):
                        if nxt.text_content:
                            parts.append(nxt.text_content.strip())
                            steps.append(nxt.text_content.strip())
                        j += 1
                        continue
                    break
                if parts:
                    seg.text_content = "\n".join(parts)
                if steps:
                    seg.solution_steps = steps
            out.append(seg)
        return out
