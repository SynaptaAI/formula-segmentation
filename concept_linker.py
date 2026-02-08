import csv
import re
from typing import List, Optional, Dict, Any, Tuple

from schemas import ConceptLink, SegmentBase
from llm_services import LLMService, generate_json_with_llm


class Concept:
    """Represents a concept from the TSV/Excel concept list."""
    def __init__(self, concept_id: str, name: str, aliases: List[str] = None,
                 description: str = "", tags: List[str] = None, level: Optional[int] = None,
                 pages: List[int] = None):
        self.concept_id = concept_id
        self.name = name
        self.aliases = aliases or []
        self.description = description
        self.tags = tags or []
        self.level = level
        self.pages = pages or []


class ConceptLinker:
    """Links segments to concepts from a concept list (TSV/Excel)."""
    
    def __init__(self, concept_list_path: Optional[str] = None, llm_service: Optional[LLMService] = None,
                 use_llm_rerank: bool = False):
        self.concepts: List[Concept] = []
        self.llm_service = llm_service
        self.use_llm_rerank = use_llm_rerank
        self._semantic_vocab: Dict[str, int] = {}
        self._concept_term_freqs: Dict[str, Dict[str, int]] = {}
        self._term_df: Dict[str, int] = {}
        self._concept_by_id: Dict[str, Concept] = {}
        if concept_list_path:
            self.load_concepts(concept_list_path)
    
    def load_concepts(self, path: str) -> None:
        """Load concepts from TSV or Excel file."""
        try:
            if path.endswith('.tsv'):
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        concept_id = row.get('concept_id', row.get('id', ''))
                        name = row.get('name', row.get('concept_name', ''))
                        aliases_str = row.get('aliases', row.get('alias', ''))
                        aliases = [a.strip() for a in aliases_str.split(',')] if aliases_str else []
                        description = row.get('description', row.get('desc', '')) or ""
                        tags_str = row.get('tags', row.get('tag', row.get('tag(s)', ''))) or ""
                        tags = [t.strip() for t in str(tags_str).replace(';', ',').split(',') if t.strip()]
                        level_val = row.get('level', row.get('Level', None))
                        level = int(level_val) if str(level_val).isdigit() else None
                        pages = self._parse_pages(row.get('pages', row.get('page', row.get('Page(s)', ''))))
                        if concept_id and name:
                            self.concepts.append(Concept(concept_id, name, aliases, description, tags=tags, level=level, pages=pages))
            elif path.endswith('.xlsx'):
                self._load_concepts_xlsx(path)
            else:
                print(f"ConceptLinker: Unsupported format {path}; use .tsv or .xlsx")
                return
            print(f"ConceptLinker: Loaded {len(self.concepts)} concepts from {path}")
            self._index_concepts()
            self._build_semantic_index()
        except Exception as e:
            print(f"Error loading concepts from {path}: {e}")
    
    def _load_concepts_xlsx(self, path: str) -> None:
        """Load concepts from Excel (.xlsx). Expects columns: Level, Concept, Tag(s), Rationale, Page(s) or similar."""
        try:
            import openpyxl
        except ImportError:
            try:
                import pandas as pd
                df = pd.read_excel(path, sheet_name=0, engine='openpyxl')
                def _cell(keys):
                    for k in keys:
                        if k in df.columns:
                            return k
                    return None
                for idx, row in df.iterrows():
                    cn = _cell(['Concept', 'concept_name', 'name', 'Name'])
                    name = row[cn] if cn else None
                    if name is None or (isinstance(name, float) and (name != name)):
                        continue
                    name = str(name).strip()
                    cid_col = _cell(['concept_id', 'id', 'ID'])
                    concept_id = row[cid_col] if cid_col else None
                    if concept_id is None or (isinstance(concept_id, float) and (concept_id != concept_id)):
                        level = row[_cell(['Level', 'level', 'L1', 'L2'])] if _cell(['Level', 'level', 'L1', 'L2']) else ""
                        concept_id = f"{level}-{idx}" if level else str(idx)
                    else:
                        concept_id = str(concept_id).strip()
                    aliases = []
                    tags = _cell(['Tag(s)', 'Tags', 'tag', 'tags', 'alias', 'aliases'])
                    if tags:
                        raw = row[tags]
                        if not (isinstance(raw, float) and raw != raw):
                            parts = str(raw).replace(';', ',').split(',')
                            aliases = [a.strip() for a in parts if a.strip()]
                    tags_list = aliases[:]
                    level_val = row[_cell(['Level', 'level'])] if _cell(['Level', 'level']) else None
                    level = int(level_val) if str(level_val).isdigit() else None
                    description = ""
                    desc_col = _cell(['Rationale', 'Description', 'Desc', 'desc', 'definition'])
                    if desc_col:
                        d = row[desc_col]
                        if not (isinstance(d, float) and d != d):
                            description = str(d).strip()
                    pages_val = row[_cell(['Page(s)', 'Pages', 'page', 'pages'])] if _cell(['Page(s)', 'Pages', 'page', 'pages']) else None
                    pages = self._parse_pages(pages_val)
                    self.concepts.append(Concept(concept_id, name, aliases, description, tags=tags_list, level=level, pages=pages))
                return
            except Exception as e:
                print(f"ConceptLinker: pandas fallback failed: {e}")
                return
        
        wb = openpyxl.load_workbook(path, data_only=True)
        sheet = wb.active
        headers = [str(c.value).strip() if c.value else "" for c in sheet[1]]
        
        def find_col(names):
            for i, h in enumerate(headers):
                if h and h.lower() in [n.lower() for n in names]:
                    return i
            return None
        
        ix_concept = find_col(['Concept', 'concept_name', 'name'])
        ix_level = find_col(['Level', 'level'])
        ix_tags = find_col(['Tag(s)', 'Tags', 'tags', 'alias', 'aliases'])
        ix_desc = find_col(['Rationale', 'Description', 'definition', 'desc'])
        ix_pages = find_col(['Page(s)', 'Pages', 'page', 'pages'])
        
        for idx, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
            if ix_concept is None or ix_concept >= len(row):
                continue
            name = row[ix_concept]
            if name is None:
                continue
            name = str(name).strip()
            if not name:
                continue
            level = row[ix_level] if ix_level is not None and ix_level < len(row) else ""
            concept_id = f"{level}-{idx}" if level else str(idx)
            aliases = []
            if ix_tags is not None and ix_tags < len(row) and row[ix_tags]:
                raw = row[ix_tags]
                if not (isinstance(raw, float) and raw != raw):
                    parts = str(raw).replace(';', ',').split(',')
                    aliases = [a.strip() for a in parts if a.strip()]
            description = ""
            if ix_desc is not None and ix_desc < len(row) and row[ix_desc]:
                d = row[ix_desc]
                if not (isinstance(d, float) and d != d):
                    description = str(d).strip()
            pages = self._parse_pages(row[ix_pages]) if ix_pages is not None and ix_pages < len(row) else []
            level_val = row[ix_level] if ix_level is not None and ix_level < len(row) else None
            level = int(level_val) if str(level_val).isdigit() else None
            self.concepts.append(Concept(concept_id, name, aliases, description, tags=aliases[:], level=level, pages=pages))
        wb.close()

    def _index_concepts(self) -> None:
        self._concept_by_id = {c.concept_id: c for c in self.concepts if c.concept_id}

    def _build_lexical_evidence(self, segment: SegmentBase, concept: Concept) -> Optional[Dict[str, Any]]:
        if not concept:
            return None
        heading_text = " ".join([
            segment.heading_path or "",
            segment.chapter_title or "",
        ]).strip()
        body_text = (segment.text_content or "").strip()
        if not heading_text and not body_text:
            return None

        def _match(text: str, phrase: str, match_type: str, source: str) -> Optional[Dict[str, Any]]:
            if phrase and text and self._phrase_in_text(text, phrase, boundary=True):
                return {
                    "type": match_type,
                    "matched_phrase": phrase,
                    "source": source
                }
            return None

        # 1) Heading/title match (highest priority)
        ev = _match(heading_text, concept.name, "heading", "heading_path")
        if ev:
            return ev
        for alias in concept.aliases or []:
            ev = _match(heading_text, alias, "heading", "heading_path")
            if ev:
                return ev

        # 2) Direct name/alias match in body
        ev = _match(body_text, concept.name, "name", "segment_text")
        if ev:
            return ev
        for alias in concept.aliases or []:
            ev = _match(body_text, alias, "alias", "segment_text")
            if ev:
                return ev

        # 3) Tag/rationale n-gram match (heading preferred, then body)
        tag_phrases = self._extract_ngrams(" ".join(concept.tags or []), min_n=2, max_n=4)
        desc_phrases = self._extract_ngrams(concept.description or "", min_n=2, max_n=4)
        for phrase in tag_phrases:
            ev = _match(heading_text, phrase, "tag", "heading_path")
            if ev:
                return ev
            ev = _match(body_text, phrase, "tag", "segment_text")
            if ev:
                return ev
        for phrase in desc_phrases:
            ev = _match(heading_text, phrase, "rationale", "heading_path")
            if ev:
                return ev
            ev = _match(body_text, phrase, "rationale", "segment_text")
            if ev:
                return ev

        return None

    def _extract_ngrams(self, text: str, min_n: int = 2, max_n: int = 4) -> List[str]:
        text = self._normalize_text(text)
        if not text:
            return []
        tokens = [t for t in text.split() if len(t) >= 3 and t not in {
            "the", "and", "for", "with", "from", "this", "that", "are", "was", "were",
            "into", "over", "under", "about", "between", "among", "through", "using"
        }]
        if len(tokens) < min_n:
            return []
        phrases: List[str] = []
        max_n = min(max_n, len(tokens))
        for n in range(min_n, max_n + 1):
            for i in range(0, len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i + n]).strip()
                if phrase:
                    phrases.append(phrase)
        # Prefer longer phrases first for precision
        phrases.sort(key=lambda p: (-len(p.split()), p))
        return phrases
    
    def link_segment_to_concepts(self, segment: SegmentBase) -> List[ConceptLink]:
        """Link a segment to concepts using exact/alias/semantic matching."""
        links: List[ConceptLink] = []
        if not self.concepts:
            return links
        
        title_text = " ".join([
            segment.heading_path or "",
            segment.chapter_title or "",
        ]).strip().lower()
        body_text = (segment.text_content or "").lower()
        page_num = getattr(segment, "page_start", None)

        scored_candidates: List[Tuple[Concept, float, str]] = []
        best_by_concept = {}
        for concept in self.concepts:
            score, method = self._score_concept(concept, title_text, body_text, page_num)
            if score <= 0:
                continue
            scored_candidates.append((concept, score, method))
            link = ConceptLink(
                concept_id=concept.concept_id,
                link_method=method,
                confidence=min(score, 1.0),
                concept_name=concept.name,
                level=concept.level,
                tags=list(concept.tags or []),
                rationale=concept.description or None,
            )
            existing = best_by_concept.get(concept.concept_id)
            if not existing or score > existing[1]:
                best_by_concept[concept.concept_id] = (link, score)

        links = [item[0] for item in best_by_concept.values()]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        if self.use_llm_rerank and self.llm_service and scored_candidates:
            llm_links = self._llm_select(segment, scored_candidates[:12])
            if llm_links:
                links = llm_links

        # Semantic similarity (TF-IDF cosine) - lightweight semantic signal
        semantic_links = self._semantic_match(segment, body_text)
        if semantic_links:
            links.extend(semantic_links)

        # Heuristic keyword overlap fallback
        if not links:
            for concept in self.concepts:
                concept_words = set(concept.name.lower().split())
                segment_words = set(body_text.split())
                overlap = len(concept_words & segment_words) / max(len(concept_words), 1)
                if overlap > 0.5:
                    links.append(ConceptLink(
                        concept_id=concept.concept_id,
                        link_method="heuristic",
                        confidence=overlap
                    ))

        # Fail-safe: attach at least one concept using heading path / chapter title
        if not links:
            fallback = self._fallback_link(segment)
            if fallback:
                links.append(fallback)

        # Remove duplicates (keep highest confidence)
        seen = {}
        for link in links:
            if link.concept_id not in seen or seen[link.concept_id].confidence < link.confidence:
                seen[link.concept_id] = link
        deduped = list(seen.values())

        # Mark segment for review if no strong (lexical + semantic) link
        semantic_methods = {"semantic", "llm_rerank"}
        lexical_methods = {"exact_match", "alias"}
        any_strong = False
        for link in deduped:
            concept = self._concept_by_id.get(link.concept_id)
            evidence = self._build_lexical_evidence(segment, concept) if concept else None
            semantic_present = link.link_method in semantic_methods
            lexical_present = bool(evidence) or link.link_method in lexical_methods
            if semantic_present and lexical_present:
                any_strong = True
                break

        if not any_strong:
            segment.needs_human_review = True

        # Keep lexical/semantic links, and allow heuristic/fallback only if at least one strong link exists
        strong_methods = {"exact_match", "alias", "semantic", "llm_rerank"}
        filtered = [l for l in deduped if l.link_method in strong_methods] or deduped

        return filtered[:5]  # Top 5 matches

    def _score_concept(self, concept: Concept, title_text: str, body_text: str, page_num: Optional[int]) -> tuple:
        name = concept.name or ""
        score = 0.0
        method = "heuristic"

        if name and self._phrase_in_text(title_text, name, boundary=True):
            score += 0.6
            method = "exact_match"
        if not score:
            for alias in concept.aliases:
                if alias and self._phrase_in_text(title_text, alias, boundary=True):
                    score += 0.5
                    method = "alias"
                    break

        if name and self._phrase_in_text(body_text, name, boundary=True):
            score += 0.35
            method = method if method != "heuristic" else "exact_match"
        if score < 0.5:
            for alias in concept.aliases:
                if alias and self._phrase_in_text(body_text, alias, boundary=True):
                    score += 0.3
                    method = method if method != "heuristic" else "alias"
                    break

        if score < 0.4 and name and len(name) >= 4 and self._phrase_in_text(body_text, name, boundary=False):
            score += 0.15
        if score < 0.4:
            for alias in concept.aliases:
                if alias and len(alias) >= 4 and self._phrase_in_text(body_text, alias, boundary=False):
                    score += 0.12
                    break

        if concept.tags:
            tag_hits = sum(1 for t in concept.tags if t.lower() in body_text or t.lower() in title_text)
            if tag_hits:
                score += min(0.2, 0.05 * tag_hits)

        if concept.description:
            if self._phrase_in_text(body_text, concept.description, boundary=False):
                score += 0.1

        if page_num and concept.pages:
            page_dist = min(abs(page_num - p) for p in concept.pages if p)
            if page_dist == 0:
                score += 0.4
            elif page_dist <= 1:
                score += 0.25
            elif page_dist <= 3:
                score += 0.1
            elif page_dist > 6:
                return (0.0, "heuristic")

        return (score, method)

    def _phrase_in_text(self, text: str, phrase: str, boundary: bool = True) -> bool:
        if not text or not phrase:
            return False
        norm_text = self._normalize_text(text)
        norm_phrase = self._normalize_text(phrase)
        if not norm_text or not norm_phrase:
            return False
        if boundary:
            # Word-boundary match on normalized text (handles punctuation/hyphens)
            if " " in norm_phrase:
                pattern = r'\b' + re.escape(norm_phrase) + r'\b'
                return bool(re.search(pattern, norm_text))
            pattern = r'\b' + re.escape(norm_phrase) + r'\b'
            return bool(re.search(pattern, norm_text))
        return norm_phrase in norm_text

    def _normalize_text(self, text: str) -> str:
        """Normalize for lexical matching."""
        t = (text or "").lower()
        t = re.sub(r"[^a-z0-9\\s-]+", " ", t)
        t = t.replace("-", " ")
        t = re.sub(r"\\s+", " ", t).strip()
        return t

    def _llm_select(self, segment: SegmentBase, candidates: List[Tuple[Concept, float, str]]) -> List[ConceptLink]:
        prompt = self._build_llm_prompt(segment, candidates)
        data = generate_json_with_llm(self.llm_service, prompt, temperature=0.2)
        if not data or "selected" not in data:
            return []

        selected = data.get("selected", [])
        if not isinstance(selected, list):
            return []

        id_map = {c.concept_id.lower(): c for c, _, _ in candidates}
        name_map = {c.name.lower(): c for c, _, _ in candidates if c.name}
        links: List[ConceptLink] = []
        for item in selected:
            if not isinstance(item, dict):
                continue
            raw_id = str(item.get("concept_id", "")).strip().lower()
            concept = id_map.get(raw_id) or name_map.get(raw_id)
            if not concept:
                continue
            conf = item.get("confidence", 0.7)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.7
            links.append(ConceptLink(
                concept_id=concept.concept_id,
                link_method="llm_rerank",
                confidence=max(0.0, min(conf, 1.0))
            ))

        return links[:8]

    def _build_llm_prompt(self, segment: SegmentBase, candidates: List[Tuple[Concept, float, str]]) -> str:
        segment_text = (segment.text_content or "").strip()
        lines = [
            "You are classifying textbook content into predefined concepts.",
            "",
            "Segment:",
            f"Type: {getattr(segment, 'segment_type', 'unknown')}",
            f"Text: \"{segment_text[:800]}\"",
            f"Chapter: {segment.chapter_number}",
            f"Section: {segment.heading_path or ''}",
            f"Page: {segment.page_start}",
            "",
            "Candidate concepts:",
        ]
        for idx, (concept, score, _) in enumerate(candidates, start=1):
            desc = concept.description or ""
            tags = ", ".join(concept.tags) if concept.tags else ""
            pages = ", ".join(str(p) for p in concept.pages[:5]) if concept.pages else ""
            lines.extend([
                f"{idx}. {concept.concept_id} — {concept.name}",
                f"   Definition: {desc}" if desc else "   Definition: (none)",
                f"   Tags: {tags}" if tags else "   Tags: (none)",
                f"   Pages: {pages}" if pages else "   Pages: (none)",
                f"   HeuristicScore: {score:.2f}",
            ])

        lines.extend([
            "",
            "Task:",
            "Select all concepts that match the segment. You may return multiple.",
            "Return JSON only.",
            "",
            "Output format:",
            "{",
            "  \"selected\": [",
            "    {\"concept_id\": \"...\", \"confidence\": 0.0}",
            "  ]",
            "}",
        ])
        return "\n".join(lines)

    def _parse_pages(self, value: Optional[str]) -> List[int]:
        if value is None:
            return []
        raw = str(value)
        pages = set()
        for part in re.split(r'[;,]', raw):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                try:
                    start_str, end_str = part.split('-', 1)
                    start = int(re.sub(r'\D', '', start_str))
                    end = int(re.sub(r'\D', '', end_str))
                    for p in range(start, end + 1):
                        pages.add(p)
                except Exception:
                    continue
            else:
                nums = re.findall(r'\d+', part)
                for n in nums:
                    pages.add(int(n))
        return sorted(pages)
    
    def _fallback_link(self, segment: SegmentBase) -> Optional[ConceptLink]:
        """Best-effort fallback to ensure at least one concept link."""
        text = " ".join([
            segment.heading_path or "",
            segment.chapter_title or "",
        ]).strip().lower()
        best = self._best_overlap_concept(text) if text else None
        if best:
            return ConceptLink(
                concept_id=best.concept_id,
                link_method="fallback",
                confidence=0.25
            )
        # Last-resort fallback to first concept in list
        if self.concepts:
            return ConceptLink(
                concept_id=self.concepts[0].concept_id,
                link_method="fallback",
                confidence=0.1
            )
        return None
    
    def _best_overlap_concept(self, text: str) -> Optional[Concept]:
        words = set(re.findall(r'[A-Za-z][A-Za-z\-]+', text.lower()))
        if not words:
            return None
        best = None
        best_score = 0.0
        for concept in self.concepts:
            c_words = set(re.findall(r'[A-Za-z][A-Za-z\-]+', concept.name.lower()))
            if not c_words:
                continue
            score = len(words & c_words) / max(len(c_words), 1)
            if score > best_score:
                best_score = score
                best = concept
        return best if best_score > 0 else None
    
    def _semantic_match(self, segment: SegmentBase, segment_text: str) -> List[ConceptLink]:
        """
        Lightweight semantic similarity using TF-IDF cosine.
        This provides a non-LLM semantic signal to satisfy lexical+semantic requirements.
        """
        if not segment_text or not self._concept_term_freqs:
            return []
        query_vec = self._build_tfidf_vector(self._tokenize(segment_text))
        if not query_vec:
            return []
        scored: List[Tuple[str, float]] = []
        for concept in self.concepts:
            concept_vec = self._build_concept_vector(concept.concept_id)
            if not concept_vec:
                continue
            score = self._cosine_similarity(query_vec, concept_vec)
            if score > 0.18:
                scored.append((concept.concept_id, score))
        if not scored:
            return []
        scored.sort(key=lambda x: x[1], reverse=True)
        links: List[ConceptLink] = []
        for concept_id, score in scored[:5]:
            links.append(ConceptLink(
                concept_id=concept_id,
                link_method="semantic",
                confidence=min(score, 1.0)
            ))
        return links

    def _build_semantic_index(self) -> None:
        """Build TF-IDF stats for semantic matching."""
        self._concept_term_freqs = {}
        self._term_df = {}
        for concept in self.concepts:
            terms = self._tokenize(" ".join([
                concept.name or "",
                " ".join(concept.aliases or []),
                " ".join(concept.tags or []),
                concept.description or ""
            ]))
            if not terms:
                continue
            tf = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            self._concept_term_freqs[concept.concept_id] = tf
            for t in set(terms):
                self._term_df[t] = self._term_df.get(t, 0) + 1

    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\\-\\s]", " ", text)
        parts = []
        for token in text.split():
            token = token.strip("-_")
            if len(token) < 3:
                continue
            if token in {"the", "and", "for", "with", "from", "this", "that", "are", "was", "were"}:
                continue
            parts.append(token)
        return parts

    def _build_tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = max(1, sum(tf.values()))
        vec: Dict[str, float] = {}
        doc_count = max(1, len(self.concepts))
        for t, count in tf.items():
            df = self._term_df.get(t, 1)
            idf = 1.0 + (doc_count / (df + 1))
            vec[t] = (count / total) * idf
        return vec

    def _build_concept_vector(self, concept_id: str) -> Dict[str, float]:
        tf = self._concept_term_freqs.get(concept_id)
        if not tf:
            return {}
        total = max(1, sum(tf.values()))
        vec: Dict[str, float] = {}
        doc_count = max(1, len(self.concepts))
        for t, count in tf.items():
            df = self._term_df.get(t, 1)
            idf = 1.0 + (doc_count / (df + 1))
            vec[t] = (count / total) * idf
        return vec

    def _cosine_similarity(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        for k, v in a.items():
            if k in b:
                dot += v * b[k]
        norm_a = sum(v * v for v in a.values()) ** 0.5
        norm_b = sum(v * v for v in b.values()) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
