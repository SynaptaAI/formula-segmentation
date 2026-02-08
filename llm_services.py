import os
import re
import time
import json
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from schemas import VariableDefinition
from segmenter.cache_manager import DiskCache


class LLMService(ABC):
    @abstractmethod
    def extract_variables(self, formula_text: str, context: str, candidate_symbols: List[str] = None) -> List[VariableDefinition]:
        pass

    @abstractmethod
    def structure_worked_example(self, text: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def extract_formula_summary(self, formula_text: str, variables: List[VariableDefinition]) -> Optional[str]:
        pass

    def convert_to_latex(self, formula_text: str, canonical_key: Optional[str] = None) -> Optional[str]:
        return None


class MockLLMService(LLMService):
    def extract_variables(self, formula_text: str, context: str, candidate_symbols: List[str] = None) -> List[VariableDefinition]:
        return []

    def structure_worked_example(self, text: str) -> Dict[str, Any]:
        return {
            "title": None,
            "problem_statement": text,
            "steps": [],
            "final_answer": None
        }

    def extract_formula_summary(self, formula_text: str, variables: List[VariableDefinition]) -> Optional[str]:
        return None

    def convert_to_latex(self, formula_text: str, canonical_key: Optional[str] = None) -> Optional[str]:
        return None


class OpenAILLMService(LLMService):
    """OpenAI-based LLM service for variable extraction and formula analysis."""
    # Class-level cache for LLM calls (shared across instances)
    _llm_cache: Dict[str, Any] = {}
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", enable_cache: bool = True,
                 base_url: str = None):
        """
        base_url: Optional custom API base URL (e.g., "https://openrouter.ai/api/v1" for OpenRouter).
                  If None, uses OpenAI default.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model
        self.enable_cache = enable_cache
        self.base_url = base_url
        self.disk_cache = DiskCache() if enable_cache else None
        # Ollama uses a local OpenAI-compatible endpoint without API key.
        is_ollama = base_url and "localhost:11434" in base_url
        if not self.api_key and not is_ollama:
            print("Warning: OPENAI_API_KEY not found in environment variables or .env file.")
            print("Please set OPENAI_API_KEY in .env file or as an environment variable.")
            print("LLM features will fail or fallback.")
            self.client = None
        else:
            try:
                from openai import OpenAI
                kwargs = {}
                # Ollama uses a local endpoint; other providers require a key.
                if not is_ollama:
                    kwargs["api_key"] = self.api_key
                elif self.api_key == "ollama":
                    # Ignore placeholder key for Ollama.
                    pass
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self.client = OpenAI(**kwargs)
            except ImportError:
                print("Warning: openai package not installed. Install with: pip install openai")
                self.client = None
    
    def _get_cache_key(self, prompt: str, method: str) -> str:
        """Generate cache key from prompt and method."""
        cache_str = f"{method}:{prompt}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _cached_call(self, cache_key: str, call_func):
        """Execute LLM call with caching."""
        if not self.enable_cache or not self.disk_cache:
            return call_func()
        
        cached = self.disk_cache.get(cache_key)
        if cached is not None:
             return cached
        
        result = call_func()
        if result is not None:
            self.disk_cache.set(cache_key, result)
        return result

    def extract_variables(self, formula_text: str, context: str, candidate_symbols: List[str] = None) -> List[VariableDefinition]:
        """
        Extract variable meanings using LLM (meaning inference only, no JSON structure).
        
        Flow:
        1. Use candidate_symbols (already extracted by heuristic)
        2. LLM infers meanings (free text, no JSON requirement)
        3. Code parses response and structures as VariableDefinition
        """
        if not self.client:
            return []
        
        # Step 1: Use candidate symbols (already extracted by heuristic)
        if not candidate_symbols:
            return []
        
        # Step 2: LLM infers meanings only (free text, no JSON)
        prompt = f"""Given this formula and context, explain what each variable means.
Treat subscripted variables as single symbols (e.g. R_M, r_f are one variable each).

Formula: {formula_text}
Context: {context}

Variables in the formula: {', '.join(candidate_symbols[:15])}

For each variable, explain its meaning in the context of this formula. 
You can write in natural language, one variable per line, or in any clear format.
Focus on what each variable represents in financial/economic terms.

Example format (but you can use any clear format):
- r: risk-free rate
- E(r_M): expected return on market portfolio
- σ: standard deviation of returns"""

        # Cache key for this call
        cache_key = self._get_cache_key(prompt, "extract_variables")
        
        def _do_extract():
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # NO JSON format requirement - free text response
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3  # Lower temperature for more consistent extraction
                    )
                    
                    content = response.choices[0].message.content
                    
                    # Step 3: Parse free text response and structure as VariableDefinition
                    result = self._parse_meanings_response(content, candidate_symbols, formula_text)
                    
                    return result
                    
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        wait_time = 2 * (attempt + 1)
                        print(f"Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    print(f"LLM Error (extract_variables): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return []
            
            return []
        
        return self._cached_call(cache_key, _do_extract)
    
    def _parse_meanings_response(self, response_text: str, candidate_symbols: List[str], formula_text: str) -> List[VariableDefinition]:
        """
        Parse LLM's free-text response about variable meanings.
        Improved parsing: block-based, symbol-presence based (not format-based).
        
        Handles:
        - Multi-line explanations
        - Bullet points
        - Various formats (with/without colons)
        - Partial matches
        """
        variables = []
        
        # Step 1: Split into blocks (paragraphs separated by blank lines)
        blocks = re.split(r'\n\s*\n', response_text.strip())
        # Also consider single lines as blocks
        if len(blocks) == 1:
            blocks = response_text.split('\n')
        
        # Step 2: For each candidate symbol, find its meaning block
        for symbol in candidate_symbols:
            meaning = None
            units = None
            
            # Strategy 1: Look for explicit patterns (colon, dash, etc.)
            patterns = [
                rf'(?:^|\n|[-•])\s*{re.escape(symbol)}\s*[:=]\s*([^\n]+(?:\n[^\n]+)*)',  # Multi-line after colon
                rf'{re.escape(symbol)}\s+is\s+([^\n]+(?:\n[^\n]+)*)',  # "symbol is ..."
                rf'{re.escape(symbol)}\s+represents\s+([^\n]+(?:\n[^\n]+)*)',  # "symbol represents ..."
                rf'{re.escape(symbol)}\s*\([^)]*\)\s*[:=]?\s*([^\n]+(?:\n[^\n]+)*)',  # "symbol (description): ..."
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    meaning = match.group(1).strip()
                    # Clean up: remove leading dashes, bullets
                    meaning = re.sub(r'^[-•]\s*', '', meaning)
                    # Take first sentence/paragraph (up to period or newline)
                    meaning = re.split(r'[.\n]', meaning)[0].strip()
                    if meaning:
                        break
            
            # Strategy 2: Block-based matching (symbol appears in block)
            if not meaning:
                for block in blocks:
                    block_lower = block.lower()
                    symbol_lower = symbol.lower()
                    
                    # Check if symbol appears in this block
                    if symbol_lower in block_lower and len(block) > len(symbol) + 10:
                        # Extract meaning: everything after symbol (with flexible patterns)
                        # Try different separators
                        for sep in [':', '-', 'is', 'represents', '(', '=']:
                            parts = re.split(rf'{re.escape(symbol)}\s*{re.escape(sep)}\s*', block, flags=re.IGNORECASE, maxsplit=1)
                            if len(parts) > 1:
                                meaning = parts[1].strip()
                                # Clean up: remove leading dashes, bullets
                                meaning = re.sub(r'^[-•]\s*', '', meaning)
                                # Take first sentence/paragraph
                                meaning = re.split(r'[.\n]', meaning)[0].strip()
                                if meaning and len(meaning) > 3:
                                    break
                        
                        if meaning:
                            break
                        
                        # Fallback: if symbol is at start of block, take rest of block
                        if block.strip().lower().startswith(symbol_lower):
                            meaning = block[len(symbol):].strip()
                            meaning = re.sub(r'^[-•:\s]+', '', meaning)
                            meaning = re.split(r'[.\n]', meaning)[0].strip()
                            if meaning and len(meaning) > 3:
                                break
            
            # Strategy 3: Line-based matching (last resort)
            if not meaning:
                lines = response_text.split('\n')
                for line in lines:
                    line_lower = line.lower()
                    if symbol_lower in line_lower and len(line) > len(symbol) + 5:
                        # Extract after symbol
                        parts = re.split(rf'{re.escape(symbol)}\s*[:=]?\s*', line, flags=re.IGNORECASE, maxsplit=1)
                        if len(parts) > 1:
                            meaning = parts[1].strip()
                            meaning = re.sub(r'^[-•]\s*', '', meaning)
                            meaning = re.split(r'[.,;]', meaning)[0].strip()
                            if meaning and len(meaning) > 3:
                                break
            
            # Clean up meaning
            if meaning:
                # Remove trailing punctuation
                meaning = re.sub(r'[.,;]+$', '', meaning).strip()
                # Remove markdown formatting
                meaning = re.sub(r'\*\*([^*]+)\*\*', r'\1', meaning)  # Bold
                meaning = re.sub(r'\*([^*]+)\*', r'\1', meaning)  # Italic
                
                # Extract units if mentioned
                units_match = re.search(r'\(in\s+([^)]+)\)|\(units?:\s*([^)]+)\)', meaning, re.IGNORECASE)
                if units_match:
                    units = units_match.group(1) or units_match.group(2)
                    meaning = re.sub(r'\(in\s+[^)]+\)|\(units?:\s*[^)]+\)', '', meaning, flags=re.IGNORECASE).strip()
            
            # Only add if we found a meaningful description (at least 5 chars)
            if meaning and len(meaning) >= 5:
                variables.append(VariableDefinition(
                    symbol=symbol,
                    meaning=meaning,
                    units=units,
                    inferred=True,
                    source="llm",
                ))
        
        return variables
    
    def structure_worked_example(self, text: str) -> Dict[str, Any]:
        if not self.client:
            return {"problem_statement": text}

        prompt = f"""Analyze this Worked Example text and extract structured information:

---
{text}
---

Return a JSON object with:
- "title": title if present, or null
- "problem_statement": the problem/question being solved
- "steps": array of solution steps (strings)
- "final_answer": the final answer or result

Return ONLY valid JSON, no other text."""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                data = parse_json_response(response.choices[0].message.content)
                return data or {"problem_statement": text}
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait_time = 2 * (attempt + 1)
                    time.sleep(wait_time)
                    continue
                print(f"LLM Error (structure_worked_example): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return {"problem_statement": text}
        
        return {"problem_statement": text}
    
    def extract_formula_summary(self, formula_text: str, variables: List[VariableDefinition]) -> Optional[str]:
        """Extract a short summary of what the formula calculates."""
        if not self.client:
            # Fallback: use variables if available
            if variables:
                var_names = [v.symbol for v in variables[:3]]
                return f"Formula computing {', '.join(var_names)}"
            return None
        
        var_info = ""
        if variables:
            var_info = f"\nVariables: {', '.join([f'{v.symbol} ({v.meaning})' for v in variables[:5]])}"
        else:
            var_info = "\nVariables: Not yet identified"
        
        prompt = f"""What does this formula calculate? Provide a concise 1-2 sentence summary.

Formula: {formula_text}
{var_info}

Focus on: What quantity does it compute? What is it used for in finance/economics?
Return ONLY the summary text, no JSON, no explanation, no markdown."""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=100
                )
                
                summary = response.choices[0].message.content.strip()
                # Remove markdown formatting if present
                summary = summary.replace("**", "").replace("*", "").strip()
                
                if summary and len(summary) < 200:
                    return summary
                
                # Fallback if summary is too long or empty
                if variables:
                    var_names = [v.symbol for v in variables[:3]]
                    return f"Formula computing {', '.join(var_names)}"
                
                return None
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait_time = 2 * (attempt + 1)
                    time.sleep(wait_time)
                    continue
                print(f"LLM Error (extract_formula_summary): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                
                # Fallback
                if variables:
                    var_names = [v.symbol for v in variables[:3]]
                    return f"Formula computing {', '.join(var_names)}"
                return None
        
        return None

    def convert_to_latex(self, formula_text: str, canonical_key: Optional[str] = None) -> Optional[str]:
        if not self.client:
            return None
        prompt = f"""Convert this formula to LaTeX. Return ONLY valid LaTeX, no markdown.

Formula: {formula_text}
"""
        cache_key = None
        if canonical_key:
            cache_key = self._get_cache_key(f"latex:{canonical_key}", "convert_to_latex")

        def _do_convert():
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300
                )
                latex = response.choices[0].message.content.strip()
                latex = latex.replace("```", "").strip()
                return latex if latex else None
            except Exception:
                return None

        if cache_key:
            return self._cached_call(cache_key, _do_convert)
        return _do_convert()

    def extract_variables_batch(
        self,
        items: List[Tuple[str, str, Optional[List[str]]]],
    ) -> List[Tuple[List[VariableDefinition], Optional[str]]]:
        """Batch variable extraction + one-line summary per formula. Returns [(variables, summary), ...]."""
        if not self.client or items is None or len(items) == 0:
            return [([], None) for _ in (items or [])]
        parts = []
        for idx, (formula_text, context, candidate_symbols) in enumerate(items, 1):
            hint = ""
            if candidate_symbols:
                hint = f" Candidate symbols: {', '.join(candidate_symbols[:10])}."
            parts.append(f"[{idx}]\nFormula: {formula_text}\nContext: {context}{hint}")
        prompt = """Extract variables for each of the following formulas. For each formula, use its Context and Candidate symbols to infer variable meanings. Also provide a one-line "summary": what does this formula calculate (e.g. "Black-Scholes call option price").

Important: Treat subscripted variables as single symbols. E.g. R_M, r_f, beta_GE, sigma_i are one variable each—use the symbol exactly as in the formula (with subscript), not as separate R and M, r and f, etc.

""" + "\n\n---\n\n".join(parts) + """

Return a JSON object with a single key "results" whose value is an array of objects. Each object has "variables" (array of {"symbol", "meaning", "units"?}) and "summary" (one-line string, or null). Same order as [1], [2], ... above.
Example: {"results": [{"variables": [{"symbol":"r_f","meaning":"risk-free rate","units":null}], "summary": "Present value of a bond"}, {"variables": [], "summary": null}]}
Return ONLY valid JSON."""

        cache_key = self._get_cache_key(prompt, "extract_variables_batch")
        def _do_batch():
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                data = parse_json_response(response.choices[0].message.content)
                if isinstance(data, dict):
                    results = data.get("results") or []
                elif isinstance(data, list):
                    results = data
                else:
                    results = []
                if not isinstance(results, list):
                    results = []
                out: List[Tuple[List[VariableDefinition], Optional[str]]] = []
                for i, r in enumerate(results):
                    if i >= len(items):
                        break
                    vars_list = r.get("variables", []) if isinstance(r, dict) else []
                    summary = r.get("summary") if isinstance(r, dict) else None
                    if summary and not isinstance(summary, str):
                        summary = str(summary)[:200] if summary else None
                    out.append((build_variable_definitions(vars_list), summary))
                while len(out) < len(items):
                    out.append(([], None))
                return out[: len(items)]
            except Exception as e:
                print(f"LLM Error (extract_variables_batch): {e}")
                return [([], None) for _ in items]
        
        if not self.enable_cache:
            return _do_batch()
            
        # For batch, we cache the entire result using the hash of all items.
        # This is simple but means any change invalidates the whole batch. 
        # Given we use canonical keys + deterministic batching, this is acceptable.
        return self._cached_call(cache_key, _do_batch)


class GeminiLLMService(LLMService):
    """Gemini service using google.genai (new SDK). GEMINI_API_KEY required."""
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash", enable_cache: bool = True):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.enable_cache = enable_cache
        self.client = None
        self.disk_cache = DiskCache() if enable_cache else None
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables or .env file.")
            print("Please set GEMINI_API_KEY in .env file or as an environment variable.")
        else:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except ImportError:
                print("Warning: google-genai package not installed. Install with: pip install google-genai")
                print("Gemini LLM features will fail or fallback.")
            except Exception as e:
                print(f"Warning: Gemini client initialization failed: {e}")
                print("Gemini LLM features will fail or fallback.")

    def _generate(self, prompt: str, json_mode: bool = False) -> Optional[str]:
        if not self.client:
            return None
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            # Try response.text first (new SDK)
            if hasattr(response, "text") and response.text:
                return response.text
            # Fallback: old SDK structure
            if getattr(response, "candidates", None) and response.candidates:
                c = response.candidates[0]
                if getattr(c, "content", None) and getattr(c.content, "parts", None) and c.content.parts:
                    return getattr(c.content.parts[0], "text", None)
            # If no text found, check for errors
            if hasattr(response, "prompt_feedback"):
                fb = response.prompt_feedback
                if hasattr(fb, "block_reason") and fb.block_reason:
                    raise Exception(f"Gemini blocked: {fb.block_reason}")
            return None
        except Exception as e:
            # Include extra error details when available.
            error_msg = str(e)
            if hasattr(e, "status_code"):
                error_msg += f" (status: {e.status_code})"
            if hasattr(e, "message"):
                error_msg += f" (message: {e.message})"
            raise Exception(f"Gemini API error: {error_msg}") from e

    def _get_cache_key(self, prompt: str, method: str) -> str:
        """Generate cache key from prompt and method."""
        raw = f"{self.model_name}::{method}::{prompt}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cached_call(self, cache_key: str, call_func):
        if not self.enable_cache or not self.disk_cache:
            return call_func()
        cached = self.disk_cache.get(cache_key)
        if cached is not None:
             return cached
        result = call_func()
        if result is not None:
            self.disk_cache.set(cache_key, result)
        return result

    def extract_variables(self, formula_text: str, context: str, candidate_symbols: List[str] = None) -> List[VariableDefinition]:
        """Extract variable meanings using Gemini (meaning inference only, no JSON structure)."""
        if not self.client:
            return []
        
        if not candidate_symbols:
            return []
        
        # LLM infers meanings only (free text, no JSON)
        prompt = f"""Given this formula and context, explain what each variable means.
Treat subscripted variables as single symbols (e.g. R_M, r_f are one variable each).

Formula: {formula_text}
Context: {context}

Variables in the formula: {', '.join(candidate_symbols[:15])}

For each variable, explain its meaning in the context of this formula. 
You can write in natural language, one variable per line, or in any clear format.
Focus on what each variable represents in financial/economic terms.

Example format (but you can use any clear format):
- r: risk-free rate
- E(r_M): expected return on market portfolio
- σ: standard deviation of returns"""
        

        
        cache_key = self._get_cache_key(prompt, "extract_variables")
        
        def _do_extract():
             for attempt in range(3):
                try:
                    text = self._generate(prompt, json_mode=False)  # Free text, not JSON
                    if not text:
                        continue
                    # Parse free text response
                    result = self._parse_meanings_response(text, candidate_symbols, formula_text)
                    return result
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        time.sleep(2 * (attempt + 1))
                        continue
                    print(f"LLM Error (extract_variables): {e}")
                    return []
             return []

        return self._cached_call(cache_key, _do_extract)
    
    def _parse_meanings_response(self, response_text: str, candidate_symbols: List[str], formula_text: str) -> List[VariableDefinition]:
        """Parse LLM's free-text response about variable meanings."""
        variables = []
        
        for symbol in candidate_symbols:
            meaning = None
            units = None
            
            # Pattern 1: "- symbol: meaning" or "symbol: meaning"
            patterns = [
                rf'(?:^|\n|[-•])\s*{re.escape(symbol)}\s*[:=]\s*([^\n]+)',
                rf'{re.escape(symbol)}\s+is\s+([^\n,.;]+)',
                rf'{re.escape(symbol)}\s+represents\s+([^\n,.;]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    meaning = match.group(1).strip()
                    meaning = re.sub(r'[.,;]+$', '', meaning).strip()
                    break
            
            # Pattern 2: Look for symbol in context
            if not meaning:
                lines = response_text.split('\n')
                for line in lines:
                    if symbol.lower() in line.lower() and len(line) > len(symbol) + 5:
                        parts = re.split(rf'{re.escape(symbol)}\s*[:=]?\s*', line, flags=re.IGNORECASE)
                        if len(parts) > 1:
                            meaning = parts[1].strip()
                            meaning = re.sub(r'[.,;]+$', '', meaning).strip()
                            break
            
            # Extract units if mentioned
            if meaning:
                units_match = re.search(r'\(in\s+([^)]+)\)|\(units?:\s*([^)]+)\)', meaning, re.IGNORECASE)
                if units_match:
                    units = units_match.group(1) or units_match.group(2)
                    meaning = re.sub(r'\(in\s+[^)]+\)|\(units?:\s*[^)]+\)', '', meaning, flags=re.IGNORECASE).strip()
            
            if meaning and len(meaning) > 2:
                variables.append(VariableDefinition(
                    symbol=symbol,
                    meaning=meaning,
                    units=units,
                    inferred=True,
                    source="llm",
                ))
        
        return variables

    def structure_worked_example(self, text: str) -> Dict[str, Any]:
        if not self.client:
            return {"problem_statement": text}
        prompt = f"""Analyze this Worked Example text and extract structured information:

---
{text}
---

Return a JSON object with: "title", "problem_statement", "steps" (array of strings), "final_answer". Return ONLY valid JSON."""
        for attempt in range(3):
            try:
                raw = self._generate(prompt, json_mode=True)
                if raw:
                    data = parse_json_response(raw)
                    return data or {"problem_statement": text}
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(2 * (attempt + 1))
                    continue
                print(f"LLM Error (structure_worked_example): {e}")
        return {"problem_statement": text}

    def extract_formula_summary(self, formula_text: str, variables: List[VariableDefinition]) -> Optional[str]:
        if not self.client:
            return f"Formula computing {', '.join([v.symbol for v in variables[:3]])}" if variables else None
        var_info = "\nVariables: " + ", ".join([f"{v.symbol} ({v.meaning})" for v in variables[:5]]) if variables else "\nVariables: Not yet identified"
        prompt = f"""What does this formula calculate? One or two short sentences. No JSON, no markdown.

Formula: {formula_text}
{var_info}"""
        for attempt in range(3):
            try:
                summary = self._generate(prompt, json_mode=False)
                if summary:
                    summary = summary.replace("**", "").replace("*", "").strip()[:200]
                    return summary if summary else (f"Formula computing {', '.join([v.symbol for v in variables[:3]])}" if variables else None)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(2 * (attempt + 1))
                    continue
                print(f"LLM Error (extract_formula_summary): {e}")
        return f"Formula computing {', '.join([v.symbol for v in variables[:3]])}" if variables else None

    def convert_to_latex(self, formula_text: str, canonical_key: Optional[str] = None) -> Optional[str]:
        prompt = f"""Convert this formula to LaTeX. Return ONLY valid LaTeX, no markdown.

Formula: {formula_text}
"""
        text = self._generate(prompt, json_mode=False)
        if not text:
            return None
        latex = text.replace("```", "").strip()
        return latex if latex else None

    def extract_variables_batch(
        self,
        items: List[Tuple[str, str, Optional[List[str]]]],
    ) -> List[Tuple[List[VariableDefinition], Optional[str]]]:
        """Batch variable extraction + one-line summary per formula. Returns [(variables, summary), ...]."""
        if not self.client or items is None or len(items) == 0:
            return [([], None) for _ in (items or [])]
        parts = []
        for idx, (formula_text, context, candidate_symbols) in enumerate(items, 1):
            hint = ""
            if candidate_symbols:
                hint = f" Candidate symbols: {', '.join(candidate_symbols[:10])}."
            parts.append(f"[{idx}]\nFormula: {formula_text}\nContext: {context}{hint}")
        prompt = """Extract variables for each of the following formulas. For each formula, use its Context and Candidate symbols to infer variable meanings. Also provide a one-line "summary": what does this formula calculate.

Important: Treat subscripted variables as single symbols. E.g. R_M, r_f, beta_GE, sigma_i are one variable each—use the symbol exactly as in the formula (with subscript), not as separate R and M, r and f, etc.

""" + "\n\n---\n\n".join(parts) + """

Return a JSON object with a single key "results" whose value is an array of objects. Each object has "variables" (array of {"symbol", "meaning", "units"?}) and "summary" (one-line string, or null). Same order as [1], [2], ... above.
Example: {"results": [{"variables": [{"symbol":"r_f","meaning":"risk-free rate","units":null}], "summary": "Present value of a bond"}, {"variables": [], "summary": null}]}
Return ONLY valid JSON."""
        
        # Check cache
        cache_key = None
        if self.enable_cache and self.disk_cache:
            cache_key = self._get_cache_key(prompt, "extract_variables_batch")
            cached = self.disk_cache.get(cache_key)
            if cached:
                # print("DEBUG: Batch cache hit")
                return cached

        for attempt in range(3):
            try:
                if attempt > 0:
                    print(f"    Retry {attempt + 1}/3...")
                text = self._generate(prompt, json_mode=True)
                if not text:
                    if attempt == 0:
                        print(f"    Warning: Gemini returned empty response")
                    continue
                # Try to parse JSON (may have markdown code blocks)
                json_text = text.strip()
                if "```" in json_text:
                    import re
                    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", json_text, re.DOTALL)
                    if match:
                        json_text = match.group(1)
                data = parse_json_response(json_text)
                if isinstance(data, dict):
                    results = data.get("results") or []
                elif isinstance(data, list):
                    results = data
                else:
                    results = []
                if not isinstance(results, list):
                    results = []
                out: List[Tuple[List[VariableDefinition], Optional[str]]] = []
                for i, r in enumerate(results):
                    if i >= len(items):
                        break
                    vars_list = r.get("variables", []) if isinstance(r, dict) else []
                    summary = r.get("summary") if isinstance(r, dict) else None
                    if summary and not isinstance(summary, str):
                        summary = str(summary)[:200] if summary else None
                    out.append((build_variable_definitions(vars_list), summary))
                while len(out) < len(items):
                    out.append(([], None))
                
                result = out[:len(items)]
                if self.enable_cache and self.disk_cache and cache_key and result:
                    self.disk_cache.set(cache_key, result)
                return result
            except json.JSONDecodeError as e:
                if attempt == 0:
                    print(f"    Warning: JSON parse failed. Response preview: {text[:200] if text else 'None'}...")
                if attempt < 2:
                    continue
                return [([], None) for _ in items]
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower() or "rate_limit" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                    wait = 2 * (attempt + 1)
                    print(f"    Rate limit, waiting {wait}s...")
                    if attempt == 0:
                        print(f"    Full error: {error_str[:300]}")
                    time.sleep(wait)
                    continue
                if attempt == 0:
                    print(f"    Error: {type(e).__name__}: {error_str[:300]}")
                    # Include response details when available.
                    if hasattr(e, "response"):
                        print(f"    Response status: {getattr(e.response, 'status_code', 'N/A')}")
                if attempt < 2:
                    time.sleep(1)
                    continue
                return [([], None) for _ in items]
        return [([], None) for _ in items]


class AnthropicLLMService(LLMService):
    """Claude(Anthropic) API for variable extraction, summary, worked_example. ANTHROPIC_API_KEY or CLAUDE_API_KEY required."""
    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20240620",
                 fallback_models: Optional[List[str]] = None, enable_cache: bool = True):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        self.model_name = model
        self.fallback_models = fallback_models or ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        self.enable_cache = enable_cache
        self.client = None
        self.disk_cache = DiskCache() if enable_cache else None
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("Warning: anthropic not installed. Install with: pip install anthropic")
            except Exception as e:
                print(f"Warning: Anthropic client init failed: {e}")
        if not self.client:
            print("Warning: ANTHROPIC_API_KEY/CLAUDE_API_KEY not found. LLM features will fail or fallback.")

        if self.api_key:
             try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
             except ImportError:
                 pass

    def _get_cache_key(self, prompt: str, method: str) -> str:
        """Generate cache key from prompt and method."""
        raw = f"{self.model_name}::{method}::{prompt}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cached_call(self, cache_key: str, call_func):
        if not self.enable_cache or not self.disk_cache:
            return call_func()
        cached = self.disk_cache.get(cache_key)
        if cached is not None:
             return cached
        result = call_func()
        if result is not None:
            self.disk_cache.set(cache_key, result)
        return result

    def _generate(self, prompt: str, max_tokens: int = 1024) -> Optional[str]:
        if not self.client:
            return None
        model_chain = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        last_error = None
        for model in model_chain:
            try:
                message = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                if message.content and len(message.content) > 0:
                    return message.content[0].text
                return None
            except Exception as e:
                last_error = e
                if not self._is_model_not_found(e):
                    raise
        if last_error:
            raise last_error
        return None

    def _is_model_not_found(self, error: Exception) -> bool:
        error_str = str(error).lower()
        return "not_found" in error_str or "model_not_found" in error_str or "404" in error_str

    def extract_variables(self, formula_text: str, context: str, candidate_symbols: List[str] = None) -> List[VariableDefinition]:
        """Extract variable meanings using Claude (meaning inference only, no JSON structure)."""
        if not self.client:
            return []
        
        if not candidate_symbols:
            return []
        
        # LLM infers meanings only (free text, no JSON)
        prompt = f"""Given this formula and context, explain what each variable means.
Treat subscripted variables as single symbols (e.g. R_M, r_f are one variable each).

Formula: {formula_text}
Context: {context}

Variables in the formula: {', '.join(candidate_symbols[:15])}

For each variable, explain its meaning in the context of this formula. 
You can write in natural language, one variable per line, or in any clear format.
Focus on what each variable represents in financial/economic terms.

Example format (but you can use any clear format):
- r: risk-free rate
- E(r_M): expected return on market portfolio
- σ: standard deviation of returns"""
        
        for attempt in range(3):
            try:
                raw = self._generate(prompt)
                if not raw:
                    continue
                # Parse free text response
                result = self._parse_meanings_response(raw, candidate_symbols, formula_text)
                return result
            except Exception as e:
                if "429" in str(e) or "overloaded" in str(e).lower():
                    time.sleep(2 * (attempt + 1))
                    continue
                print(f"LLM Error (Anthropic extract_variables): {e}")
                return []
        return []
    
    def _parse_meanings_response(self, response_text: str, candidate_symbols: List[str], formula_text: str) -> List[VariableDefinition]:
        """Parse LLM's free-text response about variable meanings."""
        variables = []
        
        for symbol in candidate_symbols:
            meaning = None
            units = None
            
            # Pattern 1: "- symbol: meaning" or "symbol: meaning"
            patterns = [
                rf'(?:^|\n|[-•])\s*{re.escape(symbol)}\s*[:=]\s*([^\n]+)',
                rf'{re.escape(symbol)}\s+is\s+([^\n,.;]+)',
                rf'{re.escape(symbol)}\s+represents\s+([^\n,.;]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    meaning = match.group(1).strip()
                    meaning = re.sub(r'[.,;]+$', '', meaning).strip()
                    break
            
            # Pattern 2: Look for symbol in context
            if not meaning:
                lines = response_text.split('\n')
                for line in lines:
                    if symbol.lower() in line.lower() and len(line) > len(symbol) + 5:
                        parts = re.split(rf'{re.escape(symbol)}\s*[:=]?\s*', line, flags=re.IGNORECASE)
                        if len(parts) > 1:
                            meaning = parts[1].strip()
                            meaning = re.sub(r'[.,;]+$', '', meaning).strip()
                            break
            
            # Extract units if mentioned
            if meaning:
                units_match = re.search(r'\(in\s+([^)]+)\)|\(units?:\s*([^)]+)\)', meaning, re.IGNORECASE)
                if units_match:
                    units = units_match.group(1) or units_match.group(2)
                    meaning = re.sub(r'\(in\s+[^)]+\)|\(units?:\s*[^)]+\)', '', meaning, flags=re.IGNORECASE).strip()
            
            if meaning and len(meaning) > 2:
                variables.append(VariableDefinition(
                    symbol=symbol,
                    meaning=meaning,
                    units=units,
                    inferred=True,
                    source="llm",
                ))
        
        return variables

    def structure_worked_example(self, text: str) -> Dict[str, Any]:
        if not self.client:
            return {"problem_statement": text}
        prompt = f"""Analyze this Worked Example text. Return a JSON object with: "title", "problem_statement", "steps" (array of strings), "final_answer". Return ONLY valid JSON.

---
{text}
---"""
        for attempt in range(3):
            try:
                raw = self._generate(prompt)
                if raw:
                    data = parse_json_response(raw)
                    return data if isinstance(data, dict) else {"problem_statement": text}
            except Exception as e:
                if "429" in str(e) or "overloaded" in str(e).lower():
                    time.sleep(2 * (attempt + 1))
                    continue
                print(f"LLM Error (Anthropic structure_worked_example): {e}")
        return {"problem_statement": text}

    def extract_formula_summary(self, formula_text: str, variables: List[VariableDefinition]) -> Optional[str]:
        if not self.client:
            return f"Formula computing {', '.join([v.symbol for v in variables[:3]])}" if variables else None
        var_info = "\nVariables: " + ", ".join([f"{v.symbol} ({v.meaning})" for v in variables[:5]]) if variables else "\nVariables: Not yet identified"
        prompt = f"""What does this formula calculate? One or two short sentences. No JSON, no markdown.

Formula: {formula_text}
{var_info}"""
        for attempt in range(3):
            try:
                summary = self._generate(prompt, max_tokens=100)
                if summary:
                    summary = summary.replace("**", "").replace("*", "").strip()[:200]
                    return summary or (f"Formula computing {', '.join([v.symbol for v in variables[:3]])}" if variables else None)
            except Exception as e:
                if "429" in str(e) or "overloaded" in str(e).lower():
                    time.sleep(2 * (attempt + 1))
                    continue
                print(f"LLM Error (Anthropic extract_formula_summary): {e}")
        return f"Formula computing {', '.join([v.symbol for v in variables[:3]])}" if variables else None

    def _serialize_batch_for_cache(
        self, result: List[Tuple[List[VariableDefinition], Optional[str]]]
    ) -> List[Tuple[List[Dict[str, Any]], Optional[str]]]:
        """Make batch result JSON-serializable for disk cache."""
        return [([(v.model_dump() if hasattr(v, 'model_dump') else v.dict()) for v in vars_list], summary) for (vars_list, summary) in result]

    def _deserialize_batch_cached(
        self, cached: List[Tuple[List[Dict[str, Any]], Optional[str]]]
    ) -> List[Tuple[List[VariableDefinition], Optional[str]]]:
        """Restore batch result from disk cache."""
        return [(build_variable_definitions(list(vars_list)), summary) for (vars_list, summary) in cached]

    def extract_variables_batch(
        self,
        items: List[Tuple[str, str, Optional[List[str]]]],
    ) -> List[Tuple[List[VariableDefinition], Optional[str]]]:
        """Batch variable extraction + one-line summary per formula. Returns [(variables, summary), ...]. Uses disk cache when enable_cache=True."""
        if not self.client or items is None or len(items) == 0:
            return [([], None) for _ in (items or [])]
        parts = []
        for idx, (formula_text, context, candidate_symbols) in enumerate(items, 1):
            hint = ""
            if candidate_symbols:
                hint = f" Candidate symbols: {', '.join(candidate_symbols[:10])}."
            parts.append(f"[{idx}]\nFormula: {formula_text}\nContext: {context}{hint}")
        prompt = """Extract variables for each of the following formulas. For each formula, use its Context and Candidate symbols to infer variable meanings. Also provide a one-line "summary": what does this formula calculate.

Important: Treat subscripted variables as single symbols. E.g. R_M, r_f, beta_GE, sigma_i are one variable each—use the symbol exactly as in the formula (with subscript), not as separate R and M, r and f, etc.

""" + "\n\n---\n\n".join(parts) + """

Return a JSON object with a single key "results" whose value is an array of objects. Each object has "variables" (array of {"symbol", "meaning", "units"?}) and "summary" (one-line string, or null). Same order as [1], [2], ... above.
Example: {"results": [{"variables": [{"symbol":"r_f","meaning":"risk-free rate","units":null}], "summary": "Present value of a bond"}, {"variables": [], "summary": null}]}
Return ONLY valid JSON."""
        cache_key = self._get_cache_key(prompt, "extract_variables_batch")
        if self.enable_cache and self.disk_cache:
            cached = self.disk_cache.get(cache_key)
            if cached is not None:
                return self._deserialize_batch_cached(cached)
        for attempt in range(3):
            try:
                if attempt > 0:
                    print(f"    Retry {attempt + 1}/3...")
                raw = self._generate(prompt, max_tokens=2048)
                if not raw:
                    if attempt == 0:
                        print(f"    Warning: Claude returned empty response")
                    continue
                # Try to parse JSON (may have markdown code blocks)
                json_text = raw.strip()
                if "```" in json_text:
                    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", json_text, re.DOTALL)
                    if match:
                        json_text = match.group(1)
                start = json_text.find("{")
                if start >= 0:
                    json_text = json_text[start: json_text.rfind("}") + 1] if "}" in json_text[start:] else json_text[start:]
                data = parse_json_response(json_text)
                if isinstance(data, dict):
                    results = data.get("results") or []
                elif isinstance(data, list):
                    results = data
                else:
                    results = []
                if not isinstance(results, list):
                    results = []
                out: List[Tuple[List[VariableDefinition], Optional[str]]] = []
                for i, r in enumerate(results):
                    if i >= len(items):
                        break
                    vars_list = r.get("variables", []) if isinstance(r, dict) else []
                    summary = r.get("summary") if isinstance(r, dict) else None
                    if summary and not isinstance(summary, str):
                        summary = str(summary)[:200] if summary else None
                    out.append((build_variable_definitions(vars_list), summary))
                while len(out) < len(items):
                    out.append(([], None))
                result = out[:len(items)]
                if self.enable_cache and self.disk_cache and result:
                    try:
                        self.disk_cache.set(cache_key, self._serialize_batch_for_cache(result))
                    except Exception as e:
                        print(f"    Cache write (batch) skipped: {e}")
                return result
            except json.JSONDecodeError as e:
                if attempt == 0:
                    print(f"    Warning: JSON parse failed. Response preview: {raw[:200] if raw else 'None'}...")
                if attempt < 2:
                    continue
                return [([], None) for _ in items]
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "overloaded" in error_str.lower():
                    wait = 2 * (attempt + 1)
                    print(f"    Rate limit, waiting {wait}s...")
                    if attempt == 0:
                        print(f"    Full error: {error_str[:300]}")
                    time.sleep(wait)
                    continue
                if attempt == 0:
                    print(f"    Error: {type(e).__name__}: {error_str[:300]}")
                if attempt < 2:
                    time.sleep(1)
                    continue
                return [([], None) for _ in items]
        return [([], None) for _ in items]


class ClaudeLLMService:
    """Claude API service for solution generation and validation."""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if not self.api_key:
            print("Warning: CLAUDE_API_KEY or ANTHROPIC_API_KEY not found in environment variables or .env file.")
            print("Please set CLAUDE_API_KEY or ANTHROPIC_API_KEY in .env file or as an environment variable.")
            print("Claude features will fail.")
            self.client = None
        else:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("Warning: anthropic package not installed. Install with: pip install anthropic")
                self.client = None
    
    def generate_solution(self, question_text: str, formulas: List[str] = None, 
                         context: str = "") -> Dict[str, Any]:
        """Generate a solution candidate using Claude."""
        if not self.client:
            return {"solution_steps": [], "final_answer": None, "error": "Claude client not available"}
        
        formulas_hint = ""
        if formulas:
            formulas_hint = f"\n\nRelevant formulas:\n" + "\n".join([f"- {f}" for f in formulas[:5]])
        
        prompt = f"""Solve this question step by step.

Question: {question_text}
{formulas_hint}
Context: {context}

Provide a detailed solution with:
1. Clear step-by-step reasoning
2. Use of relevant formulas where applicable
3. Final answer

Return JSON with:
- "solution_steps": array of step strings
- "final_answer": the final answer
- "reasoning": brief explanation of approach"""
        
        try:
            # Try latest model first, fallback to older if not available
            model_name = "claude-3-5-sonnet-20240229"
            try:
                message = self.client.messages.create(
                    model=model_name,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
            except Exception as model_error:
                # Fallback to simpler model name if date version fails
                error_str = str(model_error)
                if "404" in error_str or "not_found" in error_str.lower():
                    # Try simpler model name once, then give up
                    try:
                        model_name = "claude-3-5-sonnet"
                        message = self.client.messages.create(
                            model=model_name,
                            max_tokens=2000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                    except:
                        # If both fail, return empty result silently
                        return {"solution_steps": [], "final_answer": None, "error": "Claude model not available"}
                else:
                    raise
            
            content = message.content[0].text
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data
            else:
                # Fallback: parse as text
                return {
                    "solution_steps": [content],
                    "final_answer": None,
                    "reasoning": content[:200]
                }
        except Exception as e:
            error_str = str(e)
            # Don't spam errors if model not found - just return empty
            if "404" in error_str or "not_found" in error_str.lower():
                print(f"Claude Error: Model not available. Skipping solution generation.")
                return {"solution_steps": [], "final_answer": None, "error": "Claude model not available"}
            print(f"Claude Error (generate_solution): {e}")
            return {"solution_steps": [], "final_answer": None, "error": str(e)}
    
    def validate_solution(self, question_text: str, solution_text: str, 
                         formulas: List[str] = None) -> Dict[str, Any]:
        """Validate a solution using Claude."""
        if not self.client:
            return {"score": 0.0, "issues": ["Claude client not available"]}
        
        formulas_hint = ""
        if formulas:
            formulas_hint = f"\nRelevant formulas: {', '.join(formulas[:3])}"
        
        prompt = f"""Evaluate this solution for correctness and completeness.

Question: {question_text}
{formulas_hint}

Solution:
{solution_text}

Rate the solution on:
1. Correctness (0-1): Is the answer correct?
2. Completeness (0-1): Are all steps shown?
3. Formula usage (0-1): Are correct formulas used?
4. Clarity (0-1): Is the explanation clear?

Return JSON with:
- "overall_score": 0.0-1.0
- "correctness": 0.0-1.0
- "completeness": 0.0-1.0
- "formula_usage": 0.0-1.0
- "clarity": 0.0-1.0
- "issues": array of specific problems found
- "strengths": array of what's good"""
        
        try:
            # Try latest model first, fallback to older if not available
            model_name = "claude-3-5-sonnet-20240229"
            try:
                message = self.client.messages.create(
                    model=model_name,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
            except Exception as model_error:
                # Fallback to simpler model name if date version fails
                error_str = str(model_error)
                if "404" in error_str or "not_found" in error_str.lower():
                    # Try simpler model name once, then give up
                    try:
                        model_name = "claude-3-5-sonnet"
                        message = self.client.messages.create(
                            model=model_name,
                            max_tokens=1000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                    except:
                        # If both fail, return low score silently
                        return {"score": 0.0, "issues": ["Claude model not available"], "error": "Claude model not available"}
                else:
                    raise
            
            content = message.content[0].text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data
            else:
                return {"score": 0.5, "issues": ["Could not parse validation response"]}
        except Exception as e:
            print(f"Claude Error (validate_solution): {e}")
            return {"score": 0.0, "issues": [str(e)]}


def _extract_json_span(raw: str) -> Optional[str]:
    """Extract the first complete JSON object {...} or array [...] by brace matching."""
    raw = raw.strip()
    i_obj = raw.find("{")
    i_arr = raw.find("[")
    if i_obj < 0 and i_arr < 0:
        return None
    if i_arr < 0 or (i_obj >= 0 and i_obj < i_arr):
        open_c, close_c = "{", "}"
        i = i_obj
    else:
        open_c, close_c = "[", "]"
        i = i_arr
    depth = 0
    in_string = None
    escape = False
    for j in range(i, len(raw)):
        c = raw[j]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if in_string:
            if c == in_string:
                in_string = None
            continue
        if c in ('"', "'"):
            in_string = c
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return raw[i : j + 1]
    return None


def parse_json_response(text: Optional[str]) -> Optional[Any]:
    if not text:
        return None
    raw = text.strip()
    if "```" in raw:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()
    # Try full string first
    try:
        data = json.loads(raw)
        return data if isinstance(data, (dict, list)) else None
    except Exception:
        pass
    # Try brace-matched extraction (handles leading/trailing prose)
    span = _extract_json_span(raw)
    if span:
        try:
            data = json.loads(span)
            return data if isinstance(data, (dict, list)) else None
        except Exception:
            pass
    # Fallback: first { to last } (may be wrong with multiple top-level objects)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            return data if isinstance(data, (dict, list)) else None
        except Exception:
            pass
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            return data if isinstance(data, list) else None
        except Exception:
            pass
    return None


def build_variable_definitions(vars_list: List[Dict[str, Any]]) -> List[VariableDefinition]:
    defs: List[VariableDefinition] = []
    for item in vars_list:
        try:
            if not isinstance(item, dict):
                continue
            if "symbol" in item and "meaning" in item:
                defs.append(VariableDefinition(
                    symbol=str(item["symbol"]),
                    meaning=str(item["meaning"]),
                    units=item.get("units"),
                    inferred=True,
                    source="llm",
                ))
        except Exception:
            continue
    return defs


def _generate_json_anthropic(llm_service: "AnthropicLLMService", prompt: str, temperature: float = 0.3) -> Optional[str]:
    """Call Anthropic with a strict JSON-only system instruction to improve parse success."""
    if not llm_service.client:
        return None
    model_chain = [llm_service.model_name] + [m for m in getattr(llm_service, "fallback_models", []) if m != llm_service.model_name]
    system = "You must respond with only valid JSON. No markdown code blocks, no explanation, no preamble. Output nothing before or after the JSON."
    for model in model_chain:
        try:
            message = llm_service.client.messages.create(
                model=model,
                max_tokens=4096,  # haiku/sonnet cap; opus allows more but 4096 is safe for all
                system=system,
                messages=[{"role": "user", "content": prompt}],
                # No stop_sequences: they can cut off mid-JSON (e.g. "```" in solution text)
            )
            if message.content and len(message.content) > 0:
                return message.content[0].text
        except Exception as e:
            if "not_found" in str(e).lower() or "404" in str(e):
                continue
            raise
    return None


def generate_json_with_llm(llm_service: LLMService, prompt: str, temperature: float = 0.3) -> Optional[Dict[str, Any]]:
    if isinstance(llm_service, OpenAILLMService):
        if not llm_service.client:
            return None
        try:
            response = llm_service.client.chat.completions.create(
                model=llm_service.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature
            )
            return parse_json_response(response.choices[0].message.content)
        except Exception:
            return None
    if isinstance(llm_service, GeminiLLMService):
        text = llm_service._generate(prompt, json_mode=True)
        return parse_json_response(text)
    if isinstance(llm_service, AnthropicLLMService):
        text = _generate_json_anthropic(llm_service, prompt, temperature)
        if not text:
            return None
        data = parse_json_response(text)
        if data is None:
            # Log a short preview to help debug Claude response format
            preview = (text.strip() or "(empty)")[:450]
            print(f"Claude JSON parse failed. Response preview: {preview}...")
            return None
        # Claude sometimes returns a top-level array; wrap as {"solutions": [...]}
        if isinstance(data, list):
            return {"solutions": data}
        return data
    return None
