# Synapta Formula Segmentation (Prototype)

This repository contains a **v1 prototype pipeline** for extracting and structuring
formula-driven learning objects from textbook PDFs for Synapta.

## Scope
- Formula / equation detection
- Variable dictionary extraction
- Derivations / calculation blocks
- Worked examples
- End-of-chapter questions & solutions
- Concept-ID linking (via taxonomy TSV)

## Status
- Focused on validated baseline, not perfect accuracy
- Will be iterated and potentially migrated into Synapta org repo

## Output
Structured JSON segments with:
- page traceability
- heading paths
- segment types
- adjacency + concept links
