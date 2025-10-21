# Schema.org Class & Property Extraction

Explore the Schema.org vocabulary against your own RDF dataset. This tool:
- Parses the Schema.org JSON-LD to capture class/property hierarchies and ranges.
- Queries your local triple store to find which classes have instances and which properties appear for each class.
- Saves intermediate results so long runs can be resumed.
- Writes both full-URI and prefix-shortened outputs.

> **Important:** Make sure your SPARQL endpoint (triple store) is running on your **localhost** (or set `--endpoint` to match where it runs). The default in the examples below is `http://localhost:7878/query`.

---

## Files

- `yago_schema_with_value_preprocess_refactored.py` — main script (CLI) with modular functions, logging, and resumable saves.
- `partial_results.json` — incremental results (class counts and properties by class, full URIs).
- `partial_results_prefixed.json` — same data with CURIE-style prefixes (e.g., `schema:Person`).

---

## Quick Start

1. **Have a Schema.org JSON-LD file** (e.g., `schemaorg-current-https.jsonld`).
2. **Run your triple store locally** and load your dataset.
3. **Run the script** (adjust paths and flags as needed):

```bash
python notebooks/yago_schema_with_value_preprocessing.py \
  --jsonld schemaorg-current-https.jsonld \
  --endpoint http://localhost:7878/query \
  --out partial_results.json \
  --out-prefixed partial_results_prefixed.json
