#!/usr/bin/env python3
"""
Schema.org Preprocessor for YAGO/Local Triple Stores (Refactored)
================================================================

This script performs the first stage of the value extraction pipeline. It
parses a Schema.org JSON‑LD file to obtain class and property hierarchies,
then queries a SPARQL endpoint to discover which classes actually have
instances in your triple store and what properties are present for each
class. The results are saved in two forms: using full URIs and using
CURIE‑style prefixes for more readable keys.

The script can resume from previous runs, saving intermediate results
incrementally. It is designed to work with any SPARQL endpoint, but a
friendly reminder is printed if you specify a localhost endpoint.

Example Usage
-------------
```
python yago_schema_with_value_preprocess.py \
  --jsonld RFDs/schemaorg-current-https.jsonld \
  --endpoint http://localhost:7878/query \
  --out partial_results.json \
  --out-prefixed partial_results_prefixed.json
```

"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Any, Optional, Tuple

try:
    from SPARQLWrapper import SPARQLWrapper, JSON  # type: ignore
except Exception:
    SPARQLWrapper = None  # type: ignore
    JSON = None  # type: ignore

from pathlib import Path


def parse_jsonld_hierarchy(jsonld_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse a Schema.org JSON‑LD file into class and property metadata."""
    def extract_hierarchy(items, type_filter: str, subclass_key: str, dest: Dict[str, Any]):
        for item in items:
            if "@type" in item and type_filter in item["@type"]:
                iri = item.get("@id")
                parents = item.get(subclass_key)
                if isinstance(parents, dict):
                    dest[iri] = parents.get("@id")
                elif isinstance(parents, list):
                    dest[iri] = [p.get("@id") for p in parents if isinstance(p, dict) and "@id" in p]
                else:
                    dest[iri] = None

    def extract_rel(items, field: str) -> Dict[str, List[str]]:
        rel: Dict[str, List[str]] = {}
        for item in items:
            if "@type" in item and "Property" in item["@type"]:
                prop = item.get("@id")
                domains_or_ranges = item.get(field)
                if isinstance(domains_or_ranges, dict) and "@id" in domains_or_ranges:
                    rel[prop] = [domains_or_ranges["@id"]]
                elif isinstance(domains_or_ranges, list):
                    rel[prop] = [d["@id"] for d in domains_or_ranges if isinstance(d, dict) and "@id" in d]
        return rel

    with open(jsonld_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    graph = data.get("@graph", [])
    classes: Dict[str, Any] = {}
    properties: Dict[str, Any] = {}
    extract_hierarchy(graph, "Class", "rdfs:subClassOf", classes)
    extract_hierarchy(graph, "Property", "rdfs:subPropertyOf", properties)
    property_class_relations = extract_rel(graph, "schema:domainIncludes")
    property_ranges = extract_rel(graph, "schema:rangeIncludes")
    return {
        "classes": classes,
        "properties": properties,
        "property_class_relations": property_class_relations,
        "property_ranges": property_ranges,
    }


def count_instances_for_class(cls_uri: str, endpoint: str) -> int:
    """Return the number of instances of a given class in the triple store."""
    if SPARQLWrapper is None:
        raise RuntimeError("SPARQLWrapper library is required but not installed")
    sparql = SPARQLWrapper(endpoint)
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT (COUNT(?s) AS ?c)
    WHERE {{
      ?s a <{cls_uri}> .
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return int(results["results"]["bindings"][0]["c"]["value"]) if results["results"]["bindings"] else 0
    except Exception:
        return 0


def distinct_properties_for_class(cls_uri: str, endpoint: str) -> List[str]:
    """Return a list of distinct properties used by instances of a given class."""
    if SPARQLWrapper is None:
        raise RuntimeError("SPARQLWrapper library is required but not installed")
    sparql = SPARQLWrapper(endpoint)
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT DISTINCT ?p
    WHERE {{
      ?s a <{cls_uri}> ; ?p ?o .
    }}
    ORDER BY ?p
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return [row["p"]["value"] for row in results.get("results", {}).get("bindings", [])]
    except Exception:
        return []


def to_prefixed_uri(uri: str, prefix_map: Dict[str, str]) -> str:
    """Convert a full URI to its prefixed form if a matching prefix is found."""
    for base_uri, prefix in prefix_map.items():
        if uri.startswith(base_uri):
            return prefix + uri[len(base_uri):]
    return uri


def build_prefix_map() -> Dict[str, str]:
    """Return a default mapping from full namespace URIs to CURIE prefixes."""
    return {
        "http://schema.org/": "schema:",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
        "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
        "http://www.w3.org/2002/07/owl#": "owl:",
        "http://purl.org/science/owl/sciencecommons/": "sc:",
        "http://yago-knowledge.org/resource/": "yago:",
    }


def save_json(path: str, data: Any) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess(
    jsonld_path: str,
    endpoint: str,
    out_path: str,
    out_prefixed_path: str,
) -> None:
    """Main driver for preprocessing Schema.org classes and properties."""
    # Friendly reminder for localhost endpoints
    if "localhost" in endpoint or "127.0.0.1" in endpoint:
        print("[Reminder] Using a localhost endpoint; make sure your triple store is running and loaded.")
    # Parse JSON‑LD
    hierarchy = parse_jsonld_hierarchy(jsonld_path)
    classes = hierarchy["classes"]
    # Attempt to resume from existing file
    valid_classes: Dict[str, int] = {}
    class_properties: Dict[str, List[str]] = {}
    if os.path.exists(out_path):
        try:
            prev = load_json(out_path)
            valid_classes = prev.get("valid_classes", {})
            class_properties = prev.get("class_properties", {})
        except Exception:
            pass
    # Iterate over all classes
    for cls_uri in classes.keys():
        if cls_uri in class_properties:
            continue
        # Skip extremely generic class if present
        if cls_uri.endswith("Thing"):
            continue
        count = count_instances_for_class(cls_uri, endpoint)
        if count > 0:
            valid_classes[cls_uri] = count
            props = distinct_properties_for_class(cls_uri, endpoint)
            class_properties[cls_uri] = props
        # Save after each class to allow resuming on interruption
        save_json(out_path, {"valid_classes": valid_classes, "class_properties": class_properties})
        print(f"Processed {cls_uri}: count={count}, properties={len(class_properties.get(cls_uri, []))}")
    # Build prefixed version
    prefix_map = build_prefix_map()
    prefixed_properties: Dict[str, List[str]] = {}
    for cls_uri, props in class_properties.items():
        prefixed_class = to_prefixed_uri(cls_uri, prefix_map)
        prefixed_properties[prefixed_class] = [to_prefixed_uri(p, prefix_map) for p in props]
    prefixed_data = {"valid_classes": valid_classes, "class_properties": prefixed_properties}
    save_json(out_prefixed_path, prefixed_data)
    print(f"[OK] Saved raw results -> {out_path}")
    print(f"[OK] Saved prefixed results -> {out_prefixed_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Schema.org classes and properties using a SPARQL endpoint")
    parser.add_argument("--jsonld", required=True, help="Path to the Schema.org JSON‑LD file")
    parser.add_argument("--endpoint", required=True, help="SPARQL endpoint URL (e.g., http://localhost:7878/query)")
    parser.add_argument("--out", required=True, help="Output JSON path for full URIs (resumable)")
    parser.add_argument("--out-prefixed", required=True, help="Output JSON path for prefixed URIs")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    preprocess(
        jsonld_path=args.jsonld,
        endpoint=args.endpoint,
        out_path=args.out,
        out_prefixed_path=args.out_prefixed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
