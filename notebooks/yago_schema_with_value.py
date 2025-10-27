#!/usr/bin/env python3
"""
Value Harvesting and Instance Generation for Schema.org/YAGO (Refactored)
=======================================================================

This script coordinates the later stages of the data generation pipeline. It
provides subcommands to harvest example values for each class/property pair
from a SPARQL endpoint, to split those values into train/test subsets, and
to generate random JSON instances using the harvested values. The aim is to
produce data suitable for training and evaluating models that operate on
Schema.org‑like JSON structures.

Key Subcommands
---------------
```
harvest  : For each class and property, collect example values from the SPARQL endpoint.
split    : Split the harvested values into train/test sets by class/property.
generate : Generate random JSON documents for each class using the (train) values.
```

Example Workflow
----------------
1. **Harvest values**
   ````
   python yago_schema_with_value.py harvest \
     --partial-results partial_results_prefixed.json \
     --jsonld RFDs/schemaorg-current-https.jsonld \
     --endpoint http://localhost:7878/query \
     --out class_property_values.json \
     --limit-per-prop 200
   ````

2. **Split values into train and test sets**
   ````
   python yago_schema_with_value.py split \
     --values class_property_values.json \
     --train-out class_property_values_train.json \
     --test-out class_property_values_test.json \
     --train-ratio 0.8
   ````

3. **Generate instances from schemas**
   ````
   python yago_schema_with_value.py generate \
     --partial-results partial_results_prefixed.json \
     --values class_property_values_train.json \
     --jsonld RFDs/schemaorg-current-https.jsonld \
     --num-instances 1000 \
     --probability 0.35 \
     --expand-depth 1 \
     --train-out generated_instances_train.json \
     --test-out generated_instances_test.json
   ````

All three subcommands accept a `--endpoint` option, and they will remind
you if you specify a localhost endpoint to ensure your triple store is
running and loaded with data.

"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from SPARQLWrapper import SPARQLWrapper, JSON  # type: ignore
except Exception:
    SPARQLWrapper = None  # type: ignore
    JSON = None  # type: ignore

# -----------------------------------------------------------------------------
# JSON‑LD parsing helper
# -----------------------------------------------------------------------------

def parse_jsonld_hierarchy(jsonld_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse a Schema.org JSON‑LD file to extract class and property metadata.

    Returns a dictionary containing ``classes``, ``properties``,
    ``property_class_relations``, and ``property_ranges``.
    """
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


# -----------------------------------------------------------------------------
# Utility functions for SPARQL
# -----------------------------------------------------------------------------

def get_property_values(
    cls_uri: str,
    prop_uri: str,
    property_ranges: Dict[str, List[str]],
    endpoint: str,
    limit: int,
) -> List[str]:
    """
    Retrieve example values for a given (class, property) pair. If the
    property has a range including QuantitativeValue, we look for a nested
    schema:value; otherwise we retrieve the direct object of the triple.
    """
    if SPARQLWrapper is None:
        raise RuntimeError("SPARQLWrapper library is required but not installed")
    sparql = SPARQLWrapper(endpoint)
    # Determine whether to query for ?value or ?uri
    ranges = property_ranges.get(prop_uri, [])
    if "schema:QuantitativeValue" in ranges:
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX schema: <http://schema.org/>
        SELECT DISTINCT ?value WHERE {{
          ?s a {cls_uri} .
          ?s {prop_uri} ?uri .
          ?uri schema:value ?value .
        }} LIMIT {limit}
        """
        var_name = "value"
    else:
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?uri WHERE {{
          ?s a {cls_uri} .
          ?s {prop_uri} ?uri .
        }} LIMIT {limit}
        """
        var_name = "uri"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return [row[var_name]["value"] for row in results.get("results", {}).get("bindings", [])]
    except Exception:
        return []


def get_schema_org_classes_for_resource(resource_uri: str, endpoint: str) -> List[str]:
    """Return Schema.org classes for a resource URI (only those under schema.org)."""
    if SPARQLWrapper is None:
        raise RuntimeError("SPARQLWrapper library is required but not installed")
    sparql = SPARQLWrapper(endpoint)
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX schema: <http://schema.org/>
    SELECT DISTINCT ?class WHERE {{
      <{resource_uri}> rdf:type ?class .
      FILTER (STRSTARTS(STR(?class), "http://schema.org/"))
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return [row["class"]["value"] for row in results.get("results", {}).get("bindings", [])]
    except Exception:
        return []


def get_rdfs_label(resource_uri: str, endpoint: str) -> Optional[str]:
    """Retrieve the English rdfs:label for a YAGO resource, if available."""
    if SPARQLWrapper is None:
        raise RuntimeError("SPARQLWrapper library is required but not installed")
    sparql = SPARQLWrapper(endpoint)
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?label WHERE {{
      <{resource_uri}> rdfs:label ?label .
      FILTER (lang(?label) = "en")
    }}
    LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["label"]["value"]
        return None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Helper functions for harvesting, splitting, and generating
# -----------------------------------------------------------------------------

def harvest_values(
    partial_results_path: str,
    jsonld_path: str,
    endpoint: str,
    out_path: str,
    limit_per_prop: int = 100,
) -> None:
    """Harvest example values for each class/property pair in the partial results."""
    if "localhost" in endpoint or "127.0.0.1" in endpoint:
        print("[Reminder] Using a localhost endpoint; ensure your triple store is running.")
    # Load class_properties and valid_classes from partial results
    with open(partial_results_path, "r", encoding="utf-8") as f:
        partial = json.load(f)
    class_properties: Dict[str, List[str]] = partial.get("class_properties", {})
    valid_classes: Dict[str, int] = partial.get("valid_classes", {})  # not used directly but loaded for completeness
    # Parse JSON‑LD to get property ranges
    hierarchy = parse_jsonld_hierarchy(jsonld_path)
    property_ranges: Dict[str, List[str]] = hierarchy["property_ranges"]
    # Load or initialize the output
    if os.path.exists(out_path):
        try:
            values_by_class: Dict[str, Dict[str, List[str]]] = json.load(open(out_path, "r", encoding="utf-8"))
        except Exception:
            values_by_class = {}
    else:
        values_by_class = {}
    # Iterate through classes and properties
    for cls_uri in class_properties:
        if cls_uri not in values_by_class:
            values_by_class[cls_uri] = {}
        for prop_uri in class_properties[cls_uri]:
            if prop_uri in values_by_class[cls_uri]:
                continue  # skip already harvested
            vals = get_property_values(
                cls_uri=cls_uri,
                prop_uri=prop_uri,
                property_ranges=property_ranges,
                endpoint=endpoint,
                limit=limit_per_prop,
            )
            values_by_class[cls_uri][prop_uri] = vals
            # Save incrementally to allow resuming
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(values_by_class, f, indent=2)
            print(f"Harvested {len(vals)} values for {cls_uri} -> {prop_uri}")
    print(f"[OK] Wrote harvested values -> {out_path}")


def split_values(
    values_path: str,
    train_out: str,
    test_out: str,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
) -> None:
    """Split values into train and test sets for each class/property list."""
    if seed is not None:
        random.seed(seed)
    with open(values_path, "r", encoding="utf-8") as f:
        values = json.load(f)
    train_split: Dict[str, Dict[str, List[str]]] = {}
    test_split: Dict[str, Dict[str, List[str]]] = {}
    for cls_uri, prop_dict in values.items():
        train_split[cls_uri] = {}
        test_split[cls_uri] = {}
        for prop_uri, val_list in prop_dict.items():
            shuffled = val_list[:]
            random.shuffle(shuffled)
            split_index = int(len(shuffled) * train_ratio)
            train_split[cls_uri][prop_uri] = shuffled[:split_index]
            test_split[cls_uri][prop_uri] = shuffled[split_index:]
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2)
    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_split, f, indent=2)
    print(f"[OK] Wrote train split -> {train_out}")
    print(f"[OK] Wrote test split -> {test_out}")


def generate_instance(
    cls_uri: str,
    class_properties: Dict[str, List[str]],
    values_by_class: Dict[str, Dict[str, List[str]]],
    property_ranges: Dict[str, List[str]],
    probability: float,
    endpoint: str,
    expand_depth: int,
) -> Dict[str, Any]:
    """Generate a random JSON instance for a given class, optionally expanding nested values."""
    inst: Dict[str, Any] = {"@type": cls_uri}
    props = class_properties.get(cls_uri, [])
    for prop_uri in props:
        # Always include rdfs:label (if present) or include with given probability
        include = prop_uri == "rdfs:label" or random.random() <= probability
        if not include:
            continue
        possible_vals = values_by_class.get(cls_uri, {}).get(prop_uri, [])
        if not possible_vals:
            continue
        val = random.choice(possible_vals)
        # If the value is a YAGO resource, decide whether to expand it
        if expand_depth > 0 and isinstance(val, str) and val.startswith("http://yago-knowledge.org/resource/"):
            # Optionally expand into nested object or label
            # Expand if property range permits and random chance
            rngs = property_ranges.get(prop_uri, [])
            # If the range includes schema:Thing only, treat as label
            if rngs and ("schema:Thing" not in rngs or len(rngs) > 1):
                # Choose a candidate type to expand
                cand_types = [rt for rt in rngs if rt != "schema:Thing"] or rngs
                chosen_type = random.choice(cand_types)
                # Recursively generate nested instance
                nested = generate_instance(
                    cls_uri=chosen_type,
                    class_properties=class_properties,
                    values_by_class=values_by_class,
                    property_ranges=property_ranges,
                    probability=probability,
                    endpoint=endpoint,
                    expand_depth=expand_depth - 1,
                )
                inst[prop_uri] = nested
            else:
                # Otherwise try to fetch rdfs:label
                label = get_rdfs_label(val, endpoint)
                inst[prop_uri] = label if label else val.replace("http://yago-knowledge.org/resource/", "")
        else:
            # Use the chosen value directly
            inst[prop_uri] = val
    return inst


def tidy_value(value: Any, endpoint: str) -> Any:
    """Coerce numeric strings and dereference YAGO/Wikidata values when possible."""
    if isinstance(value, str):
        # Convert to int or float if possible
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            pass
        # Replace YAGO resource with label
        if value.startswith("http://yago-knowledge.org/resource/"):
            label = get_rdfs_label(value, endpoint)
            return label if label else value.replace("http://yago-knowledge.org/resource/", "")
        # Otherwise return string as is
        return value
    # For lists or dicts, tidy recursively
    if isinstance(value, dict):
        return {k: tidy_value(v, endpoint) for k, v in value.items()}
    if isinstance(value, list):
        return [tidy_value(v, endpoint) for v in value]
    return value


def generate_instances(
    partial_results_path: str,
    values_path: str,
    jsonld_path: str,
    endpoint: str,
    num_instances: int,
    probability: float,
    expand_depth: int,
    train_out: str,
    test_out: str,
    seed: Optional[int] = None,
) -> None:
    """Generate random JSON documents for each class using harvested values."""
    if seed is not None:
        random.seed(seed)
    # Load class_properties and valid_classes
    partial = json.load(open(partial_results_path, "r", encoding="utf-8"))
    class_properties: Dict[str, List[str]] = partial.get("class_properties", {})
    # Load values
    values = json.load(open(values_path, "r", encoding="utf-8"))
    # Parse JSON‑LD to get property ranges
    hierarchy = parse_jsonld_hierarchy(jsonld_path)
    property_ranges = hierarchy["property_ranges"]
    # Prepare containers
    all_train: List[Dict[str, Any]] = []
    all_test: List[Dict[str, Any]] = []
    classes = list(class_properties.keys())
    print(f"Generating instances for {len(classes)} classes…")
    for cls_uri in classes:
        for _ in range(num_instances):
            inst = generate_instance(
                cls_uri=cls_uri,
                class_properties=class_properties,
                values_by_class=values,
                property_ranges=property_ranges,
                probability=probability,
                endpoint=endpoint,
                expand_depth=expand_depth,
            )
            # Randomly assign to train/test (80/20) for each instance
            if random.random() < 0.8:
                all_train.append(inst)
            else:
                all_test.append(inst)
    # Tidy up values (coerce numbers, labels)
    print("Tidying up generated documents…")
    all_train = [tidy_value(doc, endpoint) for doc in all_train]
    all_test = [tidy_value(doc, endpoint) for doc in all_test]
    # Save
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(all_train, f, indent=2)
    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(all_test, f, indent=2)
    print(f"[OK] Wrote {len(all_train)} train instances -> {train_out}")
    print(f"[OK] Wrote {len(all_test)} test instances -> {test_out}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Harvest values and generate instances from Schema.org/YAGO data")
    sub = p.add_subparsers(dest="cmd", required=True)
    # harvest
    hp = sub.add_parser("harvest", help="Harvest example values from a SPARQL endpoint")
    hp.add_argument("--partial-results", required=True, help="Path to partial_results_prefixed.json from preprocessing")
    hp.add_argument("--jsonld", required=True, help="Path to Schema.org JSON‑LD file (for property ranges)")
    hp.add_argument("--endpoint", required=True, help="SPARQL endpoint URL")
    hp.add_argument("--out", required=True, help="Output JSON file for harvested values")
    hp.add_argument("--limit-per-prop", type=int, default=100, help="Maximum number of values per property")
    hp.set_defaults(func=lambda args: harvest_values(
        partial_results_path=args.partial_results,
        jsonld_path=args.jsonld,
        endpoint=args.endpoint,
        out_path=args.out,
        limit_per_prop=args.limit_per_prop,
    ))
    # split
    sp = sub.add_parser("split", help="Split harvested values into train/test sets")
    sp.add_argument("--values", required=True, help="Input class_property_values.json")
    sp.add_argument("--train-out", required=True, help="Output JSON for train values")
    sp.add_argument("--test-out", required=True, help="Output JSON for test values")
    sp.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of values to allocate to training")
    sp.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    sp.set_defaults(func=lambda args: split_values(
        values_path=args.values,
        train_out=args.train_out,
        test_out=args.test_out,
        train_ratio=args.train_ratio,
        seed=args.seed,
    ))
    # generate
    gp = sub.add_parser("generate", help="Generate random JSON documents using harvested values")
    gp.add_argument("--partial-results", required=True, help="Path to partial_results_prefixed.json from preprocessing")
    gp.add_argument("--values", required=True, help="Path to (train) class_property_values.json")
    gp.add_argument("--jsonld", required=True, help="Path to Schema.org JSON‑LD file (for property ranges)")
    gp.add_argument("--endpoint", required=True, help="SPARQL endpoint URL")
    gp.add_argument("--num-instances", type=int, default=1000, help="Number of instances per class to generate")
    gp.add_argument("--probability", type=float, default=0.5, help="Probability of including each property")
    gp.add_argument("--expand-depth", type=int, default=1, help="Depth of nested object expansion when values are URIs")
    gp.add_argument("--train-out", required=True, help="Output JSON file for generated train instances")
    gp.add_argument("--test-out", required=True, help="Output JSON file for generated test instances")
    gp.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    gp.set_defaults(func=lambda args: generate_instances(
        partial_results_path=args.partial_results,
        values_path=args.values,
        jsonld_path=args.jsonld,
        endpoint=args.endpoint,
        num_instances=args.num_instances,
        probability=args.probability,
        expand_depth=args.expand_depth,
        train_out=args.train_out,
        test_out=args.test_out,
        seed=args.seed,
    ))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
