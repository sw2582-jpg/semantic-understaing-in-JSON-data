#!/usr/bin/env python3
"""
Schema Maker (Refactored)
=========================

This script generates random JSON Schemas centered on Schema.org classes that
are known to have instances in your RDF dataset. It can optionally fill
generated schemas with instance URIs from a SPARQL endpoint. The aim is
to provide a reproducible and configurable way to create a set of example
schemas based on the Schema.org vocabulary and the data actually present
in your triple store.

Key Features
------------
* Parses a Schema.org JSON‑LD file to build class and property hierarchies,
  as well as domain and range relationships for properties.
* Accepts a mapping of Schema.org classes to the number of instances
  present in your dataset (e.g., produced by the preprocessing script).
* Randomly selects classes with instances and properties with appropriate
  ranges to build each schema. You can control the number of properties and
  the maximum depth of nested objects.
* Optionally fills the generated schemas with real instance URIs drawn
  from a SPARQL endpoint, allowing you to see what actual data might look like.
* All paths and parameters are provided via command‑line arguments.

Example Usage
-------------
```
# Generate 10 random schemas without filling values
python schema_making.py \
  --jsonld RFDs/schemaorg-current-https.jsonld \
  --schemas-with-instances schemas_with_instances.json \
  --num-schemas 10 \
  --out generated_schemas.json

# Generate 5 schemas and fill them with URIs using a local SPARQL endpoint
python schema_making.py \
  --jsonld RFDs/schemaorg-current-https.jsonld \
  --schemas-with-instances schemas_with_instances.json \
  --num-schemas 5 \
  --out generated_schemas.json \
  --fill-out filled_schemas.json \
  --endpoint http://localhost:7878/query
```

"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

try:
    import requests  # type: ignore
except Exception:
    requests = None


def parse_jsonld_hierarchy(jsonld_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse a Schema.org JSON‑LD file to extract class and property hierarchies.

    The returned dictionary contains four keys:
      - ``classes``: mapping of class IRI to parent class or list of parents.
      - ``properties``: mapping of property IRI to parent property or list of parents.
      - ``property_class_relations``: mapping of property IRI to domain classes.
      - ``property_ranges``: mapping of property IRI to range classes.
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


DEFAULT_STOP_CLASSES = {"Thing", "Place", "QuantitativeValue", "QualitativeValue", "Date"}
STATIC_OVERRIDES = {"name": "string", "description": "string", "identifier": "string"}
NO_EXPAND_PROPS = {"description"}


@dataclass
class SchemaGenConfig:
    """Configuration settings for the JSON schema generator."""
    hierarchy: Dict[str, Dict[str, Any]]
    schemas_with_instances: Dict[str, int]
    stop_classes: set | None = None
    max_depth: int = 2
    seed: Optional[int] = None

    def __post_init__(self):
        if self.stop_classes is None:
            self.stop_classes = set(DEFAULT_STOP_CLASSES)
        if self.seed is not None:
            random.seed(self.seed)


class JSONSchemaGenerator:
    """Generate random JSON Schemas centered on classes with instances."""
    def __init__(self, cfg: SchemaGenConfig):
        self.cfg = cfg

    def _select_central_class(self) -> str:
        classes = list(self.cfg.schemas_with_instances.keys())
        if not classes:
            raise ValueError("No classes with instances are available for schema generation.")
        return random.choice(classes)

    def _relevant_properties(self, central_class: str, min_properties: int) -> List[str]:
        """Collect a list of candidate properties for a given central class."""
        props: List[str] = []
        visited: set = set()
        current = central_class if central_class.startswith("schema:") else f"schema:{central_class}"
        while current and len(props) < min_properties:
            if current in visited or current.split(":")[-1] in self.cfg.stop_classes:
                break
            visited.add(current)
            current_props = [
                p for p, domains in self.cfg.hierarchy["property_class_relations"].items() if current in domains
            ]
            valid: List[str] = []
            for p in current_props:
                ranges = self.cfg.hierarchy["property_ranges"].get(p, [])
                if any(rt in self.cfg.schemas_with_instances for rt in ranges):
                    valid.append(p)
            random.shuffle(valid)
            props.extend(valid)
            # deduplicate while preserving order
            props = list(dict.fromkeys(props))
            parents = self.cfg.hierarchy["classes"].get(current)
            if isinstance(parents, list) and parents:
                current = random.choice(parents)
            else:
                current = parents
        if len(props) < min_properties:
            fillers = ["name", "description", "identifier"]
            random.shuffle(fillers)
            for f in fillers:
                if f not in props:
                    props.append(f)
                if len(props) >= min_properties:
                    break
        random.shuffle(props)
        return props[:min_properties]

    def _prop_definition(self, prop: str, current_depth: int) -> Dict[str, Any]:
        if prop in STATIC_OVERRIDES:
            return {"type": STATIC_OVERRIDES[prop]}
        ranges = self.cfg.hierarchy["property_ranges"].get(prop, [])
        chosen_type = random.choice(ranges).split(":")[-1] if ranges else "string"
        if prop in NO_EXPAND_PROPS or chosen_type in self.cfg.stop_classes:
            return {"type": chosen_type}
        # Expand nested objects
        if chosen_type not in {"string", "Number", "boolean", "array", "URL", "Text"} and current_depth < self.cfg.max_depth:
            nested = self.generate_schema(num_properties=max(2, random.randint(2, 4)), central_class=chosen_type, current_depth=current_depth + 1)
            # Add a name placeholder for nested objects
            nested["properties"]["name"] = {"type": "name_of_object"}
            return {"type": "object", "properties": nested["properties"]}
        return {"type": chosen_type}

    def generate_schema(self, num_properties: int, central_class: Optional[str] = None, current_depth: int = 0) -> Dict[str, Any]:
        if not central_class:
            central_class = self._select_central_class()
        props = self._relevant_properties(central_class, num_properties)
        schema: Dict[str, Any] = {
            "$schema": "http://json-schema.org/schema#",
            "title": central_class if central_class.startswith("schema:") else f"schema:{central_class}",
            "type": "object",
            "properties": {},
            "required": [],
        }
        for p in props:
            key = p.split(":")[-1]
            schema["properties"][key] = self._prop_definition(p, current_depth)
            if random.choice([True, False]):
                schema["required"].append(key)
        schema["required"] = sorted(set(schema["required"]))
        return schema


# ----------------------------
# SPARQL helper for filling
# ----------------------------

DEFAULT_ENDPOINT = "https://yago-knowledge.org/sparql/query"


def _require_requests():
    if requests is None:
        raise RuntimeError("The 'requests' library is required for SPARQL operations. Install it or omit --fill-out.")


def query_instances_for_type(schema_type: str, endpoint: str, limit: int = 100) -> List[str]:
    """Query the SPARQL endpoint for distinct instance URIs of a given schema type."""
    _require_requests()
    sparql = f"""
    PREFIX schema: <http://schema.org/>
    SELECT DISTINCT ?instance
    WHERE {{
      ?instance a schema:{schema_type} .
    }}
    LIMIT {limit}
    """
    headers = {
        "Content-Type": "application/sparql-query",
        "Accept": "application/sparql-results+json",
    }
    try:
        resp = requests.post(endpoint, data=sparql, headers=headers, timeout=60)
        if not resp.ok:
            return []
        data = resp.json()
        return [b["instance"]["value"] for b in data.get("results", {}).get("bindings", []) if "instance" in b]
    except Exception:
        return []


def fill_schema_values(schema_def: Dict[str, Any], endpoint: str) -> Any:
    """Recursively fill a generated schema with instance URIs from a SPARQL endpoint."""
    t = schema_def.get("type")
    if t == "object":
        return {k: fill_schema_values(v, endpoint) for k, v in schema_def.get("properties", {}).items()}
    if t:
        candidates = query_instances_for_type(t, endpoint=endpoint, limit=100)
        return random.choice(candidates) if candidates else f"NoInstanceFoundFor_{t}"
    return "NoTypeSpecified"


# ----------------------------
# CLI Implementation
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate random JSON Schemas from Schema.org classes")
    p.add_argument("--jsonld", required=True, help="Path to the Schema.org JSON‑LD file")
    p.add_argument("--schemas-with-instances", required=True, help="JSON mapping {schema:Class -> count} from preprocessing")
    p.add_argument("--num-schemas", type=int, default=10, help="Number of schemas to generate")
    p.add_argument("--out", required=True, help="Output path for generated schemas (JSON list)")
    p.add_argument("--max-depth", type=int, default=2, help="Maximum nested object depth")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    # Optional filling
    p.add_argument("--fill-out", help="If specified, write a filled version of schemas here")
    p.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"SPARQL endpoint for filling. Default: {DEFAULT_ENDPOINT}. If you use localhost (e.g., http://localhost:7878/query), ensure your triple store is running.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # Friendly reminder for localhost
    if "localhost" in args.endpoint or "127.0.0.1" in args.endpoint:
        print("[Reminder] You're using a localhost endpoint. Ensure your triple store is running and loaded.")
    # 1) Parse hierarchy
    hierarchy = parse_jsonld_hierarchy(args.jsonld)
    # 2) Load class instance counts
    with open(args.schemas_with_instances, "r", encoding="utf-8") as f:
        schemas_with_instances = json.load(f)
    # 3) Generate schemas
    cfg = SchemaGenConfig(
        hierarchy=hierarchy,
        schemas_with_instances=schemas_with_instances,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    gen = JSONSchemaGenerator(cfg)
    schemas: List[Dict[str, Any]] = []
    for _ in range(args.num_schemas):
        num_props = random.randint(2, 5)
        schemas.append(gen.generate_schema(num_properties=num_props))
    # Write schemas
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2)
    print(f"[OK] Wrote {len(schemas)} schemas -> {args.out}")
    # 4) Optionally fill
    if args.fill_out:
        if requests is None:
            print("[WARN] requests library not available; cannot fill schemas. Skipping.")
        else:
            filled = [fill_schema_values(s, args.endpoint) for s in schemas]
            with open(args.fill_out, "w", encoding="utf-8") as f:
                json.dump(filled, f, indent=2)
            print(f"[OK] Wrote filled schemas -> {args.fill_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
