#!/usr/bin/env python3
"""
Schema Maker (Refactored)
=========================

Features
--------
1) Parse Schema.org JSON-LD: classes, properties, domainIncludes, rangeIncludes
2) Generate randomized JSON Schemas centered on classes that have instances
3) Optionally fill generated schemas with instance URIs from a SPARQL endpoint
   (default: YAGO public endpoint, but you can point to localhost)

Usage
-----
# 1) Generate schemas (no fill), saving incremental results:
python schema_making.py \
  --jsonld RFDs/schemaorg-current-https.jsonld \
  --schemas-with-instances schemas_with_instances.json \
  --num-schemas 10 \
  --out generated_schemas.json \
  --seed 42

# 2) Generate and then fill with URIs (SPARQL):
python schema_making_refactored.py \
  --jsonld RFDs/schemaorg-current-https.jsonld \
  --schemas-with-instances schemas_with_instances.json \
  --num-schemas 5 \
  --out generated_schemas.json \
  --fill-out filled_schemas.json \
  --endpoint http://localhost:7878/query \
  --seed 7

IMPORTANT
---------
- If you use a localhost endpoint (e.g., http://localhost:7878/query),
  be sure your triple-store is running locally and is loaded with data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Optional import for SPARQL filling; only required if --fill-out is used
try:
    import requests  # noqa: F401
except Exception:
    requests = None


# =========================
# JSON-LD parsing utilities
# =========================
def parse_jsonld_hierarchy(jsonld_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse a Schema.org JSON-LD file to extract:
      - classes: name -> parents (str | list[str] | None)
      - properties: property -> parents
      - property_class_relations: property -> [domainIncludes...]
      - property_ranges: property -> [rangeIncludes...]

    Parameters
    ----------
    jsonld_path : str
        Path to Schema.org JSON-LD file.

    Returns
    -------
    dict
        {
          "classes": {...},
          "properties": {...},
          "property_class_relations": {...},
          "property_ranges": {...}
        }
    """
    def extract_hierarchy(items, type_filter: str, subclass_key: str, hierarchy: Dict[str, Any]):
        for item in items:
            if "@type" in item and type_filter in item["@type"]:
                name = item.get("@id")
                parents = item.get(subclass_key)
                if isinstance(parents, dict):
                    hierarchy[name] = parents.get("@id")
                elif isinstance(parents, list):
                    hierarchy[name] = [p.get("@id") for p in parents if isinstance(p, dict) and "@id" in p]
                else:
                    hierarchy[name] = None

    def extract_property_relations(items, field: str) -> Dict[str, List[str]]:
        rel = {}
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

    g = data.get("@graph", [])
    classes = {}
    properties = {}

    extract_hierarchy(g, "Class", "rdfs:subClassOf", classes)
    extract_hierarchy(g, "Property", "rdfs:subPropertyOf", properties)
    property_class_relations = extract_property_relations(g, "schema:domainIncludes")
    property_ranges = extract_property_relations(g, "schema:rangeIncludes")

    return {
        "classes": classes,
        "properties": properties,
        "property_class_relations": property_class_relations,
        "property_ranges": property_ranges,
    }


# ======================
# Schema generation core
# ======================
DEFAULT_STOP_CLASSES = {"Thing", "Place", "QuantitativeValue", "QualitativeValue", "Date"}
STATIC_OVERRIDES = {"name": "string", "description": "string", "identifier": "string"}
NO_EXPAND_PROPS = {"description"}


@dataclass
class SchemaGenConfig:
    hierarchy: Dict[str, Dict[str, Any]]
    schemas_with_instances: Dict[str, int]
    stop_classes: set = None
    max_depth: int = 2
    seed: Optional[int] = None

    def __post_init__(self):
        if self.stop_classes is None:
            self.stop_classes = set(DEFAULT_STOP_CLASSES)
        if self.seed is not None:
            random.seed(self.seed)


class JSONSchemaGenerator:
    """
    Generate randomized JSON Schemas centered on classes known to have instances
    (as indicated by `schemas_with_instances`).
    """
    def __init__(self, cfg: SchemaGenConfig):
        self.cfg = cfg

    # ---- Helpers ----
    def _select_central_class(self) -> str:
        classes = list(self.cfg.schemas_with_instances.keys())
        if not classes:
            raise ValueError("No classes with instances are available.")
        return random.choice(classes)

    def _relevant_properties(self, central_class: str, min_properties: int) -> List[str]:
        """
        Collect a list of properties relevant to `central_class` (and ancestors),
        ensuring property ranges include at least one class that has instances.
        """
        props: List[str] = []
        visited: set = set()

        cc = central_class if central_class.startswith("schema:") else f"schema:{central_class}"

        while cc and len(props) < min_properties:
            if cc in visited or cc in self.cfg.stop_classes:
                break
            visited.add(cc)

            current = [
                prop for prop, domains in self.cfg.hierarchy["property_class_relations"].items()
                if cc in domains
            ]

            valid = []
            for p in current:
                ranges = self.cfg.hierarchy["property_ranges"].get(p, [])
                if any(rt in self.cfg.schemas_with_instances for rt in ranges):
                    valid.append(p)

            random.shuffle(valid)
            props.extend(valid)
            props = list(dict.fromkeys(props))  # stable dedupe

            parents = self.cfg.hierarchy["classes"].get(cc)
            if isinstance(parents, list):
                cc = random.choice(parents) if parents else None
            else:
                cc = parents

        if len(props) < min_properties:
            fillers = ["name", "description", "identifier"]
            random.shuffle(fillers)
            props.extend(fillers[: min_properties - len(props)])

        random.shuffle(props)
        return props[:min_properties]

    def _prop_definition(self, prop: str, current_depth: int) -> Dict[str, Any]:
        # static overrides
        if prop in STATIC_OVERRIDES:
            return {"type": STATIC_OVERRIDES[prop]}

        ranges = self.cfg.hierarchy["property_ranges"].get(prop, [])
        if ranges:
            chosen_type = random.choice(ranges).split(":")[-1]
        else:
            chosen_type = "string"

        if prop in NO_EXPAND_PROPS or chosen_type in self.cfg.stop_classes:
            return {"type": chosen_type}

        # Expand complex types into nested objects, up to max_depth
        if chosen_type not in {"string", "Number", "boolean", "array", "URL", "Text"} and current_depth < self.cfg.max_depth:
            nested = self.generate_schema(
                num_properties=max(2, random.randint(2, 4)),
                central_class=chosen_type,
                current_depth=current_depth + 1,
            )
            # cosmetic hint for nested objects:
            nested["properties"]["name"] = {"type": "name_of_object"}
            return {"type": "object", "properties": nested["properties"]}
        return {"type": chosen_type}

    # ---- Public API ----
    def generate_schema(self, num_properties: int, central_class: Optional[str] = None, current_depth: int = 0) -> Dict[str, Any]:
        if not central_class:
            central_class = self._select_central_class()
        props = self._relevant_properties(central_class, num_properties)

        schema = {
            "$schema": "http://json-schema.org/schema#",
            "title": central_class if central_class.startswith("schema:") else f"schema:{central_class}",
            "type": "object",
            "properties": {},
            "required": [],
        }

        for p in props:
            clean = p.split(":")[-1]
            schema["properties"][clean] = self._prop_definition(p, current_depth)
            # randomly mark required
            if random.choice([True, False]):
                schema["required"].append(clean)

        schema["required"] = sorted(set(schema["required"]))
        return schema


# ===============================
# SPARQL-backed value filler (opt)
# ===============================
DEFAULT_ENDPOINT = "https://yago-knowledge.org/sparql/query"

def _require_requests():
    if requests is None:
        raise RuntimeError("The 'requests' library is required for SPARQL filling. Install it or omit --fill-out.")


def query_instances_for_type(schema_type: str, endpoint: str, limit: int = 100) -> List[str]:
    """
    Query SPARQL endpoint for ?instance a schema:<schema_type>.
    Returns a list of URIs.
    """
    _require_requests()
    sparql = f"""
    PREFIX schema: <http://schema.org/>
    SELECT DISTINCT ?instance
    WHERE {{
      ?instance a schema:{schema_type} .
    }}
    LIMIT {limit}
    """
    headers = {"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"}
    try:
        resp = requests.post(endpoint, data=sparql, headers=headers, timeout=60)
        if not resp.ok:
            return []
        data = resp.json()
        out = []
        for b in data.get("results", {}).get("bindings", []):
            uri = b.get("instance", {}).get("value")
            if uri:
                out.append(uri)
        return out
    except Exception:
        return []


def fill_schema_values(schema_def: Dict[str, Any], endpoint: str) -> Any:
    """
    Recursively fill a generated schema with instance URIs.
    - For "object" types, recurse into properties.
    - For leaf types (e.g., CreativeWork, URL, Text), fetch a random instance.
    """
    t = schema_def.get("type")
    if t == "object":
        return {k: fill_schema_values(v, endpoint) for k, v in schema_def.get("properties", {}).items()}
    if t:
        # Note: many Schema.org types won't return instances on your dataset;
        # this function will fall back to a stub string if none found.
        candidates = query_instances_for_type(t, endpoint=endpoint, limit=100)
        return random.choice(candidates) if candidates else f"NoInstanceFoundFor_{t}"
    return "NoTypeSpecified"


# ============
# I/O helpers
# ============
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ============
# CLI
# ============
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Schema Maker (refactored)")
    p.add_argument("--jsonld", required=True, help="Path to Schema.org JSON-LD (e.g., schemaorg-current-https.jsonld)")
    p.add_argument("--schemas-with-instances", required=True,
                   help="JSON mapping { schema:Class -> instance_count }. Used to pick central classes.")
    p.add_argument("--num-schemas", type=int, default=10, help="How many schemas to generate")
    p.add_argument("--out", required=True, help="Where to write generated schemas (JSON list)")
    p.add_argument("--max-depth", type=int, default=2, help="Max nested object expansion depth")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # Optional filling
    p.add_argument("--fill-out", help="If set, write a filled version of the schemas here")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                   help=f"SPARQL endpoint for filling (default: {DEFAULT_ENDPOINT}). "
                        "If you use localhost (e.g., http://localhost:7878/query), ensure your triple-store is running.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Friendly reminder about localhost usage
    if "localhost" in args.endpoint or "127.0.0.1" in args.endpoint:
        print("[Reminder] You're using a localhost endpoint. Make sure your triple-store is running and loaded.")

    # 1) Parse Schema.org JSON-LD
    hierarchy = parse_jsonld_hierarchy(args.jsonld)

    # 2) Load {schema:Class -> count} for picking central classes
    schemas_with_instances = read_json(args.schemas_with_instances)

    # 3) Generate schemas
    cfg = SchemaGenConfig(
        hierarchy=hierarchy,
        schemas_with_instances=schemas_with_instances,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    gen = JSONSchemaGenerator(cfg)

    out_schemas: List[Dict[str, Any]] = []
    for _ in range(args.num_schemas):
        num_props = random.randint(2, 5)
        out_schemas.append(gen.generate_schema(num_properties=num_props))

    write_json(args.out, out_schemas)
    print(f"[OK] Wrote {len(out_schemas)} schemas -> {args.out}")

    # 4) Optionally fill schemas via SPARQL
    if args.fill_out:
        if requests is None:
            print("[WARN] 'requests' not installed; cannot fill schemas. Skipping.")
        else:
            filled = []
            for sch in out_schemas:
                filled.append(fill_schema_values(sch, endpoint=args.endpoint))
            write_json(args.fill_out, filled)
            print(f"[OK] Wrote filled schemas -> {args.fill_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
