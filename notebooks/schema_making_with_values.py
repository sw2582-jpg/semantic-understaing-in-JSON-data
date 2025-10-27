#!/usr/bin/env python3
"""
Unified Schema.org -> Values -> Instances Tool (Refactored)
==========================================================

This refactors the original `yago_schema_with_value.py` into a modular CLI with
four stages you can run end-to-end or individually:

  1) preprocess:  Parse Schema.org JSON-LD and write class/property metadata
  2) harvest:     Query a SPARQL endpoint to gather example values per property
  3) schema:      Generate JSON Schemas centered on classes with instances
  4) instances:   Generate synthetic JSON instances (optionally deep expand)

Key improvements:
- All paths/parameters are flags (no hard-coding)
- Clear, typed functions with docstrings
- Resumable outputs where applicable
- Safe SPARQL wrappers and robust JSON I/O
- Helpful logging and a localhost reminder if you point to 127.0.0.1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from typing import Dict, List, Optional, Any, Iterable

# Optional: SPARQL via requests
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


# =====================
# General JSON utilities
# =====================
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# 1) Preprocess JSON-LD meta
# =========================
def parse_jsonld_hierarchy(jsonld_path: str) -> Dict[str, Dict[str, Any]]:
    """Extract Schema.org metadata from JSON-LD (@graph):
    - classes: {classIRI -> parents}
    - properties: {propIRI -> parents}
    - property_class_relations: {propIRI -> [domainIncludes]}
    - property_ranges: {propIRI -> [rangeIncludes]}
    """
    def extract_hierarchy(items: Iterable[dict], type_filter: str, subclass_key: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for item in items:
            if "@type" in item and type_filter in item["@type"]:
                cid = item.get("@id")
                parents = item.get(subclass_key)
                if isinstance(parents, dict):
                    out[cid] = parents.get("@id")
                elif isinstance(parents, list):
                    out[cid] = [p.get("@id") for p in parents if isinstance(p, dict) and "@id" in p]
                else:
                    out[cid] = None
        return out

    def extract_rel(items: Iterable[dict], field: str) -> Dict[str, List[str]]:
        rel: Dict[str, List[str]] = {}
        for item in items:
            if "@type" in item and "Property" in item["@type"]:
                pid = item.get("@id")
                dom = item.get(field)
                if isinstance(dom, dict) and "@id" in dom:
                    rel[pid] = [dom["@id"]]
                elif isinstance(dom, list):
                    rel[pid] = [d["@id"] for d in dom if isinstance(d, dict) and "@id" in d]
        return rel

    data = read_json(jsonld_path)
    g = data.get("@graph", [])
    return {
        "classes": extract_hierarchy(g, "Class", "rdfs:subClassOf"),
        "properties": extract_hierarchy(g, "Property", "rdfs:subPropertyOf"),
        "property_class_relations": extract_rel(g, "schema:domainIncludes"),
        "property_ranges": extract_rel(g, "schema:rangeIncludes"),
    }


# ===============================
# 2) SPARQL helper and harvesting
# ===============================
DEFAULT_ENDPOINT = "http://localhost:7878/query"


def _need_requests():
    if requests is None:
        raise RuntimeError("The 'requests' package is required for SPARQL operations.")


def sparql_select(endpoint: str, query: str, timeout: int = 90) -> Dict[str, Any]:
    _need_requests()
    headers = {"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"}
    resp = requests.post(endpoint, data=query, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def count_instances_for_class(cls_uri: str, endpoint: str) -> int:
    q = f"""
    SELECT (COUNT(?s) AS ?c)
    WHERE {{ ?s a <{cls_uri}> . }}
    """
    data = sparql_select(endpoint, q)
    return int(data["results"]["bindings"][0]["c"]["value"]) if data["results"]["bindings"] else 0


def distinct_properties_for_class(cls_uri: str, endpoint: str) -> List[str]:
    q = f"""
    SELECT DISTINCT ?p
    WHERE {{ ?s a <{cls_uri}> ; ?p ?o . }} ORDER BY ?p
    """
    data = sparql_select(endpoint, q)
    return [b["p"]["value"] for b in data.get("results", {}).get("bindings", [])]


def harvest_example_values(properties: List[str], endpoint: str, limit_per_prop: int = 100) -> Dict[str, List[str]]:
    """Fetch up to N example values for each property (as string or URI)."""
    out: Dict[str, List[str]] = {}
    for p in properties:
        q = f"""
        SELECT DISTINCT ?o
        WHERE {{ ?s <{p}> ?o . }}
        LIMIT {limit_per_prop}
        """
        try:
            data = sparql_select(endpoint, q)
            out[p] = [b["o"]["value"] for b in data.get("results", {}).get("bindings", [])]
        except Exception:
            out[p] = []
    return out


# =========================
# 3) JSON Schema generation
# =========================
STOP_CLASSES = {"http://schema.org/Thing", "http://schema.org/Place", "http://schema.org/Date"}
STATIC_OVERRIDES = {"http://schema.org/name": "string", "http://schema.org/description": "string"}


def _strip_prefix(iri: str) -> str:
    return iri.split("/")[-1] if iri.startswith("http") else iri.split(":")[-1]


def build_schema_for_class(central_class_iri: str,
                           meta: Dict[str, Dict[str, Any]],
                           class_has_instances: Dict[str, int],
                           max_depth: int = 2,
                           min_props: int = 3) -> Dict[str, Any]:
    """Create a JSON Schema centered on `central_class_iri` by pulling properties
    whose ranges are classes with instances (when possible), and expanding nested
    objects up to `max_depth`.
    """
    def pick_props(cls: str) -> List[str]:
        props: List[str] = []
        seen: set = set()
        cur = cls
        while cur and len(props) < min_props:
            if cur in seen or cur in STOP_CLASSES:
                break
            seen.add(cur)
            current = [p for p, doms in meta["property_class_relations"].items() if cur in doms]
            valid = []
            for p in current:
                rng = meta["property_ranges"].get(p, [])
                if any(r in class_has_instances for r in rng):
                    valid.append(p)
            random.shuffle(valid)
            props.extend(valid)
            props = list(dict.fromkeys(props))
            parent = meta["classes"].get(cur)
            cur = parent[0] if isinstance(parent, list) and parent else (parent or None)
        # fill if not enough
        if len(props) < min_props:
            fillers = ["http://schema.org/name", "http://schema.org/description"]
            for f in fillers:
                if f not in props:
                    props.append(f)
                if len(props) >= min_props:
                    break
        return props[:min_props]

    def prop_def(p: str, depth: int) -> Dict[str, Any]:
        if p in STATIC_OVERRIDES:
            return {"type": STATIC_OVERRIDES[p]}
        rng = meta["property_ranges"].get(p, [])
        chosen = random.choice(rng) if rng else "http://schema.org/Text"
        if chosen in STOP_CLASSES or depth >= max_depth:
            return {"type": _strip_prefix(chosen)}
        # nested expansion
        nested = build_schema_for_class(chosen, meta, class_has_instances, max_depth=max_depth, min_props=2)
        return {"type": "object", "properties": nested["properties"]}

    props = pick_props(central_class_iri)
    schema = {
        "$schema": "http://json-schema.org/schema#",
        "title": _strip_prefix(central_class_iri),
        "type": "object",
        "properties": {},
        "required": [],
    }
    for p in props:
        key = _strip_prefix(p)
        schema["properties"][key] = prop_def(p, depth=0)
        if random.choice([True, False]):
            schema["required"].append(key)
    schema["required"] = sorted(set(schema["required"]))
    return schema


# ==============================
# 4) Instance synthesis & tidy-up
# ==============================
def tidy_value(v: str) -> Any:
    """Attempt to coerce numeric strings; otherwise return original string."""
    try:
        if v.isdigit():
            return int(v)
        if any(ch in v for ch in [".", "e", "E"]):
            return float(v)
    except Exception:
        pass
    return v


def generate_instance_from_values(schema: Dict[str, Any], values_by_prop: Dict[str, List[str]], expand_depth: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, spec in schema.get("properties", {}).items():
        t = spec.get("type")
        if t == "object" and expand_depth > 0:
            out[k] = generate_instance_from_values({"properties": spec.get("properties", {})}, values_by_prop, expand_depth-1)
        else:
            choices = values_by_prop.get(k, [])
            out[k] = tidy_value(random.choice(choices)) if choices else f"{k}_value"
    return out


# =========
# CLI tasks
# =========
def cmd_preprocess(args: argparse.Namespace) -> None:
    meta = parse_jsonld_hierarchy(args.jsonld)
    write_json(args.out, meta)
    print(f"[OK] wrote metadata -> {args.out}")


def cmd_harvest(args: argparse.Namespace) -> None:
    if "localhost" in args.endpoint or "127.0.0.1" in args.endpoint:
        print("[Reminder] Using localhost endpoint; ensure your triple store is running.")
    meta = read_json(args.meta)

    # Collect distinct properties across all classes (domains)
    all_props = sorted(set(meta["property_class_relations"].keys()))
    print(f"[INFO] harvesting example values for {len(all_props)} properties (limit {args.limit_per_prop})")
    values = harvest_example_values(all_props, endpoint=args.endpoint, limit_per_prop=args.limit_per_prop)
    write_json(args.out, values)
    print(f"[OK] wrote values -> {args.out}")


def cmd_schema(args: argparse.Namespace) -> None:
    meta = read_json(args.meta)
    class_counts = read_json(args.class_counts)  # {classIRI -> count}
    out: List[Dict[str, Any]] = []
    for _ in range(args.num):
        c = random.choice(list(class_counts.keys()))
        out.append(build_schema_for_class(c, meta, class_counts, max_depth=args.max_depth))
    write_json(args.out, out)
    print(f"[OK] wrote {len(out)} schemas -> {args.out}")


def cmd_instances(args: argparse.Namespace) -> None:
    schemas = read_json(args.schemas)
    values_raw = read_json(args.values)  # {propIRI or short -> [vals]}

    # normalize keys to short names for matching schema property keys
    values_by_short: Dict[str, List[str]] = {}
    for iri, arr in values_raw.items():
        key = iri.split("/")[-1] if iri.startswith("http") else iri.split(":")[-1]
        values_by_short[key] = list(arr) if isinstance(arr, list) else []

    out = []
    for sch in schemas:
        out.append({
            "title": sch.get("title"),
            "instance": generate_instance_from_values(sch, values_by_short, expand_depth=args.expand_depth)
        })
    write_json(args.out, out)
    print(f"[OK] wrote {len(out)} instances -> {args.out}")


# ======
# Parser
# ======
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Schema.org -> Values -> Instances (Refactored)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preprocess", help="Parse Schema.org JSON-LD into metadata")
    sp.add_argument("--jsonld", required=True, help="Path to Schema.org JSON-LD")
    sp.add_argument("--out", required=True, help="Output metadata JSON path")
    sp.set_defaults(func=cmd_preprocess)

    sh = sub.add_parser("harvest", help="Harvest example values from a SPARQL endpoint")
    sh.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"SPARQL endpoint (default: {DEFAULT_ENDPOINT})")
    sh.add_argument("--meta", required=True, help="Metadata JSON from preprocess stage")
    sh.add_argument("--out", required=True, help="Output values JSON path")
    sh.add_argument("--limit-per-prop", type=int, default=100, help="Max example values per property")
    sh.set_defaults(func=cmd_harvest)

    ss = sub.add_parser("schema", help="Generate JSON Schemas for classes with instances")
    ss.add_argument("--meta", required=True, help="Metadata JSON from preprocess stage")
    ss.add_argument("--class-counts", required=True, help="JSON mapping of classIRI -> instance count")
    ss.add_argument("--num", type=int, default=10, help="How many schemas to generate")
    ss.add_argument("--out", required=True, help="Output schemas JSON path")
    ss.add_argument("--max-depth", type=int, default=2, help="Max nested expansion depth")
    ss.set_defaults(func=cmd_schema)

    si = sub.add_parser("instances", help="Generate JSON instances from schemas and harvested values")
    si.add_argument("--schemas", required=True, help="Input schemas JSON path")
    si.add_argument("--values", required=True, help="Input harvested values JSON path")
    si.add_argument("--out", required=True, help="Output instances JSON path")
    si.add_argument("--expand-depth", type=int, default=0, help="Nested object expansion depth for instance generation")
    si.set_defaults(func=cmd_instances)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Friendly localhost reminder
    endpoint = getattr(args, "endpoint", None)
    if isinstance(endpoint, str) and ("localhost" in endpoint or "127.0.0.1" in endpoint):
        print("[Reminder] Using localhost endpoint; ensure your triple store is running.")

    random.seed(int(time.time()))
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
