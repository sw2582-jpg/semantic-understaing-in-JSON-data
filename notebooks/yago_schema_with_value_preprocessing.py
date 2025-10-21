"""
Refactored utilities for working with Schema.org JSON‑LD and SPARQL queries
=======================================================================

This module provides a set of helper functions to parse Schema.org’s JSON‑LD
hierarchy, inspect classes and their properties in a local triple store, and
persist intermediate results to disk.  The original version of this script was
split across multiple Jupyter cells and included several hard‑coded paths and
magic strings.  This refactored module consolidates that functionality into
clear, reusable functions with sensible defaults that can be overridden as
needed.

The functions in this module assume that a SPARQL endpoint is available on
``http://localhost:7878/query`` by default.  You can override this by
supplying a different ``endpoint`` argument.  The endpoint must be
running and pointed at a dataset that contains the Schema.org data; if it is
not running locally, the queries will fail.  See the included README for
additional usage information.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

from SPARQLWrapper import JSON as SPARQLJSON
from SPARQLWrapper import SPARQLWrapper


def parse_jsonld_hierarchy(jsonld_path: str) -> Dict[str, Dict[str, Optional[str]]]:
    """Parse Schema.org classes, properties, and their hierarchies from a JSON‑LD file.

    Given a JSON‑LD document (such as the Schema.org ``schemaorg-current-https.jsonld``
    file), this function extracts all classes and properties, captures their
    subclass and subproperty relationships, and records the domain and range
    relations for each property.  The result is a dictionary with the
    following keys:

    * ``"classes"``: Mapping of class URI → its parent class URI(s) or ``None``.
    * ``"properties"``: Mapping of property URI → its parent property URI(s) or ``None``.
    * ``"property_class_relations"``: Mapping of property URI → list of domain class URIs.
    * ``"property_ranges"``: Mapping of property URI → list of range class URIs.

    Args:
        jsonld_path: Path to a JSON‑LD file exported from Schema.org.

    Returns:
        A dictionary describing the class and property hierarchies.
    """

    def extract_hierarchy(
        items: Iterable[dict], type_filter: str, subclass_key: str, hierarchy: Dict[str, Optional[Iterable[str]]]
    ) -> None:
        """Extract subclass or subproperty relations from a list of graph items.

        The Schema.org JSON‑LD file contains many objects.  Some of these
        objects represent classes or properties.  This helper iterates over
        them and stores their immediate parent(s) in the supplied ``hierarchy``
        dictionary.

        Args:
            items: An iterable of JSON objects from the ``@graph``.
            type_filter: A substring that must be present in an item’s
                ``@type`` for it to be considered (e.g. ``"Class"`` or
                ``"Property"``).
            subclass_key: The key that contains parent information (e.g.
                ``"rdfs:subClassOf"`` or ``"rdfs:subPropertyOf"``).
            hierarchy: A dictionary that will be mutated to map each URI to
                either ``None``, a single parent URI, or a list of parent URIs.
        """
        for item in items:
            if "@type" in item and type_filter in item["@type"]:
                name = item["@id"]
                parents = item.get(subclass_key)
                if isinstance(parents, dict):
                    # Single parent specified as an object
                    hierarchy[name] = parents.get("@id")
                elif isinstance(parents, list):
                    # Multiple parents specified as a list of objects
                    hierarchy[name] = [parent.get("@id") for parent in parents if "@id" in parent]
                else:
                    # No parent specified; top‑level class/property
                    hierarchy[name] = None

    def extract_property_relations(items: Iterable[dict], field: str) -> Dict[str, List[str]]:
        """Extract domain/range relations for properties.

        Args:
            items: An iterable of JSON objects from the ``@graph``.
            field: The key containing the domain or range definitions
                (e.g. ``"schema:domainIncludes"`` or ``"schema:rangeIncludes"``).

        Returns:
            A mapping from each property URI to a list of class URIs that
            appear in the given field.
        """
        relations: Dict[str, List[str]] = {}
        for item in items:
            if "@type" in item and "Property" in item["@type"]:
                prop = item["@id"]
                domains_or_ranges = item.get(field)
                if domains_or_ranges:
                    if isinstance(domains_or_ranges, dict):
                        relations[prop] = [domains_or_ranges["@id"]]
                    elif isinstance(domains_or_ranges, list):
                        relations[prop] = [d["@id"] for d in domains_or_ranges if "@id" in d]
        return relations

    with open(jsonld_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = data.get("@graph", [])
    classes_hierarchy: Dict[str, Optional[Iterable[str]]] = {}
    properties_hierarchy: Dict[str, Optional[Iterable[str]]] = {}

    extract_hierarchy(graph, "Class", "rdfs:subClassOf", classes_hierarchy)
    extract_hierarchy(graph, "Property", "rdfs:subPropertyOf", properties_hierarchy)
    property_class_relations = extract_property_relations(graph, "schema:domainIncludes")
    property_ranges = extract_property_relations(graph, "schema:rangeIncludes")

    return {
        "classes": classes_hierarchy,
        "properties": properties_hierarchy,
        "property_class_relations": property_class_relations,
        "property_ranges": property_ranges,
    }


def get_count_for_class(cls_uri: str, endpoint: str = "http://localhost:7878/query") -> int:
    """Count the number of instances of a given class in a SPARQL triple store.

    This function sends a simple SPARQL query that counts the number of
    resources of the supplied class.  The default endpoint assumes that the
    triple store is running locally.  If your SPARQL service is running on a
    different host or port, supply a different ``endpoint`` value.

    Args:
        cls_uri: The full URI of the class (e.g. ``"http://schema.org/Person"``).
        endpoint: URL of the SPARQL endpoint accepting POST/GET requests.

    Returns:
        An integer representing the total count of resources of the given class.
    """
    sparql = SPARQLWrapper(endpoint)
    query = f"""
        PREFIX sc: <http://purl.org/science/owl/sciencecommons/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>
        PREFIX yago: <http://yago-knowledge.org/resource/>

        SELECT (COUNT(?instance) AS ?count)
        WHERE {{
          ?instance a {cls_uri} .
        }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(SPARQLJSON)
    results = sparql.query().convert()
    try:
        count_str = results["results"]["bindings"][0]["count"]["value"]
        return int(count_str)
    except (IndexError, KeyError, ValueError):
        # If the query fails or returns an unexpected structure, return zero
        return 0


def get_properties_for_class(cls_uri: str, endpoint: str = "http://localhost:7878/query") -> List[str]:
    """Retrieve all distinct properties used by instances of a given class.

    The SPARQL query selects distinct predicates for all triples where the
    subject is of type ``cls_uri``.  Because the triple store may contain
    numerous predicates, the results are ordered alphabetically for
    reproducibility.

    Args:
        cls_uri: The full URI of the class to query.
        endpoint: URL of the SPARQL endpoint.

    Returns:
        A list of property URIs used by resources of the supplied class.
    """
    sparql = SPARQLWrapper(endpoint)
    query = f"""
        PREFIX sc: <http://purl.org/science/owl/sciencecommons/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>
        PREFIX yago: <http://yago-knowledge.org/resource/>

        SELECT DISTINCT ?p
        WHERE {{
          ?instance a {cls_uri} ;
                    ?p ?o .
        }}
        ORDER BY ?p
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(SPARQLJSON)
    results = sparql.query().convert()
    properties: List[str] = []
    for row in results.get("results", {}).get("bindings", []):
        prop = row.get("p", {}).get("value")
        if prop:
            properties.append(prop)
    return properties


def save_partial_results(valid_classes: Dict[str, int], class_properties: Dict[str, List[str]], filepath: str) -> None:
    """Write intermediate results to disk as JSON.

    When querying large vocabularies, it is common to save partial results to
    avoid re‑querying the same classes repeatedly.  This helper serializes
    two dictionaries – one mapping classes to their instance counts and one
    mapping classes to their property lists – into a single JSON file.  The
    file is written with an indentation of two spaces for readability.

    Args:
        valid_classes: A mapping of class URIs to the number of instances
            found in the triple store.
        class_properties: A mapping of class URIs to a list of properties
            used by instances of the class.
        filepath: Destination filename for the JSON data.
    """
    data = {
        "valid_classes": valid_classes,
        "class_properties": class_properties,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Prefix definitions used when converting full URIs into compact prefixed names
PREFIX_MAP: Dict[str, str] = {
    "http://schema.org/": "schema:",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
    "http://www.w3.org/2002/07/owl#": "owl:",
    "http://purl.org/science/owl/sciencecommons/": "sc:",
    "http://yago-knowledge.org/resource/": "yago:",
    # Additional prefixes may be added here as needed
}


def to_prefixed_uri(uri: str, prefix_map: Dict[str, str] = PREFIX_MAP) -> str:
    """Convert a full URI into a compact prefixed representation.

    For example, ``"http://schema.org/Person"`` becomes ``"schema:Person"``.
    If no matching prefix is found, the original URI is returned unchanged.

    Args:
        uri: A full URI.
        prefix_map: A mapping of base URIs to prefixes.

    Returns:
        A string containing the prefixed name or the original URI.
    """
    for base_uri, prefix in prefix_map.items():
        if uri.startswith(base_uri):
            return prefix + uri[len(base_uri) :]
    return uri


def convert_to_prefixed(class_properties: Dict[str, List[str]], prefix_map: Dict[str, str] = PREFIX_MAP) -> Dict[str, List[str]]:
    """Apply ``to_prefixed_uri`` to both keys and values of a mapping.

    This helper takes a dictionary mapping full class URIs to lists of
    full property URIs and returns a new dictionary where both the class and
    property URIs have been shortened to use the prefix map.

    Args:
        class_properties: Mapping of full class URIs to lists of full property
            URIs.
        prefix_map: Mapping of base URIs to prefixes for compact names.

    Returns:
        A new dictionary with the same structure but using prefixed names.
    """
    new_mapping: Dict[str, List[str]] = {}
    for cls_uri, props in class_properties.items():
        prefixed_cls = to_prefixed_uri(cls_uri, prefix_map=prefix_map)
        prefixed_props = [to_prefixed_uri(prop, prefix_map=prefix_map) for prop in props]
        new_mapping[prefixed_cls] = prefixed_props
    return new_mapping


def collect_class_info(
    jsonld_path: str,
    endpoint: str = "http://localhost:7878/query",
    partial_results_file: str = "partial_results.json",
    prefixed_results_file: str = "partial_results_prefixed.json",
) -> None:
    """Query a SPARQL endpoint for all Schema.org classes and save their properties.

    This convenience function orchestrates the typical workflow of parsing
    Schema.org’s JSON‑LD vocabulary, counting the instances of each class in
    a local triple store, retrieving the properties used by each class, and
    writing both the raw and prefixed results to disk.  Intermediate results
    can be resumed if the process is interrupted; if a ``partial_results.json``
    file exists it will be loaded and updated rather than overwritten.

    Args:
        jsonld_path: Path to the Schema.org JSON‑LD file.
        endpoint: URL of the SPARQL endpoint (defaults to local host).
        partial_results_file: Destination filename for saving raw results.
        prefixed_results_file: Destination filename for saving prefixed results.
    """
    # Parse the JSON‑LD to obtain class URIs
    hierarchies = parse_jsonld_hierarchy(jsonld_path)
    class_uris = list(hierarchies["classes"].keys())

    # Load existing results if available
    if os.path.exists(partial_results_file):
        with open(partial_results_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        valid_classes = existing.get("valid_classes", {})
        class_properties = existing.get("class_properties", {})
    else:
        valid_classes = {}
        class_properties = {}

    # Iterate through all classes and query the triple store
    for cls_uri in class_uris:
        # Skip classes that have already been processed
        if cls_uri in class_properties:
            continue
        # Skip the generic ``schema:Thing`` class; it would match every triple
        if cls_uri.rstrip("/").endswith("Thing"):
            continue
        count = get_count_for_class(cls_uri, endpoint=endpoint)
        if count > 0:
            valid_classes[cls_uri] = count
            props = get_properties_for_class(cls_uri, endpoint=endpoint)
            class_properties[cls_uri] = props
            # Save progress after each successful query to avoid losing work
            save_partial_results(valid_classes, class_properties, partial_results_file)

    # Convert the results to prefixed form and write them to a separate file
    prefixed_properties = convert_to_prefixed(class_properties)
    prefixed_data = {
        "valid_classes": valid_classes,
        "class_properties": prefixed_properties,
    }
    with open(prefixed_results_file, "w", encoding="utf-8") as f:
        json.dump(prefixed_data, f, indent=2)

    print(
        f"Saved raw results to {partial_results_file} and prefixed results to {prefixed_results_file}."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect instance counts and properties for Schema.org classes using a local SPARQL endpoint."
    )
    parser.add_argument(
        "jsonld_path",
        type=str,
        help="Path to a Schema.org JSON‑LD file (e.g. schemaorg-current-https.jsonld).",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:7878/query",
        help="URL of the SPARQL endpoint (default: http://localhost:7878/query).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="partial_results.json",
        help="Path to save intermediate results (default: partial_results.json).",
    )
    parser.add_argument(
        "--prefixed-output",
        type=str,
        default="partial_results_prefixed.json",
        help="Path to save results with prefixed URIs (default: partial_results_prefixed.json).",
    )
    args = parser.parse_args()

    collect_class_info(
        jsonld_path=args.jsonld_path,
        endpoint=args.endpoint,
        partial_results_file=args.output,
        prefixed_results_file=args.prefixed_output,
    )