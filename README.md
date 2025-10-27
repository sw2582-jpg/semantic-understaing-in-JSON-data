Semantic Understanding in JSON Data

This repository contains a set of scripts to explore and generate
Schema.org‑compliant JSON data based on the contents of an RDF triple store
such as YAGO. The goal is to enable researchers and practitioners to
automatically build realistic JSON Schemas and corresponding synthetic data
instances, as well as extract CodeBERT embeddings for downstream machine
learning tasks.

Overview

The pipeline consists of the following stages:

Preprocess Schema.org vocabulary and your dataset.
Use yago_schema_with_value_preprocess.py to parse a Schema.org JSON‑LD
file, discover which classes have instances in your triple store, and
record the properties used for each class. The script also writes a
CURIE‑prefixed version of the results for easier reading.

Generate JSON Schemas.
Use schema_making.py to randomly assemble JSON Schema definitions
centered on classes with instances. You can control the number of
properties per schema, the maximum depth of nested objects and, if
desired, fill the schemas with real instance URIs from a SPARQL endpoint.

Harvest example values.
Use the harvest subcommand in yago_schema_with_value.py to collect
example property values for each class/property pair. This stage
distinguishes numeric QuantitativeValue ranges from other object
ranges.

Split values into train/test sets.
Use the split subcommand in yago_schema_with_value.py to partition
the harvested values into independent training and test sets.

Generate random instances.
Use the generate subcommand in yago_schema_with_value.py to create
synthetic JSON documents for each class, drawing on the harvested values
and expanding nested resources when appropriate. The documents are
saved into separate train and test files.

Compute CodeBERT embeddings.
Use codebert.py to compute fixed‑size embeddings from a column of text
in a CSV file using a pre‑trained CodeBERT model. This is useful for
downstream tasks such as entity matching or classification.

Installation

All scripts are pure Python and rely on a few external libraries:
transformers, torch, pandas, numpy, tqdm, and SPARQLWrapper. You
can install them with pip:

pip install transformers torch pandas numpy tqdm SPARQLWrapper


If you plan to fill schemas with real data, make sure requests is
available as well.

Running the Pipeline

Below is a high‑level walkthrough showing how to use these scripts together.
Assume you have:

A Schema.org JSON‑LD file (e.g., RFDs/schemaorg-current-https.jsonld).

A SPARQL endpoint hosting your dataset (e.g., YAGO) at
http://localhost:7878/query. Adjust accordingly if your endpoint is
remote.

1. Preprocess
python yago_schema_with_value_preprocess.py \
  --jsonld RFDs/schemaorg-current-https.jsonld \
  --endpoint http://localhost:7878/query \
  --out partial_results.json \
  --out-prefixed partial_results_prefixed.json


This step saves two files:

partial_results.json — class → instance count and class → properties
(full URIs). The script saves progress incrementally so you can resume
interrupted runs.

partial_results_prefixed.json — the same data but with readable
prefixes (e.g., schema:Person).

2. Generate Schemas

First prepare a JSON mapping of classes to their instance counts (the
valid_classes dictionary) from the previous step. You can extract it
directly from partial_results.json:

import json
with open('partial_results.json') as f:
    data = json.load(f)
valid_classes = data['valid_classes']
with open('schemas_with_instances.json', 'w') as f:
    json.dump(valid_classes, f, indent=2)


Then run:

python schema_making.py \
  --jsonld RFDs/schemaorg-current-https.jsonld \
  --schemas-with-instances schemas_with_instances.json \
  --num-schemas 10 \
  --out generated_schemas.json \
  --max-depth 2 \
  --seed 42


To also fill the schemas with instance URIs, add --fill-out and
--endpoint:

python schema_making.py \
  --jsonld RFDs/schemaorg-current-https.jsonld \
  --schemas-with-instances schemas_with_instances.json \
  --num-schemas 5 \
  --out generated_schemas.json \
  --fill-out filled_schemas.json \
  --endpoint http://localhost:7878/query

3. Harvest Property Values
python yago_schema_with_value.py harvest \
  --partial-results partial_results_prefixed.json \
  --jsonld RFDs/schemaorg-current-https.jsonld \
  --endpoint http://localhost:7878/query \
  --out class_property_values.json \
  --limit-per-prop 200


This stage queries your triple store for example values of each property
attached to each class. It distinguishes numeric QuantitativeValue
properties from others and extracts the numeric literal where appropriate.
The results are stored as a nested dictionary:

{
  "schema:Person": {
    "schema:name": ["Alice", "Bob", ...],
    "schema:birthDate": ["1962-01-01", ...],
    ...
  },
  ...
}

4. Split Train/Test Sets
python yago_schema_with_value.py split \
  --values class_property_values.json \
  --train-out class_property_values_train.json \
  --test-out class_property_values_test.json \
  --train-ratio 0.8


This splits each list of values into training and test subsets, preserving
all classes and properties.

5. Generate Random Instances
python yago_schema_with_value.py generate \
  --partial-results partial_results_prefixed.json \
  --values class_property_values_train.json \
  --jsonld RFDs/schemaorg-current-https.jsonld \
  --endpoint http://localhost:7878/query \
  --num-instances 1000 \
  --probability 0.35 \
  --expand-depth 1 \
  --train-out generated_instances_train.json \
  --test-out generated_instances_test.json


This command creates a specified number of random instances per class,
including each property with a given probability. When a property value is a
YAGO resource and nested expansion is allowed, the script will either
replace it with its English label or generate a nested object using the
property’s range information. The resulting documents are stored in
separate training and test files and can be further tidied or processed
downstream.

6. Compute CodeBERT Embeddings
python codebert.py \
  --input col_yago_train.csv \
  --text-col values_concat \
  --out train_embeddings.npy


This reads the specified text column from a CSV and produces a NumPy array
of embeddings along with a .meta.json file describing the settings used
(model, pooling, etc.). Adjust parameters such as --pooling, --max-len,
and --batch-size as needed.

Notes

Triple Store: Many of the scripts assume you have loaded your RDF
dataset into a SPARQL endpoint (e.g., Blazegraph) running on
http://localhost:7878/query. If your endpoint is elsewhere, set the
--endpoint flag accordingly. The scripts will remind you to start your
triple store if you specify a localhost URL.

Prefixes: The preprocessing script writes a CURIE‑prefixed version
of class/property relationships. All subsequent scripts expect these
prefixed identifiers (e.g., schema:Person, schema:name).

Resuming: Both the preprocessing and harvesting stages save their
progress incrementally. If interrupted, simply rerun the same command
and the scripts will skip completed classes or properties.

Extensibility: Feel free to modify the prefix_map in
yago_schema_with_value_preprocess.py or adjust the sampling logic in
schema_making.py and yago_schema_with_value.py to suit your needs.

License

This project is provided as‑is without any warranty. You are free to use,
modify and distribute the scripts for research or commercial purposes.