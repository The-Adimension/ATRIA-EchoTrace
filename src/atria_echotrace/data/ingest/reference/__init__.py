"""Vendored reference preprocessors — the scripts that built the training corpus.

Copied verbatim from ``datasets/data_processing_scripts/``; see ``README.md`` in this
directory. Nothing here may be edited: `atria ingest` reproduces the adapters' training
data precisely because these files are unmodified.

The modules are deliberately **not** imported here. ``preprocess_echonet`` imports
OpenCV at module scope, so importing this package eagerly would make the whole
application depend on the ``ingest`` extra. Callers import the submodule they need.
"""
