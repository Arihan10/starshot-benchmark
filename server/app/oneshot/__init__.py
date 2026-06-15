"""One-shot spatial benchmark — an experimental, fully isolated track.

One LLM call designs the ENTIRE scene (a flat list of world-frame objects),
bypassing the recursive divider/generation harness and its prompt versioning
completely. Everything specific to this track lives in this package: its own
slots and models (`slots.py`), its own editable prompt (`prompts/*.txt`), its
own pipeline (`pipeline.py`), and its own HTTP surface (`routes.py`, mounted
under `/oneshot`). Only shared infrastructure is reused: the LLM/mesh/image
services, the event log (SlotLog), the canonical types, and the template
variable resolver.
"""
