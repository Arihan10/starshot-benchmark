#!/usr/bin/env python3
"""Extract sentences containing numbers from LLM reasoning traces.

Walks the given runs under ``runs/``, reads every ``events.jsonl``, pulls the
``reasoning`` field out of each ``cache.llm`` event, splits it into sentences,
and keeps the sentences that contain at least one digit. Results (with full
provenance: run / scene / model / node / pipeline step) are written to JSON.
"""
import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

DEFAULT_RUNS = ["against-the-gods", "against-the-shooters", "compare"]

# Abbreviations whose trailing period must not be treated as a sentence end.
_ABBREVS = [
    "e.g.", "i.e.", "etc.", "vs.", "approx.", "cf.", "al.",
    "Dr.", "Mr.", "Mrs.", "Ms.", "Fig.", "No.", "St.",
]
_DOT_PLACEHOLDER = "\uE000"
_decimal_dot = re.compile(r"(?<=\d)\.(?=\d)")
_sentence_boundary = re.compile(r"(?<=[.!?])\s+")
_has_digit = re.compile(r"\d")
# Signed integers / decimals not glued to a word char, plus the trailing
# operands of dimension notation like "40x50" or "20x22x10".
_number_token = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?|(?<=\d[xX\u00d7*])\d+(?:\.\d+)?")


def _protect(text: str) -> str:
    text = _decimal_dot.sub(_DOT_PLACEHOLDER, text)
    for abbrev in _ABBREVS:
        text = text.replace(abbrev, abbrev.replace(".", _DOT_PLACEHOLDER))
    return text


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in re.split(r"[\r\n]+", text):
        paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
        if not paragraph:
            continue
        for part in _sentence_boundary.split(_protect(paragraph)):
            sentence = part.replace(_DOT_PLACEHOLDER, ".").strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def derive_provenance(events_path: Path, runs_dir: Path) -> dict:
    rel = events_path.relative_to(runs_dir)
    middle = rel.parts[1:-1]  # between <run> and events.jsonl
    return {
        "run": rel.parts[0],
        "scene": middle[0] if middle else "",
        "model": "/".join(middle[1:]) if len(middle) > 1 else "",
        "source": rel.as_posix(),
    }


def iter_reasoning_events(path: Path):
    """Yield (line_no, event) for cache.llm events with non-empty reasoning."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            if '"cache.llm"' not in line:  # cheap prefilter, avoids parsing
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") != "cache.llm":
                continue
            reasoning = event.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                yield line_no, event, reasoning


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=DEFAULT_RUNS,
                        help="Run folder names under the runs directory.")
    parser.add_argument("--runs-dir", type=Path, default=script_dir / "runs",
                        help="Directory that contains the run folders.")
    parser.add_argument("--out", type=Path,
                        default=script_dir / "reasoning_number_sentences.json",
                        help="Output JSON path.")
    args = parser.parse_args()

    runs = args.runs or DEFAULT_RUNS
    runs_dir: Path = args.runs_dir

    files_out: list[dict] = []
    n_files = 0
    n_reasoning_events = 0
    n_sentences = 0
    n_kept = 0
    by_run: Counter = Counter()
    by_model: Counter = Counter()
    by_step: Counter = Counter()

    for run in runs:
        run_dir = runs_dir / run
        if not run_dir.is_dir():
            print(f"[warn] run folder not found: {run_dir}")
            continue
        for events_path in sorted(run_dir.rglob("events.jsonl")):
            n_files += 1
            prov = derive_provenance(events_path, runs_dir)
            model_id = None
            sentences_out: list[dict] = []
            for line_no, event, reasoning in iter_reasoning_events(events_path):
                n_reasoning_events += 1
                if model_id is None:
                    model_id = event.get("model")
                for sentence in split_sentences(reasoning):
                    n_sentences += 1
                    if not _has_digit.search(sentence):
                        continue
                    n_kept += 1
                    sentences_out.append({
                        "event_index": event.get("index", line_no),
                        "node": event.get("node"),
                        "step": event.get("step"),
                        "sentence": sentence,
                        "numbers": _number_token.findall(sentence),
                    })
                    by_run[prov["run"]] += 1
                    by_model[prov["model"]] += 1
                    by_step[event.get("step")] += 1
            files_out.append({
                **prov,
                "model_id": model_id,
                "sentence_count": len(sentences_out),
                "sentences": sentences_out,
            })

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs_dir": runs_dir.as_posix(),
        "runs": runs,
        "source_field": "reasoning (cache.llm events)",
        "inclusion_rule": "sentence contains at least one digit (0-9)",
        "counts": {
            "events_files": n_files,
            "reasoning_events": n_reasoning_events,
            "sentences_scanned": n_sentences,
            "sentences_with_numbers": n_kept,
        },
        "breakdown": {
            "by_run": dict(by_run.most_common()),
            "by_model": dict(by_model.most_common()),
            "by_step": dict(by_step.most_common()),
        },
        "files": files_out,
    }

    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"Files scanned          : {n_files}")
    print(f"Reasoning events        : {n_reasoning_events}")
    print(f"Sentences scanned       : {n_sentences}")
    print(f"Sentences with numbers  : {n_kept}")
    print(f"Wrote                   : {args.out}")


if __name__ == "__main__":
    main()
