# Note on this directory

`expected_metrics.csv` and `../metrics_manifest.json` (referenced by
`README.md` and `scripts/generate_metrics_manifest.py`) held numbers from
an earlier submission cycle and have been removed rather than shipped
stale — they no longer matched the ICAART 2027 paper's final numbers
(e.g. B6 ASR 24.0% here vs. 12.0% in the paper, post the two measurement-bug
fixes and the list-argument policy fix described in the paper's Results and
Implementation sections).

The authoritative source for every number in the paper is the curated set
of directories under `data/results/` (see the top-level `README.md`'s
results table for the exact file-to-row mapping). Regenerating
`generate_metrics_manifest.py` against the current `data/results/` layout
and the current paper is tracked as follow-up work, not yet done.
