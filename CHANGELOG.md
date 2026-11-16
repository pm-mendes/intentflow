# Changelog

## After the tag

- README, Known limitations: the request-text rule of `ProvenanceTracker.track_argument` (step 2), which
  labels `user_intent` any value contained in the request text, for any field and before the documents are
  consulted. Kept unchanged in the cited version: removing it lowers IF-Hardened's Bench-v2 benign success
  from 200/200 to 183/200.
- ARCHITECTURE.md: the reproduction entry point is `artifact/scripts/run_all.sh --offline`;
  `scripts/run_all_experiments.sh` (API key, regenerates the data) is marked obsolete.
- Zenodo DOI added to README.md and CITATION.cff: 10.5281/zenodo.24233094 (archive of the tag, identical to
  it), concept DOI 10.5281/zenodo.24233093.

## icaart2027-camera-ready (November 2026)

Version cited by the ICAART 2027 camera-ready. Zenodo: 10.5281/zenodo.24233094.

- **Security fix, value–intent matching** (`provenance/tracker.py::_matches_intent_parameters`): a value now
  matches the intent contract only if it equals an authorized value after role-specific normalization
  (`normalize_argument`), or belongs to an authorized list. The submitted version also accepted substring
  inclusion in both directions, so `manager@company.com.evil.io` inherited `user_intent` when the contract
  authorized `manager@company.com`. Checked on every controlled experiment of the paper (16,437
  decisions: Bench-v2 seeds 42/123/777 under B5, IF-Core, IF-Hardened and IF-Confirm; B4; Disambiguation-v2;
  hard-benign; mixed provenance; AgentDojo-derived replay; adaptive attacks; endpoint ablation; SOTA
  re-implementations): no allow/block decision, executed flag, blocked argument or provenance origin
  changes. Unit tests: `tests/unit/test_provenance.py::TestExactIntentMatch`.
- Camera-ready results, regeneration scripts and `scripts/check_paper_numbers.py` (see README).
- Authors and links restored after double-blind review.
- `CITATION.cff`: authors, affiliations, ORCID iDs and license for the GitHub citation button and the
  Zenodo archive.

## icaart2027-submission (28 September 2026)

Anonymized artifact of the submitted version (commit `aec9b68`).
