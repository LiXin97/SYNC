"""B0 standard-benchmark harness (the study protocol §5, first 48h).

Modules
-------
``build_bench``   OpenBookQA / SciQ / ARC-Challenge -> our episode JSONL schema,
                  with the §1.2 leakage screens computed and logged per item.
``run_headroom``  the G0 headroom probe (§4 "Dev gates"): packet-free, two arms
                  (no_comm, skyline), letter-logit forced-choice readout.

Nothing in this package touches the packet machinery: G0 must be measurable
before any injection cell is paid for (§4, "[non-English note removed] receiver [non-English note removed]; [non-English note removed] ->
[non-English note removed] benchmark [non-English note removed], confirmatory [non-English note removed]").
"""
