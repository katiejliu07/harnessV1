This half-precision Metal pilot is preserved for debugging. It generated
nonsensical text for every condition, although ordinary/no-op/alpha-zero and
cached/uncached paths agreed on the short parity fixture. **Do not use its
timings, correctness, or token lengths to evaluate activation patching.**
The runner now also requires unpatched semantic sanity and rejects nonfinite
logits. The initial run preceded these additional prerequisites.
