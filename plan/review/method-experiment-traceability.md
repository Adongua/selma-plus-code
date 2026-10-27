# Method–Experiment Traceability

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| Ruler-sReLU | `s-vit/srelu/protocol.py` | Three-party CUDA operator microbenchmark | Planned sReLU operator table | Online cost and approximation error under the reported caveats | Server rerun required |
| Baseline fairness | `s-vit/srelu/baselines.py` | Correctness-gated baseline comparison | Same operator table | Only PASS rows may support efficiency claims | Implemented locally |
| Reduced communication | Ring3 counters in `benchmark_crypten_cuda.py` | Same-shape sent-payload comparison | Same operator table | Communication reduction, if supported by real logs | Previous logs exist; gated rerun required |
| hAttention integration | `s-vit/hattention/` | Full RSA/LSA/Mixer benchmark | Future attention table | End-to-end attention improvement | Not yet implemented |
