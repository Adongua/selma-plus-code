# Table Schema

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| sReLU operator microbenchmark | Compare valid three-party operator implementations | CrypTen-compatible full-width, adapted Bicoptor2, adapted HummingBird, sReLU | mean time, sent KB, rounds, MAE, max error, correctness gate | `comparison.json` formal non-warmup records | sReLU benchmark script |

Aggregation uses the arithmetic mean over measured repeats.  A row that fails
its declared-reference gate must display `INVALID` and must not expose a timing
value in the publication table.  Standard deviation will be added when the
final result table is generated from server logs.
