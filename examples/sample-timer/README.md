# WordChipper Encode/Decode Side-by-Side Benchmarks

```terminaloutput
$ RAYON_NUM_THREADS=48 cargo run --release -p sample-timer -- \
    --dataset-dir $DATASET_CACHE_DIR --decode
Model: "openai/o200k_harmony"
- shards: [0, 1]
- batch_size: 1024

Samples Summary:
- num batches: 104
- avg bytes/sample: 4777
- avg bytes/token: 4.8

Encoder Batch Timing:
- "wordchipper"
  - batch:      37.1ms
  - sample:     36.3µs
  - bps:    125.68 MiB/s
- "tiktoken-rs"
  - batch:      37.2ms
  - sample:     36.3µs
  - bps:    125.57 MiB/s
- "tokenizers"
  - batch:     215.1ms
  - sample:    210.1µs
  - bps:    21.69 MiB/s
```
