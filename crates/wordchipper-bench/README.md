# wordchipper-bench

Benchmarks comparing wordchipper's `SpanEncoder` variants against each other and against tiktoken-rs
and HuggingFace tokenizers.

## Benchmarks

| Bench               | What it measures                              |
| ------------------- | --------------------------------------------- |
| `encoding_single`   | Single-string encoding (no parallelism)       |
| `encoding_parallel` | Batch encoding via rayon (`try_encode_batch`) |
| `decoding_single`   | Single-string decoding                        |
| `spanning`          | Text spanning (regex vs logos DFA)            |

### Encoder Variants

- **`incremental_sweep`** - O(n^2) linear-scan BPE merge (current default)
- **`merge_heap`** - O(n^2) with parallel pair-rank tracking
- **`priority_merge`** - O(n log n) binary min-heap with doubly-linked list

## Running

```bash
# All benchmarks
cargo bench -p wordchipper-bench

# Individual benchmarks
cargo bench -p wordchipper-bench --bench encoding_single
cargo bench -p wordchipper-bench --bench encoding_parallel
cargo bench -p wordchipper-bench --bench decoding_single
cargo bench -p wordchipper-bench --bench spanning

# Filter by name
cargo bench -p wordchipper-bench --bench encoding_single -- diverse
cargo bench -p wordchipper-bench --bench encoding_parallel -- priority_merge
```

## Results

Collected on Apple M4 Pro. Corpus: `english.txt` (~7 KB x10) and `multilingual.txt` (~9 KB x10).

### Single-String Encoding (median MB/s)

| Encoder            | diverse cl100k | diverse o200k | english cl100k | english o200k |
| ------------------ | -------------- | ------------- | -------------- | ------------- |
| incremental_sweep  | 57             | 28            | 89             | 83            |
| merge_heap         | 63             | 38            | 100            | 98            |
| **priority_merge** | **94**         | **85**        | **130**        | **124**       |
| tiktoken-rs        | 11             | 11            | 11             | 11            |
| HF tokenizers      | 7              | 7             | 7              | 7             |

### Parallel Batch Encoding (median MB/s)

| Encoder            | diverse cl100k | diverse o200k | english cl100k | english o200k |
| ------------------ | -------------- | ------------- | -------------- | ------------- |
| incremental_sweep  | 57             | 29            | 95             | 89            |
| merge_heap         | 88             | 46            | 100            | 97            |
| **priority_merge** | **108**        | **95**        | **99**         | **96**        |
| tiktoken-rs        | 11             | 11            | 11             | 11            |
| HF tokenizers      | 6              | 6             | 6              | 6             |

The `priority_merge` encoder is 3x faster than `incremental_sweep` on diverse/multilingual text with
o200k, where longer multi-byte spans expose the O(n^2) vs O(n log n) gap.
