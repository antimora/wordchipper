#![allow(missing_docs)]

use std::sync::{Arc, LazyLock};

use divan::{Bencher, black_box, counter::BytesCount};
use tiktoken_rs::CoreBPE;
use tokenizers::Tokenizer;
use wordchipper::{
    TokenEncoder,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    encoders::span_encoders::{
        IncrementalSweepSpanEncoder,
        MergeHeapSpanEncoder,
        PriorityMergeSpanEncoder,
        SpanEncoder,
        TokenSpanEncoder,
    },
    pretrained::openai::OATokenizer,
    spanning::TextSpannerBuilder,
    support::concurrency::rayon::ParallelRayonEncoder,
};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

static DIVERSE_CORPUS: &str = include_str!("data/multilingual.txt");
static ENGLISH_CORPUS: &str = include_str!("data/english.txt");

struct Batch {
    lines: Vec<String>,
    total_bytes: usize,
}

impl Batch {
    fn strs(&self) -> Vec<&str> {
        self.lines.iter().map(|s| s.as_str()).collect()
    }
}

fn english_batch() -> Batch {
    let lines: Vec<String> = ENGLISH_CORPUS.lines().map(String::from).collect();
    let total_bytes = lines.iter().map(|s| s.len()).sum();
    Batch { lines, total_bytes }
}

fn diverse_batch() -> Batch {
    let lines: Vec<String> = DIVERSE_CORPUS.lines().map(String::from).collect();
    let total_bytes = lines.iter().map(|s| s.len()).sum();
    Batch { lines, total_bytes }
}

fn load_wc_variant(
    model: OATokenizer,
    se_builder: Arc<dyn Fn() -> Box<dyn SpanEncoder<u32>> + Send + Sync>,
) -> Arc<dyn TokenEncoder<u32>> {
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: Arc<UnifiedTokenVocab<u32>> =
        model.load_vocab::<u32>(&mut disk_cache).unwrap().into();
    let spanner = TextSpannerBuilder::from_vocab(&vocab).build();
    let inner: Arc<dyn TokenEncoder<u32>> =
        Arc::new(TokenSpanEncoder::<u32>::new(spanner, vocab, se_builder));
    Arc::new(ParallelRayonEncoder::new(inner))
}

// cl100k variants
static WC_SWEEP_CL100K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::Cl100kBase,
        Arc::new(|| Box::new(IncrementalSweepSpanEncoder::<u32>::default())),
    )
});
static WC_HEAP_CL100K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::Cl100kBase,
        Arc::new(|| Box::new(MergeHeapSpanEncoder::<u32>::default())),
    )
});
static WC_PMERGE_CL100K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::Cl100kBase,
        Arc::new(|| Box::new(PriorityMergeSpanEncoder::<u32>::default())),
    )
});

// o200k variants
static WC_SWEEP_O200K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::O200kBase,
        Arc::new(|| Box::new(IncrementalSweepSpanEncoder::<u32>::default())),
    )
});
static WC_HEAP_O200K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::O200kBase,
        Arc::new(|| Box::new(MergeHeapSpanEncoder::<u32>::default())),
    )
});
static WC_PMERGE_O200K: LazyLock<Arc<dyn TokenEncoder<u32>>> = LazyLock::new(|| {
    load_wc_variant(
        OATokenizer::O200kBase,
        Arc::new(|| Box::new(PriorityMergeSpanEncoder::<u32>::default())),
    )
});

struct TiktokenFixture {
    bpe: Arc<CoreBPE>,
}

static TT_CL100K: LazyLock<TiktokenFixture> = LazyLock::new(|| TiktokenFixture {
    bpe: Arc::new(tiktoken_rs::cl100k_base().unwrap()),
});
static TT_O200K: LazyLock<TiktokenFixture> = LazyLock::new(|| TiktokenFixture {
    bpe: Arc::new(tiktoken_rs::o200k_base().unwrap()),
});

static HF_CL100K: LazyLock<Arc<Tokenizer>> = LazyLock::new(|| {
    Arc::new(Tokenizer::from_pretrained("Xenova/text-embedding-ada-002", None).unwrap())
});
static HF_O200K: LazyLock<Arc<Tokenizer>> =
    LazyLock::new(|| Arc::new(Tokenizer::from_pretrained("Xenova/gpt-4o", None).unwrap()));

mod english {
    use super::*;

    mod incremental_sweep {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let batch = english_batch();
            let strs = batch.strs();
            let encoder = &*WC_SWEEP_CL100K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let batch = english_batch();
            let strs = batch.strs();
            let encoder = &*WC_SWEEP_O200K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }
    }

    mod merge_heap {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let batch = english_batch();
            let strs = batch.strs();
            let encoder = &*WC_HEAP_CL100K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let batch = english_batch();
            let strs = batch.strs();
            let encoder = &*WC_HEAP_O200K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }
    }

    mod priority_merge {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let batch = english_batch();
            let strs = batch.strs();
            let encoder = &*WC_PMERGE_CL100K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let batch = english_batch();
            let strs = batch.strs();
            let encoder = &*WC_PMERGE_O200K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let batch = english_batch();
            let bpe = &TT_CL100K.bpe;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| {
                    for s in &batch.lines {
                        black_box(bpe.encode_with_special_tokens(s));
                    }
                });
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let batch = english_batch();
            let bpe = &TT_O200K.bpe;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| {
                    for s in &batch.lines {
                        black_box(bpe.encode_with_special_tokens(s));
                    }
                });
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let batch = english_batch();
            let tok = &*HF_CL100K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| {
                    for s in &batch.lines {
                        black_box(tok.encode(s.as_str(), true).unwrap());
                    }
                });
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let batch = english_batch();
            let tok = &*HF_O200K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| {
                    for s in &batch.lines {
                        black_box(tok.encode(s.as_str(), true).unwrap());
                    }
                });
        }
    }
}

mod diverse {
    use super::*;

    mod incremental_sweep {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let batch = diverse_batch();
            let strs = batch.strs();
            let encoder = &*WC_SWEEP_CL100K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let batch = diverse_batch();
            let strs = batch.strs();
            let encoder = &*WC_SWEEP_O200K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }
    }

    mod merge_heap {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let batch = diverse_batch();
            let strs = batch.strs();
            let encoder = &*WC_HEAP_CL100K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let batch = diverse_batch();
            let strs = batch.strs();
            let encoder = &*WC_HEAP_O200K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }
    }

    mod priority_merge {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let batch = diverse_batch();
            let strs = batch.strs();
            let encoder = &*WC_PMERGE_CL100K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let batch = diverse_batch();
            let strs = batch.strs();
            let encoder = &*WC_PMERGE_O200K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| encoder.try_encode_batch(black_box(&strs)).unwrap());
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let batch = diverse_batch();
            let bpe = &TT_CL100K.bpe;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| {
                    for s in &batch.lines {
                        black_box(bpe.encode_with_special_tokens(s));
                    }
                });
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let batch = diverse_batch();
            let bpe = &TT_O200K.bpe;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| {
                    for s in &batch.lines {
                        black_box(bpe.encode_with_special_tokens(s));
                    }
                });
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let batch = diverse_batch();
            let tok = &*HF_CL100K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| {
                    for s in &batch.lines {
                        black_box(tok.encode(s.as_str(), true).unwrap());
                    }
                });
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let batch = diverse_batch();
            let tok = &*HF_O200K;
            bencher
                .counter(BytesCount::new(batch.total_bytes))
                .bench(|| {
                    for s in &batch.lines {
                        black_box(tok.encode(s.as_str(), true).unwrap());
                    }
                });
        }
    }
}
