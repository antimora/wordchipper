#![allow(missing_docs)]

use std::sync::{Arc, LazyLock};

use divan::{Bencher, black_box, counter::BytesCount};
use tiktoken_rs::CoreBPE;
use tokenizers::Tokenizer;
use wordchipper::{
    TokenEncoder,
    UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    pretrained::openai::OATokenizer,
};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

static CORPUS: &str = include_str!("data/corpus.txt");

fn diverse_text() -> String {
    CORPUS.repeat(10)
}

fn english_text() -> String {
    let paragraph = "The quick brown fox jumps over the lazy dog. \
        It's a beautiful day, and I'll be taking my 3 dogs for a walk. \
        Don't forget: the temperature is 72 degrees! \
        We've been waiting since 10:30am.\n\
        \n\
        In 2024, artificial intelligence continued to advance rapidly. \
        Large language models like GPT-4 and Claude demonstrated remarkable capabilities. \
        The researchers couldn't believe the results they'd achieved.\n";
    paragraph.repeat(100)
}

fn load_wc_encoder(model: OATokenizer) -> Arc<dyn TokenEncoder<u32>> {
    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: UnifiedTokenVocab<u32> = model.load_vocab(&mut disk_cache).unwrap();
    vocab.to_default_encoder()
}

struct TiktokenFixture {
    bpe: Arc<CoreBPE>,
}

static WC_CL100K: LazyLock<Arc<dyn TokenEncoder<u32>>> =
    LazyLock::new(|| load_wc_encoder(OATokenizer::Cl100kBase));

static WC_O200K: LazyLock<Arc<dyn TokenEncoder<u32>>> =
    LazyLock::new(|| load_wc_encoder(OATokenizer::O200kBase));

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

    mod wordchipper {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let encoder = &*WC_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let encoder = &*WC_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let bpe = &TT_CL100K.bpe;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.encode_with_special_tokens(black_box(&text)));
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let bpe = &TT_O200K.bpe;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.encode_with_special_tokens(black_box(&text)));
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = english_text();
            let tok = &*HF_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.encode(black_box(text.as_str()), true).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = english_text();
            let tok = &*HF_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.encode(black_box(text.as_str()), true).unwrap());
        }
    }
}

mod diverse {
    use super::*;

    mod wordchipper {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let encoder = &*WC_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let encoder = &*WC_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| encoder.try_encode(black_box(&text)).unwrap());
        }
    }

    mod tiktoken {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let bpe = &TT_CL100K.bpe;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.encode_with_special_tokens(black_box(&text)));
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let bpe = &TT_O200K.bpe;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| bpe.encode_with_special_tokens(black_box(&text)));
        }
    }

    mod tokenizers {
        use super::*;

        #[divan::bench]
        fn cl100k(bencher: Bencher) {
            let text = diverse_text();
            let tok = &*HF_CL100K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.encode(black_box(text.as_str()), true).unwrap());
        }

        #[divan::bench]
        fn o200k(bencher: Bencher) {
            let text = diverse_text();
            let tok = &*HF_O200K;
            bencher
                .counter(BytesCount::new(text.len()))
                .bench(|| tok.encode(black_box(text.as_str()), true).unwrap());
        }
    }
}
