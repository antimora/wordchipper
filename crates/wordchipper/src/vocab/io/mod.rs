//! # Vocabulary IO
//!
//! ## Loading A Vocab
//!
//! ```rust,no_run
//! use wordchipper::{
//!     decoders::DefaultTokenDecoder,
//!     encoders::DefaultTokenEncoder,
//!     pretrained::openai::patterns::OA_GPT5_O220K_WORD_PATTERN,
//!     spanning::TextSpanningConfig,
//!     vocab::{SpanMapVocab, SpanTokenMap, UnifiedTokenVocab, io::load_tiktoken_vocab_path},
//! };
//!
//! fn example() -> anyhow::Result<(DefaultTokenEncoder<u32>, DefaultTokenDecoder<u32>)> {
//!     type T = u32;
//!     let text_config = TextSpanningConfig::from_pattern(OA_GPT5_O220K_WORD_PATTERN);
//!     let span_map: SpanTokenMap<T> = load_tiktoken_vocab_path("vocab.tiktoken")?;
//!     let span_vocab: SpanMapVocab<T> = span_map.into();
//!
//!     let vocab: UnifiedTokenVocab<T> =
//!         UnifiedTokenVocab::from_span_vocab(text_config, span_vocab);
//!
//!     let encoder: DefaultTokenEncoder<T> = DefaultTokenEncoder::new(vocab.clone(), None);
//!     let decoder: DefaultTokenDecoder<T> = DefaultTokenDecoder::from_unified_vocab(vocab);
//!
//!     Ok((encoder, decoder))
//! }
//! ```

mod base64_vocab;
mod tiktoken_io;

#[doc(inline)]
pub use base64_vocab::{
    load_base64_span_map_path,
    read_base64_span_map,
    save_base64_span_map_path,
    write_base64_span_map,
};
#[doc(inline)]
pub use tiktoken_io::{
    load_tiktoken_vocab_path,
    read_tiktoken_vocab,
    save_tiktoken_vocab_path,
    write_tiktoken_vocab,
};
