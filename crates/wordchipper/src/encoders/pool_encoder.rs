//! # Parallel Encoder

use crate::concurrency::pool_toy::PoolToy;
use crate::encoders::TokenEncoder;
use crate::segmentation::TextSegmentor;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialVocab;
use std::num::NonZeroUsize;

/// Batch-Level Parallel Encoder Wrapper.
///
/// Enables ``rayon`` encoding of batches when available.
#[derive(Clone)]
pub struct PoolEncoder<T: TokenType, D: TokenEncoder<T>> {
    /// Inner token encoder.
    pub pool: PoolToy<D>,

    _marker: std::marker::PhantomData<T>,
}

impl<T, D> PoolEncoder<T, D>
where
    T: TokenType,
    D: TokenEncoder<T>,
{
    /// Create a new parallel encoder.
    ///
    /// ## Arguments
    /// * `inner` - The token encoder to wrap.
    ///
    /// ## Returns
    /// A new `ParallelRayonEncoder` instance.
    pub fn new(
        inner: D,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        let pool = PoolToy::init(inner, max_pool);
        Self {
            pool,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, D> TokenEncoder<T> for PoolEncoder<T, D>
where
    T: TokenType,
    D: TokenEncoder<T>,
{
    fn segmentor(&self) -> &TextSegmentor {
        self.pool.get().segmentor()
    }

    fn special_vocab(&self) -> &SpecialVocab<T> {
        self.pool.get().special_vocab()
    }

    fn try_encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) -> anyhow::Result<()> {
        self.pool.get().try_encode_append(text, tokens)
    }
}

#[cfg(test)]
mod tests {
    use crate::encoders::test_utils::{common_encoder_test_vocab, common_encoder_tests};
    use crate::encoders::{DefaultTokenEncoder, TokenEncoder};
    use crate::rayon::ParallelRayonEncoder;
    use crate::regex::RegexSupplier;
    use crate::types::TokenType;

    fn test_encoder<T: TokenType>() {
        let vocab = common_encoder_test_vocab();

        let encoder = DefaultTokenEncoder::<T>::init(vocab.clone().into());
        let encoder = ParallelRayonEncoder::new(encoder);

        assert_eq!(
            encoder.segmentor().span_re().get_pattern().as_str(),
            vocab.segmentation.pattern.as_str()
        );
        assert_eq!(encoder.special_vocab(), encoder.inner.special_vocab());

        common_encoder_tests(vocab, &encoder)
    }

    #[test]
    fn test_encoder_u16() {
        test_encoder::<u16>();
    }

    #[test]
    fn test_encoder_u32() {
        test_encoder::<u32>();
    }
}
