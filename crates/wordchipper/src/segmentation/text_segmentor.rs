//! # Text Segmentor

use crate::alloc::string::String;
use crate::alloc::vec::Vec;
use crate::regex::exact_match_union::exact_match_union_regex_pattern;
use crate::regex::{RegexSupplier, RegexWrapper, RegexWrapperPattern};
use crate::segmentation::segmentation_config::SegmentationConfig;
use crate::types::TokenType;
use crate::vocab::TokenVocab;
use crate::vocab::size_hints::EXPECTED_BYTES_PER_TOKEN;
use core::ops::Range;

/// Word Reference for [`TextSegmentor`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpanRef {
    /// A normal word reference.
    Normal(Range<usize>),

    /// A special word reference.
    Special(Range<usize>),
}

impl From<SpanRef> for Range<usize> {
    fn from(span: SpanRef) -> Self {
        match span {
            SpanRef::Normal(range) => range,
            SpanRef::Special(range) => range,
        }
    }
}

fn offset_range(
    range: Range<usize>,
    offset: usize,
) -> Range<usize> {
    Range {
        start: range.start + offset,
        end: range.end + offset,
    }
}

/// Word Split + Special Words Segmentor
#[derive(Clone)]
pub struct TextSegmentor {
    /// Regex for splitting words.
    pub span_re: RegexWrapper,

    /// Regex for matching special words.
    pub special_re: Option<RegexWrapper>,
}

impl TextSegmentor {
    /// Create a new text segmentor from the given configuration.
    ///
    /// ## Arguments
    /// * `config` - The segmentation configuration.
    ///
    /// ## Returns
    /// A new `TextSegmentor` instance.
    pub fn from_config<T>(config: SegmentationConfig<T>) -> Self
    where
        T: TokenType,
    {
        let specials = config
            .special_vocab()
            .span_pairs()
            .map(|(span, _)| String::from_utf8(span.clone()).unwrap())
            .collect::<Vec<_>>();

        Self::init(config.pattern, &specials)
    }

    /// Create a new text segmentor with the given regex pattern and special words.
    ///
    /// ## Arguments
    /// * `pattern` - The word split pattern.
    /// * `specials` - A slice of special word strings.
    ///
    /// ## Returns
    /// A new `TextSegmentor` instance.
    pub fn init<P, S>(
        pattern: P,
        specials: &[S],
    ) -> Self
    where
        P: Into<RegexWrapperPattern>,
        S: AsRef<str>,
    {
        let span_re = pattern.into().compile().unwrap();

        let special_re = if specials.is_empty() {
            None
        } else {
            Some(exact_match_union_regex_pattern(specials).compile().unwrap())
        };

        Self::new(span_re, special_re)
    }

    /// Create a new text segmentor with the given regex suppliers.
    ///
    /// ## Arguments
    /// * `word_re` - The regex for word splitting.
    /// * `special_re` - The optional regex for special word matching.
    ///
    /// ## Returns
    /// A new `TextSegmentor` instance.
    pub fn new(
        span_re: RegexWrapper,
        special_re: Option<RegexWrapper>,
    ) -> Self {
        /*
        let span_re = RegexWrapperPool::from(span_r);
        let special_re = special_re.map(RegexWrapperPool::from);
         */

        Self {
            span_re,
            special_re,
        }
    }

    /// Get the span split regex.
    pub fn span_re(&self) -> &RegexWrapper {
        self.span_re.get_regex()
    }

    /// Get the optional special split regex.
    pub fn special_re(&self) -> Option<&RegexWrapper> {
        match &self.special_re {
            None => None,
            Some(special_re) => Some(special_re.get_regex()),
        }
    }

    /// Find the next special span in the text.
    ///
    /// ## Arguments
    /// * `text` - The text to search in.
    ///
    /// ## Returns
    /// * `Some(Range<usize>)` if a special span is found,
    /// * `None` otherwise.
    pub fn next_special_span<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Option<Range<usize>> {
        match self.special_re {
            None => None,
            Some(ref special_re) => {
                let mut iter = special_re.get_regex().find_iter(text.as_ref());
                iter.next().map(|m| m.range())
            }
        }
    }

    /// Split a chunk of text into [`SpanRef::Normal`], appending to the `words` buffer.
    ///
    /// ## Arguments
    /// * `text` - The text to split.
    /// * `words` - The target buffer to append to.
    fn split_append_normal_words(
        &self,
        text: &str,
        offset: usize,
        words: &mut Vec<SpanRef>,
    ) {
        words.extend(
            self.span_re
                .get_regex()
                .find_iter(text)
                .map(|m| SpanRef::Normal(offset_range(m.range(), offset))),
        )
    }

    /// Split a chunk of text into spans, appending to the `words` buffer.
    ///
    /// ## Arguments
    /// * `text` - The text to split.
    /// * `words` - The target buffer to append to.
    pub fn split_append_spans(
        &self,
        text: &str,
        words: &mut Vec<SpanRef>,
    ) {
        let mut current = text;
        let mut offset = 0;

        while let Some(range) = self.next_special_span(current) {
            let pre = &current[..range.start];
            self.split_append_normal_words(pre, offset, words);

            words.push(SpanRef::Special(offset_range(range.clone(), offset)));

            current = &current[range.end..];
            offset += range.end;
        }

        if !current.is_empty() {
            self.split_append_normal_words(current, offset, words);
        }
    }

    /// Split text into spans.
    ///
    /// ## Arguments
    /// * `text` - The text to split.
    ///
    /// ## Returns
    /// A vector of `SpanRef` items.
    pub fn split_spans(
        &self,
        text: &str,
    ) -> Vec<SpanRef> {
        let capacity = text.len() as f64 / (EXPECTED_BYTES_PER_TOKEN * 0.8);
        let mut words = Vec::with_capacity(capacity as usize);

        self.split_append_spans(text, &mut words);
        words
    }

    /// Rewrite text by splitting and re-joining with spaces.
    ///
    /// ## Arguments
    /// * `text` - The text to rewrite.
    ///
    /// ## Returns
    /// The rewritten string.
    pub fn rewrite<S: AsRef<str>>(
        &self,
        text: S,
    ) -> String {
        let text = text.as_ref();
        let mut words = Vec::new();
        self.split_append_spans(text, &mut words);
        words
            .into_iter()
            .map(|w| &text[Range::<usize>::from(w)])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::vec;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;

    #[test]
    fn test_split_words() {
        type T = u32;

        let config: SegmentationConfig<T> =
            SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN)
                .with_special_words([("<|FNORD|>", 4000), ("<|NORP|>", 4001)]);

        let segmentor = TextSegmentor::from_config(config);

        let buf = "hello<|FNORD|> wor<|NORP|>ld!";

        assert_eq!(
            &segmentor.split_spans(buf),
            &vec![
                SpanRef::Normal(0..5),
                SpanRef::Special(5..14),
                SpanRef::Normal(14..18),
                SpanRef::Special(18..26),
                SpanRef::Normal(26..28),
                SpanRef::Normal(28..buf.len()),
            ]
        );
    }

    #[test]
    fn test_rewrite() {
        type T = u32;

        let config: SegmentationConfig<T> = SegmentationConfig::from_pattern(r"\w+");

        let segmentor = TextSegmentor::from_config(config);

        let buf = "hello world!";
        assert_eq!(segmentor.rewrite(buf), "helloworld");
    }
}
