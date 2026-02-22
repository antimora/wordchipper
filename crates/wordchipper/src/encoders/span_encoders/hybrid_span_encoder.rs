//! # Hybrid sweep/heap [`SpanEncoder`].
//!
//! Short spans use an inline linear sweep (low overhead, O(m*n)).
//! Long spans switch to a min-heap + linked-list algorithm (O(m log n)).

use core::cmp::Reverse;

use crate::{
    TokenType,
    alloc::{collections::BinaryHeap, vec::Vec},
    encoders::span_encoders::span_encoder::SpanEncoder,
    vocab::UnifiedTokenVocab,
};

/// Spans with at most this many bytes use the linear sweep path.
const SWEEP_THRESHOLD: usize = 16;

/// Sentinel value for "no neighbor" in the linked list.
const SENTINEL: u32 = u32::MAX;

/// A heap entry: (merge_rank, position, generation_at_push_time).
///
/// Wrapped in [`Reverse`] so the [`BinaryHeap`] acts as a min-heap by rank,
/// with ties broken by position (leftmost first).
type HeapEntry<T> = Reverse<(T, u32, u8)>;

/// A hybrid [`SpanEncoder`] that picks the best merge strategy per span.
///
/// Short spans (up to [`SWEEP_THRESHOLD`] bytes) use a simple linear sweep
/// with `Vec::remove`, matching [`super::IncrementalSweepSpanEncoder`].
///
/// Longer spans use a [`BinaryHeap`] for O(log n) min-finding and a
/// doubly-linked list for O(1) token removal, with lazy staleness detection
/// via per-position generation counters.
///
/// Working buffers are reused across calls to avoid repeated allocation.
#[derive(Default, Debug, Clone)]
pub struct HybridSpanEncoder<T: TokenType> {
    next: Vec<u32>,
    prev: Vec<u32>,
    generation: Vec<u8>,
    heap: BinaryHeap<HeapEntry<T>>,
}

impl<T: TokenType> HybridSpanEncoder<T> {
    /// Linear sweep merge, identical to [`super::IncrementalSweepSpanEncoder`].
    fn sweep(
        vocab: &UnifiedTokenVocab<T>,
        tokens: &mut Vec<T>,
        start: usize,
    ) {
        let stop = start + 2;
        while tokens.len() >= stop {
            if let Some((token, idx)) = tokens[start..]
                .windows(2)
                .enumerate()
                .filter_map(|(idx, w)| vocab.lookup_pair(&(w[0], w[1])).map(|token| (token, idx)))
                .min()
            {
                let idx = start + idx;
                tokens[idx] = token;
                tokens.remove(idx + 1);
            } else {
                break;
            }
        }
    }

    /// Heap-based merge for long spans.
    fn heap_merge(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        tokens: &mut Vec<T>,
        start: usize,
    ) {
        let n = tokens.len() - start;

        // Initialize linked-list arrays and generation counters.
        self.next.clear();
        self.next
            .extend((1..=n as u32).map(|i| if i < n as u32 { i } else { SENTINEL }));

        self.prev.clear();
        self.prev.push(SENTINEL);
        self.prev.extend(0..n as u32 - 1);

        self.generation.clear();
        self.generation.resize(n, 0);

        // Seed the heap with all adjacent pairs.
        self.heap.clear();
        let mut pos = 0u32;
        while self.next[pos as usize] != SENTINEL {
            let j = self.next[pos as usize];
            if let Some(rank) =
                vocab.lookup_pair(&(tokens[start + pos as usize], tokens[start + j as usize]))
            {
                self.heap.push(Reverse((rank, pos, 0)));
            }
            pos = j;
        }

        // Merge loop.
        while let Some(Reverse((rank, i, entry_gen))) = self.heap.pop() {
            let ii = i as usize;

            if entry_gen != self.generation[ii] {
                continue;
            }
            let j = self.next[ii];
            if j == SENTINEL {
                continue;
            }
            let jj = j as usize;

            tokens[start + ii] = rank;

            let k = self.next[jj];
            self.next[ii] = k;
            if k != SENTINEL {
                self.prev[k as usize] = i;
            }
            self.next[jj] = SENTINEL;

            self.generation[ii] = self.generation[ii].wrapping_add(1);

            let p = self.prev[ii];
            if p != SENTINEL {
                let pp = p as usize;
                self.generation[pp] = self.generation[pp].wrapping_add(1);
                if let Some(new_rank) =
                    vocab.lookup_pair(&(tokens[start + pp], tokens[start + ii]))
                {
                    self.heap
                        .push(Reverse((new_rank, p, self.generation[pp])));
                }
            }

            if k != SENTINEL {
                if let Some(new_rank) =
                    vocab.lookup_pair(&(tokens[start + ii], tokens[start + k as usize]))
                {
                    self.heap
                        .push(Reverse((new_rank, i, self.generation[ii])));
                }
            }
        }

        // Compact live tokens in-place by walking the linked list.
        let mut write = start;
        let mut pos = 0u32;
        loop {
            tokens[write] = tokens[start + pos as usize];
            write += 1;
            let nxt = self.next[pos as usize];
            if nxt == SENTINEL {
                break;
            }
            pos = nxt;
        }
        tokens.truncate(write);
    }
}

impl<T: TokenType> SpanEncoder<T> for HybridSpanEncoder<T> {
    fn encode_append_compound_span(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        let start = tokens.len();
        vocab.byte_vocab().append_tokens(span, tokens);

        let n = tokens.len() - start;
        if n <= 1 {
            return;
        }

        if n <= SWEEP_THRESHOLD {
            Self::sweep(vocab, tokens, start);
        } else {
            self.heap_merge(vocab, tokens, start);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TokenType,
        alloc::{boxed::Box, sync::Arc},
        encoders::{
            span_encoders::TokenSpanEncoder,
            testing::{common_encoder_test_vocab, common_encoder_tests},
        },
        spanning::TextSpannerBuilder,
    };

    fn test_encoder<T: TokenType>() {
        let vocab: Arc<UnifiedTokenVocab<T>> = common_encoder_test_vocab().into();
        let encoder = TokenSpanEncoder::<T>::new(
            TextSpannerBuilder::default(&vocab),
            vocab.clone(),
            Arc::new(|| Box::new(HybridSpanEncoder::<T>::default())),
        );
        common_encoder_tests(vocab.into(), encoder)
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
