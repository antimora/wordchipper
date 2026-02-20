//! # Span Lexer Trait

/// Trait for finding the next occurrence of a span.
///
/// ## Implementation Notes
///
/// This trait is typically implemented on concrete types like [`RegexWrapper`](crate::regex::RegexWrapper).
///
/// Smart pointer types that implement `Deref<Target: SpanLexer>` (such as `Arc<T>`, `Box<T>`,
/// and [`PoolToy<T>`](crate::concurrency::PoolToy)) automatically implement `SpanLexer` through
/// a blanket implementation. This is the idiomatic Rust pattern used by the standard library
/// for traits like `Iterator` and `Future`.
pub trait SpanLexer: Send + Sync {
    /// Find the next occurrence of a span.
    ///
    /// ## Arguments
    /// * `text` - the text to scan over.
    /// * `offset` - the offset to start scanning from.
    ///
    /// ## Returns
    /// The span bounds, if found, relative to `text`.
    fn next_span(
        &self,
        text: &str,
        offset: usize,
    ) -> Option<(usize, usize)>;
}

// Blanket implementation for any type that derefs to a SpanLexer.
// This allows Arc<T>, Box<T>, PoolToy<T>, etc. to automatically implement SpanLexer.
impl<D> SpanLexer for D
where
    D: core::ops::Deref + Send + Sync,
    D::Target: SpanLexer,
{
    fn next_span(
        &self,
        text: &str,
        offset: usize,
    ) -> Option<(usize, usize)> {
        self.deref().next_span(text, offset)
    }
}
