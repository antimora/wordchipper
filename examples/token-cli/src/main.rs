use crate::tokenizer_timer::FullMontyTokenizer;
use arrow::array::StringArray;
use clap::Parser;
use rayon::prelude::*;
use similar::{ChangeTag, TextDiff};
use std::time::Duration;
use wordchipper::decoders::{DictionaryDecoder, TokenDecoder};
use wordchipper::disk_cache::WordchipperDiskCache;
use wordchipper::encoders::{DefaultTokenEncoder, TokenEncoder};
use wordchipper::rayon::{ParallelRayonDecoder, ParallelRayonEncoder};
use wordchipper::segmentation::TextSegmentor;
use wordchipper::vocab::UnifiedTokenVocab;
use wordchipper::vocab::public::openai::load_o200k_harmony_vocab;
use wordchipper_data::dataset::DatasetCacheConfig;

mod tokenizer_timer;

fn timeit<F, R>(f: F) -> (Duration, R)
where
    F: FnOnce() -> R,
{
    let t0 = std::time::Instant::now();
    let ret = f();
    let t1 = std::time::Instant::now();
    (t1 - t0, ret)
}

/// Example encoders trainer.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// Enable verbose output.
    #[arg(long, default_value = "false")]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Parser, Debug)]
pub enum Command {
    /// Load a tokenizer.
    Load {},
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.verbose {
        println!("{:#?}", args);
    }

    match &args.command {
        Some(Command::Load { .. }) => {
            run_load(&args)?;
        }
        None => unreachable!(),
    }

    Ok(())
}

#[allow(unused)]
fn run_load(args: &Args) -> anyhow::Result<()> {
    type T = u32;

    let mut dataset_cache = DatasetCacheConfig::new()
        .with_cache_dir(args.dataset_dir.clone())
        .init()?;

    let tt_bpe = tiktoken_rs::o200k_harmony()?;

    let mut disk_cache = WordchipperDiskCache::default();
    let vocab: UnifiedTokenVocab<T> = load_o200k_harmony_vocab(&mut disk_cache)?;

    let wc_tokenizer = FullMontyTokenizer::init(
        ParallelRayonEncoder::new(DefaultTokenEncoder::<T>::init(vocab.clone())),
        ParallelRayonDecoder::new(DictionaryDecoder::from_unified_vocab(vocab.clone())),
    );

    let shards: Vec<usize> = vec![0];
    let num_timing_batches = 20;
    let batch_size = 512;

    println!("Loading Shards: {shards:?}");
    println!("...");
    dataset_cache.load_shards(&shards)?;

    let mut samples = Vec::new();
    {
        for batch in dataset_cache
            .read_cached_batches(shards[0])?
            .take(num_timing_batches)
        {
            let batch = batch?;
            let column = batch
                .column_by_name("text")
                .expect("failed to find 'text' column in batch")
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            for val in column {
                let val = val.unwrap().to_string();
                samples.push(val);
            }
        }
    }

    println!();
    println!("Samples Summary:");
    let sample_count = samples.len();
    println!("- count: {}", sample_count);
    let total_sample_bytes = samples.iter().map(|s| s.len()).sum::<usize>();
    println!("- total size: {}", total_sample_bytes);
    let avg_sample_size = total_sample_bytes / sample_count;
    println!("- avg size: {avg_sample_size}");

    let sample_batches: Vec<&[String]> = samples.chunks(batch_size).collect::<Vec<_>>();
    let num_batches = sample_batches.len();

    let avg_batch_size_bytes = total_sample_bytes / num_batches;
    println!("- avg batch size bytes: {avg_batch_size_bytes}");

    println!();
    println!("Timing Config:");
    println!("- batch size: {}", batch_size);
    println!("- num batches: {}", num_batches);

    println!();
    println!("Timing Encode:");
    let mut wc_token_batches: Vec<Vec<Vec<T>>> = Default::default();
    let mut wc_total_token_count = 0;
    let mut tt_total_token_count = 0;
    let mut wc_batch_durations = vec![];
    let mut tt_batch_durations = vec![];
    for (idx, batch) in sample_batches.iter().enumerate() {
        let batch = batch.iter().map(|s| s.as_str()).collect::<Vec<_>>();

        let (durationn, wc_batch_tokens) = timeit(|| {
            if true {
                wc_tokenizer.encoder.try_encode_batch(&batch).unwrap()
            } else {
                batch
                    .par_iter()
                    .map(|s| wc_tokenizer.encoder.try_encode(s))
                    .collect::<anyhow::Result<Vec<Vec<T>>>>()
                    .unwrap()
            }
        });
        wc_batch_durations.push(durationn);

        wc_total_token_count += wc_batch_tokens
            .iter()
            .map(|tokens| tokens.len())
            .sum::<usize>();

        {
            let (duration, tt_batch_tokens) = timeit(|| {
                batch
                    .par_iter()
                    .map(|s| tt_bpe.encode_with_special_tokens(s))
                    .collect::<Vec<_>>()
            });
            tt_batch_durations.push(duration);

            tt_total_token_count += tt_batch_tokens
                .iter()
                .map(|tokens| tokens.len())
                .sum::<usize>();
        }

        wc_token_batches.push(wc_batch_tokens);
    }

    for (name, durations) in [
        ("wordchipper", &wc_batch_durations),
        ("tiktoken-rs", &tt_batch_durations),
    ] {
        let mean_time = durations.iter().sum::<Duration>() / num_batches as u32;
        let bps = avg_batch_size_bytes as f64 / mean_time.as_secs_f64();

        println!("- {name}:\t{bps:.1e}b/s, {mean_time:10.1?}");
    }

    println!();
    println!("Observed Bytes/Token Stats:");
    for (name, token_count) in [
        ("wordchipper", wc_total_token_count),
        ("tiktoken-rs", tt_total_token_count),
    ] {
        println!("- {name} token count: {}", token_count);
        println!(
            "- {name} byte/token: {:.2}",
            total_sample_bytes as f64 / token_count as f64
        );
    }

    println!();
    println!("Timing Decode:");

    let segmentor: TextSegmentor = TextSegmentor::from_config(vocab.segmentation.clone());

    let mut wc_batch_decode_durations = vec![];
    let mut tt_batch_decode_durations = vec![];
    for (idx, sample) in sample_batches.iter().enumerate() {
        let batch = &wc_token_batches[idx];

        let expected = sample
            .iter()
            .map(|s| segmentor.rewrite(s))
            .collect::<Vec<_>>();

        {
            let (duration, wc_decoded) = timeit(|| {
                wc_tokenizer
                    .decoder
                    .try_decode_batch_to_strings(batch)
                    .unwrap()
            });
            wc_batch_decode_durations.push(duration);

            verify_decode(&expected, &wc_decoded);
        }

        {
            let (duration, tt_decoded) = timeit(|| {
                batch
                    .par_iter()
                    .map(|tokens| tt_bpe.decode(tokens.clone()).unwrap())
                    .collect::<Vec<_>>()
            });

            tt_batch_decode_durations.push(duration);

            verify_decode(&expected, &tt_decoded);
        }
    }

    for (name, durations) in [
        ("wordchipper", &wc_batch_decode_durations),
        ("tiktoken-rs", &tt_batch_decode_durations),
    ] {
        let mean_time = durations.iter().sum::<Duration>() / num_batches as u32;
        println!("- {name}: batch {mean_time:10.1?}");
    }

    Ok(())
}

pub fn verify_decode(
    samples: &[String],
    decoded: &[String],
) {
    for (s, d) in samples.iter().zip(decoded.iter()) {
        if s != d {
            let diff = TextDiff::from_lines(s, d);

            for change in diff.iter_all_changes() {
                let sign = match change.tag() {
                    ChangeTag::Delete => "-",
                    ChangeTag::Insert => "+",
                    ChangeTag::Equal => " ",
                };
                print!("{}{}", sign, change);
            }
            panic!("MISMATCH");
        }
    }
}

/*
pub fn batch_score(
    actual: &[String],
    expected: &[String],
) -> f64 {
    score_batch(actual, expected).iter().sum::<f64>() / actual.len() as f64
}

pub fn score_batch(
    actual: &[String],
    expected: &[String],
) -> Vec<f64> {
    use rayon::prelude::*;
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected.iter())
        .collect::<Vec<_>>()
        .par_iter()
        .map(|(a, e)| edit_score(a, e))
        .collect::<Vec<_>>()
}

pub fn edit_score(
    actual: &str,
    expected: &str,
) -> f64 {
    let distance = edit_distance(actual, expected);
    let size = expected.len();

    (size as isize - distance as isize).abs() as f64 / (size as f64)
}
*/
