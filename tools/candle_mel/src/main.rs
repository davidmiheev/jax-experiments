//! candle-mel PCM_F32LE FILTERS_F32LE N_MEL OUT_F32LE
//!
//! Reads 16 kHz mono f32 samples and an [n_mel, 201] f32 filter bank (raw little-endian),
//! runs candle's `pcm_to_mel` exactly as candle's whisper example does, and writes the
//! [n_mel, n_frames] log-mel features as raw little-endian f32.

/// The parts of candle_transformers::models::whisper that audio.rs refers to (`super::...`).
#[allow(dead_code)]
mod whisper {
    pub const SAMPLE_RATE: usize = 16000;
    pub const N_FFT: usize = 400;
    pub const HOP_LENGTH: usize = 160;
    pub const CHUNK_LENGTH: usize = 30;
    pub struct Config {
        pub num_mel_bins: usize,
    }
    pub mod audio {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/vendor/audio.rs"));
    }
}

fn read_f32(path: &str) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 5, "usage: candle-mel PCM FILTERS N_MEL OUT");
    let pcm = read_f32(&args[1])?;
    let filters = read_f32(&args[2])?;
    let config = whisper::Config { num_mel_bins: args[3].parse()? };
    let mel = whisper::audio::pcm_to_mel(&config, &pcm, &filters);
    let bytes: Vec<u8> = mel.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&args[4], bytes)?;
    eprintln!("{} samples -> {} mel values ({} frames)", pcm.len(), mel.len(), mel.len() / config.num_mel_bins);
    Ok(())
}
