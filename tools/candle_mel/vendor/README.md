# vendor/audio.rs

`audio.rs` is candle's Whisper front end, copied **byte for byte** from
`candle-transformers/src/models/whisper/audio.rs` in
[davidmiheev/candle](https://github.com/davidmiheev/candle) at commit
`9aa02f1069b7fd9ad018bd12cdca1024dcede579`: the commit ml-ladder's I-part
Whisper runs used. SHA-256
`ed5f1c81a0bebf0e67f8066f51d87cf8312a48aadb3a9a3a126ab7c09714880b`
(`experiments/o7_mel/run.py` checks it before use).

Why vendored, not depended on: at that commit `candle-transformers` does not
compile without CUDA (gemma4, qwen3_5 and unlimited_ocr import the CUDA-only
`candle_nn::fused::static_decode`). `audio.rs` itself needs only `candle-core`
(for `utils::get_num_threads`) and `num-traits`, so `src/main.rs` compiles the
unmodified file with `include!` inside a minimal `whisper` module.

The file is candle's (upstream: huggingface/candle, "adapted from whisper.cpp"),
under MIT OR Apache-2.0; `LICENSE-MIT` is the fork's copy.
