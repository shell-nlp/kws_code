import argparse
import sys
from pathlib import Path
import sounddevice as sd
import sherpa_onnx
import numpy as np
import sys

sys.path.append(__file__)

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="../sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt",
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default="../sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        help="Path to the transducer decoder model",
        default="../sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        help="Path to the transducer joiner model",
        default="../sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=4,
        help="""
        It specifies number of active paths to keep during decoding.
        """,
    )

    parser.add_argument(
        "--num-trailing-blanks",
        type=int,
        default=1,
        help="""The number of trailing blanks a keyword should be followed. Setting
        to a larger value (e.g. 8) when your keywords has overlapping tokens
        between each other.
        """,
    )

    parser.add_argument(
        "--keywords-file",
        type=str,
        help="""
        The file containing keywords, one words/phrases per line, and for each
        phrase the bpe/cjkchar/pinyin are separated by a space. For example:

        ▁HE LL O ▁WORLD
        x iǎo ài t óng x ué 
        """,
        default="../sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt",
    )

    parser.add_argument(
        "--keywords-score",
        type=float,
        default=1.0,
        help="""
        The boosting score of each token for keywords. The larger the easier to
        survive beam search.
        """,
    )

    parser.add_argument(
        "--keywords-threshold",
        type=float,
        default=0.25,
        help="""
        The trigger threshold (i.e. probability) of the keyword. The larger the
        harder to trigger.
        """,
    )
    parser.add_argument(
        "--silero-vad-model",
        type=str,
        default="../silero_vad.onnx",
        help="Path to silero_vad.onnx",
    )

    return parser.parse_args()


def main():
    args = get_args()

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    assert Path(
        args.keywords_file
    ).is_file(), (
        f"keywords_file : {args.keywords_file} not exist, please provide a valid path."
    )
    sample_rate = 16000
    # 加载KWS模型
    keyword_spotter = sherpa_onnx.KeywordSpotter(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=args.num_threads,
        max_active_paths=args.max_active_paths,
        keywords_file=args.keywords_file,
        keywords_score=args.keywords_score,
        keywords_threshold=args.keywords_threshold,
        num_trailing_blanks=args.num_trailing_blanks,
        provider=args.provider,
    )
    # 加载VAD模型
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = args.silero_vad_model
    vad_config.silero_vad.min_silence_duration = 0.25
    vad_config.silero_vad.min_speech_duration = 0.25
    vad_config.sample_rate = sample_rate

    window_size = vad_config.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)
    # 加载ASR模型
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model="../sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
        tokens="../sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
        num_threads=2,
        use_itn=True,
        debug=False,
    )

    print("Started! Please speak")

    idx = 0

    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    kws_stream = keyword_spotter.create_stream()
    # VAD变量
    kws_flag = False
    buffer = []
    texts = []
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            if not kws_flag:  # 如果还没有检测到关键词测一直执行 kws模型
                kws_stream.accept_waveform(sample_rate, samples)
                if keyword_spotter.is_ready(kws_stream):
                    keyword_spotter.decode_stream(kws_stream)
                    result = keyword_spotter.get_result(kws_stream)
                    if result:  # 语音唤醒检测到关键词
                        print(f" KWS {idx}: {result }")
                        idx += 1
                        # Remember to reset stream right after detecting a keyword
                        keyword_spotter.reset_stream(kws_stream)
                        kws_flag = True
            else:  # 已经检测到关键词，语音已经唤醒
                print("开始VAD")
                # ------------
                buffer = np.concatenate([buffer, samples])
                while len(buffer) > window_size:
                    vad.accept_waveform(buffer[:window_size])
                    buffer = buffer[window_size:]
                while not vad.empty():
                    if len(vad.front.samples) < 0.5 * sample_rate:
                        # this segment is too short, skip it
                        vad.pop()
                        continue
                    # 调用 asr模型
                    print("开始ASR")
                    asr_stream = recognizer.create_stream()
                    asr_stream.accept_waveform(sample_rate, vad.front.samples)
                    vad.pop()
                    recognizer.decode_stream(asr_stream)
                    text = asr_stream.result.text.strip().lower()
                    if len(text):
                        idx = len(texts)
                        texts.append(text)
                        print(f"ASR {idx}: {text}")
                    kws_flag = False


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
