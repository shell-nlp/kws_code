import argparse
import asyncio
import sys
from pathlib import Path
import time
import sounddevice as sd
import sherpa_onnx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import soundfile as sf
import logging

model_root_path = "/home/dev/liuyu/project/kws_code"
app = FastAPI()


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default=f"{model_root_path}/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt",
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default=f"{model_root_path}/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        help="Path to the transducer decoder model",
        default=f"{model_root_path}/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        help="Path to the transducer joiner model",
        default=f"{model_root_path}/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
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
        default=f"{model_root_path}/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt",
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
        default=f"{model_root_path}/silero_vad.onnx",
        help="Path to silero_vad.onnx",
    )

    return parser.parse_args()


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
    model=f"{model_root_path}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
    tokens=f"{model_root_path}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
    num_threads=2,
    use_itn=True,
    debug=False,
)
tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=f"{model_root_path}/kokoro-multi-lang-v1_1/model.onnx",
            voices=f"{model_root_path}/kokoro-multi-lang-v1_1/voices.bin",
            tokens=f"{model_root_path}/kokoro-multi-lang-v1_1/tokens.txt",
            data_dir=f"{model_root_path}/kokoro-multi-lang-v1_1/espeak-ng-data",
            dict_dir=f"{model_root_path}/kokoro-multi-lang-v1_1/dict",
            lexicon=f"{model_root_path}/kokoro-multi-lang-v1_1/lexicon-zh.txt",
        ),
        provider="cpu",
        debug=False,
        num_threads=2,
    ),
    rule_fsts=f"{model_root_path}/kokoro-multi-lang-v1_1/phone-zh.fst,{model_root_path}/kokoro-multi-lang-v1_1/date-zh.fst,{model_root_path}/kokoro-multi-lang-v1_1/number-zh.fst",
    max_num_sentences=1,
)
tts = sherpa_onnx.OfflineTts(tts_config)
print("Started! Please speak")

import io
import wave


@app.websocket("/ws")
async def main(websocket: WebSocket):
    def generated_audio_callback(samples: np.ndarray, progress: float):
        """This function is called whenever max_num_sentences sentences
        have been processed.

        Note that it is passed to C++ and is invoked in C++.

        Args:
        samples:
            A 1-D np.float32 array containing audio samples
        """
        global first_message_time
        if first_message_time is None:
            first_message_time = time.time()
        # 将numpy数组转换为字节数据
        audio_bytes = samples.tobytes()
        print("开始发送")
        asyncio.create_task(websocket.send_bytes(audio_bytes))
        # buffer.put(samples)
        global started

        if started is False:
            logging.info("Start playing ...")
        started = True

        # 1 means to keep generating
        # 0 means to stop generating
        if killed:
            return 0

        return 1

    await websocket.accept()
    idx = 0
    # samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    kws_stream = keyword_spotter.create_stream()
    # VAD变量
    kws_flag = False
    buffer = []
    texts = []
    try:
        while True:
            data = await websocket.receive_bytes()
            wav_file = io.BytesIO(data)
            with wave.open(wav_file, "rb") as wf:
                # 获取 WAV 文件的基本信息
                num_channels = wf.getnchannels()  # 声道数
                sample_width = wf.getsampwidth()  # 采样宽度（字节数）
                frame_rate = wf.getframerate()  # 采样率
                num_frames = wf.getnframes()  # 总帧数
                # 读取音频数据
                raw_data = wf.readframes(num_frames)
                # print(num_channels, sample_width, frame_rate, num_frames)
                # 根据采样宽度选择 NumPy 数据类型
                if sample_width == 2:  # 16bit PCM
                    dtype = np.int16
                elif sample_width == 4:  # 32bit PCM
                    dtype = np.int32
                else:
                    raise ValueError(f"不支持的采样宽度: {sample_width}")
                data = raw_data
            # 将原始字节数据转换为 NumPy 数组
            samples = np.frombuffer(data, dtype=dtype)
            # 如果是立体声，转换为单声道
            if num_channels > 1:
                samples = samples.reshape((-1, num_channels))
                samples = samples.mean(axis=1)  # 取平均值
            # 归一化到 [-1.0, 1.0] 范围，并转换为 float32
            if dtype == np.int16:
                samples = samples.astype(np.float32) / 32768.0
            elif dtype == np.int32:
                samples = samples.astype(np.float32) / 2147483648.0
            else:
                raise ValueError("不支持的数据类型转换")
            if not kws_flag:  # 如果还没有检测到关键词测一直执行 kws模型
                kws_stream.accept_waveform(sample_rate, samples)
                if keyword_spotter.is_ready(kws_stream):
                    keyword_spotter.decode_stream(kws_stream)
                    result = keyword_spotter.get_result(kws_stream)
                    if result:  # 语音唤醒检测到关键词
                        # await websocket.send_text(
                        #     json.dumps(
                        #         {"result": "KWS {idx}: {result }"}, ensure_ascii=False
                        #     )
                        # )

                        print(f"KWS {idx}: {result }")
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
                        # TODO ------- add TTS--------
                        start = time.time()
                        """47->zf_xiaoxiao, 48->zf_xiaoyi,49->zm_yunjian, 50->zm_yunxi,
                        51->zm_yunxia, 52->zm_yunyang,"""
                        for i in range(47, 48):
                            sid = i
                            audio = tts.generate(
                                "如果在解压过程中遇到错误，可能是因为文件损坏或权限不足。请确保文件完整性，并以适当权限运行命令。",
                                sid=sid,
                                speed=1.0,
                                callback=generated_audio_callback,
                            )
                            end = time.time()
                            if len(audio.samples) == 0:
                                print(
                                    "Error in generating audios. Please read previous error messages."
                                )
                                return
                            elapsed_seconds = end - start
                            audio_duration = len(audio.samples) / audio.sample_rate
                            real_time_factor = elapsed_seconds / audio_duration

                            sf.write(
                                f"./adudi_sid_{sid}.wav",
                                audio.samples,
                                samplerate=audio.sample_rate,
                                subtype="PCM_16",
                            )
                            print(f"Elapsed seconds: {elapsed_seconds:.3f}")
                            print(f"Audio duration in seconds: {audio_duration:.3f}")
                            print(
                                f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
                            )

                    kws_flag = False
    except WebSocketDisconnect:
        print("Client disconnected")


import queue
import threading

# buffer saves audio samples to be played
buffer = queue.Queue()

# started is set to True once generated_audio_callback is called.
started = False

# stopped is set to True once all the text has been processed
stopped = False

# killed is set to True once ctrl + C is pressed
killed = False

# Note: When started is True, and stopped is True, and buffer is empty,
# we will exit the program since all audio samples have been played.


event = threading.Event()

first_message_time = None


# 启动 FastAPI 应用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8887)
