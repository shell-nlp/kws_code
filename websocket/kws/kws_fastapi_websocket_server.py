import argparse
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import sherpa_onnx
import numpy as np

model_root_path = "/home/dev/liuyu/project/kws_code"
app = FastAPI()


# 定义参数解析函数
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
        help="It specifies number of active paths to keep during decoding.",
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

    return parser.parse_args()


# 初始化参数
args = get_args()

# 检查文件是否存在
assert Path(
    args.keywords_file
).is_file(), (
    f"keywords_file : {args.keywords_file}  not exist, please provide a valid path."
)

# 初始化 KeywordSpotter
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream = keyword_spotter.create_stream()
    sample_rate = 16000

    try:
        while True:
            data = await websocket.receive_bytes()
            samples = np.frombuffer(data, dtype=np.float32)
            stream.accept_waveform(sample_rate, samples)
            if keyword_spotter.is_ready(stream):
                keyword_spotter.decode_stream(stream)
                result = keyword_spotter.get_result(stream)
                if result:
                    await websocket.send_text(
                        json.dumps({"result": result}, ensure_ascii=False)
                    )
                    print(result)
                    # Remember to reset stream right after detecting a keyword
                    keyword_spotter.reset_stream(stream)
                else:
                    await websocket.send_text(json.dumps({"result": None}))
            else:
                await websocket.send_text(json.dumps({"result": None}))
    except WebSocketDisconnect:
        print("Client disconnected")


# 启动 FastAPI 应用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8887)
