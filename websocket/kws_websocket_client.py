import asyncio
import json
import sys
import sounddevice as sd
import websockets
import sys
import os

# os.path.abspath(__file__)
sys.path.append(__file__)


async def send_audio(websocket, sample_rate, samples_per_read):
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as stream:
        while True:
            samples, _ = stream.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            await websocket.send(samples.tobytes())
            # 接收来自服务器的文本消息
            message = await websocket.recv()
            if message:
                if json.loads(message)["result"]:
                    print(f"Received from server: {message}")


async def main():
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    print("Started! Please speak")

    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    uri = "ws://localhost:8887/ws"  # 替换为实际的IP地址
    async with websockets.connect(uri) as websocket:
        await send_audio(websocket, sample_rate, samples_per_read)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
