"""
Records the PC's audio output (WASAPI loopback of the default device) to a 32-bit float stereo WAV, and writes the
UTC time of the first sample next to it, so lab recordings line up with the DayZ pose / beacon CSVs.
Needs PyAudioWPatch (pip install --user PyAudioWPatch).
Usage: python record_loopback.py <seconds> <out.wav>
"""
import sys, time, wave, struct, datetime, json
import numpy as np
import pyaudiowpatch as pa

seconds, out = float(sys.argv[1]), sys.argv[2]
p = pa.PyAudio()
dev = p.get_default_wasapi_loopback()
rate, ch = int(dev["defaultSampleRate"]), dev["maxInputChannels"]
frames_total = int(seconds * rate)
chunks = []
got = 0
stream = p.open(format=pa.paFloat32, channels=ch, rate=rate, input=True, input_device_index=dev["index"], frames_per_buffer=1024)
start_utc = datetime.datetime.now(datetime.timezone.utc)
while got < frames_total:
    data = stream.read(1024, exception_on_overflow=False)
    chunks.append(np.frombuffer(data, dtype=np.float32))
    got += 1024
stream.stop_stream(); stream.close(); p.terminate()
pcm = np.concatenate(chunks).reshape(-1, ch)[:frames_total]
# Written in chunks: one np.save of 230 MB to the D: share failed on NV1080 with WinError 1450 / "not enough memory resources" and left a
# truncated file (600 s recording, 2026-11-04). 360 s (138 MB) had worked.
with open(out.replace(".wav", ".npy"), "wb") as npy:
    np.lib.format.write_array_header_1_0(npy, np.lib.format.header_data_from_array_1_0(pcm))
    for start in range(0, len(pcm), 1 << 20):
        npy.write(pcm[start: start + (1 << 20)].tobytes())
with open(out.replace(".wav", ".json"), "w") as f:
    json.dump({"device": dev["name"], "rate": rate, "channels": ch, "start_utc": start_utc.isoformat(), "frames": len(pcm)}, f)
# 16-bit WAV for listening; analysis uses the float .npy
with wave.open(out, "wb") as w:
    w.setnchannels(ch); w.setsampwidth(2); w.setframerate(rate)
    for start in range(0, len(pcm), 1 << 20):
        w.writeframes((np.clip(pcm[start: start + (1 << 20)], -1, 1) * 32767).astype("<i2").tobytes())
print(f"recorded {len(pcm) / rate:.1f} s from {dev['name']} @ {rate} Hz, start {start_utc.isoformat()}")
