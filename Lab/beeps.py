"""Beeps from another process, on the default output, with their UTC times. If the app hears them it is getting system audio, not the game window's own. Usage: beeps.py <out csv> <count> <every seconds> [tone.csv]; with tone.csv, it starts half a period after tone_window.py's pulses, so the two never overlap. Start this from its own terminal, not from tone_window.py."""
import datetime, math, os, struct, sys, time, wave, winsound
out, count, every = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
wav = os.path.join(os.path.dirname(out), "beep.wav")
rate = 48000
with wave.open(wav, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate)
    n = int(rate * 0.35)
    data = bytearray()
    for i in range(n):
        v = int(0.6 * 32767 * math.sin(2 * math.pi * 1000 * i / rate) * min(1, (n - i) / 2000))
        data += struct.pack("<hh", v, v)
    w.writeframes(bytes(data))
if len(sys.argv) > 4:
    while not (os.path.exists(sys.argv[4]) and open(sys.argv[4]).readline().strip()):
        time.sleep(0.05)
    first = datetime.datetime.fromisoformat(open(sys.argv[4]).readline().strip()).timestamp()
    now = time.time()
    time.sleep(first + (int((now - first) / every) + 1) * every + every / 2 - now)
with open(out, "w") as f:
    for _ in range(count):
        t = datetime.datetime.now(datetime.timezone.utc)
        winsound.PlaySound(wav, winsound.SND_FILENAME)
        f.write(t.isoformat() + "\n")
        f.flush()
        time.sleep(max(0, every - 0.35))
print("beeps done", count)
