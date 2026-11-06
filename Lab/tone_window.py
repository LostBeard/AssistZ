"""
Positive control for the windowAudio test: a visible window titled AZTONE that plays its own pulses from this process, logging when. Share AZTONE: the app must hear these whatever the setting. Pass for window audio: these heard AND beeps.py's not.
Run beeps.py yourself from another terminal, with this tone.csv as its last argument, so it plays half a period later. It must not be started from here: Chrome's app audio toggle takes "all its windows and related apps", so a child of this process could count as this app.
Usage: python tone_window.py <out dir> [count] [every seconds]
Writes <out dir>/tone.csv. Plays through the default output.
"""
import datetime, math, os, struct, sys, threading, time, tkinter, wave, winsound
out = sys.argv[1]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 30
every = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0
os.makedirs(out, exist_ok=True)
wav = os.path.join(out, "tone.wav")
rate = 48000
with wave.open(wav, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate)
    n = int(rate * 0.35)
    data = bytearray()
    for i in range(n):
        v = int(0.6 * 32767 * math.sin(2 * math.pi * 440 * i / rate) * min(1, (n - i) / 2000))
        data += struct.pack("<hh", v, v)
    w.writeframes(bytes(data))
root = tkinter.Tk()
root.title("AZTONE")
root.geometry("420x240+80+80")
tkinter.Label(root, text="AssistZ windowAudio test: this window plays 440 Hz pulses", wraplength=380).pack(expand=True)


def play():
    time.sleep(1)
    started = time.time()
    with open(os.path.join(out, "tone.csv"), "w") as f:
        for k in range(count):
            while time.time() < started + k * every:
                time.sleep(0.01)
            f.write(datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
            f.flush()
            winsound.PlaySound(wav, winsound.SND_FILENAME)
    time.sleep(every)
    root.after(0, root.destroy)


threading.Thread(target=play, daemon=True).start()
root.mainloop()
print("done")
