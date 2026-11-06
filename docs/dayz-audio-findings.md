# DayZ audio: measured behaviour (AssistZ lab)

Measured with the AssistZ lab (DayZDiag 1.30.164200, Chernarus, offline server + client): grenades detonated at known
bearings/distances from a standing player (`Lab/assistz_pose_logger.c` beacon), recorded by WASAPI loopback
(`Lab/record_loopback.py`) and by AssistZ's own GPU pipeline, analysed with `Lab/analyze_beacons.py`.

| Question | Result | Evidence |
|---|---|---|
| What is the listener? | The **head**: body yaw + free-look (`aim_lr`, + = right) | 8/8 beacons fold correctly only with the free-look offset applied (2026-11-02 run 1) |
| Pan method | **Level only**; no inter-channel delay | cross-correlation peak at lag 0 for every blast |
| Pan law | AssistZ's constant-power inverse reads **~0.91x** the true folded angle | 36->32.4, 81->75.6, 54->48.7, 9->8.1 (and mirrored) |
| Channel difference | 45 deg: 6.8 dB; 90 deg: ~27 dB | run 2, 50 m |
| Front/back cue in the mix | **None**: mirror pairs identical within 0.3 dB in every octave 63 Hz-16 kHz | 45/135, 315/225, 0/180 at 50 m |
| Propagation delay | **Yes: 3.05 ms/m = 328 m/s** | 35 m 112 ms ... 280 m 857 ms |
| Level vs distance | **-2.7 dB per doubling** (not inverse square) | 35/50/70/140/280 m at 90 deg |
| Air absorption | **None measurable** (HF tilt flat over 35-280 m) | 8k+16k vs 250+500 band tilt |
| Own footsteps | Centred; cadence tracks speed: 0.33-0.36 s at ~4 m/s, 0.26-0.28 s at 6.7 m/s | run 0, pose CSV speed |

Consequences for AssistZ:
- Direction from a stereo stream = left/right only; front/back has to come from **head motion** (a source ahead and
  its mirror behind move in opposite directions as the view turns; the video gives the turn).
- Distance: **timing** (supersonic crack vs muzzle report, muzzle flash vs sound) is the strong cue; level needs
  the source's loudness, i.e. a classifier first.
