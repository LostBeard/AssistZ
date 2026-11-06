# AssistZ lab

Ground truth for everything AssistZ infers from a video/audio stream. Findings: [`../docs/dayz-audio-findings.md`](../docs/dayz-audio-findings.md).

**Before any GPU run on the shared dev PC, ask every agent that might be using the GPU (board: `_DevComms/board.md`).**
A game is a GPU-heavy job; a timing run needs a quiet GPU.

## DayZ (offline, DayZDiag)

| Script | What it does |
|---|---|
| `make_lab_missions.ps1 -Mode 0\|1\|2` | Builds `..\..\Lab\missions\assistzLab.<map>` from Bohemia's `dayzOffline.<map>` (clone of BohemiaInteractive/DayZ-Central-Economy, ADPL-SA, never committed) with `assistz_pose_logger.c` injected. Mode 0 = pose log only, 1 = grenade sound beacon, 2 = server-scripted view turns (only turns the SERVER's copy of the player). |
| `launch_dayzdiag.ps1` / `stop_dayzdiag.ps1` | DayZDiag `-server` with the mission + a DayZDiag client joining 127.0.0.1; PIDs recorded, only those are ever stopped. `serverDZ.cfg` must keep `allowFilePatching = 1`. |
| `drive_turns.py` | Turns the client view with SendInput mouse moves (one click first so the window captures the mouse; mouse only, foreground PID checked every step). |
| `record_loopback.py` | WASAPI loopback recording (PyAudioWPatch) with the UTC start time. |
| `analyze_beacons.py`, `analyze_yaw.py` | Recording vs beacon log; tracker vs pose log. |
| `watch_youtube.py` | Unattended watch-along: plays a YouTube result in the lab Chrome, pins the player, shares the tab (with audio) into AssistZ in its own window. |

Server-side outputs (`$profile`): `assistz_pose_<utc>.csv` (10 Hz: game_ms, utc, position, yaw/pitch/roll, aim_lr/ud),
`assistz_beacon_<utc>.csv`, `assistz_turns_<utc>.csv`.

## Browser harness (`harness/`)

All drive a separate Chrome (own profile, CDP port 9333; never the user's browser) against a release publish served by
`spa_serve.py <publish>\wwwroot 5390`. Work folder: `$env:ASSISTZ_LAB_WORK` or `%TEMP%\assistz-lab`.

| Script | What it does |
|---|---|
| `run_m0.ps1` + `verify_m0.py` | `/testsignal` (stereo pulses at known pan) shared into AssistZ: PASS needs every angle bucket, the pulse ORDER (catches a left/right swap) and spectral azimuth == broadband azimuth. |
| `ab_spectral.ps1` + `ab_analyze.py` | Audio batch latency with the SpectralFrontend path on/off (`?spectral=0`); the analysis refuses a run whose page did not run the requested mode. |
| `lab_yaw_run.ps1 -Driver mouse\|server` | Full unattended DayZ yaw-tracker run: publish, launch, wait for spawn, capture, turn, collect, stop. |
| `run_live.ps1` + `capture_live.py` | Capture a window/tab source and dump telemetry, events and screenshots. |

Gotchas learned the hard way:
- Chrome's `--auto-select-*-capture-source` flags skip the picker's "Share audio" toggle: a window capture then has no
  audio. Windows DO share audio when a person ticks it. (Tab auto-select does include audio.)
- A second tab in the same window hides the source tab and Chrome stops decoding its video (black frames, live audio):
  open AssistZ in its own window.
- CDP `/json/new?<url>` cuts the URL at the first `&`: use `Target.createTarget`.
