# AssistZ roadmap

What AssistZ should track from a game's video and audio stream (DayZ first), with what is done, what is next, and
the lab ground truth each needs. Measured facts: [`dayz-audio-findings.md`](dayz-audio-findings.md).

## Done (verified in the lab)
- Stereo direction of sound events, calibrated for DayZ (pan 0.91; listener = head).
- View-turn tracking from video (8.8 px/deg, coarse-to-fine) and world-anchored radar events.
- Front/back resolution for repeating sounds while the view turns (bearing error p90 59 -> 13 deg).
- Spectral features per hop on SpawnDev.ILGPU.ML's SpectralFrontend.

## Next
| Feature | Why | Ground truth / dependency |
|---|---|---|
| **Gunshots: "being shot at"** | DayZ plays a close-only bullet noise (bullet_flyby / sonicCrack_Close) that you do not hear for far shots (TJ) | Lab mode 4: dummy-survivor shooter, near (2 m) vs wide (25 m) misses |
| **Gunshot distance** | crack-to-report gap (~1.7 ms/m) and the Close/Mid/Dist report layers | Lab mode 4 range ladder |
| **Footsteps as tracked sources** | extremely useful (TJ); players and zombies sound different | Classifier (CLAP in ML, Tuvok); lab: spawned zombies/players walking known paths |
| **Zombies** | tracking infected matters even when they are not attacking (TJ) | Lab: spawn infected at known positions (server `CreateObjectEx` + ECE_INITAI) |
| **Animals** | wolves/bears are threats, deer/chickens are food (TJ) | Lab: spawn animals at known positions |
| **Direction from the stars** | DayZ's night sky can give a compass heading (TJ) | Vision: star field vs the in-game date/time (the pose log knows the true heading) |
| **Weather** | inclement weather tracking is useful (TJ) | Vision + audio (rain, wind, thunder); the server knows the weather state (HostZLive logs rain/fog) |
| **Labels on the radar** | gunshot / steps / engine / zombie / animal | CLAP zero-shot (Tuvok, after the ML release) |

## CLAP labels (Tuvok, 2026-11-03)

Word CLAP labels with the training template: "This is a sound of " + label. A LibriSpeech speech clip scored "music" 0.971 vs bare "a person speaking" 0.007; "This is a sound of speech" scores 0.913 in transformers.
