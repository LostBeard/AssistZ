# AssistZ lab handoff (2026-11-03, Riker, moving DayZ lab work to the NV1080 PC: i7-4770, 32 GB, AMD RX 590 8 GB)

READ docs/PLAN.md FIRST: the goal and where each part stands. Then read this on the new PC. Repo: `LostBeard/AssistZ` (private), branch `master`, last commit at handoff below.
TJ's direction for the project: an ML showcase for SpawnDev.ILGPU.ML in the browser (not an accessibility tool); sleek,
unobtrusive UI; ML needs go to Tuvok; automate everything; never model or assume how the game makes sounds: measure it.

## Where things are
- App: `AssistZ/` (Blazor WASM). ML package `SpawnDev.ILGPU.ML 5.8.2-local.8` from the local feed, audio tower capture off by default (`?capture=1`)
  `D:\users\SpawnDevPackages`; delete `bin/obj` after any package switch. Tests: `dotnet test AssistZ.Tests` (33).
- Lab scripts: `Lab/` (this repo). Lab DATA lives OUTSIDE the repo in `D:\users\tj\Projects\AssistZ\Lab\`:
  `missions\` (built each run), `mods\@AssistZLab` (packed each run), `profiles\{server,client}`, `runs\<run>`,
  `DayZ-Central-Economy\` (Bohemia, ADPL-SA: local use only, never commit), `expansion_lab_settings\AIPatrolSettings.json`
  (Expansion's generated settings with Enabled = 0: map patrols off; Expansion content, never commit).
- Expansion AI (CC BY-NC-ND 4.0): used unmodified from the Workshop folder, never repacked, altered or committed.
  Needs Workshop subscriptions on THIS PC's Steam: @CF, Dabs Framework, @DayZ-Expansion-Core, @DayZ-Expansion-AI.
- `L:\dayz_nasdara` (HostZ) is READ-ONLY reference: another agent works on those files.
- Tuvok (ML): CLAP runs on WebGPU in Chrome on this PC. If anything uses ML's OpenCL lane on the AMD RX 590, tell
  Tuvok first: there have been AMD-OpenCL-specific bugs (an async host-copy race) and he wants to gate CLAP on it.

## New-PC checks before the first run
1. `Lab/launch_dayzdiag.ps1 -DayZ <path>`: the default is `E:\SteamLibrary\steamapps\common\DayZ` (this PC's). Pass this
   PC's DayZ folder, or change the default if it differs. DayZDiag_x64.exe must exist there.
2. Workshop mods present under `<DayZ>\!Workshop` (the launcher throws with a clear message if Dabs is missing).
3. Python: numpy, scipy, websockets, PyAudioWPatch (WASAPI loopback). `record_loopback.py` records the default output
   device; check it picks this PC's speakers/headset.
4. .NET SDK 10.0.403, Chrome (the lab Chrome uses its own profile and port 9333).
5. Set `$env:ASSISTZ_LAB_WORK` to a scratch folder (publish output, lab Chrome profile).
6. One instance at a time on the lab DATA folders: if `D:` is shared with the main PC, do not run labs from both.

## NV1080 as set up (2026-11-03)
- DayZ is at `L:\SteamLibrary\steamapps\common\DayZ`; the launcher finds it on `E:` or `L:`. @CF, Dabs, Expansion-Core and
  Expansion-AI are in its `!Workshop`.
- Work folder: `$env:ASSISTZ_LAB_WORK = 'L:\AssistZLabWork'` (publish, lab Chrome profile with the CLAP towers in OPFS, and the
  lab MISSIONS: DayZ's economy database fails on a network share, "disk I/O error", then the server stops itself). It is not set
  persistently: set it in each command. The scripts refuse a mission folder on a network drive.
- Python 3.11: `c:\python311\Scripts` is read-only without elevation, so numpy and scipy are installed with `pip --user`.
  `py -3.13` does not exist here: run `_TestStats\log_run.py` with `python`.
- `D:` is the share `\\Yoda\TData` and fsync fails on it, so the Write and Edit tools fail there and change nothing. Write in the
  scratchpad and copy, or use a script, and read back.
  Git too: add and commit fail on loose objects, so run `git -c core.fsyncObjectFiles=false add` and `commit`, then push and check.
- NOT checked yet: the default output is `Speakers (Steam Streaming Speakers)` and `record_loopback.py` records the default
  device. Prove a DayZ recording is not silent before trusting a run.

## How to run
- DayZ lab (mode 6 = Expansion raiders at 50/100/200/400 m; mode 7 = wide misses, East shooter at a West decoy beside
  a Civilian-group listener): `Lab/harness/lab_sniper_run.ps1 -Expansion` or `-WideMiss`, `-RecordSeconds 190`.
  Server-first compile gate; PIDs recorded in `dayzdiag.pids.json`; `Lab/stop_dayzdiag.ps1` stops only those PIDs.
- After a run: `python Lab/aishots_to_shots.py <run>`, `python Lab/analyze_crack.py <run>`,
  `python Lab/analyze_crack_timing.py <run>` (calibrates the run's clock on the data).
- CLAP labels on a recording: publish the app, serve it (`Lab/harness/spa_serve.py <publish>\wwwroot 5390`), then
  `Lab/harness/run_labels.ps1 -Run <run dir> -Seconds <len> [-Query "&clap=larger"] [-Tag _x]`.
- Exact labels: `Lab/harness/run_replay.ps1 -Run <run dir> -Skips "0,256" -Tag _x`, then `python Lab/compare_replay.py <A.jsonl> <B.jsonl>` and
  `python Lab/score_replay.py <run dir> <A.jsonl>`. Two real-time label runs: `python Lab/compare_labels.py <run dir> <labels A> <labels B>`.
- M0 timing: `Lab/harness/run_m0.ps1 -Query "lab=1" -ConsoleOut <file>`, then `python Lab/harness/analyze_m0.py <file>`.
- Log EVERY run to `D:\users\tj\Projects\_TestStats\AssistZ.jsonl` (append only). Commit and push to master often.

## State at handoff
- Gunfire: Expansion raiders fire at the listener when he makes an Expansion noise (eAINoiseSystem) every 3 s; they
  only line-of-sight check their current target, against the look direction and FOV.
- CLAP in the app: every detected event gets a label; gunfire split into single / automatic by CadenceMeter
  (fire mode 24/24, 39/39, 44/44). Default CLAP checkpoint stays (the larger one hears speech as "dog barking").
- Ground truth logs per run: client + server shot logs (EEFired), AI pose (speed, distance, FSM state), client bullet
  IMPACTS (`DayZGame.FirearmEffects`: position, surface, speed, distance to head).
- Crack: a > 3 kHz precursor ~35 ms after the fire, well before the report. At a fixed ~45 m shooter it is present at
  every pass distance 1-30 m (16.6 dB median), so it is NOT a close-pass cue; its strength follows the shooter distance.
  Its lead over the report gives the shooter distance. DayZ also defines `bulletFlyBy_SoundSet` (range 30 m), not yet
  isolated.

## Parity result (2026-11-03, 540d344, RX 590 vs RTX 4070, logged in _TestStats)
- No GPU-code finding, so nothing for Tuvok or ILGPU. M0 passes on `amd gcn-4` (WebGPU), spectral az == broadband az. CLAP on run
  `sniper_20261103_153510`: every one of the 58 events after 5 s is detected on both GPUs. Labels differ on 12 and 7 of 58 between
  NVIDIA and AMD, and on 13 of 58 between two AMD runs, so the vendor difference is inside run-to-run noise. On NVIDIA's clock the
  AMD runs score gunfire 38/50 and 39/50 against 39/50, fire mode all right. CLAP median 75.5 ms against 71.7 ms.
- Numeric parity is now measured (Data, RTX 4070, driver 596.21, d8ec3eb, replay files `replay_rtx4070_s0.jsonl`): 12 recordings, 1,111 events, all on the same clip, same CLAP label on all,
  max |score difference| 5.37e-06. Float noise, not a GPU finding. One CLAP call took 50,780 ms on that PC in run 002001 (results identical); watch for a repeat.

## Labels and clocks (2026-11-03, RX 590)
- Replay (`?replay=1`, `Lab/harness/run_replay.ps1`): the file's samples go through the production analyzer, detector, cadence and
  CLAP in fixed steps. Same samples give bit-identical scores (61/61 events). Use it for every label score: no capture clock in it.
- Label flips had one cause: clips were cut relative to the hop grid. Same clip = bit-identical; 64/256/512 samples later flipped
  6/15/7 events of about 60. Clips are now anchored on the sound's peak (`SoundLabeler.AnchorOnPeak`, `?align=0` to A/B): 0/0/1 on
  run 4, 284/288 paired events across all five recordings at skip 0 vs 256. Two real-time runs: 58/58 labels, p difference 0.000.
  Cost: CLAP median about +4 ms, and a label can come up to 86 ms later.
- Fire mode looks 3 s back, 1 s ahead (`CadenceMeter.IsAutomatic`): AUG 480/480 at 16 hop phases, other weapons wrongly automatic
  24 of 2224 (was 38; two SVD shots, run 7 at 62.41 and 62.73 s, 12 of 16 phases each). There is only one AUG burst in the data: a burst that ends and is followed within
  3 s by single shots is untested.
- Warm-up (`StereoAnalyzer.WarmUpAsync`, `?warm=0` to A/B): skipped windows 0 in 4/4 runs, 10-14 in 4/4 without. Capture starts
  about 0.8 s later instead (M0 29 events, 29-31 without). The compile itself still takes 730-830 ms here (RTX 4070 log: 49.6 ms
  max) and a batch of about 110 ms remains; neither is traced.
- Shot clock (`Lab/lab_clock.py`) searches +-5 s. Run 7 (190530) is -2365 ms, not the -35 ms in earlier notes, which was a false peak
  in a +-0.5 s search. Shots with an event went 8/42 -> 21/37. Its cause is not traced, and the 14:10 near-vs-wide note used the
  wrong clock (already retracted). On the right clock run 7's precursor agrees with the crack statement above.
- Real-time scoring: `verify_labels.py` reads the app's playclock lines. Offset between runs about 12 ms, but about 80 ms from replay's
  exact positions. Use replay when position matters. `-Seconds` longer than the recording wraps: events past the end are left out.
- Long recordings: one np.save of 230 MB to the D: share failed ("not enough memory resources", truncated file), so `record_loopback.py` writes in chunks. WASAPI loopback delivers nothing while the device is silent, so testing it without the game needs a sound playing, or it blocks forever. `check_audio.py` does not memory-map either.
- `Lab/` runs on Python 3.11 now (a 3.12-only f-string in analyze_crack.py broke it).
- Score on the final build, skip 0, shots named gunfire: run 4 34/44, run 5 15/20, run 6 13/18, run 7 21/37, run 8 (200816) 44/44.
  Fire mode wrong on one event at skip 0 (run 7). The next lab run uses decoy offsets 1/2/25/30 m.

## Flyby (2026-11-03, `Lab/analyze_flyby.py`): the effect replicated, the single-shot rule did not
- Discovery, three recordings: 12-20 kHz peak around the bullet's arrival, passes under 5 m about 26 dB, far passes about 10 dB (7 close
  shots, band chosen after seeing them).
- Holdout, rule committed first (a298456): 12-20 kHz, arrival -15..+10 ms, 20 dB. Confirmed if at least 80% of passes under 5 m and at
  most about 25% of passes at 15 m or more exceed it. Three new runs, one invalid: audio stopped at 32.1 s (`invalid_silent_sniper_20261103_232810`).
  As written with all three: 15/32 close, 6/33 far. On the two valid runs: 15/22 (68%) close, 6/26 (23%) far. NOT MET.
- What did replicate, two valid runs: close median 25.5 dB, far 12.7 dB, p = 0.0002, n = 22 and 26; Spearman with pass distance -0.45.
  Follows the pass, not the landing (under 5 m and landed far: 25.9 dB, n = 20; landed near: 20.2 dB, n = 2). Same band before the fire:
  AUC 0.46. Per run AUC 0.83 and 0.67. The close group has a hump at the precursor and a larger one at the arrival (-5 ms).
- So: close passes are louder up there on average, and one shot does not reliably tell. Do not retune on these runs.
- What it is, from the game's own files (`Lab/flyby/`, read-only; samples stay local, never commit them). `CfgAmmo/Bullet_Base` in weapons_projectiles.pbo sets
  `soundSetSuperSonic` = SuperSonicCrack (range 500 m) + Tail (100 m) and `soundSetBulletFly` = bulletFlyBy (30 m). Searching recordings for the samples themselves (133 shots, pitch steps 0.3%):
  - Flyby samples are NOT the cue: 0 of 25 passes under 5 m, even with the crack subtracted. One clear match in 133 shots, at 0.661, on a 21.4 m pass. (I first called the flyby the answer because its
    samples have 20 dB more 12-20 kHz than the crack's. The search disproved that.)
  - The crack IS found: pitch 0.995, correlation up to 0.77, in two groups (around 0.2 and 0.5), so it is on or off. Chance of on: under 8 m 76%, 8-15 m 50%, 15-25 m 35%, beyond 20%
    (without run 190530, 114 shots, p = 8e-5). 12-20 kHz is mostly this on/off: 20.3 vs 6.9 dB (p = 1e-13), and once on it does not follow pass distance (-0.03). That is why the threshold rule stopped
    at 68% and 23%: the game itself plays it that way.
  - When on: its time follows the bullet (1.12 ms per m of shooter, bullet 1.27, sound 3.03; pass distance adds nothing), and its level does not follow the pass (about -15 dB in every bin, sd 2 dB).
  - Not known: what decides on or off. One crack blocking the next is ruled out by these gaps (close passes on 15/20 and 7/8 at 0.45-0.8 s, 4/6 beyond 3 s). "On" comes in streaks: 96% within 0.8 s of an on, 17% after 5 s.
    Run 190530 (shooter about 60 m, passes 9-45 m) is 0/19. Left/right level tells the bullet's side only 67% of the time.
- Holdout, rule committed first (76af9ab): correlation >= 0.35 at pitch 0.85-1.18 = on, new mode 7 runs only, confirmed if on-rate is at least 60% under 8 m and at most 40% at 15 m or more. Two new runs: 11/16 (69%) and
  13/33 (39%). MET, narrowly, and a Fisher test on those two groups alone is only p = 0.07. Flyby again absent, none of 66 above the control. Pooled, 180 shots (bins are exploration): 76, 67, 70, 75, 69% in 3 m bins up to 15 m,
  53% at 15-18 m, 27-33% at 18-30 m; under 15 m 64/89 vs over 32/91 (p = 8e-7). A plateau and a step near 15-18 m rather than a falloff.
- AssistZ cannot ship the game's samples. If this holds, they are a way to label clips for a probe, not something to embed.
- 12-20 kHz after subtracting the crack only drops 3-6 dB, but high-band subtraction is limited by pitch precision, so this does not show another sound is present.
- Audio stops when the game window loses focus: digital silence, while the game logs a normal run, and sound returns with focus. TJ pointed at it after two mode 15 runs went silent while TJ was at the PC. Measured with Notepad taking focus for 60 s; silent with `pauseMode` 2 + `-noPause` (how the lab ran), `-noPauseSound` added (does nothing), 0 + `-noPause`, 0 alone; no silence with 2 and no `-noPause`, so that is now the launcher's default for the listening client. The menu's "Update In Background" is `pauseMode`. Probably what happened in run 232810 too (not provable now). `Lab/check_audio.py` still fails any such run.

## Direction, pre-registered (written before mode 25 ran)
Every earlier shooter sat in a narrow arc ahead, because `AzClearSpot` takes the first clear bearing nearest straight ahead (spread about 12 deg; 30 of 1579 shots behind), so direction has not been tested. `lab_sniper_run.ps1 -DirectionTest -RecordSeconds 450` (mode 25): one AKM at 50 m at 12 bearings 30 deg apart, two runs at different spawns. Truth: the logged muzzle position and the listener's pose. Stereo cannot tell front from back, so angles are folded to the front half. Limits, `Lab/direction_truth.py`, with bearings that got fewer than 5 heard shots reported, not scored: left or right right at least 95% where the true direction is at least 15 deg off centre; folded angle median error at most 15 deg and p90 at most 30 deg, overall and at each bearing. Front or back is reported. It is not expected to work: the app has no cue for it.

**Result: FAILED at the sides, close overall** (`sniper_20261105_000715`, `001851`; 1370 shots, 717 with an event; `Lab/direction_truth.py`). Left or right 627/633 (99.1%). Folded angle median 10.1 deg, p90 29.6, correlation 0.97, over a truth spread of 55 deg. Over the limit at true +-60 to +-120: 90 deg reads about 60. Systematic, so correctable. (`Lab/direction_next.py`) A curve fixed on one run, tested on the other: median 10.0 -> 6.7 and 10.1 -> 8.9, p90 29.5 -> 19.8 and 29.7 -> 21.0, worst bearing median 25 -> 15 and 28 -> 14.
Front or back: not in the stereo mix. Level differs at most 0.7 dB, above 4 kHz at most 1.1 dB, and an embedding probe trained on one run and tested on the other reads 54% and 55% balanced against a coin's 50%. So it needs another source. Heading from the video feed plus the player turning would do it: the angle's change tells front from back. To be tested with video.
Detection, a correction: I told TJ the detector misses most shots in storms. It does not. Bursts heard 83/83 across 9 runs in every weather. Shots without their own event come from pacing: at most one event each 0.17 s against shots about each 0.15 s. Distant single shots remain open: older SVD runs 15/20, 14/18, 21/42.

### Pre-registered: azimuth curve (written before its fresh run)
Curve fixed on both runs, mirrored, az to true: [(0.0, 0.0), (5.0, 3.0), (15.0, 20.1), (25.0, 29.8), (35.0, 58.6), (45.0, 61.2), (55.0, 68.8), (65.0, 77.2), (75.0, 83.2), (90.0, 90.0)]. Fresh `-DirectionTest` run, never used to fit. Limits on the fresh run with the curve, against the same run raw: folded median at most 10 deg; p90 at most 25; worst bearing's median at most 18; left or right not below 95%. Pass: the HUD uses the curve (raw az stays in the logs).

**Result: missed by 0.2 deg, applied to gunfire only** (`sniper_20261105_003257`; `Lab/curve_gate.py`). Raw against curve: median 9.0 -> 5.3, p90 32.9 -> 25.2 (limit 25), worst bearing 28.7 -> 15.5, left or right 99.0% both. Applied despite the miss, recorded as such: it moves a radar arc and cannot hide a sound. Gunfire only, because `GameProfile` already holds 0.91 measured on grenades (`dayz-audio-findings.md`), so the curve would harm other sounds. Known limits: one weapon at one distance; a bumpy table (raw 25 to 35 moves 29.8 to 58.6).
Correction: this round re-measured what `roadmap.md` and `dayz-audio-findings.md` already held, because they were not read after a context reset. Already done before today: view-turn tracking from video (`YawTracker`), world-anchored radar events, front or back for repeating sounds while the view turns (`SourceTracker`, bearing error p90 59 -> 13 deg, lab run `frontback_20261103_130111`), and "no front or back cue in the mix". READ THOSE TWO DOCS FIRST. Not done: front or back for gunfire (only repeating beacons were tested); reading the mini-map or sitrep from video.

## Vision Module (the plan: `../dayz_assistive_system_overview.md`, one folder above this repo; READ IT)
Where the player is and which way they face, from video alone. Tuvok, 2026-11-05: CLIP ViT-B/32 vision exists (`ZeroShotClassificationPipeline.EncodeImageAsync`, 512-d, no per-frame time measured yet), but it knows "a forest road" better than this forest road; if it falls short a DINOv2 place embedding is a gap he would fill. No OCR: a gap, likely PP-OCR, when told how many pixels a town sign's letters are at 1080p. So keep raw frames, on `L:\AssistZLabWork\frames`, never committed. `ClapProbe` works on any normalised vectors, for region classes; position and heading come from retrieval. Collect: `lab_sniper_run.ps1 -StormLab 26 -RecordSeconds N -FrameDir <dir>`.

## Next
From TJ, 2026-11-04, not started:
- Distance from sound. Direction and label exist; distance is the missing piece for the mini-map and sitrep, which the video feed will also feed. Test, with gates committed first, against shot logs (40-150 m) and world events (300-2500 m): crack-to-report gap (only when a crack is heard), a probe on the embedding, level (confounded by weapon). Likely bands, not numbers.
- Nasdara as a held-out map: every result so far is Chernarus. It has new animals and dust storms; dust storms could become a word.
- Birds: TJ by ear, the 121745 clip is a crow-like "caw caw caw" at 3-5 and 15-17 s, plus a second bird farther off, faster and higher, 2 caws about every 8-9 s. Confirm against the game's samples. DayZ has many animal sounds, so test "crow cawing" as a prompt, and check it does not shift other words.

0. Stable is not right. Ten AUG shots in run 4 are named footsteps, the same every run (p 0.51-0.89). Ask Tuvok before changing
   prompts or the model. A burst-level vote needs footstep ground truth first (the AI pose log has it).
   Shots with no event at all: 5/20 run 5, 4/18 run 6, 16/37 run 7. Not looked at yet: detector threshold, or clock on run 7.
1. Flyby candidate, see below; holdout offsets 1/2/1/30 m (AzDecoyOffset). Original plan: mode 7 at a fixed 45 m shooter (many < 5 m and many
   20-30 m passes), then compare a window around the bullet's arrival (d / 785 s after the fire).
2. Weather and infected: `env.csv` per run, once a second, every lab mode (`AzEnvTick`: overcast, rain, fog, wind; living infected, animals, vehicles within 200 m). No lightning event exists in the script API. Three runs so far, all dry
   (overcast 0.30-0.62, rain 0), infected never closer than 107 m, no animals: so these show false words only: 3 "zombie groaning" (p 0.26-0.33), 6 "wolf howling", 1 "rain", 3 "dog barking". No run yet has rain, storms or infected near, so
   Mode 12 forces them: `lab_sniper_run.ps1 -WeatherLab -RecordSeconds 800`, phases in `world.csv`. Results under "Weather and infected" below.
3. World events are in the audio. The low rumble is a gas strike: TJ recognised it by ear ("a gas bomb being dropped"), and the samples prove it (`Lab/flyby/world_windows.py`): `Artillery_Close` matches 0.60-0.98 in 7 of 8 rumbles, 0.06-0.07 in all 5 control
   windows, at most 0.275 reversed. `ContaminatedArea_Dynamic` plays this sequence, range up to 5 km. In Expansion runs the server log shows the zone created 4-9 s after the rumble starts. These 7 strikes were the prompts' false "machine gun fire"; the probe's "not
   gunfire" was right. Still unexplained: the rumble in 20261103_142532 at 52 s (no match, among sniper shots). Heli and sleigh crashes exist (`HeliCrash_Distant`, `SledgeCrash_Distant`, range 3 km; StaticHeliCrash, StaticSantaCrash) but none matched in these windows
   (at most 0.19); nothing beyond those 13 windows was searched. `Artillery_Distant` does not discriminate here (0.36 on controls, higher reversed).
   Next: a labelled set of world events for the probe, gas strike and heli crash as their own classes. They would make good showcase labels. 7 strikes are too few, and samples must not ship.
   `lab_sniper_run.ps1 -WorldEvents -RecordSeconds 1000` (mode 11) triggers them; `world.csv` logs each one. First run (051244, a one-minute schedule): crash samples matched (sleigh 0.52-0.66; heli 0.54, 0.75 at 300 and 1000 m), and gas sounded 182-208 s after its
   trigger, on top of the crashes, so that run cannot score heli or gas words. The probe called 4 of its 147 events gunfire (prompts 46): loud world events can fool it, and it never trained on them.
3. Answered for lab modes 8 and 9 (a decoy target): the AI needs both the provoke noise and an armed target at 90 m (54 shots; noise off 0;
   unarmed 0-2), and fires at 0 shots from 110, 130 and 150 m (AKM and SVD; active threat 0.032, not targeted). Why a patrol's AUG fired
   from 150 m in run 4 is not known.
4. AUG named footsteps: FIXED in the app, see "The probe in the app" below. Labelled set: `D:\users\tj\Projects\AssistZ\Lab\label_set_20261104.jsonl`.

## Weather and infected (2026-11-04, mode 12, `Lab/weather_words.py`, run `sniper_20261104_081350`)
- Conditions took: rain 0.9 then 1.0, overcast 1.0, infected at 10, 25 and 50 m. Read on the app at 2a13266, 130 events in 800 s.
- No gunfire word in any phase. Steady rain gives one event in 151 s: the detector passes transients, so the app is nearly silent in rain, and CLAP's "rain" word (23 in all) comes from the storm's bursts.
- Storm: 39 events in 7 clusters 17-30 s apart, about -25 dB, none like them in the rain phase. CLAP says rain and wind, never thunder (0 in the whole run). The probe called 10 of them "sleigh crash": a false world event the app shows today.
- Are the clusters thunder? Not proven. The game's thunder samples match one cluster well (0.745 at 360.8 s) and one weakly (0.342 at 410.7 s); the other five 0.17-0.20, no better than the controls (0.14-0.225). The spacing suits a minimum time of 15 s, which is what the lab set.
- 3 "gas strike" words at 470.5 s, 5 s after the storm was switched off: unexplained. Artillery_Close 0.231 (rain control 0.055, but real strikes 0.40 and up); thunder 0.292 (controls 0.21-0.225); 12.1 s after the previous cluster, under the minimum.
- Infected: "zombie groaning" 8 times, p 0.27-0.43, 0 of 11 events at 10 m, 2 of 10 at 25 m, 5 of 45 from 50 m on, and 1 at 35 s with none within 84 m. Footsteps or wind are the usual words. Nobody checked yet whether the infected made a sound at all (no sample match run).

### Pre-registered: storm test and a thunder class (written before modes 13 and 14 ran)
Prediction, per run (`Lab/storm_gaps.py`, clusters are events under 6 s apart): with the minimum time at 45 s the smallest gap between clusters is at least 30 s (margin for sound travel) and the median at least 45 s; at 15 s the median is under 35 s; in rain with no storm, no more than 1 cluster after the first 20 s. Both orders must agree (modes 13 and 14). If they do not, the clusters are not shown to be thunder and I say so; the class does not ship under that name.
**Result: the prediction FAILED as written on mode 13.** Runs `sniper_20261104_084227` (mode 13) and `085810` (mode 14). Mode 14 passed (45 s: min 58.5, median 64.4; 15 s: median 21.3; rain 0 clusters before an infected attack). Mode 13 did not: at 45 s min 10.3, median 29.1 (needed 30 and 45); rain 3 clusters (at most 1). So by the rule above the clusters are not shown to be thunder yet.
Why, found after looking (so not proof): a strike's sharp start is 3-7 events inside about a second, and a quieter follow-up, usually 2 "rain" events, comes about 9-11 s later and counted as a cluster of its own. Counting only bursts (at least 3 events, each under 0.5 s after the last; `Lab/storm_gaps.py` now), no gap is under its setting in any of the 5 storm phases: 45 s gives 57.4-68.6, 15 s gives 15.7-30.1; the 3 lone rain events in mode 13 are not bursts. Whole-sample match (`Lab/flyby/storm_diag2.py`): 0.114-0.170 in 10 storm windows, 0.061-0.077 in 3 controls, in 9 of the 10 `Norm_Far_3` starting 8-10 s after the burst. Modest, but the same sample and delay repeat.

Second prediction, registered before mode 15 runs (`lab_sniper_run.ps1 -StormLab 15 -RecordSeconds 1000`, settings 90 and 30 s, never used to form the definition). Definition and script frozen at this commit. All must hold: at 90 s every burst gap at least 90, at 30 s every burst gap at least 30 and the median 30-60; rain with no storm: 0 bursts while no infected is within 3 m. If it fails, the class does not ship as thunder; it could still ship under a truthful name such as "storm".

**Result: the second prediction PASSED** (`sniper_20261104_110232`, mode 15, script unchanged since 5688621). 90 s: gaps 136.2, 125.2. 30 s: 39.7-51.7, median 44.5. Rain with no storm: 0 bursts, no infected within 3 m. Across all four settings the spacing follows: 15 gives 15.7-30.1, 30 gives 39.7-51.7, 45 gives 57.4-68.6, 90 gives 125.2-136.2. So the bursts are thunder.

Accidental find, mode 14's last 145 s: an infected reached the invulnerable player at 655 s (`env.csv`: 11 m to 0.64 m, then stays). 215 events: "zombie groaning" 86, rain 64, wolf howling 56, footsteps 8; bursts about 2 s apart. So CLAP's zombie word does respond to an attack; mode 12's spawned infected at 10, 25 and 50 m stayed where they spawned and nothing shows they made a sound. An `infected` class needs its own run with attacks at known times.

Probe, if the second prediction passes: add a `thunder` class. Label: every event in a storm phase; the first 20 s after a storm is switched off are left out of training (the 470.5 s words in 081350).
 Train on 081350 and the mode 13 run; the mode 14 run is never trained on. No refit after looking. To ship, in the real app:
- Held-out storm: at least 80% of storm phase events named thunder; at most 5% named gas strike, heli crash or sleigh crash; 0 gunfire.
- The five recordings never trained on: gunfire 133/133, 150 m AUG 30/30, no shot called gunfire 0/201, at most 2 thunder words, both natural strikes in 021109 still gas strike.
- World runs left out in turn, against the current 84/97, 54/60, 43/49: at most 3 events worse in each.

**Result: every gate above passed, and the probe was NOT shipped.** In the app, fitted on 081350 and 084227 (hash E1B38EF52572, not committed): held-out storms 160/163 thunder (98.2%), 0 gas or crash, 0 gunfire. Five recordings: gunfire 133/133, no shot 0/201, 0 thunder words, app = JSON 385/385, strikes in 021109 all gas strike (10/10, 15/15). World: 84/97, 53/60, 43/49. One new false "heli crash" at 45.8 s in 010112.
The gates missed the real failure: in mode 14's natural infected attack it says thunder on 96 of 215 events, where the five-class probe's CLAP word was zombie groaning 86 times. Reading every phase, not only the gated ones, is what showed it. So gates now cover every phase.

### Pre-registered: an `infected attack` class (written before modes 16 and 17 ran)
Modes 16 and 17 (`lab_sniper_run.ps1 -StormLab 16|17 -RecordSeconds 600`): six infected, one at a time, spawned 2 m from the invulnerable listener for 45 s with 20 s between, then 120 s of storm. Mode 17 uses other infected types. Short test (`aggro_test_sniper_20261104_115929`, distance only, its audio not read or trained on): within 6 s the infected is at 0.64 m and stays, the same distance as the natural attack.
Label: `infected_attack` from 3 s after a spawn until its removal, while env.csv's nearest infected is under 3 m; events in the 3 s after a spawn or removal are left out; storm phases as before, with the same 20 s left out after; everything else other.
Train on the mode 16 run plus 081350 and 084227. Never trained on: the mode 17 run, 085810, 110232, the five recordings. No refit after looking; a miss means new data and new gates. To ship, in the real app:
- Attacks: the mode 17 run, and 085810 from 655 s (natural), each: at least 70% named infected attack; at most 5% thunder; at most 5% gas strike, heli crash or sleigh crash; 0 gunfire.
- Storms: mode 17, 085810, 110232 together: at least 80% thunder; at most 5% gas or crash; at most 5% infected attack; 0 gunfire.
- Every other phase of those runs: at most 5% named thunder or infected attack; 0 gunfire.
- The five recordings: gunfire 133/133, 150 m AUG 30/30, no shot called gunfire 0/201, at most 2 thunder words and 2 infected attack words, both natural strikes in 021109 still gas strike.
- World runs left out in turn: at least 81/97, 51/60, 40/49.

**Result: FAILED, not shipped** (`sniper_20261104_120527` trained, `121745` held out; candidate hash A4C5F2EB65F5, not committed). Clear-weather attacks, new types: 277/319 (86.8%). Storms 192/195 thunder. Five recordings: gunfire 133/133, AUG 30/30, 0/201, 0 thunder, 0 infected attack, strikes 10/10 and 15/15. World 84/97, 53/60, 42/49. But the natural attack in 085810: 113/215 infected attack (52.6%, needed 70), 92 thunder (42.8%, at most 5). It is in heavy rain; every lab attack was in clear weather. "Thunder" may partly mean "a sharp sound in heavy rain". Not proven.
Second finding, in the SHIPPED five-class probe too: 74 "sleigh crash" words in the 473 events of run 121745, same clips, same count. They come from one direction (about az 36), about -40 dB, recurring about every 10 s through every phase; crash samples match 0.04-0.07 (controls 0.04-0.05), so no crash played. CLAP says gunshot, birds, wolf. The earlier "1 false of 166" only covered a few places. The gate missed it because it limited thunder and infected attack only.

### Pre-registered, round 3 (written before modes 18-21 ran)
New data. Mode 18: attacks in heavy rain (mode 16's types), then a storm. Mode 19: survey of 12 sites in clear weather, 55 s each, trains. Mode 20: survey of 12 other sites, including the Kamyshovo area of run 121745, never trained. Mode 21: attacks in rain, mode 17's types, never trained.
Labels. Attack and storm as at cb5cf01; in modes 18 and 21, rest phases and the rain before the first attack are other. Survey: other, leaving out 8 s after each move, anything with an infected under 3 m, and anything from -2 to +12 s around an Artillery_Close match of 0.4 or more.
Train on everything at cb5cf01's list plus modes 18 and 19. Never trained on: 20, 21, 121745, 085810, 110232, the five recordings (121745, 085810 and 110232 have been looked at; none was trained on). Fit once. No refit after looking. To ship, in the real app, on local.7 or a later package shown bit-identical:
- Attacks, each of 121745 (clear), mode 21 (rain), 085810 natural (rain): at least 70% infected attack; at most 5% thunder; at most 5% gas strike, heli crash or sleigh crash; 0 gunfire.
- Storms, 121745 + mode 21 + 085810 + 110232: at least 80% thunder; at most 5% gas or crash; at most 5% infected attack; 0 gunfire.
- Every other phase of those runs and all of mode 20 together: at most 5% of events named gas strike, heli crash, sleigh crash, thunder or infected attack; 0 gunfire. Run 121745's 74 false sleigh crashes count here. No single survey site over 20%.
- The five recordings: gunfire 133/133, 150 m AUG 30/30, no shot called gunfire 0/201, at most 2 thunder words and 2 infected attack words, both natural strikes in 021109 still gas strike.
- World runs left out in turn: at least 81/97, 51/60, 40/49.

**Result: FAILED, not shipped** (`sniper_20261104_130955`, `132217` trained; `135013`, `133615` held out; candidate 2E602B8D0E43, not committed). The rain confusion is gone: natural attack 214/215 (99.5%), 0 thunder. Clear attacks 280/319. Storms 232/235 thunder. Five recordings: gunfire 133/133, AUG 30/30, 0/201, 0 thunder, 1 infected attack, strikes 10/10 and 15/15. World 85/97, 54/60, 42/49. Survey sites: worst 0 of 2 events, but only 11 scored events across 12 sites, so a weak test. Failed: other phases 26/109 false words (23.9%), 21 of them sleigh crash; and false sleigh crash on the five rose from 1 to 7. Mode 21's attacks could not be scored: 0 events, the infected wandered 2-7 m off and never attacked there (they did at every other spawn), reason unknown.
Why sleigh crash goes: the lab builds from `dayzOffline.*`, where `StaticSantaCrash` is `active=0`; it is on only in the Christmas missions. So the class spends false alarms on something normal play does not have. A confidence floor was checked and not used: false sleigh p 0.35-0.55, but at 0.6 only 24/42 true events survive (`Lab/flyby/sleigh_p.py`). Mystery source in 121745: a triple knock every 12 s or so from one direction, 1-4 kHz, not identified; clip sent to TJ.

### Pre-registered, round 4 (written before its fresh runs)
Classes: other, gunfire, gas_strike, heli_crash, thunder, infected_attack. Real sleigh crash rows dropped from training. Training and labels as round 3. Fresh, never trained: two mode 21 runs and one mode 17 run, on local.8. Also never trained, but looked at: 121745, 135013, 085810, 110232, 133615, the five. Fit once; no refit after looking. To ship, in the real app:
- Rain attacks, fresh mode 21 runs together, with at least 100 events scored or more runs are recorded first: at least 70% infected attack; at most 5% thunder; at most 5% gas strike or heli crash; 0 gunfire.
- Clear attacks, the fresh mode 17 run and 121745, each; and 085810's natural attack: the same.
- Storms, all of those plus 110232: at least 80% thunder; at most 5% gas or heli; at most 5% infected attack; 0 gunfire.
- Every other phase of all of those, plus survey 133615: at most 5% named gas strike, heli crash, thunder or infected attack; 0 gunfire; no survey site over 20%.
- The five recordings: gunfire 133/133, 150 m AUG 30/30, no shot called gunfire 0/201, at most 2 thunder, at most 2 infected attack, at most 3 gas or heli words outside proven strikes, both natural strikes in 021109 still gas strike.
- World runs left out in turn: at least 81/97 gas, 51/60 heli. What real sleigh crashes get called is reported, not gated.

**Result: FAILED by one event, not shipped** (candidate 2E5E6A463D2B, not committed; fresh `sniper_20261104_144529`, `145757`, `151013`, `152249`; `151013` and `135013`'s infected did not attack). Rain attacks 399/405 (98.5%). Clear attacks 199/223 and 275/319. Natural attack 214/215. Storms 351/354. Five recordings: gunfire 133/133, AUG 30/30, 0/201, 0 thunder, 1 infected attack, 0 gas or heli outside proven strikes, strikes 10/10 and 15/15. World 83/97, 55/60. The crow clip: 0 crash words. Failed: other phases 7/133 false (5.3%, limit 5): 110232 gas strike 2 before 11 s, infected attack 2 in rain; 121745 thunder 2 at rest; 135013 infected attack 1 at rest. Real sleigh crashes (not gated) read other 34, gas 6, heli 5, gunfire 4; the class is gone, so this is what a Christmas mission would get.
The limit stays. Moving it after looking is what these gates exist to stop.

### Pre-registered, replication of the failed gate (written before its runs)
Candidate frozen: hash 2E5E6A463D2B, no refit, nothing about it changes. Fresh sites never heard: survey modes 22 and 23, 12 sites each, sharing no site with 19 or 20. Run 22; add 23 only if under 100 scored events. Labels as round 3. Gate, `Lab/replicate_gate.py`: at least 100 scored events; at most 5% named gas strike, heli crash, thunder or infected attack; 0 gunfire; no site with at least 5 events over 20%. Pass: this candidate ships, re-verified in the app (`check_app_classes`, live path, M0, tests). Fail: no ship, and the failure goes to TJ with the words that caused it.

**REVERTED: this probe hid real gunfire in heavy rain; see round 6's result.**
**Result: PASSED, SHIPPED** (`sniper_20261104_160723` mode 22, 78 scored; `162444` mode 23, 137 scored). 215 scored; false words 2 (0.9%), both infected attack at 688.9 and 689.4 s in 162444; 0 gunfire; worst site with 5+ events 4%. Words: birds chirping 100, footsteps 58, dog barking 18, wolf howling 14, wind 13, zombie groaning 9. A natural strike found by sample match at 251.0 s in 160723 (0.481) reads gas strike on 6 of 7 events.
Shipped weights 2E5E6A463D2B on local.8. App = JSON 385/385, gunfire 133/133, 0/201, bit-identical to the gate replays on the five, tests 42/42, M0 passes. Live path: run 4 50/50, fire mode 50/50, CLAP median 75.5 ms. Rain attack run 145757, 150 s: infected attack 75/75, from 32.8 s (first spawn about 20 s earlier, the infected reaches the listener after). Storm run 110232, 150 s: thunder 6 from the first burst at 72.7 s, plus 1 stray infected attack at 140.1 s.
Limits: Chernarus only; no attack was recorded with the listener fighting back or moving, and no attack with a storm overhead; a real sleigh crash now reads other 34, gas 6, heli 5, gunfire 4 of 49 events; a few false infected attack and thunder words remain (7 of 133 in the looked-at phases).
### Pre-registered: Nasdara, the held-out map (written before any Nasdara audio)
The shipped probe 2E5E6A463D2B has heard only Chernarus. A measurement of how far it travels, not a ship gate: it stays shipped either way, and a failure means Nasdara data goes into the next fit, with its own held-out runs. `lab_sniper_run.ps1 -Map nasdara -StormLab 17|21 -RecordSeconds 600`, repeated until the infected attack. Same limits as round 4: attacks at least 70% infected attack, at most 5% thunder, at most 5% gas or heli, 0 gunfire; storms at least 80% thunder, at most 5% gas or heli, at most 5% infected attack, 0 gunfire; other phases at most 5% event words, 0 gunfire. Every word in every phase is reported, since new animals and dust storms are exactly what this should show.

**Result on Nasdara: attacks travel, storms do not** (`sniper_20261104_170612` mode 17, `171851` mode 21; `Lab/nasdara_gates.py`). Attacks 536/582 infected attack (92.1%), 0 thunder, 0 gas or heli, 0 gunfire. Storms 37/126 thunder (29.4%, needed 80): heli crash 60, wind 17, gunshot 5. Other phases 1/17 (5.9%), only 17 events. The probe stays shipped, as registered; a player on Nasdara sees false heli crash and some false gunshot in storms.
Why (`Lab/flyby/nasdara_storm.py`): it is the same thunder. ThunderNorm_Near_4 matches 0.859 at 428.5 s, against 0.07 before the storm; bursts 20-26 s apart with the 15 s setting in 170612. Rain stops at 0.3 there. So 10 s of Nasdara storm is 55-93% of its energy below 200 Hz, where Chernarus's is about 80% hiss above 1 kHz. Every thunder example so far was under rain 1.0, so the class learned thunder plus rain. Its other half is unknown: thunder in dry weather on Chernarus was never recorded.

### Pre-registered, round 5 (written before modes 24's runs)
New: mode 24, storm with no rain. Train on the current list plus Nasdara's 170612 and 171851 and one Chernarus mode 24 run. Never trained: one fresh Chernarus mode 24 run, one fresh Nasdara mode 24 run, one fresh Nasdara mode 21 run, plus everything never trained before. Same labels: every event in a storm phase is thunder. Fit once, no refit after looking, on local.8. To replace the shipped probe, in the real app:
- Fresh dry storms, each map: at least 80% thunder; at most 5% gas or heli; at most 5% infected attack; 0 gunfire.
- Fresh Nasdara mode 21: attacks at least 70% infected attack, at most 5% thunder, at most 5% gas or heli, 0 gunfire; its storm with the dry storms' limits.
- Regression, never trained: rain attacks 144529 + 145757, clear attacks 152249 and 121745, natural attack 085810: the same attack limits; rain storms 121745, 135013, 085810, 110232, 144529, 145757, 151013, 152249 together: the same storm limits.
- Every other phase of all of those, plus surveys 133615, 160723, 162444 together: at most 5% gas, heli, thunder or infected attack; 0 gunfire; no survey site with 5+ events over 20%.
- The five recordings: gunfire 133/133, AUG 30/30, 0/201, at most 2 thunder, at most 2 infected attack, at most 3 gas or heli outside proven strikes, strikes still gas.
- World runs left out in turn: at least 81/97 gas, 51/60 heli.

**Result: FAILED, not shipped** (candidate 1C8D5D88E143, not committed; `sniper_20261104_174018` trained; `175558`, `181149`, `182730` never trained). The shipped probe on dry thunder, before: Chernarus 9/216 thunder (4.2%), gas or heli 188, gunshot 7; Nasdara 80/221 (36.2%), gas or heli 75, gunshot 8. So dry thunder is badly misread on both maps by what ships today. The candidate: Chernarus 182/216 (84.3%), gas or heli 18 (8.3%, limit 5), gunshot 7 (limit 0); Nasdara 172/221 (77.8%, need 80), gunshot 6 (limit 0). All else passed: Nasdara attacks 264/271, Nasdara storm with rain 0.3 54/59, rain attacks 398/405, clear 210/223 and 288/319, natural 213/215, rain storms 351/354, other phases 16/386 (4.1%), five recordings as before, world 84/97, 54/60. The blocker is false gunshots from close thunder, which more thunder data did not remove.
Where they sit (`Lab/flyby/false_gunshot.py`): 11 of the 13 are the first event of a thunder burst, about -37 dB, every neighbour thunder. A confidence floor does not separate them: real gunfire goes down to p 0.34, and a 0.7 floor loses 22 of 183 real events.

### Pre-registered, round 6 (written before its runs)
Code: a gunfire word outnumbered by thunder in its own burst takes thunder (`BurstVote.GunfireYields`); a tie keeps the gunfire; it yields to no other word. Off in the app until these gates pass (`?yield=1`, which every gate replay below uses). Its risk is a real shot lost in a thunder burst, so new data measures exactly that: `lab_sniper_run.ps1 -LabelSet -Weather 1|2 -RecordSeconds 360`, labelled AI gunfire under a storm.
Train on round 5's list plus 175558, 181149 and 182730 (looked at, now training). Never trained: fresh Chernarus mode 24, fresh Nasdara mode 24, fresh Nasdara mode 21, the two gunfire-in-a-storm runs, and everything else never trained. Fit once, no refit after looking, on local.8. To replace the shipped probe, in the real app:
- Gunfire in a storm, each of rain and dry: at least 90% of events near a logged shot named gunfire.
- Fresh dry storms, each map: at least 80% thunder; at most 5% gas or heli; at most 5% infected attack; 0 gunfire.
- Fresh Nasdara mode 21: attacks at least 70% infected attack, at most 5% thunder, at most 5% gas or heli, 0 gunfire; its storm with the dry storms' limits.
- Regression: rain attacks 144529 + 145757, clear attacks 152249 and 121745, natural attack 085810: the same attack limits; rain storms 121745, 135013, 085810, 110232, 144529, 145757, 151013, 152249 together: the same storm limits.
- Every other phase of all of those, plus surveys 133615, 160723, 162444 together: at most 5% gas, heli, thunder or infected attack; 0 gunfire; no survey site with 5+ events over 20%.
- The five recordings: gunfire 133/133, AUG 30/30, 0/201, at most 2 thunder, at most 2 infected attack, at most 3 gas or heli outside proven strikes, strikes still gas.
- World runs left out in turn: at least 81/97 gas, 51/60 heli.
- Live path, reported: how long a false gunshot shows before the burst corrects it.

**Result: FAILED, and it exposed a regression in what shipped.** (candidate 0FABE3DE0B0C, not committed; `sniper_20261104_193426` gunfire in heavy rain, 295 shots; `194250` gunfire in a dry storm, 193 shots; fresh `195113`, `200652`, `202233`.) Real gunfire in heavy rain, events near a shot, named gunfire: five-class probe from before e71a4a7 82/96; six-class e71a4a7 27/96, thunder 67; this candidate 28/96 with the yield off, 18/96 on. In the dry storm all three about 103/108. So e71a4a7 was reverted to the five-class weights: its thunder class had only heard rain-soaked thunder and took anything under heavy rain, gunfire included, and no gate had gunfire in rain. The yield also costs real shots there (28 to 18), though it removes false gunshots from dry thunder (6 to 0).
Also failed: Chernarus dry storm gas 13/206 (6.3%, limit 5), other phases 20/378 (5.3%). Passed: Nasdara dry storm 111/118, Nasdara attacks 329/331, Nasdara storm 49/56, rain attacks 397/405, clear 210/223 and 286/319, natural 211/215, rain storms 351/354, five recordings, world 84/97 and 54/60.
Not looked at yet: the detector misses most shots under a storm. Shots with an event near: 104/295 in heavy rain, 133/193 dry. For example AKM at about 25 m 20/119, 75 m 23/28.
Rule for the next gates: gunfire is checked in every weather before anything else.

### Pre-registered, round 7 (written before its runs)
New: `-Weather 3`, heavy rain with no storm. Labels in labelled-gunfire runs under weather (`gunstorm_rows`): near a logged shot gunfire; over 1.5 s from any shot, thunder once a storm is on, other otherwise; in between left out.
Train on round 6's list plus 193426, 194250, one fresh `-LabelSet -Weather 3` run, and 195113, 200652, 202233 (looked at, now training). Never trained, all fresh: `-LabelSet -Weather 1`, `2` and `3`; a Chernarus mode 24; a Nasdara mode 24; a Nasdara mode 21. Fit once, no refit after looking, on local.8. Both yield settings are scored; if both pass everything, the yield stays off. To replace the shipped five-class probe, in the real app, gunfire first:
- Gunfire in each of rain storm, dry storm, rain with no storm: at least 90% of events near a logged shot named gunfire. In addition, on the same clips, no worse than the shipped probe by more than 3 events.
- Fresh dry storms, each map: at least 80% thunder; at most 5% gas or heli; at most 5% infected attack; 0 gunfire.
- Fresh Nasdara mode 21: attacks at least 70% infected attack, at most 5% thunder, at most 5% gas or heli, 0 gunfire; its storm with the dry storms' limits.
- Regression: rain attacks 144529 + 145757, clear attacks 152249 and 121745, natural attack 085810: the same attack limits; rain storms 121745, 135013, 085810, 110232, 144529, 145757, 151013, 152249 together: the same storm limits.
- Every other phase of all of those, plus surveys 133615, 160723, 162444 together: at most 5% gas, heli, thunder or infected attack; 0 gunfire; no survey site with 5+ events over 20%.
- The five recordings: gunfire 133/133, AUG 30/30, 0/201, at most 2 thunder, at most 2 infected attack, at most 3 gas or heli outside proven strikes, strikes still gas.
- World runs left out in turn: at least 81/97 gas, 51/60 heli.
Anything a probe could hide is listed before training this time. Still not recorded, so a limit of any shipped result: attacks during a storm, gunfire on Nasdara, gas strikes and crashes in weather.

**Result: FAILED, not shipped; last storm round, by TJ's priority** (candidate 347BA848BDD7, not committed; never trained `sniper_20261104_215004`, `215827`, `220700`, `221544`, `223129`, `224711`). Gunfire, candidate against shipped on the same clips: rain storm 97/103 against 63/103; dry storm 116/118 against 115/118; rain with no storm 127/127 against 74/127. So the earlier 82/96 for the shipped probe was one kind recording. Failed: Chernarus dry storm thunder 184/238 (77.3%, 79.4% with the yield), gas 26 (10.9%, limit 5), gunshot 5 (0 with the yield); Nasdara dry storm gunshot 3 (0 with the yield); false gunshot in attacks 4 of 265 on Nasdara and 1 of 405 in rain; 1 in rain storms. Passed: Nasdara thunder 154/160, attacks 92-99% everywhere, rain storms 350/354, other phases 7/375 (1.9%), five recordings, world 84/97, 54/60. Full side by side: `Lab/flyby/side_by_side.py`. **SHIPPED on TJ's decision, 2026-11-04, despite the missed limits, with the yield on (`?yield=0` to compare).** It names gunfire far better than what it replaces. The missed limits stand as known flaws: Chernarus dry storm thunder 79.4%, false gas strike 10.9%, false gunshots in attacks 5 of 670.
TJ, 2026-11-04: progress on what a player sees has stalled; direction, distance, weapon names and video come next. First numbers, not gated (`Lab/quick_truth.py`), 19 recordings, each run left out in turn: shots with an event 1579/2271; left or right 796/821 where at least 15 deg off centre; angle not proven, shooters spanned about 12 deg and only 30 of 1579 were behind; no front or back cue; weapon 68% (commonest 31%): SVD 97%, AKM 65%, AUG 64%, M4A1 54%; distance band 75% (always "near" 65%), 0 of 33 beyond 120 m.

Crows: the game has Crow and Raven samples (`sounds_ai`, `sounds_environment`); they match TJ's clip at 0.24-0.29 against 0.13-0.16 in controls (`Lab/flyby/crow_check.py`). Supports TJ's ear; weaker than the gas strike's 0.98.

## The probe in the app (2026-11-04, early results on SpawnDev.ILGPU.ML 5.8.2-local.3; the shipped six-class probe and its numbers are in "Weather and infected" above)
- local.5 against local.3, RX 590, run 081350, publish folder emptied first so only local.5 could load: 130/130 same clip and label, scores and embeddings bit-identical (max difference 0). So local.3 results carry over. Expected memory cut not seen: peak dedicated 1229 -> 1217 MB, private 2290 -> 2280 MB. Tuvok: reuse does engage for CLAP (712 planned rents per run, 0 deviations), the pool only goes 36 to 20 MiB, so that is about the expected drop.
- Memory leak, in the library, on local.3 and local.5: `?memlog=1` (`Lab/mem_read.py`) shows `PadKernel.Forward < PadOperator.Execute` holding 1212 live buffers at event 100 and 3612 at event 300: 12 per event, 88 bytes each, outside BufferPool (allocations flat at 181). Chrome's dedicated memory climbs with it, 1050 to 1353 MB over 319 events, about 0.95 MB per event; FIXED in local.6 (Tuvok: PadKernel's parameter buffers cached by content). Same replay: live buffers flat at 4480 from event 3 to 319, dedicated flat at 1128 MB (peak 1141, was 1353), 319/319 same clip and label, scores and embeddings bit-identical to local.5. Measure: `replay_mem.ps1 -Tag _mem -Query '&embed=1&memlog=1'` on 085810.
- local.8 (split-K in small Linear layers; capture off by default in the library): same replay against local.7, capture off: median 73.8 -> 70.6 ms, p90 83.8 -> 78.9, max 144.2 -> 140.9; same clip and label 319/319; scores differ by at most 7.1e-07, embeddings 1.8e-07; dedicated 1128 -> 1142 MB.
- local.7 records the audio tower and replays it. On the RX 590, same replay: live on WebGPU (2995 dispatches), bit-identical 319/319, but median 74.4 -> 73.9 ms, p90 84.6 -> 89.7 ms, max 149.7 -> 112.6 ms, dedicated 1128 -> 1830 MB, storage 654.6 -> 992.1 MB, buffers 4480 -> 8917. Tuvok's RTX 4070 went 40 -> 22-23 ms, so it depends on the card. Off by default here, `?capture=1` to test; with Tuvok.
- The probe names gunfire, gas strike, heli crash and sleigh crash; the prompts name everything else; cadence splits single and automatic. `ClassifyWithEmbeddingAsync` gives both from one tower run. `?probe=0` compares. The weights ship in `wwwroot/models/sound-probe.json`, fitted by
  `Lab/flyby/fit_sound_probe.py` (needs the game's samples locally: `Lab/flyby/world_events.py` extracts them) on 10 lab runs plus 3 world event runs; re-verify in the app after any refit. A probe is tied to its checkpoint, so with `?clap=larger` the prompts decide.
- Naming world events (`Lab/flyby/world_classes.py`, `world_fp.py`), each world run held out in turn: gas strike 84/97 events, heli crash 54/60, sleigh crash 43/49. Gunfire no worse: 133/133, 0/208. On the five unseen recordings it flagged 37 no-shot events: 35 in real natural gas strikes proven by sample
  match (all three strikes there, one I had not known about), 2 false of 173 (1.2%). 4 of the 35 got the wrong event word. Also: the 166 s false alarm below was the sleigh crash at 300 m. Limits: 27 triggered events in all, one map, and strikes the game fires naturally sit in the training "other".
  Heli at 2500 m mostly gives no detector event at all, and one 2505 m strike was not located.
- In the shipped app (with the two fixes below): decisions equal the JSON's 385/385, gunfire 133/133, 0/201, ?probe=0 bit-identical. Live, both natural strikes in 021109 are named "gas strike" start to finish while CLAP's own word is machine gun fire. Left: one false "sleigh crash" at 49.0 s in 010112 (1 of 166 no-shot events outside a proven strike).
- `BurstVote`: events under 1 s apart share the burst's most common world event word; gunfire takes no part. Simulated held out: mixed bursts 9/62 -> 0, gas 91/97, heli 57/60, sleigh 43/49. The app logs `relabel` lines; the replay files keep `raw` too.
- The detector's background starts at the first audio, not at the -70 dB floor: the 1-5 false events at 0.2-1.2 s in most recordings are gone, every later event's clip and scores are bit-identical, M0 passes twice. Red-checked on real audio.
- Five recordings it never trained on: gunfire 133/133 (CLAP's word 123/133), 150 m AUG burst 30/30 (20/30), no shot 0/208 (36/208). Live path, run 4, twice: 50/50 near a report (was 40/50), fire mode 50/50, labels 58/58 between runs.
- World events, three runs, each left out in turn: 4/477 called gunfire (trained without them 11/477; prompts 160/477). The one left in most runs is at about 166 s, always "machine gun fire": the sleigh crash at 300 m, its final impact. Smallest gunfire margin +0.35 (was +0.58).
- The three world runs are `sniper_20261104_053416, 055325, 061239`, with `world.csv`; gas times come from the Artillery_Close match, since a strike sounds minutes after its trigger. Only tested on DayZ.

## Gunfire naming (2026-11-04)
- Cause (Tuvok): transformers on the same clips gives 31/31 identical labels, p within 0.019: the checkpoint, not the port.
- Collected with the game's logs as truth: `lab_sniper_run.ps1 -LabelSet -RecordSeconds 360` (mode 8: AUG, AKM, M4A1, armed decoy, AI cadence 0.15-0.28 s;
  597 shots in 3 runs, nearly all at 40 and 65 m) and `-Footsteps -RecordSeconds 190` (mode 10: an unarmed AI on a waypoint loop).
- Probe: class-balanced logistic regression on CLAP's 512-d audio embedding (`?embed=1` dump, `Lab/probe_eval.py`). On data it did not train on:
  gunfire 618/619 (prompts 616/619); no shot called gunfire 1/169 (33/169); run 4's 150 m AUG burst 30/30 (20/30); real footsteps 0/55.
- Could not break it: shuffled labels fall to near chance; weapons never trained on AUG 166/166, AKM 122/122, M4A1 100/100, and SVD-only training gets
  388/388 automatics and 25/30 on the burst (`Lab/probe_checks.py`). Not loudness: footsteps +30 dB 0/66 where a level threshold says 66/66; run 4 at -20 dB
  38/43 where level says 0/43 (`Lab/probe_gain_test.py`).
- Clip normalization to -12 dBFS (`?norm=-12`, chosen on run 4 and footsteps, scored on the other ten): burst 29/30, gunfire 617/619, no shot 30/169, footsteps
  2/62, and more wrong words ("dog barking" 5 -> 17). Cheap, and it shows the prompts' error is tied to clip level, but it does not touch the false alarms. Off by
  default. Tuvok reproduced the level effect in transformers on the same 30 clips (raw 20/30, -24 dBFS 9/30, -18 18/30, -12 29/30, -6 30/30): the checkpoint, not
  the port. CLAP's features are log-mel dB with no per-clip normalization, so gain shifts every feature and quiet transients read as footsteps. He advises against
  normalizing for the prompts because it erases loudness as a cue, and letting the probe decide gunfire.
- Tuvok's plan, after his sweep: `ClassifyWithEmbeddingAsync` (scores and embedding from one tower run), `ClapProbe` with Fit, Predict, ToJson, FromJson. His gate is
  the table above, reproduced from the labelled set on his backends.
- Fresh test, three recordings collected after the probe was fixed (sniper_20261104_021109, 021644, 024720), `Lab/analyze_env.py`: gunfire 89/89 (prompts 89/89); no shot called gunfire 0/137 (prompts 36/137); AI
  walking 0/59. 024720 has no shots at all in 600 s: prompts flag 11 of 100, the probe none.
- Limits: one burst beyond 90 m; non-gunfire is only footsteps, wind-like sounds and one low rumble, with no vehicles, speech or music, so the probe answers
  gunfire or not, on DayZ only. Not tested on another game or a stream. The rumble at 170-183 s in three recordings is not gunfire (slow 4 s rise, centroid
  100-230 Hz vs 790 Hz) and is not identified. Real-time path not measured with the probe.
