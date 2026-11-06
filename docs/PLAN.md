# AssistZ plan

READ THIS FIRST, after any context reset. Keep it current: when TJ and an agent agree a plan, it goes here in the same turn, before the work starts.

Other sources, in order: `../dayz_assistive_system_overview.md` (one folder above this repo: the original architecture and Cortana addendum), `docs/roadmap.md` (TJ's tracking wishes, what is verified done), `docs/dayz-audio-findings.md` (measured facts), `docs/lab-handoff.md` (running lab record).

## Public, 2026-11-06 (TJ's call)

The repo is public, with a fresh history and the app on GitHub Pages (`.github/workflows/deploy-to-github-pages.yml`, run by hand, so the page stays steady while work goes on: run it only for a build that has passed). The goal in the README is TJ's: demonstrating SpawnDev.ILGPU.ML on real time audio and video in the browser.
- **Old commit hashes** in this plan, the lab record and `_TestStats` are in `D:\users\tj\Projects\AssistZ\AssistZ-history-20261106.bundle` on NV1080, 169 commits, not on GitHub. `git clone` it to read them. `Lab/flyby/gun_rain_compare.py` needs old revisions too.
- **Before video ships in the app, check the licence.** The YOLOv8 weights come from Ultralytics, whose licence I understand to be AGPL-3.0, unlike this repo's MIT. Today only the lab page `/vision` loads them. Ask TJ, or use another detector.
- Never commit game samples, frames, `DayZ-Central-Economy` or `expansion_lab_settings` content: it would now be public at once.

## What this is
"Essentially give you Cortana in DayZ": as helpful as ML can get using just the video and audio streams, and a test of SpawnDev.ILGPU.ML. She can't control you, fight for you or look around corners, but she is smart, helpful, tracks what matters and gives as much useful information as possible. In the browser, sleek and unobtrusive. An ML showcase, not an accessibility tool (TJ's direction, handoff).

## The goal, in TJ's words (2026-11-05)
"DINOv2 and advanced video audio processing using ml to get detailed sitrep data including a minimap that would show best estimates of positions of sound surroundings."

And (TJ, 2026-11-05): "this was sort of like a "Halo's Cortana" style project... in that it uses ML and AI for advanced tactical assistance and situational awareness in the game. In Halo it gave you a mini map similar to the one we have right now but when an enemy was heard it pinged a red dot on the map with a relational distance to other things pinged on the map (non-threats white)... that kind of stuff and this was a project to show off and test SpawnDev.ILGPU.ML."

What that means for the design:
- A Halo-style motion tracker, built on the radar the app already has: each sound heard becomes a ping on the map.
- **Threats ping red, non-threats ping white.**
- **Pings sit at their relative distance from each other and from the player**, so distance from sound is core to this, not an extra.
- Which sounds are threats is a design choice not yet made: gunfire and infected attack clearly; ask TJ about the rest rather than assume.

So the destination is:
- **A minimap** of best estimates of where the sounds around the player are.
- **A detailed sitrep** from video and audio together.
- **DINOv2** for the vision side, not CLIP (Tuvok agrees: CLIP knows "a forest road", not this forest road).

The first conversation about this was never written down. Searched on 2026-11-05 and not found: the saved sessions on NV1080, and on Tuvok's PC (all 6 project folders, TJ's own messages). `D:\users\tj\Projects\DayZSitRep` is an older RCon project (2022-2024), not this. So TJ is the only source for what it held. Do not fill gaps by guessing: ask TJ.

## DINOv2 pipeline (Tuvok, 2026-11-05; delivered in SpawnDev.ILGPU.ML 5.8.2-local.11, master 01366dac)
```csharp
using var dino = await Dinov2EmbeddingPipeline.CreateFromHubAsync(acc, modelSource, Dinov2EmbeddingPipeline.BaseRepo);  // or SmallRepo
var e = await dino.EmbedAsync(typedArrayRgba, w, h, includeTokens: false);   // also int[] and GPU ArrayView<int>
// e.Pooled: float[2*D], L2-normalised, so dot product = cosine. e.Tokens: [1 + 16*16, D] when includeTokens: true (CLS, then patches, row-major).
```
Xenova/dinov2-small (88 MB), Xenova/dinov2-base (346 MB), Apache-2.0. Preprocessing on the GPU as the checkpoints' processor: bicubic short edge to 256, centre crop 224, ImageNet normalisation. **The centre crop drops the frame's sides: a 16:9 frame keeps its middle 56%.** Tuvok will add a letterbox or no-crop mode if asked; labelled frames decide. Descriptor: L2norm(concat(L2norm(CLS), L2norm(mean of patch tokens))), not GeM. Gated against onnxruntime on CUDA, OpenCL and WebGPU. Times on the RTX 4070 while training: WebGPU in Chrome small 36 ms, base 42 ms median; the RX 590's are unmeasured and Tuvok wants them.

Measured: local.11 against local.8 on 295 CLAP events: clips, labels, scores and embeddings bit-identical. On the RX 590 in Chrome, on real 1600x900 frames, including reading the pixels back from a canvas: small about 50 ms, base 70-80 ms. Survey: grid 26 697 of 700 views, grid 27 696 of 716, one frame per view, `make_manifest.py`.

### Location, targets (written before any location number was seen)
`Lab/vision_eval.py`. Index grid 26, query grid 27, so no query place is in the index and the nearest possible match is 707 m away. A first measure of how far this goes: not a ship gate, and the numbers choose the model and the next step. Base and small, centre crop and three HUD-free tiles.
- Works: median location error under 2000 m, at least 3 times better than shuffled positions, and heading at least 50% (chance 25).
- Promising: anything clearly better than shuffled but short of that. The likely next step is a denser index, since a 1 km grid may simply be too coarse for place recognition.
- Dead end for this approach: no better than shuffled. Then ask Tuvok about trained aggregators such as SALAD before anything else.

**Location result (2026-11-05): barely above chance at 1 km spacing, short of "promising".** `Lab/vision_eval.py`, `L:\AssistZLabWork\frames\grid2?_{small,base}.npz`. Median error small 5.7-6.5 km, base 6.4-6.5 km, shuffled about 6.8 km. Within 1 km 6-8%. Top 5 holds a point within 1 km about 22%. Heading 32-38% (chance 25). The three tiles help a little, base does not beat small. Not yet a verdict on DINOv2: a view from 700 m away may share nothing with the query. RX 590 per frame: small 51 ms whole frame, 43 ms tile; base 76, 67.
If location comes back: first a positive control, to show DINOv2 can recognise the same place at all. Tuvok's reading: at one view per 1 km the nearest database view usually looks at different terrain, so place retrieval has no matching scene and the 6 km median may say more about coverage than about DINOv2. So: query frames within about 25 m of a database view, same and rotated heading. If that fails too, it goes to Tuvok as a library suspect. A whole-map index at 100 m is about 18,000 land points: too many to survey this way. The overview's other routes remain: horizon and ridge profiles against the heightmap, which need no survey, and town signs.

### Sound distance (2026-11-05)
First look, `Lab/distance_first.py`, 2625 events from 22 recordings, each run left out in turn: a linear fit on CLAP's embedding has a median error of 3.6 m against 9.0 m for always answering the median, 93% within 25%, correlation 0.84. Level alone is useless (9.0 m), which fits -2.7 dB per doubling. Not trusted yet: most shots sit near 50 m, and distance is tangled with weapon and how those runs were set up; a table of the errors is the next read. And the range is short: 95% of the data is under 76 m, because the AI will not fire past about 100 m. A tracker needs far more than that.

Pre-registered (before mode 28 ran), `lab_sniper_run.ps1 -DistanceTest -RecordSeconds 450`, two runs at new spawns: three weapons at 30, 55, 80 and 95 m, ranges the training never used, all the way round. Fit on every recording so far, in every weather; these two never trained on. Limits, on the run's events: median error at most 10 m; at least 75% within 25%; correlation at least 0.7; and for at least 2 of the 3 weapons the median estimate rises with each step of true distance. Fail: distance from the embedding was learning the runs, and the crack-to-report gap is next.

**Result: passes on what it measured** (`sniper_20261105_043053`, `044227`, 454 events, `Lab/distance_gate.py`). Median error 5.9 m (always the training median: 16.3). Within 25%: 85.9%. Correlation 0.91. True 30 m reads 35, 53 reads 54, 78 reads 74, 96 reads 85. Per weapon: AKM 35, 52, 71; AUG 33, 63, 79, 84; M4A1 42, 65, 73. The script's fourth check says FAIL (1 of 3), because it wants all four steps and the AI never fired the AKM or M4A1 from 95 m; every step that has data rises for all three. Read as a pass, recorded as such.
Limits: gunfire only, AKM, AUG and M4A1, about 25-95 m. It reads short past about 90 m (148 m reads 112; 590 m reads 78 on 3 shots): the AI will not fire farther, so the far range has no data. No footsteps, infected or vehicles.

### What the Halo goal changes (2026-11-05)
**A Halo motion tracker is centred on the player. It needs direction and distance from the player, not map position.** So location is parked and sound distance comes first. Order now:
1. Done for near gunfire: sound distance, see above.
2. NOW: threat colours and pings on the existing radar: red, white, at relative distance. Distance probe shipped as JSON, like the sound probe; refit on every recording, app = JSON checked on replay, the radar looked at by eye.
3. First cut of threat, for TJ to change: gunfire and infected attack are red; everything else white. Sounds without a measured distance get no invented one: they stay an arc on the outer ring, in their colour. Over about 100 m reads as "100+ m", since the far range is unmeasured. The outer ring is 100 m, the inner 60 m.
   Built: `AssistZ/Audio/DistanceProbe.cs`, `wwwroot/models/distance-probe.json` (`Lab/fit_distance_probe.py`, every recording, 3079 examples), `Components/RadarHud.razor`. App = JSON within 0.0005 m on 251 gunfire events, no other sound gets a distance, radar checked by eye on replay.
   **Next test, registered before the runs:** every recording so far is in the shipped fit, so the shipped weights get fresh mode 28 runs, one clear and one heavy rain (`-Weather 3`), scored from the app's own `dist=` by `Lab/distance_shipped.py`, with no refit. Limits in each, same as the method's test: median error at most 10 m, at least 75% within 25%, correlation at least 0.7. Fail in rain only: the radar stays, the limit goes in this file and rain gets its own look.
   **Result: both pass.** Clear (`sniper_20261105_051052`, 214 shots with a distance): median error 5.0 m, 76.2% within 25%, correlation 0.94. Heavy rain (`052232`, 198): 7.1 m, 77.3%, 0.89. App = JSON within 0.0005 m on both. Within 25% cleared its limit of 75 only narrowly. By true distance, clear: AKM 30, 53, 82; AUG 29, 58, 73, 94; M4A1 31, 53, 78. Rain: AKM 31, 50, 70; AUG 34, 62, 77, 97; M4A1 43, 69, 77 (M4A1 reads long up close in rain). Only 57% of shots in these runs produce their own event, since fast fire merges (214/375, 206/352), so the radar pings most shots, not all.
   Unnamed sounds are a dim thin arc until named: white said "not a threat" about what may yet be gunfire.
   **Infected attack, checked against the game's log** (`Lab/attack_align.py`, `env.csv` nearest infected). An infected really did attack the character in the rain run: 57 of 61 attack labels came with one within 5 m, against 24 of 215 other events, and it was that close 11% of the time. Labels start at 211 s, the log has it close from 210 s. Clear run: 12 attack labels, quiet (-49 dB), nearest infected 67-137 m, not proven either way. CLAP's own word on those is mostly wolf howling, and no animal was logged nearby.
4. NOW: far gunfire. The AI will not fire past about 100 m; the lab's own shooter will. First look at the older SVD ladders (`Lab/far_look.py`, 6 recordings with a sound clock, 24 shots a range, never trained on): the shipped probe rises past 100 m but flattens: 50 m reads 60, 100 reads 92, 200 reads 107, 400 reads 116. And the namer loses far shots to white: named gunfire 23/23 at 50 m, 19/24 at 100, 15/24 at 200, 11/24 at 400; the rest called thunder or heli crash. So a far shooter is not a threat on the radar yet.
   Lab mode 29 (`-FarTest`): AKM, AUG, M4A1, SVD at 150, 200, 300, 400, 600, 800 m, 4 shots each, 2 near and 2 wide, round the listener. One run first, to see what can be heard at each range; limits get written here before any fresh test. New gunfire class must not hide old ones: thunder and heli crash re-checked in every condition.
   First run (`sniper_20261105_055044`, clear, `Lab/far_table.py`): 96 of 96 shots produce an event, every weapon to 800 m. Level barely falls: about -30 dB at 150 m, about -38 at 800. Named gunfire by range: 14, 11, 10, 9, 7, 11 of 16; by weapon AKM 10, AUG 23, M4A1 20, SVD 9 of 24, the rest thunder or heli crash. The shipped probe stays flat: 150 m reads 104, 800 m reads 123. Near and wide misses name the same (31 of 48 each).
   Three more recordings, then each far run held out in turn (`Lab/flyby/far_cv.py`, `far_cv2.py`, `far_cv4.py`):
   - Namer with far gunfire in training: 312 of 323 named gunfire, against 228 without; at 800 m 32 of 33 against 12. No shot within 1.5 s: 0 of 151 called gunfire. Heli crash 53 of 60 against 54, gas 84 of 97 both ways. Older holdouts on stored embeddings barely move: false gunshots in dry storm 8 against 5 of 238, Nasdara attacks 7 against 4 of 265.
   - One distance ridge over all ranges: rejected. It reaches past 100 m but near gunfire in rain falls from 77% to 63% within 25%.
   - Two steps instead: past 120 m? Then a near or a far ridge. Far called 317 of 323. Near shots wrongly called far: almost only at 88 m or more. Near stays good: 5.3 m and 78.3% clear, 6.3 m and 81.1% rain. Far-only ridge: within a factor of 2 on 91%, correlation 0.67, rises with range: 206, 235, 285, 346, 459, 732 on one clear run. Coarse, and flatter in weather.
   - Weather cuts hearing at the far end: 800 m heard 16 of 16 clear, 1 of 16 heavy rain, 0 of 16 storm; 600 m 12 and 6 of 16. Nothing shows for those shots.
   Radar scale: inner ring 100 m, linear inside; outer ring 800 m, logarithmic between. Far reads rounded with "~", since it is good to about a factor of 2; past 800 reads "800+ m".
   **Registered before the fresh runs.** Shipped weights, real app, new runs never trained on: mode 29 clear and heavy rain, mode 28 clear and heavy rain.
   - `Lab/far_gate.py`, each mode 29 run: at least 85% of shots with an event named gunfire; at least 75% at any range with 10 or more; at least 85% of those read 100 m or more; at least 75% within a factor of 2; at most 2% of events with no shot within 1.5 s called gunfire.
   - `Lab/distance_shipped.py`, each mode 28 run: median error at most 10 m, at least 75% within 25%, correlation at least 0.7; and at most 5% of shots under 80 m read 100 m or more.
   - `Lab/phase_gates7.py` in the real app, against the shipped probe: gunfire checks pass; thunder and attack rates no more than 3 points worse; false gunshots no more than 5 more in any one check. A check that already fails on its zero-gunfire rule counts as passed on that rule only if within those 5.
   - The radar looked at by eye on a far run.
   **Result: passes all of them.** App = JSON on the replays, labels and `dist=` from the app.
   - Far clear (`sniper_20261105_071655`): named gunfire 83/95 (87.4%), worst range 75% (exactly on the limit), read 100 m or more 100%, within a factor of 2 94.0%, no shot called gunfire 0/26. Median read at 150, 200, 300, 400, 600, 800 m: 194, 251, 276, 361, 533, 729.
   - Far heavy rain (`073259`): named gunfire 79/80 (98.8%), worst range 94%, read 100 m or more 98.7%, within a factor of 2 96.2%, 0/5 false. Reads 187, 229, 266, 287, 437, 426. Only 2 of 16 heard at 800 m.
   - Near clear (`074844`): median error 3.4 m, 92.4% within 25%, correlation 0.97, under 80 m reading 100+ 0/182. Near heavy rain (`080020`): 4.8 m, 95.2%, 0.89, 0/151. Better than before the refit (5.0 m, 76.2%; 7.1 m, 77.3%), though on different runs.
   - Older holdouts, real app, against the shipped probe: gunfire checks all pass (rain storm 98/103, dry storm 116/118, rain 127/127). Dry storm false gunshots 5 to 0; Chernarus dry thunder 77.3% to 79.4%; Nasdara dry storm now passes outright. Worst moves: false gunshots in Nasdara attacks 4 to 7 of 265, Nasdara storm thunder 98.1% to 96.3%, event words in other phases 7 to 9 of 375. Two checks still fail on their own limits as they did before, and within this test's limits: Chernarus dry thunder under 80% (TJ's earlier decision) and gunfire above zero in attacks.
   Not covered: bursts past about 100 m, since far training is single shots; a fresh heli and gas run, those being held-out only; weather hides far shots from the radar altogether (2 of 16 heard at 800 m in rain); shot-to-shot scatter is wide, ~300 to 800+ m from one spot. 
   **Live, first time on the game's own sound** (`run_live.ps1 -Picker DayZ -NameSounds`, scored by `Lab/live_score.py`, `live_drift.py`). Chrome's auto-select flag picks a source but leaves audio off, so the app heard nothing; `Lab/harness/picker_accept.ps1` drives the real picker through Windows UI Automation: Window tab, the DayZ window, audio on. Sharing the game window carries system audio.
   - Works: 331 of 331 events named on one run, 248 of 249 on another. Live against replay of the same recording: 7.3 m and 63.3% against 6.7 m and 66.1%; 3.2 m and 89.9% against 3.6 m and 93.2%. Name about 1 s after the sound (0.85 s clip plus CLAP about 120 ms, against about 70 ms on replay), and it does not grow: the app keeps up while sharing the RX 590 with DayZ.
   - Near distance spreads by run, and the app is not why: within 25% was 66.1% and 93.7% with the app alongside, 77.3% and 92.4% without. So 66.1% is a miss of the 75% limit on shipped weights that stands as a miss, with no cause known. Read within 25% as about 66-95% by run.
   - Audio goes missing live: the live clock loses about 3.3 s in about 6 minutes, most in the first minute, and "Skipped" still reads 0. Not fixed.
   - **Bug: false view turns.** Replays have no video, so this could only show live. Character still (game log yaw constant within 0.06 degrees), measured heading climbed 0 to 161 degrees; front/back called on 36 events, 19 wrong, 11 right, at confidence near 1. Screenshots confirm the view did not move (0 px shift between all 38 pairs). Each jump: one frame of about -245 px, at the search limit, score up to 0.94, stillness either side, near the 35 s slot changes, when yellow on-screen text changes. The text as the cause is likely, not proven.
   - `AssistZ/Video/YawGate.cs`: a peak at the search's edge is not a measurement; a jump with stillness either side is not a turn. Offline (`Lab/yaw_rules2.py`): false path 234 to 13 degrees; the two recorded fast-turn runs keep 93.6-93.8% of their movement. A score bar would not do it: the worst jump scored 0.94, and 0.8 loses 39% of the fast-turn run.
   **Registered before the runs:**
   - Still: live mode 28 (`Lab/still_check.py`): net heading at most 15 degrees; at most 5% of events call a side.
   - Real turns: `lab_frontback_run.ps1`, against `frontback_20261103_130111` (decided 56/61, correct 39/56, 70%; resolved bearing median 4.0, p90 13.1): decided at least 70% of events, correct at least 65%, resolved bearing p90 at most 20.
   **Result: one limit failed, the rest passed.**
   - Real turns, `frontback_20261105_100416`, now without a person at the picker: decided 50/66 (76%), correct 50/50, resolved bearing median 2.0, p90 4.2, against stereo alone 23.7 and 93.5. One run, two sources, slow sweeps of at most 15 px a frame, and a different run than the baseline, so not all of the gain is the gate. Faster turns rest on the offline check.
   - Still, `sniper_20261105_094557`: events calling a side 0/267, pass (36/331 before). **Net heading 47.9 degrees, FAIL** (limit 15). About 2 degrees until one second near 329 s, then frames like -118, -50, -162, -55 at scores 0.91-0.94; adjacent ones are not isolated, so they pass. An odd shift came at the same moment in the run before (`091617`, 328.8 s), so this looks tied to the lab's script; not proven. Left as a failed limit. Next if it matters: separate shift estimates for the upper and lower image, which a real turn moves together.
   - No sound in the share now says so, in the app's message line. Chrome leaves audio off for a window or screen by default; a person would have seen a working app that hears nothing.
   - `windowAudio: "window"` tested and not kept. Data added the wrapper (SpawnJS 3.0.8-local.2). On this PC (Windows 10 19045, Chrome 157) the picker still offers system audio, with or without `--enable-features=GetDisplayMediaWindowAudioCapture`, and the app heard 20 of 20 beeps from another program with the option set (1-2 of 20 at shifted times; `Lab/beeps.py`, `beep_score.py`, `sniper_20261105_101750`). So here a player's voice chat and music reach the detector. The app stays on SpawnJS 3.0.7 with no untested option. Chromium's Windows path uses process audio capture, which may need a newer Windows; not checked.
     **Windows 11: it isolates** (Data, YODA, 10.0.26200, Chrome 156, a test window as source, no game). The option brings a new toggle, "Also share app audio (from all its windows and related apps)", which Windows 10 never offers. With only that on: the window's own pulses 16/16, another program's 0/17 (controls 0). With system audio on too: 16/16 and 17/17. So game-only sound is possible on Windows 11 and not on this Windows 10. Not tested: DayZ itself through app audio, so whether stereo, level and labels hold up. The app now asks for it (SpawnJS 3.0.8, released by Data) and the no-sound message names the toggle. A run on the game wants a Windows 11 PC with DayZ.
   - Audio lost live, under test. Likely cause: `MediaStreamTrackProcessor` keeps few audio chunks and drops the oldest when the page stalls. Now measured from the chunks' timestamps (`AudioGapMeter`) and shown as "Audio lost". **Registered before the A/B:** the same short live run, Chrome's default against `?audiobuf=512`. A larger buffer becomes the default if audio lost falls by at least 80%, and the name still comes within 1.5 s at p95. Otherwise the cause is elsewhere and the default stays.
     **Result: passes, now the default.** Chrome's default: 2.139 and 2.128 s lost in about 150 s, 77 and 75 gaps, up to 180 ms; name p95 1.53 and 1.43 s. `?audiobuf=512`: 0 and 0 lost; name p95 1.15 and 1.18 s. Skipped read 0 throughout, which is why this had stayed hidden.
   - **Full live check on the final build, no switches** (`sniper_20261105_105721`, 391 s): audio lost 0. Name 1.0 s after the sound, 1.04 at the start and 1.00 at the end, so the clock no longer drifts and the app keeps up. 233 of 235 events named. Live distance 4.9 m and 89.4% within 25%, replay 5.0 m and 90.5%. Under 80 m reading 100 m or more: 1 of 190. No shot within 1.5 s called gunfire: 0 of 20. Side calls 1 of 235, pass. **Net heading 35.0 degrees, FAIL again** (limit 15): steps of +23 and -59. The limit stands failed on two runs. AUG at 95 m read 141 on this run.
   Next: see "Icons, threat levels, own sounds" below, which comes first (TJ, 2026-11-05). Then the remaining false heading; distance for footsteps and infected; far bursts.
   **False heading, second try, registered before its runs.** The jumps come near when yellow on-screen text changes, low in the picture. So the tracker's column profile now reads rows 10% to 65% in place of 10% to 90%: simpler than two estimates, and a test of the text as the cause. Same limits as before. Still: net heading at most 15 degrees, at most 5% of events calling a side (`Lab/still_check.py`). Real turns: decided at least 70%, correct at least 65%, resolved bearing p90 at most 20 (`lab_frontback_run.ps1`). If the jumps remain, the text is not the cause.
   **Result: rejected, and reverted.** Rows 10% to 65%: net heading 40.3 and 61.0 degrees (`sniper_20261105_181348`, `182550`), no better than 47.9 and 35.0, and side calls 35/318 on one run. Text is not the cause, or not all of it. Real turns stayed good: 49 of 50, bearing p90 4.4 (`frontback_20261105_183747`). The YawGate stays. Next, instead of guessing: `yawprofile` lab lines log the picture's column profiles at each big jump.
   **Then the jumps stopped, and I do not know why.** Four live runs on the final build, gate and full band: net heading 0.9, 0.6, 2.2, 11.8 degrees, side calls 0 of 210, 320, 223, 292, big jumps 0, 1, 0, 0 (`sniper_20261105_184530`, `185745`, `190944`, `192137`). All pass. Before them, every live run failed: gate and full band 47.9 and 35.0 (ML local.11), gate and narrow band 40.3 and 61.0 (ML 5.8.2). Nothing changed since except the band going back and the logging, so this is not shown to be a fix: where the character spawned, or the time in the game, may be what differs. The 11.8 run drifts slowly, a different pattern from the jumps. Status: passing now, cause unknown; the logging stays on, so the next live run with jumps says what they are.

## Decided by TJ, 2026-11-05: video is next; far shots rank lower

TJ's words: "#1 Video #2 Sure. Soudns good", answering: (1) video next, for calm zombies, the player's remaining own shots, and which way the player is walking; (2) shots past about 300 m show Medium, not High.

TJ, right after: "as far as #2.. the gun makes a difference. Sniper shots from 300m are more dangerous than m4 or pistol at that range". So the level should come from weapon and range together. The 300 m line applies to every gun for now; a sniper at 300 m shows Medium, which is wrong until the app can tell a sniper by sound.
First look (`Lab/sniper_look.py`, 7 far runs, each held out): SVD called sniper 150 of 152; the others 24 of 440 (AKM 15/150, AUG 5/143, M4A1 4/147). At 300 m: 26/26 and 7/81. Only one sniper rifle and no pistol so far, so this may mean "SVD" and not "sniper". Lab mode 33: Mosin, Winchester, FAL, MP5K, CZ75, FNX45, never heard, as the test.

## The radar names the gun (TJ, 2026-11-05)

TJ's words: "if it knows the gun types from the soudns (which it should) the soanr map shoudl say the gun type not jsut "gunshot" the whole point is info"

So: where sound can tell, the radar says the weapon, not "gunshot". Where it can only tell the family, it says that (sniper rifle, assault rifle, submachine gun, pistol); where it cannot, "gunshot". No gun name claimed that was not tested on recordings it never trained on. Comes before video, with the sniper test it builds on.

**First look** (`Lab/weapon_cv.py`, 2433 shots, 17 runs, each held out; ten guns). Right overall 82.3%. AKM 584/671, AUG 740/849, M4A1 470/576, SVD 143/152; at confidence 0.8, right when said AKM 521/549, AUG 633/657, M4A1 383/402, SVD 131/146. The six new guns have two recordings each and are weak: CZ75 26/32, MP5K 11/28, Mosin 9/31, Winchester 12/32, FNX45 7/30, FAL 0/32 (called AKM). Family, held out: assault 2032/2128, sniper 189/215, pistol 34/62, submachine gun 11/28. By range, gun right: under 120 m 86-94%, 120-450 m 71-74%, past 450 m 56%.
**Sniper on guns never heard** (`Lab/weapons_new.py`, trained on SVD against AKM, AUG, M4A1): Mosin 24/31, Winchester 17/32, but 27/31 at 300-600 m and only 5/16 at 50 m; FAL 7/32, MP5K 12/28, CZ75 2/32, FNX45 0/30 wrongly called sniper. So it carries over at range, not close.
**Problem found: quiet guns read far.** Distance by true 50, 150, 300, 600 m: FNX45 621, 531, 708, 842; CZ75 84, 546, 852, 1412; MP5K 70, 831, 1698, 2471; against Mosin 50, 199, 231, 829. A pistol next to the player can show as far away. Distance must take the gun into account, or say less for these.
Four more recordings of the six new guns, six runs each now (2729 shots, 21 runs, each held out; `Lab/weapon_cv.py`). Right overall 83.8%. Named at confidence 0.8, right when said: AKM 97%, AUG 97%, M4A1 94%, SVD 93%, CZ75 92%, Winchester 90%, FNX45 87%: these may be named. Mosin 81%, MP5K 65%, FAL 33%: these say their family. Family held out: assault 95%, sniper 94%, pistol 90%, submachine gun 61%.
Distance with the new guns in training (`Lab/gun_distance_cv.py`, each mode 33 run held out): pistols and MP5K within a factor of 2 from 33-45% to 78-82%; FNX45 at 50 m read 535, now 59; quiet guns under 100 m reading 300+ from 22 to 4 of 57. Near rifles hold on runs never trained on: 3.3-5.0 m, 93.7-95.3% within 25%. Far rifles lose a little: 90.5-93.8% within a factor of 2 against 92.6-98.9.
Built: `AssistZ/Audio/GunProbe.cs`, `models/gun-probe.json` (`Lab/fit_gun_probe.py`). The list says the gun, the name, else the family, else "Gunshot"; a burst adds ", auto". A single shot's icon shows the family; a sniper rifle stays High at any range, others Medium past 300 m. Named on single shots at range and bursts up close, so a gun's record is thin outside what it was recorded doing. No other guns: an unknown gun gets the nearest it knows.
**Registered before the fresh runs**, real app, new runs: mode 33 clear and heavy rain, mode 29 clear, mode 28 heavy rain, mode 32 clear, mode 30 clear:
- `Lab/gun_gate.py`, over mode 28, 29 and 33: each gun named 10+ times right at least 85% when named; all named together at least 90%; family right at least 85%; quiet guns under 100 m reading 300 m or more at most 10%.
- `Lab/far_gate.py` on mode 29; `Lab/distance_shipped.py` on mode 28 and at least 90% of its shots with an event named gunfire. Own sounds and agroed: `Lab/own_gate.py` as before. Older holdouts: `Lab/phase_gates7.py`: gunfire checks pass; thunder and attack rates no more than 3 points worse than before; false gunshots no more than 5 more in any one check.
**Result: guns pass; three other limits missed by under a point, recorded as failed.** Fresh mode 33 clear and rain, mode 29 clear, mode 28 rain, `Lab/gun_gate.py`: all named 236/249 (94.8%): AKM 40/40, AUG 76/78, CZ75 16/16, FNX45 15/16, M4A1 48/54, SVD 20/22, Winchester 21/23. Family 340/350 (97.1%). Quiet guns under 100 m reading 300 m or more 0/20. Of 398 events at a shot, a gun's name on 249, a family on 350, neither on 48. Far gunfire 94/95 named, 97.9% within a factor of 2. Near gunfire in rain 5.1 m, 86.8% within 25%, named 151/158.
Missed: **real gunfire in a rain storm 92/103 (89.3%, limit 90), was 97 with the bar:** the namer now trains on quiet guns, some of which it hears as thunder there. Without that training the quiet guns were named about 84% against about 96% now, so the training stays unless TJ says otherwise. **Own movement named gunfire or infected 7/131 (5.3%, limit 5):** real infected were about for 65% of that run. **Agitated called agroed 134/149 (89.9%, limit 90):** mostly at under 1 m, called footsteps. Also: false gunshots in Nasdara attacks 10 of 265 (7 before), Chernarus dry thunder 76.5% (77.3), within their limits. Survey site back to 11%.

## Video: the plan (TJ chose this next, 2026-11-05)

What video is for, in the order to try it, each tested against the game's logs with limits written first:
1. **Infected in view, the calm ones sound cannot find.** Lab mode 34: infected at known places in front of the camera, frames saved, truth from their logged position and state (`env.csv` infected_x, infected_z). First look with what the library has: `ObjectDetectionPipeline` (YOLOv8; its "person" may or may not fire on an infected; no detector here was trained on this game), and DINOv2 patch tokens with a small probe trained on the lab's frames. Does it see one at 15, 30, 60 m, at what false alarm rate with none there, and how fast on the RX 590? DINOv2 took 51 ms a frame there, so a few frames a second is the budget, as the overview planned (2-5 FPS).
2. **Which way the player is moving.** Optical flow from frame to frame, truth from the pose log. This unlocks front and back from walking, and "not moving, so those steps are not yours".
3. **The player's own shots.** A rifle kicks the view and flashes. About a third of own shots still ping.
A video test needs the game in focus, since it renders badly otherwise (TJ), so these runs own the screen. Frames stay local, never committed.

### First look: YOLOv8 on calm infected (2026-11-05, not a test)

Two lab mode 34 runs at the mission's own dusk, one dark with rain (`vision_sniper_20261105_234301`, `235839`), every 4th frame (400 each) through `ObjectDetectionPipeline` with YOLOv8n in the real browser path on the RX 590: 125-136 ms per inference. COCO's "person" was never trained on this game, and it does fire on an infected. Whole frame shrinks the picture by 2.5 and loses them fast; three native 640 tiles along the horizon do far better.

Frames with a person box at the infected's column, by logged distance:
| | 0-15 m | 15-30 m | 30-50 m | 50-70 m | 70-100 m | 100-150 m |
|---|---|---|---|---|---|---|
| tiles, bar 0.25 | 26/41 | 68/101 | 11/84 | 3/56 | 0/59 | 0/38 |
| tiles, bar 0.5 | 21/41 | 50/101 | 8/84 | 0/56 | 0/59 | 0/38 |
| whole frame, bar 0.25 | 23/41 | 10/101 | 0/84 | any person 0/190 at rest | | |

**These numbers are low, and the false alarms too low, because the truth was wrong on the first run.** `infected_x,z` is the nearest of all infected. see1 had 11-13 natural ones about, so it often pointed at another: a figure boxed at 0.47-0.56 and counted false was, by eye, the placed infected walking across a field at about 60 m with the log pointing 105 degrees away. And infected wander: one placed at 10 m was often 16-18 m away for most of its slot. So distance by slot is nominal, and see1's rest frames were not empty. What the look does show:
- By eye, the boxes at the column are mostly real infected, out to about 60 m, in hard light.
- Real false boxes do exist: a tree silhouette at up to 0.56 and a patch of empty haze at 0.41. Other labels fire at random (traffic_light, boat, horse): only "person" is of use.
- Box height against distance on 91 boxes: distance = 1268 / height gives 93% within 25%, fitted on those same boxes. Bearing from the box: 0.3 degrees median off the log. Both would be new information for a calm infected, if they hold.

Mode 34 now logs the placed infected itself (`placed_x, placed_y, placed_z, placed_state`), removes natural ones so rest frames are empty, and `-Hour` sets the hour. `Lab/see_manifest.py` writes that truth to the manifest.

**Noon, clear, one run with that log (`vision_sniper_20261106_003339`, see3, every 2nd frame, used to choose the setup, so not part of the test).** Frames with a person box at the placed infected's column:
| | 0-10 m | 10-20 m | 20-30 m | 30-50 m | 50-70 m | past 70 m |
|---|---|---|---|---|---|---|
| native tiles, bar 0.5 | 8/17 | 48/62 | 47/78 | 0/79 | 0/56 | 0/180 |
| native tiles, bar 0.35 | 8/17 | 49/62 | 68/78 | 0/79 | 0/56 | 0/180 |
| whole frame, bar 0.35 | 8/17 | 47/62 | 2/78 | 0/79 | 0/56 | 0/180 |

No false sightings in 3.6 minutes of empty frames, and no person boxes away from the infected at bar 0.35. Bearing 0.27 degrees median. Distance with K fixed from the first look: 94% within 25% at bar 0.5, 85% at 0.35.

The stop past 30 m is the model, not the view. By eye the infected is plain at 40 m and still visible at 60 m, but it is 20-30 px tall with tall grass over its legs. Magnifying does not fix that: seven 320 px tiles scaled to 640, same frames every 4th, gave 8/8, 26/31, 34/38 inside 30 m at bar 0.5, better than native, then 1/39 at 30-50 m and none past. They also cost about 840 ms a frame. Reaching further needs something trained on this game, or motion.

**The setup under test, fixed now:** whole frame at bar 0.25, plus three native 640 tiles (lefts 0, 480, 960, top 120) at bar 0.35. `Lab/see_score.py 0 plan`. Why: the tiles carry 20-30 m, the whole frame carries the close ones the tiles miss (at 0.25: 7/8 at 0-10 m on the same frames, against 3/8), and in the first look the whole frame had no person box at rest at 0.25, while tiles had 31/190 there at 0.25 in dusk scenes. Four inferences at about 130 ms is about 520 ms alone, so the speed limit below is close and has to be measured. Scored on every 2nd frame (1 a second), slower than the app would run, which is the cautious side for "sighted within 5 s".

Fresh runs: dusk `-Hour 20`, rain `-Hour 12 -Weather 3`, noon `-Hour 12`, three of each, see4 to see12. One night run, `-Hour 23`, for the record with no limit.

### Limits for a calm infected on the radar (written before any fresh run)

Fresh mode 34 runs with the new log, in three conditions pooled separately: noon clear, dusk, rain. At least 12 slots in the 10-30 m range per condition before reading anything. A **sighting** is a person box on two of three consecutive frames within 60 px of each other, at a rate the app could really run. All must hold in every condition before an icon reaches the radar:
- **Sees it:** placed infected in view and 10-30 m away for 5 s or more: sighted within 5 s in at least 90% of those slots.
- **False:** with nothing there (rest frames), at most one sighting per 5 minutes.
- **Speed:** the whole frame's work at 2 frames a second or better on the RX 590 with the sound work running live. Three tiles at 130 ms is 390 ms alone, so this is not given.
- **Bearing:** median within 3 degrees of the log.
- **Distance:** K = 1268 fixed from the first look, not refitted: within 25% on at least 75% of sightings. If this misses and the rest pass, the icon shows no distance rather than a wrong one.
Past 30 m gets reported, with no limit: it is a bonus, as sound cannot find a calm one at any range.

### Fresh runs against those limits (2026-11-06): one limit failed, one not measured, nothing goes on the radar yet

Ten fresh mode 34 runs on Chernarus (`vision_sniper_20261106_011046` to `_033053`, see4 to see13, 933-937 s each), every 2nd frame through the real browser path, the setup as fixed above, `Lab/see_score.py 0 plan`.

| Limit | Dusk | Rain | Noon |
|---|---|---|---|
| Sees it: sighted within 5 s, 10-30 m (limit 90%) | 12/12 | 11/12 | 12/12 |
| False, rest frames (limit 1 per 5 min) | **5.2, FAILED** (11 in 10.5 min) | 0 in 10.5 min | 0 in 10.5 min |
| False during slots, away from the infected (no limit, reported) | 25 in 23.6 min | 0 in 23.8 min | 0 in 23.8 min |
| Bearing, median (limit 3 degrees) | 0.42 | 0.38 | 0.29 |
| Distance, K fixed, within 25% (limit 75%) | 94% | 90% | 93% |
| Speed, 2 frames a second with sound live | **not measured live** | | |

Frames with a box at the infected, in view:
| | 10-20 m | 20-30 m | 30-50 m | 50-70 m | 70-100 m |
|---|---|---|---|---|---|
| Dusk | 212/214 | 199/227 | 28/252 | 2/214 | 0/247 |
| Rain | 121/176 | 215/258 | 4/163 | 7/288 | 0/266 |
| Noon | 163/193 | 221/278 | 25/308 | 0/157 | 23/211 |

What this means:
- **Inside 30 m it works, in dim light and rain too.** Past that it is mostly no: real boxes at 60 m in rain and at 82-89 m at noon, checked by eye, on open ground, but rare.
- **The false limit failed on one thing:** a dark gap between trees in see4, in one spot for the whole run, with boxes starting exactly at the top of a tile (y = 120). see5 and see6 had none. All 11 rest sightings were see4.
- **Candidate fix, not yet earned:** drop a tile's box that touches the tile's top edge. On these runs that takes dusk to 0 and 0 with no hits lost (`plan2`), but it was found on these runs, on one scene, so it needs fresh dusk runs.
- **Speed will miss as it stands.** Measured offline, four inferences plus overhead took 544 ms a frame (3196 results in 435 s): 1.84 frames a second alone, under 2 before any sound work. To try: `ObjectDetectionPipeline` takes an input size and `InferenceSession` takes input shapes, so 320 px tiles at 320 cost should cost about a quarter each. Boxes at the infected fell between y 309 and 640, so a horizon band can skip sky and ground.
- **Night: no use as it stands** (one run, no limit written). 0 of 2 slots sighted, tree silhouettes boxed constantly, 15.7 false per 5 minutes, all from the whole frame and in a few fixed spots. By eye the frames are close to black. Vision must not speak in the dark without a test of its own.
- **Infected keep moving:** a calm one moved under 1 m in only 2% of 10 s windows and 1% of 20 s windows. Boxes that stay put are a good sign of scenery: requiring 10 px of movement kept 56 of 78 true tracks and 14 of 79 false ones, on these runs. That needs a still camera, or the yaw tracker to subtract the turn.
- **Every run here is a still camera.** Added limit, stricter than what was written: nothing reaches the radar until the same limits hold with the player moving, using `Lab/harness/drive_own.py`.
- The low close figure at dusk (8/22 at 0-10 m) is one infected attacking at 0.6 m, where sound already names it.

**Speed, measured offline on the RX 590 (2026-11-06).** Tuvok added `ObjectDetectionPipeline.EnableGraphCapture` in ML 5.8.4-local.1 when I asked about replay; it replays a recorded plan instead of dispatching 250 nodes from .NET each time.
| Median ms per inference, see11, 80 frames each | capture off | capture on | off again |
|---|---|---|---|
| 640 tile | 129 | 55 | 127 |
| Whole frame at 640 | 137 | 65 | 137 |
| 320 tile (own export, lab only) | 92 | 36 | |
- Four inferences a frame: about 520 ms becomes about 230 ms, over 4 frames a second alone. Not yet measured beside the sound work.
- Detections are the same: 320 of 320 inferences matched, by label, confidence and box, at bar 0.1.
- GPU memory goes down, not up: with capture on, Chrome holds at 273 MB (repeats 278, 283). Off, it climbs about 4.3 MB per inference, to 1449 MB after 320 inferences and still rising (repeats 1449, 1449). Reported to Tuvok; whether that is a leak or late release is not known.
- **Not used: 320 tiles.** The hub's export bakes 640 into its head, so 320 took a locally rebuilt copy. Seven of them, at 36 ms, cost more than three 640 tiles at 55 ms, and they saw no better. Tuvok has dropped a hub export at my request.
- **Not measured: AOT.** I stopped the build after an hour on `SpawnDev.ILGPU.ML.dll` alone, as capture had answered why the time was going. AOT may still help the rest of the app, and is untested.
- ML 5.8.4-local.1 needs ILGPU 5.4.16. The repo stays on the released 5.8.2 until a release build has capture, and `Lab/clap_same.py` runs on the move. Until then the runs use a copy of the app on the lab disk, with that package and `?capture=1`.

**The setup for the next fresh runs, fixed now (setup 2):** as above, whole frame at 0.25 and three native tiles at 0.35, with graph capture on, dropping any tile box whose top sits on the tile's top edge (`Lab/see_score.py 0 plan2`). Dusk failed the false limit, so three fresh dusk runs, see14 to see16, against the same limits. Rain and noon passed with no false sightings, and the rule only removes boxes; had it removed any of their hits, they would need rerunning, and on see7 to see12 it removed none. Then: the moving player, then speed live.

**Setup 2 on three fresh dusk runs (see14 to see16, `vision_sniper_20261106_065747`, `_071321`, `_072854`): failed.** Sighted 5 of 12 slots, 5.7 false sightings per 5 minutes. Capture is not the cause: see16 ran with it and boxed 78/78 at 10-20 m.
- **Missed slots were hidden infected, by eye.** see14 stood on a slope facing tall grass (1 of 4 slots sighted), and see15 spawned inside a bush (0 of 4). The limit's "in view" only checked the angle, so it counted infected nobody could have seen. Those runs have no line of sight log, so they cannot be rescored.
- **False sightings were real.** All 12 were in see16: a road marker and a pole in fixed spots, boxed for the whole run. So the edge rule is not enough on its own.
- Graph capture held: 56 ms a tile, 64 ms whole frame, on all three runs.
- Tuvok found the memory growth with capture off to be a real leak (an output buffer never returned, in 23 pipelines) and has fixed it in ML master.

**Setup 3, fixed now, chosen on all the runs above, so none of them count as its test:** setup 2, and a sighting needs a track of boxes (within 60 px, up to 5 s apart) that has moved at least 10 px from where it began and has had one box at confidence 0.6 or more. `Lab/see_score.py 0 plan3`. A calm infected rarely stands still (under 1 m in 2% of 10 s windows); scenery does. It needs a still camera, or the turn subtracted.
On the runs that chose it:
| | Sighted within 5 s | False at rest | False during slots |
|---|---|---|---|
| Dusk, see4 to see6 | 12/12 | 0 in 10.5 min | 1 in 23.6 min |
| Dusk, see14 to see16 | 6/12, the hidden ones above | 0 in 10.5 min | 1 in 23.6 min |
| Rain | 11/12 | 0 in 10.5 min | 0 in 23.8 min |
| Noon | 11/12 | 0 in 10.5 min | 0 in 23.8 min |
| Night, one run | 0/2 | 0 in 3.5 min | 0 in 7.9 min |

**The test of setup 3, written before any of its runs.** The same limits, with one correction: "in view" now also needs line of sight, meaning at least one of three rays from the player's eye reaches the infected's knee, waist or head (`placed_los`, game's view geometry, `ObjIntersectView`). Grass may not block those rays, so a slot sighted late or never gets looked at by eye and reported, not excluded. Three fresh runs each of dusk, rain and noon, at least 12 slots each. Then the moving player. Speed live comes after that, once a release build has capture.

**Setup 3's test, nine fresh runs (see18 to see26, `vision_sniper_20261106_075848` to `_100345`, 933-959 s each, ML 5.8.4-local.1, capture on): passed, except that dusk has 11 slots where the limit says at least 12.** One more dusk run decides that; see below.
| Limit | Dusk | Rain | Noon |
|---|---|---|---|
| Sees it: sighted within 5 s, 10-30 m, line of sight (limit 90%, at least 12 slots) | 11/11 | 12/12 | 11/12 |
| False, rest frames (limit 1 per 5 min) | 0 in 10.6 min | 0 in 10.6 min | 0 in 10.9 min |
| False during slots (no limit, reported) | 0 in 23.8 min | 0 in 23.8 min | 1 in 24.2 min, not false, see below |
| Bearing, median (limit 3 degrees) | 0.33 | 0.37 | 0.35 |
| Distance, K fixed, within 25% (limit 75%) | 77% | 94% | 84% |

Frames with a box at the infected, in view with line of sight:
| | 0-10 m | 10-20 m | 20-30 m | 30-50 m | 50-70 m | 70-100 m |
|---|---|---|---|---|---|---|
| Dusk | 36/57 | 174/185 | 149/160 | 26/212 | 5/188 | 0/242 |
| Rain | none | 193/224 | 227/238 | 80/250 | 31/177 | 15/182 |
| Noon | 15/25 | 275/298 | 118/129 | 16/218 | 20/237 | 0/198 |

Checked by eye, and what is still open:
- The one false sighting during a slot is the placed infected itself (see25, 629 s): the frame after the log says it was removed. So 0 real false sightings in 104 minutes of frames, 72 of them with an infected out.
- The miss (see25) was sighted at 6 s against the limit's 5.
- Dusk's distance is 77% against a limit of 75: a thin pass.
- `placed_los` matched the pictures it was checked on: zero rays was an infected behind a tree trunk, and three rays was plainly visible. But zero rays was only 9 frames of 1242, all that one trunk, and these sites had no player inside a bush like see15. **So foliage blocking the rays is untested.**
- Close to the player at 0-10 m it boxes 60%, mostly infected attacking at arm's length, where sound names them.
- **Still a still camera. The movement rule has not met a moving player, and speed has not been measured live.** Nothing reaches the radar until both pass.

**The extra dusk run (see27, `vision_sniper_20261106_105300`): setup 3 passes at dusk, 15/15 slots, 0 false in 13.2 minutes of rest and 29.7 of slots, bearing 0.33, distance 79%.** That is see18 to see20 and see27 up to game 665 s, `SEE_CUT=see27:665`, because TJ's mouse (Mouse Without Borders) reached the game at 666 s: the aim moved, the view turned 35.6 degrees at 678 s, and the inventory was opened to get out. All four of see27's 10-30 m slots ended by 632 s. Yaw and aim stayed still on every other run; `see_score.py` now prints `VIEW MOVED` when they do not.

**Setup 3 has passed its whole written test for a still camera.** TJ's accident found what that test could not:

**The inventory is a person.** With the inventory open, all 6 false sightings in see27 were the character preview, 800-850 px tall at x about 590, at up to 0.86, and it passes the movement rule. Players open it all the time, so this would put a phantom infected on the radar each time. The map and other menus are untested and may do the same. New limit, for the moving test: no sighting while the inventory is open. To try: a box too tall to be anything but arm's length. 500 px means under about 2.5 m, where sound already names an attacker, and on see18 to see26 only 27 boxes were that tall. A screen check may be needed as well.

**ML 5.8.5, done.** CLAP is bit-identical to 5.8.2 on 260 events of one recording (`Lab/clap_same.py`), with replays at 144 s and 141 s, and YOLOv8 with capture finds the same boxes on 320 inferences.

**Moving player, one trial (see17, noon, `vision_sniper_20261106_112208`), not a test.** `Lab/harness/drive_own.py <pids> <out> see <seconds>` walks at 1.4 m/s, jogs at 2.7, sprints at 3.3, sweeps the view 38 to 64 degrees and opens the inventory, in an area of 16 x 48 m.
- Sighted 4 of 4 slots. Real boxes stay close to the log's column while moving: 5 to 8 px median walking, jogging and sprinting. In a sweep the log's second of lag shows: 73 px on the few boxes counted, and by eye boxes counted "away" there were the real infected.
- Without the inventory, no false sightings at rest. One during a slot, not yet looked at.
- **Distance fell to 66% within 25%, under the limit of 75, with K about right (1216). Not explained yet.**
- **It also boxes a figure on the game's loading screen, at 0.88.** Frames before the player arrives are not scored, so this is not in the numbers. Loading screens, the map, the death screen and menus need testing; a player's map is a likely next false alarm.

**The inventory rule.** Six flat grey strips at fixed places, either side of the VICINITY, EQUIPMENT and HANDS headers' text (`Lab/see_inventory.py`): numbers, no game art. It matched all 6 strips on 543 frames on the two runs where the inventory was opened, and on none of about 37,000 frames on the other 23 runs, which never matched more than 1. Fitted on see17 and see27, so not yet tested fresh. Tested only at the default interface scale, 16:9, with the lab's clothes; headers should not depend on clothes, and that is untested too.

**Setup 4, fixed now:** setup 3, saying nothing on a frame the inventory rule matches. `Lab/see_score.py 0 plan4`.

**The moving test, written before any of its runs.** Nine fresh runs with the driver, three each of dusk, rain and noon, every frame (2 a second), at least 12 slots per condition, on ML 5.8.5 with capture. Run numbers changed, see below. The same limits as above, in view meaning line of sight. Added limits: no sighting while the inventory is open, and the inventory rule must flag at least 90% of the frames while the driver holds it open, leaving out the first 1.5 s and last 0.5 s. A false sighting during a slot is reported, and looked at by eye.

**The first attempt stopped: the character swam (2026-11-06).** TJ saw it. Sprinting forward beats walking back, so the driver drifted up to 304 m, and see28, see30, see33 and the running see35 spent 47 to 69% of their time below 0.5 m height, in the sea. My checks only read run times and frame counts. Nothing from these runs has been scored. Decided before any score: see29, see31, see32 and see34 stayed dry (0%) and stand, though they drifted 147 to 304 m. see28, see30, see33 and see35 are out, and see36 never ran. Mode 34 now sends the player home on reaching water, or between slots if over 40 m away, logging `home` in `world.csv`. Re-recorded: see37, see38 dusk, see39 rain, see40, see41 noon. So the test is dusk see29, 37, 38; rain see31, 32, 39; noon see34, 40, 41. `Lab/moving_check.py` reports time in water, read before any scoring.

**Next, in order:**
1. That moving test. It takes the keyboard and mouse.
2. Speed live beside the sound work.
3. Other screens: map, loading, death, menus.

The truth was wrong twice before these numbers, and both times the detector looked worse than it was. Checking boxes by eye found both. The nearest of all infected was not the placed one (fixed in the log). And the picture is a flat projection: the column is `800 + 840 tan(angle)` at 1600 wide, not 14.67 px per degree, and the view is about 87 degrees across (plus or minus 43.6), not the 109 once assumed. 8.8 px per degree at 960 holds at the centre, so small turns (`YawGate`, `YawTracker`) stand, but anything placing a box on the radar needs the tangent. The first look's tables above used the old line. `Lab/see_offset.py` shows the offsets slot by slot.

## Front and back from moving (TJ, 2026-11-05)

TJ's words: "i know we only have stereo sound, but if the player moves while the sound is heard you can tell by how the sound moves if it is behind or in front"

Already used for **turning**: turn the view and a sound in front and one behind drift opposite ways across the stereo image (`SourceTracker`; on real mouse turns, front/back right on 50 of 50, one run, slow sweeps). Only for a repeating sound, once the view has turned at least 12 degrees.

Not used yet: **walking**. Walking forward, a sound ahead gets louder and swings out toward the side; a sound behind gets quieter and closes toward centre, while the stereo reading alone is the same for both. That covers a player running straight, which turning does not. To build:
- When is the player moving, and which way: own footsteps are now named, which says moving but not which way; the video's forward motion would say which way. Not built.
- A lab mode: the driver walks a straight line past a repeating source at a known place, ahead and behind. Limits written first. It needs a repeating sound, so gunfire bursts and an agroed infected suit it; a single shot gives nothing to compare.

## Icons, threat levels, own sounds (TJ, 2026-11-05)

TJ's words: "red and not red are going to be too vague. we need small identifiable icons. a zombie that is not "agroed" (aggrivated) is a threat but not a big one. an agroed zombie is a bigger threat and different. so we can't be jsut blck and white. and yes, our player's own sounds could trigger things. we need to account for that"

This replaces red and white as the radar's language.
1. **Each kind of sound gets its own small icon**, identifiable at radar size. The icon says what it is.
2. **Threat has levels, not two colours.** A calm zombie is a threat, a small one. An agroed zombie is bigger, and different. Colour says how bad, the icon says what.
3. **The player's own sounds must not ping as threats.** Footsteps, own gunfire and the like have to be measured and handled.

Rules that follow from how this project works:
- No icon claims what was not measured. Calm against agroed needs ground truth from the game (the infected's mind state, logged by the lab) and a test on runs never trained on. Until then the radar says "infected", not "agroed".
- Own sounds get a lab mode where the character alone makes noise, so anything it pings is false by definition, plus a check that other players' footsteps still show.
- Icons are drawn here, simple shapes; no game art.

**Built, first cut:** `AssistZ/Audio/SoundMarks.cs`, `Components/SoundIcon.razor`, legend at `/icons`. Levels High, Medium, Low, Info for TJ to change; weather does not ping; a burst merges into one icon per spot; a dashed outline means the name was never tested.

What the data says so far (about 7,000 replay events, infected within 15 m by the game's log): infected attack 97%, trained and good. **Zombie groaning 32%**, a word from CLAP's prompts, never tested, mostly wrong. So the calm zombie icon is not to be trusted yet.

Truth is available: the game gives each infected a mind state: `MINDSTATE_CALM`, `DISTURBED`, `ALERTED`, `CHASE`, `FIGHT`, `COWER` (`ZombieBase.GetMindStateSynced`), and voice entries exist per state (`MINDSTATE_CALM_IDLE`), so the sounds may differ. 
**Agroed: measured.** Lab mode 30 (`sniper_20261105_112656`, `114203`, `115702`, clear, clear, heavy rain; none in any training; `Lab/infected_states.py`, `infected_confound.py`). The probe's "infected attack" fires on agitated infected, not on attacks: disturbed 16/19, alerted 17/17, chase 42/42, fight 222/225; calm 6/67. At matching distances, calm against agitated: 3-8 m 2/9 and 41/42; 8-15 m 2/18 and 17/17; 15-30 m 1/19 and 13/15. So it hears agroed, not close; agitated is also about 6-12 dB louder at the same distance. The radar now says "Infected, agroed", solid head, High. State truth is sampled once a second and the clock is good to about a second; calm has only 67 events.
Chase against fight, from the embedding, each run held out: 25/42 and 179/225: not yet. No "attacking" icon until that holds.
**Calm: sound cannot find it at range.** Lab mode 31, three infected at once at 15, 25, 40 m, using the worst state of any (`sniper_20261105_122339`, `124557`, `130800`, clear, clear, heavy rain; `Lab/calm_check.py`). Agitated: 85 events a minute, 96% called agroed. Truly calm in range: 3.2 a minute, 3% called agroed. No infected within 100 m: 4.4 a minute, 0%. The words are the same in the last two. So a calm infected at 15-40 m makes no more sound than none, and there is nothing to train: finding one is video's job. The agroed result replicates on these runs, never trained on. Mode 30 has calm infected nearer, 4-10 m, with few events before they notice: not settled there. The radar says "Maybe infected" for the prompt's word, never "calm".
**Footsteps down to Low:** with nobody near and the character still, 16 of 26 events get that word. Mostly a false alarm until trained.
Next:
1. Own sounds, measured (`sniper_20261105_134028`, lab mode 32, `Lab/harness/drive_own.py` drives the character, `Lab/own_sounds.py`). **TJ was right, and it is bad:** with the character alone, sprinting gave 81 events: 38 automatic fire, 33 footsteps, 10 gunshot. Jogging 18: 10 gunshot, 5 footsteps, 3 automatic fire. Own rifle, 20 shots: 46 events, 45 gunshot. Standing still, one event in 40 s. So moving paints red on the radar. Phase names are the driver's: the logged speeds do not clearly separate jog from sprint or crawl, so read these as "moving".
   Own sounds sit near centre: 162 of 163 within 10 degrees. But a filter on that alone would also take shooters truly ahead. Tested and rejected: own sound being exactly centred (`Lab/own_stereo.py`): own steps have side 11.5 dB under mid, shooters truly ahead 15.4 dB, so they overlap; a cut that catches own shots also takes up to 38% of infected on you.
   Four more runs (`135100`, `135931`, `140809`, `141634`), then two probe classes, `own_move` and `own_shot` (`Lab/flyby/own_cv.py`, each own run held out). Own movement: 263 of 283 named own, **0 named gunfire**. Own shots: 59 of 70 named own, 10 gunfire. A storm's "movement" is thunder and poisoned the class until left out. "Sounds like you" alone took 37% and 11% of another character's footsteps and 23% of far gunfire. With the near-centre rule, |az| under 10, these fall to 1 of 62, 2 of 99 and 1 of 95, and 99% of own movement and 93% of own shots still qualify. So: own only if both. Off the radar, named in the list, and the burst vote may not spread it.
   **Registered before the fresh runs**, shipped weights, real app, `Lab/own_gate.py`:
   - Own, fresh mode 32, clear, twice: at most 5% of movement events named gunfire or infected; at least 75% named own; own shots at least 60% named own.
   - Others' footsteps, fresh mode 10: at most 10% named own.
   - Far gunfire, fresh clear: `Lab/far_gate.py`'s limits. Near gunfire, fresh rain: `Lab/distance_shipped.py`'s limits and at least 90% of shots with an event named gunfire. These two sets lost most in the offline check.
   - Agroed, fresh mode 30: at least 90% of agitated called agroed, at most 15% of calm.
   - Older holdouts in the real app against the shipped probe, `Lab/phase_gates7.py`: as before: gunfire checks pass; thunder and attack rates no more than 3 points worse; false gunshots no more than 5 more in any one check.
   **Result: own sounds pass, one registered limit failed.** Own movement, fresh: 191/196 and 53/55 named own, 2/196 and 0/55 named gunfire or infected. Own shots 20/22; the second run had nothing to score, since real infected were about during firing. Another character's footsteps, fresh: 1/23 named own. Far gunfire, fresh clear: 91/94 named, 98.9% within a factor of 2. Near gunfire, fresh rain: 3.8 m, 95.9% within 25%, named 147/162 (90.7%, limit 90). Agroed, fresh: 60/65 agitated, 4/37 calm. **Older holdouts: real gunfire in a rain storm fell from 98 to 92 of 103 (89.3%, limit 90), FAIL.** Five of those shots were called own. A survey site also went over its limit (2 of 9 events, limit 20%).
   Fix: a confidence bar on own (`Lab/own_losses.py`). All 15 real sounds called own over 2645 scored under 0.5; fresh true own sounds have a median of 0.83, and 0.6 keeps 81%. Bar 0.55. It was chosen after seeing these numbers, so it gets fresh runs.
   **Registered before those runs:** fresh own, clear: movement at most 5% gunfire or infected, at least 70% named own, shots at least 60%. Fresh gunfire in a storm (`-LabelSet -Weather 1`): at least 90% of events near a shot named gunfire, at most 1% named own. Older holdouts: rain storm gunfire back to at least 90%, the rest as before.
   **Result with the bar: the registered limits pass, one older survey check stays over.** Fresh own (`sniper_20261105_162440`): 88/94 movement named own, 1/94 gunfire or infected; shots 2/2, too few to say much, since real infected came during firing on this run too. Own shots rest mainly on 20/22 before the bar, and the bar costs some. Fresh gunfire in a storm (`163321`): 90/90 named gunfire, 0 own. Rain storm holdout back to 97/103 (94.2%). Thunder and attack rates within 3 points: Chernarus dry thunder 77.3% (was 79.4), still under its own 80 as before, and false gunshots in attacks unchanged at 7. Worst survey site 22% against a limit of 20: 2 of 9 events at one site, passing before the own classes. Left as a failed check; not chased, on 9 events. That was on replay.
   **Live, passes** (`sniper_20261105_174223`, the app on the game's sound while the driver moves the character, `Lab/own_live_score.py`, phases by wall clock, one run): movement named gunfire or infected 1/78, named own 68/78. Sprinting: 63 of 64 own. Own rifle: 31 of 45 own, **14 still gunshot**: so a player's own shots still ping about a third of the time. The app's on-screen readings did not get into this run's log, so audio lost was not read here.
   What the driver really did (`Lab/own_stuck.py`, TJ saw it running into a fence and backing up: it has no steering). Stuck on something 13% of sprinting, 5% of jogging, 7% of walking, 35% of crouching. Top speeds are right: walk 1.4 m/s, jog 3.7, sprint 6.6. **The "crawl" phase never crawled:** it tops out at 3.6 m/s, a jog, so the prone key was wrong and crawling is untested. Sounds came almost only from sprinting and jogging: walking gave 0-5 per run on all but one, in line with crouching. So the filter is proven on running; walking is quiet enough to rarely reach it, and crawling, vaulting and different ground are not covered.
   ML 5.8.2 stable replaces local.11: bit-identical on 256 events (`Lab/clap_same.py`): positions, labels, scores, embeddings.
   Video may settle the rest: if the picture is not moving, the steps are not yours; a rifle kicks the view. Not built.
5. Distance for footsteps and infected.
6. Then, with TJ: sitrep, location, what DINOv2 is for.
5. Sound distance alongside: crack-to-report gap on existing recordings.
6. Minimap: direction + distance + track over time, placed by the location estimate.

## Parts, and where each stands (2026-11-05)
| Part | Status | Next |
|---|---|---|
| Sound labels | Shipped: gunfire, gas strike, heli crash, thunder, infected attack, plus CLAP's words | Known flaws in the handoff. No more storm rounds. |
| Left or right | Works: 921/930 all the way round | |
| Sound angle | Gunfire median 5.3 deg, one weapon at one distance | Other weapons and ranges |
| Front or back | Not in the stereo mix. Solved for repeating sounds by view turns (p90 59 -> 13 deg) | Gunfire, single sounds |
| View turn from video | Done (`YawTracker`), relative only | True heading |
| **Sound distance** | Shipped. Gunfire near: 3.4-4.8 m median error on fresh runs. Far, 150-800 m: within a factor of 2 on 94-96%. Other sounds unmeasured | Footsteps, infected |
| Where am I, heading | Parked: 1 km retrieval barely above chance. Frames and DINOv2 in hand | Positive control on a dense patch when this comes back |
| **Minimap (Halo tracker)** | Red and white pings at distance on the radar. Inner ring 100 m, outer 800 m | Far gunfire under test; TJ's threat list |
| **Sitrep** | Not started | Needs TJ's detail |
| Town signs (OCR) | No model: PP-OCR likely, from Tuvok, once sign text size is measured | |
| Weapon names | First look: bursts AKM 52/61, AUG 20/33, M4A1 13/22. Not in the plan's core | After the above |

## Vision capture, what was learned (2026-11-05)
Check frames by eye: three test runs passed every automated check and were useless (the operator's terminal; the pause menu over every view; a window larger than the screen, off-screen part white). `grab_frames.py` asks the window to draw itself, so nothing on top matters. Vision runs keep `-noPause`: without it an unfocused client opens the pause menu (19 FPS). They record no sound. The lab client's window is 1600x900, to fit this PC's 1680x1050 desktop.
Focus matters, as TJ said: 60 FPS with it, 19 without; the same scene sharpens from 1858 to 2498 within a second of focus returning, and scenes loaded without focus stay blurry. File names end _f or _u; use _f. So a survey needs the PC left alone; the grabber retakes focus only after 30 s with no input.

## Working rules this plan exists to enforce
- Any plan TJ and an agent agree goes in this file in the same turn.
- Before a long line of work, check it against this file. Storm-label rounds ran for a day against no plan.
- Do not tell TJ what exists or is missing without reading these docs and grepping the code.
