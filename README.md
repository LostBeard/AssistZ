<div align="center">

# AssistZ

**Real time machine learning on live audio and video, running entirely in your browser.**

[Try it](https://lostbeard.github.io/AssistZ/) · Built with [SpawnDev.ILGPU.ML](https://github.com/LostBeard/SpawnDev.ILGPU.ML)

</div>

## Why this exists

AssistZ is a demonstration of what [SpawnDev.ILGPU.ML](https://github.com/LostBeard/SpawnDev.ILGPU.ML) can do in a browser: neural networks on your own GPU, processing audio and video streams as they happen, in C#, with nothing sent to a server.

Machine learning is going to change video games in more ways than just how they are made. It will change how they are played as well. And real time audio and video processing has applications everywhere, real world and simulated.

DayZ is the test bed. It is a hard one: a loud, dark, rainy world where a sound might be a sniper 600 m away or your own boots. It also makes honest measurement possible, because a lab copy of the game can log where everything really was.

The goal is a tactical assistant that works from only what a player can see and hear. It cannot control you, fight for you or look around corners. It listens and watches, and tells you what is going on.

## What it does today

Share a game window with its audio, and AssistZ puts what it hears on a radar: an icon and threat level for each sound, where it came from, and how far away it was.

- **Names sounds** with [CLAP](https://arxiv.org/abs/2211.06687) on your GPU, plus small probes trained on the game's sounds: gunfire, infected attacks, thunder, gas strikes, helicopter crashes and your own movement.
- **Names the gun:** "SVD", or just "Sniper rifle" or "Assault rifle" when it can only tell the family. A sniper at 300 m ranks above a pistol at that range.
- **Estimates distance** from the sound alone, out to 800 m.
- **Tells an agitated infected from a calm one**, and tries not to ping your own footsteps and shots.
- **Follows your view** from the video, so a sound stays where it was as you turn.

### How well, measured

Each figure is from lab recordings the models never trained on, with the game's own log as the answer key.

| | Result |
|---|---|
| Gun named correctly | 236 of 249 shots (94.8%); family 340 of 350. Only 10 guns are known so far |
| Distance, under 100 m | Median error 3.4 to 5.1 m, by run |
| Distance, 150 to 800 m | Within a factor of 2 on 94 to 98%, by run |
| Agitated infected named | 134 of 149 (89.9%); calm taken for agitated 1 of 54 |
| Your own movement pinging as something else | 7 of 131 |
| Gunfire in a heavy rain storm | 92 of 103 |
| Delay from sound to name, live | About 1 second |

Those figures are for one game, mostly on one map, on one AMD RX 590. Treat them as what this setup achieved, not as a promise elsewhere.

### What it cannot do yet

- **Front or back.** Stereo audio cannot tell them apart while you stand still. Turning helps, and only for a repeating sound.
- **Video does not reach the radar yet.** In the lab, YOLOv8 spots calm infected, which make too little sound to find, within about 30 m: sighted in 38 of 39 slots across dusk, rain and noon, with no false sightings in 34 minutes of empty frames, at about 230 ms a frame on the RX 590. That was with a still camera. It is being tested on a moving player now, and has not been measured beside the sound work. It sees nothing useful at night yet.
- **Other games.** The probes were trained on DayZ.

The plan and every result, including what failed, are in [docs/PLAN.md](docs/PLAN.md).

## Try it

Use a desktop browser with WebGPU; it is developed on Chrome.

1. Open the [app](https://lostbeard.github.io/AssistZ/) and click **Name sounds**. The first time this downloads about 620 MB of models, which are then cached.
2. Click **Share window or tab**, and pick your game window.
3. Turn on audio in the share dialog. On Windows 11, Chrome offers to share the window's own audio. On Windows 10 it only offers system audio, so other sounds on your PC are heard too.

[/icons](https://lostbeard.github.io/AssistZ/icons) explains the radar's icons.

AssistZ only uses the picture and sound your browser's screen share gives it. It does not read the game's files or memory, or send any input to the game, and nothing leaves your PC apart from the model downloads. Whether a server allows tools like this is up to that server, so check its rules.

## How it fits together

| Folder | |
|---|---|
| `AssistZ/` | The Blazor WebAssembly app |
| `AssistZ.Tests/` | Unit tests |
| `Lab/` | Tools for measuring against a local copy of the game: scripted scenes that log the truth, recording, scoring |
| `docs/` | The plan, measured findings and the lab's running record |

Everything runs through [SpawnDev.ILGPU.ML](https://github.com/LostBeard/SpawnDev.ILGPU.ML), which runs models natively on the GPU from C#, with no ONNX Runtime, using WebGPU in the browser. The `Lab/` tools are for development and need your own copy of DayZ. Game assets, recordings and frames are not in this repository.

## Licence

[MIT](LICENSE.txt). Models are downloaded at run time under their own licences.

AssistZ is an independent project, not affiliated with or endorsed by Bohemia Interactive. DayZ is a trademark of Bohemia Interactive.
