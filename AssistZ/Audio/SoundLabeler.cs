using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.SpawnJS;
using System.Diagnostics;

namespace AssistZ.Audio
{
    public enum LabelerState { Off, Loading, Ready, Failed }

    /// <summary>
    /// Names what each detected sound is with CLAP (SpawnDev.ILGPU.ML's <see cref="ClapAudioClassifier"/>: zero-shot, any
    /// text labels, no training), on the same GPU as the stereo analysis. The clip is cut straight from the analyzer's
    /// GPU ring (<see cref="StereoAnalyzer.ExtractMono"/>), so the audio goes ring -> CLAP features -> audio tower without a
    /// round trip through .NET. The towers (620 MB) download once into the browser's OPFS cache.
    /// </summary>
    public sealed class SoundLabeler : IDisposable
    {
        /// <summary>CLAP's training template: bare labels score badly (Tuvok, 2026-11-03: a speech clip read "music" 0.971
        /// against "a person speaking"; "This is a sound of speech" scores 0.913).</summary>
        public const string Template = "This is a sound of ";

        /// <summary>What the HUD can name. Game sounds first; the rest catch what streams and recordings also carry.</summary>
        public static readonly string[] Labels =
        {
            "gunshot", "machine gun fire", "explosion", "footsteps", "zombie groaning", "speech", "car engine", "helicopter",
            "wolf howling", "dog barking", "birds chirping", "rain", "thunder", "wind", "door closing",
            "metal clanking", "music",
        };

        /// <summary>
        /// Gunfire labels. Measured on lab run sniper_20261103_153510 (44 logged shots): "machine gun fire" takes every single
        /// SVD shot too (0.69-0.97) but adds AUG bursts that "gunshot" missed (gunfire-family hits 32 -> 38 of 50 events at a
        /// shot report), so both prompts stay; single vs automatic is decided by the cadence, not by CLAP.
        /// </summary>
        public static bool IsGunfire(string label) => label is "gunshot" or "machine gun fire" or GunfireClass;

        /// <summary>The probe's gunfire class, also the word <see cref="ClassifyAsync"/> returns when the probe says gunfire.</summary>
        public const string GunfireClass = "gunfire";
        /// <summary>The probe's class for everything it does not name: the prompts name those.</summary>
        const string OtherClass = "other";
        /// <summary>The fitted probe, shipped with the app (Lab/flyby/fit_sound_probe.py).</summary>
        const string ProbePath = "models/sound-probe.json";

        /// <summary>True for a word only the probe's world event classes give (not gunfire, and not a prompt's word).</summary>
        public bool IsWorldEvent(string word) => _probe != null && word != GunfireClass && word != OtherClass && !SoundMarks.IsOwnClass(word.Replace(' ', '_')) && Array.IndexOf(_probe.Classes, word.Replace(' ', '_')) >= 0;

        /// <summary>What the HUD calls a probe class ("gas_strike" -> "gas strike").</summary>
        static string ClassWord(string probeClass) => probeClass.Replace('_', ' ');

        /// <summary>
        /// Let the probe name gunfire, gas strike, heli crash, thunder and infected attack, with the prompts naming everything else (on by default; ?probe=0 to compare). A linear probe on CLAP's audio embedding, trained on the game's logs. On recordings it never trained on: real gunfire in a rain storm 97/103, in a dry storm 116/118,
        /// in rain with no storm 127/127, in clear weather 133/133, no shot called gunfire 0/201; thunder on Nasdara 154/160, in rain 350/354; infected attacks 89-99% in every run. World events, each run left out in turn: gas 84/97, heli 54/60. Shipped on TJ's decision despite missed limits: Chernarus dry storm thunder 79.4% (limit 80), false gas strike 10.9% (limit 5),
        /// and 5 false gunshots in 670 infected attack events (limit 0). No sleigh crash word: it is on only in the Christmas missions and fired on crows. Not recorded: attacks in a storm, gunfire on Nasdara, gas strikes and crashes in weather. See docs/lab-handoff.md. Earlier gunfire leave-one-run-out reproduced by Tuvok on CUDA and OpenCL.
        /// </summary>
        public bool UseProbe { get; set; } = true;
        ClapProbe? _probe;

        /// <summary>Clip length: 1 s, starting <see cref="PreRoll"/> before the onset window.</summary>
        public const int ClipSamples = 48000;
        /// <summary>Samples kept before the anchor, so the attack is in the clip.</summary>
        public const int PreRoll = 4800;
        /// <summary>How far past the onset window's end the peak search reaches: the level crosses the detector's threshold on
        /// the way up, so the peak can still be ahead of the window that flagged it.</summary>
        public const int PeakSearchAfter = StereoAnalyzer.Window;

        /// <summary>
        /// Anchor each clip on the sound's peak (on by default). Off only for A/B measurement (?align=0): the clip is then cut
        /// from the onset window, which sits on the hop grid. Measured on lab run sniper_20261103_153510 in replay: identical
        /// clips give bit-identical scores, but starting the stream 64 / 256 / 512 samples later (hop = 1024) changed the label
        /// of 6 / 15 / 7 events, most of them footsteps vs machine gun fire in an AUG burst.
        /// </summary>
        public bool AnchorOnPeak { get; set; } = true;
        /// <summary>
        /// Scale each clip so the sound's peak sits at this level (dBFS) before CLAP; null = off (?norm=-6 to measure). A/B only so far.
        /// Why try: on lab run 4 shifted +12 dB the prompts name the 150 m AUG burst gunfire 30/30, unshifted 20/30, -20 dB 8/29.
        /// </summary>
        public float? NormalizePeakDb { get; set; }

        /// <summary>Lab only (?embed=1): log each clip's audio embedding, to score a probe on it.</summary>
        public bool KeepEmbedding { get; set; }
        /// <summary>
        /// Record the audio tower once and replay it (?capture=1; off by default). On the RX 590 in Chrome, 319 events on 5.8.2-local.7: bit-identical, but no faster (median 74.4 -> 73.9 ms) and about 700 MB more GPU memory (1128 -> 1830 MB). The RTX 4070 did run faster; re-measure when the package changes.
        /// </summary>
        public bool UseAudioCapture { get; set; }
        /// <summary>The last classified clip's L2-normalised audio embedding.</summary>
        public float[]? LastEmbedding { get; private set; }
        /// <summary>First sample of the last classified clip (a lab harness compares them).</summary>
        public long LastClipStart { get; private set; }

        readonly SpawnJSRuntime _js;
        readonly HttpClient _http;
        readonly string[] _prompts = Labels.Select(l => Template + l).ToArray();
        ClapAudioClassifier? _clap;
        MemoryBuffer1D<float, Stride1D.Dense>? _clip;
        Task? _loading;

        public SoundLabeler(SpawnJSRuntime js, HttpClient http)
        {
            _js = js;
            _http = http;
        }

        /// <summary>The CLAP checkpoint: the default (clap-htsat-unfused) or ClapAudioClassifier.LargerGeneralRepo. Set before
        /// <see cref="LoadAsync"/> (?clap=larger). The larger one fixes AUG bursts read as footsteps but, per Tuvok's
        /// transformers check, hears clean speech as "dog barking", so it is chosen by measurement, not by default.</summary>
        public string Repo { get; set; } = ClapAudioClassifier.DefaultRepo;

        public LabelerState State { get; private set; }
        /// <summary>Loading progress or the failure, for the UI.</summary>
        public string? Status { get; private set; }
        /// <summary>Last classification, host-side latency (extract + features + tower + readback).</summary>
        public double LastMs { get; private set; }
        public long Classified { get; private set; }
        /// <summary>The last classification's probability for every entry of <see cref="Labels"/> (a lab harness compares them).</summary>
        public float[] LastScores { get; } = new float[Labels.Length];
        public event Action? Changed;

        /// <summary>Downloads (first time) and loads CLAP on <paramref name="accelerator"/>. Safe to call repeatedly.</summary>
        DistanceProbe? _distance;
        GunProbe? _gun;
        const string GunProbePath = "models/gun-probe.json";

        /// <summary>The last classified clip's gun call. Trained on gunfire only.</summary>
        public GunCall? CallGun() => _gun != null && LastEmbedding is float[] e ? _gun.Call(e) : null;

        const string DistanceProbePath = "models/distance-probe.json";

        /// <summary>The last classified clip's estimated distance in metres; null without the probe. Measured on gunfire only: <see cref="DistanceProbe"/>.</summary>
        public double? EstimateDistanceMeters() => _distance != null && LastEmbedding is float[] e ? _distance.Estimate(e) : null;

        /// <summary>Whether the audio tower's recorded plan is live (lab logging).</summary>
        public string? AudioCaptureStatus => _clap?.AudioCaptureStatus?.ToString();

        public Task LoadAsync(Accelerator accelerator) => _loading ??= LoadCoreAsync(accelerator);

        async Task LoadCoreAsync(Accelerator accelerator)
        {
            State = LabelerState.Loading;
            Status = "Loading CLAP";
            Changed?.Invoke();
            try
            {
                var source = new HubModelSource(_js, _http);
                Console.WriteLine($"[AssistZ] clap repo={Repo}");
                _clap = await ClapAudioClassifier.CreateFromHubAsync(accelerator, source, Repo, (file, percent) =>
                {
                    Status = $"{Path.GetFileName(file)} {percent}%";
                    Changed?.Invoke();
                });
                _clap.EnableGraphCapture = UseAudioCapture;
                _clip = accelerator.Allocate1D<float>(ClipSamples);
                if (UseProbe)
                {
                    var probe = ClapProbe.FromJson(await _http.GetStringAsync(ProbePath));
                    // A probe means nothing on another checkpoint's embeddings (?clap=larger): the prompts decide there.
                    _probe = probe.Checkpoint == Repo ? probe : null;
                    Console.WriteLine($"[AssistZ] probe {(_probe != null ? "on" : "off: fitted on " + probe.Checkpoint)}");
                    var distance = DistanceProbe.FromJson(await _http.GetStringAsync(DistanceProbePath));
                    _distance = distance.Checkpoint == Repo ? distance : null;
                    Console.WriteLine($"[AssistZ] distance probe {(_distance != null ? "on" : "off: fitted on " + distance.Checkpoint)}");
                    var gun = GunProbe.FromJson(await _http.GetStringAsync(GunProbePath));
                    _gun = gun.Checkpoint == Repo ? gun : null;
                    Console.WriteLine($"[AssistZ] gun probe {(_gun != null ? "on" : "off: fitted on " + gun.Checkpoint)}");
                }
                // Embeds every label once (one text-tower run each), so the first sound is not slowed by 16 of them.
                await _clap.ClassifyAsync(new float[ClipSamples], _prompts);
                State = LabelerState.Ready;
                Status = null;
            }
            catch (Exception ex)
            {
                State = LabelerState.Failed;
                Status = ex.Message;
                _loading = null;
            }
            Changed?.Invoke();
        }

        /// <summary>Earliest first sample a clip can have for an onset window ending at <paramref name="endSample"/>; also where
        /// the clip starts when it is not anchored on the peak.</summary>
        public static long ClipStart(long endSample) => endSample - StereoAnalyzer.Window - StereoAnalyzer.MaxLag - PreRoll;

        /// <summary>Samples from <see cref="ClipStart"/> that must be written and still in the ring: the latest clip the peak
        /// search can choose.</summary>
        public const int ClipSpan = StereoAnalyzer.Window + StereoAnalyzer.MaxLag + PeakSearchAfter + ClipSamples;

        const int PeakSearchLength = StereoAnalyzer.Window + StereoAnalyzer.MaxLag + PeakSearchAfter;

        /// <summary>
        /// Classifies the clip of the onset ending at <paramref name="endSample"/>; null when the classifier is not ready or
        /// the clip has left the ring. Call once the analyzer has written the whole clip.
        /// </summary>
        public async Task<(string Word, string ClapWord, float Probability)?> ClassifyAsync(StereoAnalyzer analyzer, long endSample, double azimuthDegrees)
        {
            if (_clap == null || _clip == null || State != LabelerState.Ready) return null;
            long start = ClipStart(endSample);
            if (!analyzer.InRing(start, ClipSamples)) return null;
            var sw = Stopwatch.StartNew();
            float gain = 1f;
            if (AnchorOnPeak)
            {
                var peak = await analyzer.FindPeakAsync(start + PreRoll, PeakSearchLength);
                start = peak.Position - PreRoll;
                if (!analyzer.InRing(start, ClipSamples)) return null;
                if (NormalizePeakDb is float target && peak.Magnitude > 1e-6f)
                    gain = MathF.Pow(10f, target / 20f) / peak.Magnitude;
            }
            LastClipStart = start;
            analyzer.ExtractMono(start, ClipSamples, _clip.View, gain);
            var result = await _clap.ClassifyWithEmbeddingAsync(_clip.View, ClipSamples, _prompts);
            var scores = result.Scores;
            LastEmbedding = result.Embedding;
            LastMs = sw.Elapsed.TotalMilliseconds;
            Classified++;
            int best = 0;
            for (int i = 0; i < scores.Length; i++)
            {
                LastScores[i] = scores[i].Probability;
                if (scores[i].Probability > scores[best].Probability) best = i;
            }
            string clapWord = Labels[best];
            if (_probe == null) return (clapWord, clapWord, scores[best].Probability);
            float[] classes = _probe.Predict(result.Embedding);
            int top = SoundMarks.PickClass(_probe.Classes, classes, azimuthDegrees);
            string probeClass = _probe.Classes[top];
            if (probeClass == GunfireClass) return (GunfireClass, clapWord, classes[top]);
            if (probeClass != OtherClass) return (ClassWord(probeClass), clapWord, classes[top]);
            if (!IsGunfire(clapWord)) return (clapWord, clapWord, scores[best].Probability);
            // The prompts heard gunfire and the probe did not: the best word that is not gunfire.
            int other = -1;
            for (int i = 0; i < scores.Length; i++)
                if (!IsGunfire(Labels[i]) && (other < 0 || scores[i].Probability > scores[other].Probability)) other = i;
            return (Labels[other], clapWord, scores[other].Probability);
        }

        public void Dispose()
        {
            _clap?.Dispose();
            _clip?.Dispose();
        }
    }
}
