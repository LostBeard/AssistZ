using AssistZ.Audio;
using AssistZ.Video;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace AssistZ.Services
{
    /// <summary>
    /// One media source feeding the GPU: a shared window or tab (getDisplayMedia - the game itself, or a Twitch/YouTube
    /// tab) or a local recording (a video element's captureStream). Video and audio are read with
    /// MediaStreamTrackProcessor, so frames arrive as VideoFrame / AudioData objects that go straight to WebGPU without
    /// a canvas or an AudioWorklet in between.
    /// </summary>
    public sealed class CaptureSession : IAsyncDisposable
    {
        readonly SpawnJSRuntime _js;
        readonly GpuHost _gpu;
        MediaStream? _stream;
        MediaStreamTrack? _videoTrack, _audioTrack;
        CancellationTokenSource? _cts;
        Task? _videoLoop, _audioLoop;
        StereoAnalyzer? _analyzer;
        VideoPreview? _video;
        YawTracker? _yaw;
        readonly YawGate _yawGate = new();
        // Recording source: the element playing the file and its object URL, released on stop.
        HTMLVideoElement? _player;
        string? _playerUrl;

        public CaptureSession(SpawnJSRuntime js, GpuHost gpu, SoundLabeler labeler)
        {
            _js = js;
            _gpu = gpu;
            Labeler = labeler;
        }

        /// <summary>CLAP sound naming (loaded on demand from the UI); events get a label once it is ready.</summary>
        public SoundLabeler Labeler { get; }
        // Events waiting for their clip to be fully written, oldest first; one classification in flight at a time.
        readonly Queue<SoundEvent> _toLabel = new();
        Task? _labeling;
        BurstVote? _burstVote;
        /// <summary>Events whose clip left the ring before the classifier got to it (it was busy).</summary>
        public long LabelsDropped { get; private set; }

        public bool Running => _cts != null;
        /// <summary>What is being processed: the shared surface's label or the recording's file name.</summary>
        public string? SourceName { get; private set; }
        public string? Error { get; private set; }
        public CaptureStats Stats { get; } = new();
        public SoundEventDetector Detector { get; } = new();
        /// <summary>Transient peaks and their spacing: splits gunfire into single shots and automatic fire.</summary>
        public CadenceMeter Cadence { get; private set; } = new();
        /// <summary>Front/back resolution for repeating sources (needs the view to turn).</summary>
        public SourceTracker Tracker { get; private set; } = new();
        readonly Dictionary<int, SourceSide> _trackSides = new();
        /// <summary>The CURRENT side of a track: earlier events of a track are redrawn once it resolves.</summary>
        public SourceSide SideOf(int trackId) => _trackSides.TryGetValue(trackId, out var side) ? side : SourceSide.Unknown;
        /// <summary>Most recent stereo measurement.</summary>
        public StereoFrame? LastStereo { get; private set; }
        /// <summary>Recent events, newest last. Trimmed to the last few seconds by the UI tick.</summary>
        public List<SoundEvent> Events { get; } = new();
        /// <summary>Video frames processed per second (cap). The view-turn tracker needs every frame it can get.</summary>
        public double TargetFps { get; set; } = 30;
        /// <summary>Latest frame-to-frame view turn.</summary>
        public YawSample? LastYaw { get; private set; }
        /// <summary>Smoothed view-turn rate in preview pixels per second (+ = turning right).</summary>
        public double YawRatePixels { get; private set; }
        /// <summary>Smoothed view-turn rate in degrees per second (+ = turning right), via the game profile.</summary>
        public double YawRateDegrees { get; private set; }
        /// <summary>View heading integrated from the tracker since the capture started (degrees, + = right). Relative only:
        /// video alone cannot know north.</summary>
        public double ViewYawDegrees { get; private set; }
        /// <summary>Game-specific calibration of the stereo pan law and the view-turn scale.</summary>
        public GameProfile Profile { get; set; } = GameProfile.DayZ;
        /// <summary>Lab runs (?lab=1): one console line per measurement so a harness can line them up with the game's pose log.</summary>
        public bool LabLogging { get; set; }
        /// <summary>?memlog=1: log live WebGPU buffers and pool allocations per event, to split a memory climb between ILGPU's buffers and Chrome's.</summary>
        public bool MemLog { get; set; }
        /// <summary>A gunfire word outnumbered by thunder in its burst takes thunder (on; ?yield=0 to compare). In dry storms false gunshots 8 -> 0, real gunfire in storms unchanged on fresh runs (97/103, 116/118).</summary>
        public bool GunfireYieldsToThunder { get; set; }
        /// <summary>Audio chunks the reader may queue before Chrome drops the oldest; about 5 s at 10 ms a chunk. On Chrome's default a live run lost about 2.1 s in 150 s, in stalls of up to 180 ms, mostly in the first minute; with this, none, and names came no later (docs/PLAN.md). Null = Chrome's default: ?audiobuf=0.</summary>
        public int? AudioBufferChunks { get; set; } = 512;
        int _memEvents;

        void LogMemory()
        {
            _memEvents++;
            Console.WriteLine($"[AssistZ] mem n={_memEvents} storage={WebGPUBufferAccounting.LiveStorageBytes} staging={WebGPUBufferAccounting.LiveStagingBytes} buffers={WebGPUBufferAccounting.LiveBufferCount} pool={BufferPool.TotalDeviceAllocations}");
            if (_memEvents is 1 or 2 or 10) Console.WriteLine($"[AssistZ] memcapture n={_memEvents} {Labeler.AudioCaptureStatus}");
            if (_memEvents is 10 or 100 or 300)
                foreach (var site in WebGPUBufferAccounting.TopCreationSites(10))
                    Console.WriteLine($"[AssistZ] memsite n={_memEvents} {site}");
        }
        /// <summary>Audio spectral path on/off (A/B timing only; ?spectral=0).</summary>
        public bool SpectralEnabled { get; set; } = true;
        /// <summary>Compile the audio kernels before the first audio read (on by default). Off only for A/B timing (?warm=0).</summary>
        public bool WarmUp { get; set; } = true;
        /// <summary>Raised on the UI-relevant changes (start, stop, error).</summary>
        public event Action? Changed;

        /// <summary>Opens the browser's share picker and starts processing. Call from a click handler (user activation).</summary>
        public async Task StartDisplayAsync(string previewCanvasId)
        {
            if (Running) return;
            Error = null;
            try
            {
                // getDisplayMedia first, while the click's user activation is fresh.
                using var mediaDevices = _js.Get<MediaDevices>("navigator.mediaDevices");
                var stream = await mediaDevices.GetDisplayMedia(new DisplayMediaStreamOptions
                {
                    Video = true,
                    // Browser voice processing would destroy the stereo cues: echo cancellation and noise suppression
                    // are built for a microphone, AGC rescales levels. All off, both channels requested.
                    Audio = new MediaTrackConstraints
                    {
                        EchoCancellation = false,
                        NoiseSuppression = false,
                        AutoGainControl = false,
                        ChannelCount = 2,
                    },
                    SystemAudio = SystemAudioPreferenceEnum.Include,
                    // Windows 11 then offers "Also share app audio": the game alone, not voice chat or music (Data's test: its own sound 16/16, another program's 0/17). Windows 10 19045 offers only system audio, and the app works the same.
                    WindowAudio = WindowAudioPreferenceEnum.Window,
                    SelfBrowserSurface = SelfCapturePreferenceEnum.Exclude,
                    SurfaceSwitching = SurfaceSwitchingPreferenceEnum.Include,
                });
                if (stream == null) throw new InvalidOperationException("Nothing was shared.");
                using (var track = stream.GetFirstVideoTrack()) SourceName = DisplayName(track);
                await StartStreamAsync(stream, previewCanvasId);
                // Chrome's picker has audio off by default for a window or a screen. Without this the app looks fine and hears nothing.
                if (_audioTrack == null) Error = "No sound came with this share, so nothing can be heard. Stop, share again and turn on the audio switch in Chrome's dialog: \"Also share app audio\" for the game alone where it is offered, otherwise \"Share with system audio\".";
            }
            catch (Exception ex)
            {
                Error = ex.Message;
                await StopAsync();
            }
            Changed?.Invoke();
        }

        /// <summary>
        /// Plays a local recording (looped, audible) and processes its captureStream. Call from the file input's change
        /// handler so playback with sound is allowed.
        /// </summary>
        public async Task StartRecordingAsync(SpawnDev.SpawnJS.JSObjects.File file, string previewCanvasId)
        {
            if (Running) return;
            Error = null;
            try
            {
                SourceName = file.Name;
                using var document = _js.Get<Document>("document");
                _player = document.CreateElement<HTMLVideoElement>("video");
                _player.SetAttribute("class", "az-player");
                _player.Loop = true;
                _playerUrl = URL.CreateObjectURL(file);
                _player.Src = _playerUrl;
                using (var body = document.Body!) body.AppendChild(_player).Dispose();
                await _player.Play();
                await StartStreamAsync(_player.CaptureStream(), previewCanvasId);
            }
            catch (Exception ex)
            {
                Error = ex.Message;
                await StopAsync();
            }
            Changed?.Invoke();
        }

        /// <summary>
        /// Lab replay (?replay=1): the file's samples go through the production analyzer, detector, cadence meter and CLAP path
        /// in fixed 2048-frame chunks, each step awaited, with no media element and no real-time clock. The result depends only
        /// on the samples and the GPU, and sample positions are exact: EndSample + <paramref name="skipSamples"/> is the file's
        /// sample. <paramref name="skipSamples"/> starts later in the file, which moves every sound against the hop grid.
        /// </summary>
        public async Task ReplayAsync(SpawnDev.SpawnJS.JSObjects.File file, int skipSamples)
        {
            if (Running) return;
            Error = null;
            const int rate = 48000, chunk = 2048, hop = 1024;
            try
            {
                SourceName = file.Name;
                _cts = new CancellationTokenSource();
                var ct = _cts.Token;
                var accelerator = await _gpu.GetAcceleratorAsync();
                using var bytes = await file.ArrayBuffer();
                using var context = new OfflineAudioContext(2, 1, rate);
                using var audio = await context.DecodeAudioData(bytes);
                using var left = audio.GetChannelData(0);
                using var right = audio.GetChannelData(audio.NumberOfChannels > 1 ? 1 : 0);
                _analyzer = new StereoAnalyzer(accelerator) { SpectralEnabled = SpectralEnabled };
                if (WarmUp) await WarmUpAnalyzerAsync(rate);
                ResetRun();
                Stats.AudioSampleRate = rate;
                Stats.AudioChannels = audio.NumberOfChannels;
                long total = audio.Length;
                Console.WriteLine($"[AssistZ] replay start skip={skipSamples} frames={total} channels={audio.NumberOfChannels} rate={audio.SampleRate}");
                var sw = Stopwatch.StartNew();
                long nextAnalysis = 0;
                for (long position = skipSamples; position + chunk <= total && !ct.IsCancellationRequested; position += chunk)
                {
                    using (var l = left.SubArray(position, position + chunk))
                    using (var r = right.SubArray(position, position + chunk))
                        _analyzer.Write(l, r, chunk, rate);
                    if (nextAnalysis == 0) nextAnalysis = _analyzer.OldestAnalyzableEnd;
                    while (nextAnalysis <= _analyzer.WritePosition)
                    {
                        int count = (int)Math.Min((_analyzer.WritePosition - nextAnalysis) / hop + 1, StereoAnalyzer.MaxWindows);
                        await AnalyzeAsync(nextAnalysis, hop, count);
                        nextAnalysis += (long)count * hop;
                    }
                    while (true)
                    {
                        PumpLabels();
                        if (_labeling == null || _labeling.IsCompleted) break;
                        await _labeling;
                    }
                }
                Console.WriteLine($"[AssistZ] replay done skip={skipSamples} events={Events.Count} named={Labeler.Classified} unnamed={_toLabel.Count} dropped={LabelsDropped} windows={Stats.Windows} seconds={sw.Elapsed.TotalSeconds:0.00}");
            }
            catch (Exception ex)
            {
                Error = $"Replay: {ex.Message}";
                Console.WriteLine($"[AssistZ] replay failed {ex}");
            }
            finally
            {
                await StopAsync();
            }
        }

        // Chrome labels a shared surface with an internal id ("web-contents-media-stream://...", "window:526568:0",
        // "screen:0:0"), not a title, so the surface type is the honest name.
        static string DisplayName(MediaStreamTrack? track)
        {
            if (track == null) return "Shared surface";
            return track.GetSettings().DisplaySurface switch
            {
                "browser" => "Browser tab",
                "monitor" => "Screen",
                "window" => "Window",
                _ => "Shared surface",
            };
        }

        async Task StartStreamAsync(MediaStream stream, string previewCanvasId)
        {
            _stream = stream;
            var accelerator = await _gpu.GetAcceleratorAsync();
            _analyzer = new StereoAnalyzer(accelerator) { SpectralEnabled = SpectralEnabled };
            _video = new VideoPreview(accelerator, _js, previewCanvasId);
            _yaw = new YawTracker(accelerator) { LogBigShifts = LabLogging };
            LastYaw = null;
            YawRatePixels = 0;
            YawRateDegrees = 0;
            ViewYawDegrees = 0;
            _cts = new CancellationTokenSource();
            _videoTrack = _stream.GetFirstVideoTrack();
            _audioTrack = _stream.GetFirstAudioTrack();
            if (WarmUp && _audioTrack != null) await WarmUpAnalyzerAsync((int)(_audioTrack.GetSettings().SampleRate ?? 48000));
            ResetRun();
            if (_videoTrack != null)
            {
                Stats.VideoLabel = _videoTrack.Label;
                // "Stop sharing" in the browser bar ends the track: stop with it.
                _videoTrack.OnEnded += OnTrackEnded;
                _videoLoop = VideoLoopAsync(_videoTrack, _cts.Token);
            }
            if (_audioTrack != null)
            {
                Stats.AudioLabel = _audioTrack.Label;
                _audioLoop = AudioLoopAsync(_audioTrack, _cts.Token);
            }
            else
            {
                Stats.AudioLabel = null;
                Error = "No audio in this source. For a window or tab, turn on \"Share audio\" in the picker.";
            }
        }

        async Task WarmUpAnalyzerAsync(int sampleRate)
        {
            var sw = Stopwatch.StartNew();
            await _analyzer!.WarmUpAsync(sampleRate);
            Console.WriteLine($"[AssistZ] warmup ms={sw.Elapsed.TotalMilliseconds:0.0} rate={sampleRate}");
        }

        void ResetRun()
        {
            Stats.Reset();
            Events.Clear();
            _toLabel.Clear();
            LabelsDropped = 0;
            Cadence = new CadenceMeter();
            Tracker = new SourceTracker();
            _trackSides.Clear();
            LastStereo = null;
            _burstVote = new BurstVote(Labeler.IsWorldEvent, GunfireYieldsToThunder ? w => w is "gunshot" or "automatic fire" : null);
        }

        void OnTrackEnded(Event e) => _ = StopAsync();

        public async Task StopAsync()
        {
            var cts = _cts;
            _cts = null;
            cts?.Cancel();
            // Ending the tracks ends the processors' streams, which is what releases a loop parked in reader.Read().
            _stream?.StopAllTracks();
            try
            {
                if (_videoLoop != null) await _videoLoop;
                if (_audioLoop != null) await _audioLoop;
                if (_labeling != null) await _labeling;
            }
            catch (OperationCanceledException) { }
            _videoLoop = _audioLoop = null;
            _labeling = null;
            _toLabel.Clear();
            if (_videoTrack != null) _videoTrack.OnEnded -= OnTrackEnded;
            _videoTrack?.Dispose();
            _audioTrack?.Dispose();
            _videoTrack = _audioTrack = null;
            _stream?.StopAllTracks();
            _stream?.Dispose();
            _stream = null;
            _analyzer?.Dispose();
            _analyzer = null;
            _video?.Dispose();
            _video = null;
            _yaw?.Dispose();
            _yaw = null;
            if (_player != null)
            {
                _player.Pause();
                _player.Remove();
                _player.Dispose();
                _player = null;
            }
            if (_playerUrl != null) URL.RevokeObjectURL(_playerUrl);
            _playerUrl = null;
            cts?.Dispose();
            Changed?.Invoke();
        }

        async Task VideoLoopAsync(MediaStreamTrack track, CancellationToken ct)
        {
            using var processor = new MediaStreamTrackProcessor(new MediaStreamTrackProcessorOptions { Track = track, MaxBufferSize = 1 });
            using var readable = processor.Readable;
            using var reader = readable.GetReader();
            var sinceLast = Stopwatch.StartNew();
            var work = new Stopwatch();
            try
            {
                while (!ct.IsCancellationRequested)
                {
                    using var read = await reader.Read();
                    if (read.Done) break;
                    using var frame = read.JSRef!.Get<VideoFrame>("value");
                    try
                    {
                        Stats.VideoFramesSeen++;
                        if (sinceLast.Elapsed.TotalMilliseconds < 1000.0 / TargetFps) continue;
                        sinceLast.Restart();
                        work.Restart();
                        await _video!.ProcessAsync(frame);
                        Stats.RecordVideo(work.Elapsed.TotalMilliseconds, _video.SourceWidth, _video.SourceHeight);
                        await TrackYawAsync(frame.Timestamp / 1e6);
                    }
                    finally
                    {
                        frame.Close();
                    }
                }
            }
            catch (Exception ex) when (!ct.IsCancellationRequested)
            {
                Error = $"Video: {ex.Message}";
                Changed?.Invoke();
            }
            finally
            {
                await reader.Cancel();
            }
        }

        double _lastYawTime;

        async Task TrackYawAsync(double frameSeconds)
        {
            if (_yaw == null || _video?.Preview == null) return;
            var sample = await _yaw.TrackAsync(_video.Preview, frameSeconds);
            if (sample is not YawSample y) { _lastYawTime = frameSeconds; return; }
            double dt = frameSeconds - _lastYawTime;
            _lastYawTime = frameSeconds;
            LastYaw = y;
            // Only believed turns count: not a low match, a peak at the search's edge, or a jump with stillness either side.
            float turn = _yawGate.Next(y.ShiftPixels, y.Score);
            if (y.Score > YawGate.MinScore)
            {
                int width = _video.Preview.IntExtent.X;
                ViewYawDegrees += Profile.ShiftToDegrees(turn, width);
                if (dt > 0 && dt < 0.5)
                {
                    YawRatePixels += (turn / dt - YawRatePixels) * 0.3;
                    YawRateDegrees += (Profile.ShiftToDegrees(turn, width) / dt - YawRateDegrees) * 0.3;
                }
            }
            if (LabLogging)
                Console.WriteLine($"[AssistZ] yaw utc={DateTime.UtcNow:HH:mm:ss.fff} ts={frameSeconds:0.000} shift={y.ShiftPixels:0.00} score={y.Score:0.000}");
        }

        async Task AudioLoopAsync(MediaStreamTrack track, CancellationToken ct)
        {
            using var processor = new MediaStreamTrackProcessor(new MediaStreamTrackProcessorOptions { Track = track, MaxBufferSize = AudioBufferChunks });
            using var readable = processor.Readable;
            using var reader = readable.GetReader();
            var gaps = new AudioGapMeter();
            long nextAnalysis = 0;
            const int hop = 1024; // 21.3 ms at 48 kHz
            Task? analysis = null;
            try
            {
                while (!ct.IsCancellationRequested)
                {
                    using var read = await reader.Read();
                    if (read.Done) break;
                    using var data = read.JSRef!.Get<AudioData>("value");
                    try
                    {
                        if (Stats.AudioChunks == 0)
                        {
                            Stats.AudioFormat = data.Format;
                            Stats.AudioSampleRate = (double)data.SampleRate;
                            Stats.AudioChannels = data.NumberOfChannels;
                            Stats.AudioChunkFrames = data.NumberOfFrames;
                            if (data.Format != "f32-planar")
                                throw new NotSupportedException($"AudioData format '{data.Format}' (expected f32-planar).");
                        }
                        Stats.AudioChunks++;
                        double lost = gaps.Add(data.Timestamp, data.NumberOfFrames, (double)data.SampleRate);
                        Stats.AudioLostSeconds = gaps.LostSeconds;
                        if (lost > 0 && LabLogging)
                            Console.WriteLine($"[AssistZ] audiogap utc={DateTime.UtcNow:HH:mm:ss.fff} ms={lost * 1000:0.0} total={gaps.LostSeconds:0.000} count={gaps.Gaps} chunks={Stats.AudioChunks}");
                        await _analyzer!.WriteAsync(data);
                        // Sample 0 is wherever the track happened to start, so a harness needs the file position to put events
                        // on the recording's clock. Every 256 chunks (about 11 s) so it can also see drift or a loop.
                        if (LabLogging && _player != null && Stats.AudioChunks % 256 == 1)
                            Console.WriteLine($"[AssistZ] playclock end={_analyzer.WritePosition} ct={_player.CurrentTime:0.000000}");
                        PumpLabels();
                    }
                    finally
                    {
                        data.Close();
                    }
                    // Every hop boundary the chunks crossed gets its own window, all in one dispatch. One analysis in
                    // flight at a time; reading never waits on the GPU readback (hops pile up and go in the next batch).
                    if (analysis == null || analysis.IsCompleted)
                    {
                        if (nextAnalysis == 0) nextAnalysis = _analyzer.OldestAnalyzableEnd;
                        // Fell further behind than one batch (a stalled tab): skip to the newest windows rather than lag forever.
                        long behind = _analyzer.WritePosition - nextAnalysis;
                        if (behind >= (long)StereoAnalyzer.MaxWindows * hop)
                        {
                            long skip = (behind / hop - (StereoAnalyzer.MaxWindows - 1)) * hop;
                            Stats.SkippedWindows += skip / hop;
                            nextAnalysis += skip;
                        }
                        if (nextAnalysis <= _analyzer.WritePosition)
                        {
                            int count = (int)((_analyzer.WritePosition - nextAnalysis) / hop) + 1;
                            analysis = AnalyzeAsync(nextAnalysis, hop, count);
                            nextAnalysis += (long)count * hop;
                        }
                    }
                }
            }
            catch (Exception ex) when (!ct.IsCancellationRequested)
            {
                Error = $"Audio: {ex.Message}";
                Changed?.Invoke();
            }
            finally
            {
                if (analysis != null) await analysis;
                await reader.Cancel();
            }
        }

        async Task AnalyzeAsync(long firstEnd, int hop, int count)
        {
            var sw = Stopwatch.StartNew();
            var frames = await _analyzer!.AnalyzeAsync(firstEnd, hop, count);
            Stats.RecordAnalysis(sw.Elapsed.TotalMilliseconds, count);
            // Host-side latency of one batch: dispatches + GPU work + readback (mapAsync), not GPU time alone.
            if (LabLogging)
                Console.WriteLine($"[AssistZ] abatch ms={sw.Elapsed.TotalMilliseconds:0.000} n={count} spectral={(_analyzer.SpectralEnabled ? 1 : 0)}");
            // Each window's time comes from its sample position (the newest window = now), so windows of one batch
            // are 21 ms apart for the detector's refractory period, not all at the same instant.
            var now = DateTime.UtcNow;
            double rate = Stats.AudioSampleRate > 0 ? Stats.AudioSampleRate : 48000;
            long newest = frames[^1].EndSample;
            foreach (var frame in frames) Process(frame, rate, now - TimeSpan.FromSeconds((newest - frame.EndSample) / rate));
        }

        void Process(StereoFrame frame, double sampleRate, DateTime wallTime)
        {
            LastStereo = frame;
            Cadence.Add(frame.EndSample, frame.Rms > 1e-9f ? 20f * MathF.Log10(frame.Rms) : -180f, sampleRate);
            var ev = Detector.Process(frame, sampleRate, wallTime);
            if (ev != null)
            {
                double calibrated = Profile.Calibrate(ev.AzimuthDegrees);
                var track = Tracker.Add(calibrated, ViewYawDegrees, wallTime);
                _trackSides[track.TrackId] = track.Side;
                ev = ev with { CalibratedAzimuth = calibrated, ViewYawAtEvent = ViewYawDegrees, TrackId = track.TrackId };
                Events.Add(ev);
                if (Labeler.State == LabelerState.Ready) _toLabel.Enqueue(ev);
                // One line per event (a few per second at most): the verification harness reads these from the console.
                Console.WriteLine($"[AssistZ] event az={ev.AzimuthDegrees:0.0} level={ev.LevelDb:0.0}dB rise={ev.RiseDb:0.0}dB lag={ev.LagSamples} corr={ev.Correlation:0.00} saz={frame.SpectralAzimuthDegrees:0.0} cen={frame.SpectralCentroid:0.000} hf={frame.HighBandFraction:0.000} yaw={ViewYawDegrees:0.0} track={track.TrackId} side={track.Side} conf={track.Confidence:0.00} turned={track.TurnedDegrees:0.0}");
            }
        }

        /// <summary>Starts the next classification whose clip is complete; drops the ones whose clip already left the ring.</summary>
        void PumpLabels()
        {
            if (_analyzer == null || (_labeling != null && !_labeling.IsCompleted)) return;
            while (_toLabel.Count > 0)
            {
                var ev = _toLabel.Peek();
                long start = SoundLabeler.ClipStart(ev.EndSample);
                if (start + SoundLabeler.ClipSpan > _analyzer.WritePosition) return;   // not all written yet
                _toLabel.Dequeue();
                if (!_analyzer.InRing(start, SoundLabeler.ClipSpan)) { LabelsDropped++; continue; }
                _labeling = LabelAsync(_analyzer, ev);
                return;
            }
        }

        async Task LabelAsync(StereoAnalyzer analyzer, SoundEvent ev)
        {
            try
            {
                var result = await Labeler.ClassifyAsync(analyzer, ev.EndSample, ev.AzimuthDegrees);
                if (result is not (string word, string clap, float p)) { LabelsDropped++; return; }
                // By now the clip (0.85 s past the onset) is written, so the cadence sees the shots after it too.
                double rate = Stats.AudioSampleRate > 0 ? Stats.AudioSampleRate : 48000;
                bool? automatic = SoundLabeler.IsGunfire(word) ? Cadence.IsAutomatic(ev.EndSample, rate) : null;
                string label = SoundLabeler.IsGunfire(word) ? (automatic == true ? "automatic fire" : "gunshot") : word;
                string rawLabel = label;
                int i = Events.IndexOf(ev);
                // Now it has a name, gunfire takes its own direction curve; the radar draws from this event.
                double? distance = SoundLabeler.IsGunfire(word) ? Labeler.EstimateDistanceMeters() : null;
                var gun = SoundLabeler.IsGunfire(word) ? Labeler.CallGun() : null;
                if (i >= 0) Events[i] = ev with { Label = label, ClapLabel = clap, LabelProbability = p, CalibratedAzimuth = SoundLabeler.IsGunfire(word) ? Profile.CalibrateGunfire(ev.AzimuthDegrees) : ev.CalibratedAzimuth, DistanceMeters = distance, GunName = gun?.Name, GunFamily = gun?.Family };
                // One word per burst: earlier events take it too, on the HUD and in the log.
                string? burstWord = _burstVote?.Add(ev.EndSample, label, rate);
                if (burstWord != null)
                {
                    long burstStart = _burstVote!.BurstStart;
                    bool gunfireYields = _burstVote.GunfireYields;
                    for (int k = Events.Count - 1; k >= 0 && Events[k].EndSample >= burstStart; k--)
                    {
                        if (Events[k].Label is string old && old != burstWord && (Labeler.IsWorldEvent(old) || (gunfireYields && old is "gunshot" or "automatic fire")))
                        {
                            Events[k] = Events[k] with { Label = burstWord, DistanceMeters = null, GunName = null, GunFamily = null };   // a distance and a gun were only ever measured on gunfire
                            if (Events[k].EndSample != ev.EndSample)
                                Console.WriteLine($"[AssistZ] relabel end={Events[k].EndSample} label={burstWord}");
                        }
                    }
                    if (Labeler.IsWorldEvent(label) || (gunfireYields && label is "gunshot" or "automatic fire")) label = burstWord;
                }
                Console.WriteLine($"[AssistZ] label utc={DateTime.UtcNow:HH:mm:ss.fff} end={ev.EndSample} label={label} raw={rawLabel} clap={clap} p={p:0.000} ms={Labeler.LastMs:0.0} az={ev.AzimuthDegrees:0.0} level={ev.LevelDb:0.0}{(distance is double m ? $" dist={m.ToString("0.000", CultureInfo.InvariantCulture)}" : "")}{(gun is GunCall g ? $" gun={g.Name ?? "-"} family={g.Family ?? "-"}" : "")}");
                if (MemLog) LogMemory();
                if (LabLogging)
                {
                    var all = new StringBuilder();
                    foreach (float score in Labeler.LastScores) all.Append(score.ToString("R", CultureInfo.InvariantCulture)).Append(',');
                    Console.WriteLine($"[AssistZ] scores end={ev.EndSample} clip={Labeler.LastClipStart} v={all}");
                    if (Labeler.KeepEmbedding && Labeler.LastEmbedding is float[] embedding)
                    {
                        var dump = new StringBuilder();
                        foreach (float value in embedding) dump.Append(value.ToString("R", CultureInfo.InvariantCulture)).Append(',');
                        Console.WriteLine($"[AssistZ] embed end={ev.EndSample} v={dump}");
                    }
                }
            }
            catch (Exception ex)
            {
                Error = $"Labels: {ex.Message}";
                Changed?.Invoke();
            }
        }

        public async ValueTask DisposeAsync() => await StopAsync();
    }

    /// <summary>Throughput counters for the harness page.</summary>
    public sealed class CaptureStats
    {
        readonly Stopwatch _clock = new();
        public string? VideoLabel, AudioLabel, AudioFormat;
        public int VideoWidth, VideoHeight, AudioChannels, AudioChunkFrames;
        public double AudioSampleRate;
        public long VideoFramesSeen, VideoFramesProcessed, AudioChunks, Analyses, Windows, SkippedWindows;
        /// <summary>Seconds of audio that never reached the app (<see cref="AudioGapMeter"/>).</summary>
        public double AudioLostSeconds;
        public double LastVideoMs, MaxVideoMs, LastAnalysisMs, MaxAnalysisMs;
        public double Seconds => _clock.Elapsed.TotalSeconds;

        public void Reset()
        {
            VideoLabel = AudioLabel = AudioFormat = null;
            VideoWidth = VideoHeight = AudioChannels = AudioChunkFrames = 0;
            AudioSampleRate = 0;
            VideoFramesSeen = VideoFramesProcessed = AudioChunks = Analyses = Windows = SkippedWindows = 0;
            AudioLostSeconds = 0;
            LastVideoMs = MaxVideoMs = LastAnalysisMs = MaxAnalysisMs = 0;
            _clock.Restart();
        }

        public void RecordVideo(double ms, int width, int height)
        {
            VideoFramesProcessed++;
            LastVideoMs = ms;
            if (ms > MaxVideoMs) MaxVideoMs = ms;
            VideoWidth = width;
            VideoHeight = height;
        }

        public void RecordAnalysis(double ms, int windows)
        {
            Analyses++;
            Windows += windows;
            LastAnalysisMs = ms;
            if (ms > MaxAnalysisMs) MaxAnalysisMs = ms;
        }
    }
}
