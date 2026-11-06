namespace AssistZ.Audio
{
    /// <summary>A transient the detector flagged: when, how loud, and which way the stereo mix placed it.</summary>
    /// <param name="AzimuthDegrees">Raw stereo azimuth (constant-power inverse), -90 left .. 90 right.</param>
    /// <param name="CalibratedAzimuth">The game profile's true front-half azimuth (set by the capture session).</param>
    /// <param name="ViewYawAtEvent">The integrated view heading when the sound was heard (set by the capture session), so
    /// the radar can keep the sound where it happened while the view turns.</param>
    public sealed record SoundEvent(DateTime Time, float AzimuthDegrees, float LevelDb, float RiseDb, int LagSamples, float Correlation)
    {
        public double CalibratedAzimuth { get; init; } = AzimuthDegrees;
        public double ViewYawAtEvent { get; init; }
        /// <summary>Estimated distance in metres (<see cref="DistanceProbe"/>); null when not measured for this kind of sound, which is everything but gunfire so far.</summary>
        public double? DistanceMeters { get; init; }
        /// <summary>The gun's name, where it may be said (<see cref="GunProbe"/>); null otherwise.</summary>
        public string? GunName { get; init; }
        /// <summary>The gun's family: sniper, assault, smg, pistol; null when unsure.</summary>
        public string? GunFamily { get; init; }
        /// <summary>The repeating-source track this event joined (<see cref="SourceTracker"/>); 0 = none.</summary>
        public int TrackId { get; init; }
        /// <summary>Sample position (exclusive end) of the window that flagged the onset.</summary>
        public long EndSample { get; init; }
        /// <summary>What the HUD calls it, once classified: CLAP's label, with gunfire split into "gunshot" / "automatic
        /// fire" by the shot cadence (<see cref="CadenceMeter"/>); null before or without the classifier.</summary>
        public string? Label { get; init; }
        /// <summary>CLAP's own best label.</summary>
        public string? ClapLabel { get; init; }
        /// <summary>Confidence in <see cref="Label"/>: the probe's for gunfire, otherwise CLAP's softmax over the label set.</summary>
        public float LabelProbability { get; init; }
    }

    /// <summary>
    /// First-cut transient detector over the analyzer's per-hop levels: an event is a hop whose level jumps
    /// <see cref="RiseThresholdDb"/> above the slow background average. No classification yet (gunshot / steps /
    /// vehicle need the lab recordings); this exists so the harness shows something to calibrate against.
    /// </summary>
    public sealed class SoundEventDetector
    {
        /// <summary>Rise over the background level that counts as an onset.</summary>
        public float RiseThresholdDb { get; set; } = 9f;
        /// <summary>Absolute floor: quieter hops are never events (dBFS).</summary>
        public float FloorDb { get; set; } = -55f;
        /// <summary>Minimum time between events.</summary>
        public TimeSpan Refractory { get; set; } = TimeSpan.FromMilliseconds(150);

        /// <summary>Lowest background level tracked (dBFS); also the reference a sound out of silence rises from.</summary>
        public float BackgroundFloorDb { get; set; } = -70f;

        float _backgroundDb;
        bool _seeded;
        long _lastEventSample = long.MinValue / 2;

        /// <summary>
        /// Feeds one hop; returns an event when it is an onset. Timing (the refractory period) runs on the SAMPLE
        /// clock: a batch of windows read back late (a shader compile, a busy tab) must not shift events against each
        /// other. <paramref name="wallTime"/> only stamps the event for display.
        /// </summary>
        public SoundEvent? Process(StereoFrame frame, double sampleRate, DateTime wallTime)
        {
            // Digital silence would drag the background to -inf and make every later sound a "160 dB rise": the
            // background never goes below BackgroundFloorDb.
            float levelDb = frame.Rms > 1e-9f ? 20f * MathF.Log10(frame.Rms) : -180f;
            float trackedDb = MathF.Max(levelDb, BackgroundFloorDb);
            // The background starts at the first audio heard. Starting at the floor, every ordinary sound read as a rise for the second or so the slow upward tracking took: 1-5 false events
            // around 0.2-1.2 s in most lab recordings, then nothing for 30 s.
            if (!_seeded)
            {
                _backgroundDb = trackedDb;
                _seeded = true;
            }
            float rise = trackedDb - _backgroundDb;
            SoundEvent? result = null;
            long refractorySamples = (long)(Refractory.TotalSeconds * sampleRate);
            if (levelDb > FloorDb && rise > RiseThresholdDb && frame.EndSample - _lastEventSample > refractorySamples)
            {
                result = new SoundEvent(wallTime, frame.PanAzimuthDegrees, levelDb, rise, frame.BestLagSamples, frame.PeakCorrelation) { EndSample = frame.EndSample };
                _lastEventSample = frame.EndSample;
            }
            // Background follows quickly downward and slowly upward, so a burst does not raise it at once.
            float rate = trackedDb < _backgroundDb ? 0.3f : 0.02f;
            _backgroundDb += (trackedDb - _backgroundDb) * rate;
            return result;
        }
    }
}
