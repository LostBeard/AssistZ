namespace AssistZ.Audio
{
    /// <summary>
    /// Audio that never reached the app, from the chunks' own timestamps: a chunk that starts later than the last one ended. Sample positions cannot show it, since they only count what arrived, and "Skipped" counts analysis windows dropped after arrival. Live runs lost about 3 s in 6 minutes while Skipped read 0 (docs/PLAN.md).
    /// </summary>
    public sealed class AudioGapMeter
    {
        /// <summary>Smaller differences are timestamp jitter. A chunk is about 10 ms.</summary>
        public const double MinGapSeconds = 0.005;

        double _expected = double.NaN;

        public double LostSeconds { get; private set; }
        public int Gaps { get; private set; }

        /// <summary>Takes one chunk; returns the seconds lost just before it, 0 if none.</summary>
        public double Add(long timestampMicroseconds, int frames, double sampleRate)
        {
            double start = timestampMicroseconds / 1e6;
            double gap = double.IsNaN(_expected) ? 0 : start - _expected;
            _expected = start + frames / sampleRate;
            if (gap < MinGapSeconds) return 0;
            LostSeconds += gap;
            Gaps++;
            return gap;
        }
    }
}
