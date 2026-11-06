namespace AssistZ.Audio
{
    /// <summary>
    /// Finds the level peaks of distinct transients in the per-hop level stream and measures their spacing, so gunfire can
    /// be split into single shots and automatic fire: CLAP cannot (a 1 s clip of an SVD fired every 0.5 s already holds two
    /// shots, and it reads "machine gun fire" as readily as an AUG burst - Tuvok, 2026-11-03, with transformers too).
    /// <para>
    /// A peak counts when it stands <see cref="ProminenceDb"/> above the lowest level since the previous peak AND the level
    /// then falls that far below it again (a streaming two-sided prominence). Echo tails and the detector's extra onsets on
    /// them do not clear it. Measured on lab run sniper_20261103_153510 at the app's resolution (2048-sample RMS every 1024):
    /// prominence 10 dB found 11 peaks for 11 SVD shots (median spacing 0.523 s, true 0.53) and 27 for 30 AUG shots (0.288 s,
    /// true 0.27); 14 dB already lost most of the burst.
    /// </para>
    /// </summary>
    public sealed class CadenceMeter
    {
        public float ProminenceDb { get; set; } = 10f;
        /// <summary>Quieter peaks are ignored (dBFS).</summary>
        public float FloorDb { get; set; } = -50f;
        /// <summary>Closest two peaks may be (s): a peak closer than this to the last one replaces it when louder.</summary>
        public double MinSpacingSeconds { get; set; } = 0.12;
        /// <summary>Median spacing under this is automatic fire (AUG 0.25-0.3 s vs SVD 0.48-0.6 s, measured).</summary>
        public double AutomaticBelowSeconds { get; set; } = 0.38;
        /// <summary>
        /// A peak this close (s) to a much louder one (<see cref="CompanionDb"/>) is part of the same shot and is dropped.
        /// Lab run sniper_20261103_160950: a transient 14 dB below the first SVD report and 150 ms before it (possibly the
        /// bullet's crack) made the burst start read 0.15 / 0.49 / 0.34 s spacing = "automatic". AUG shots are all equally
        /// loud, so a burst is not thinned.
        /// </summary>
        public double CompanionSeconds { get; set; } = 0.25;
        public float CompanionDb { get; set; } = 8f;
        /// <summary>How long peaks are kept (s).</summary>
        public double HistorySeconds { get; set; } = 10;

        readonly List<long> _peaks = new();
        // Lowest level since the last confirmed peak; the candidate peak (-1 = none) and its level, waiting for the fall.
        float _minSincePeak = float.MaxValue;
        long _candidate = -1;
        float _candidateDb, _lastPeakDb;

        /// <summary>Confirmed peak positions (sample of the hop's window end), oldest first.</summary>
        public IReadOnlyList<long> Peaks => _peaks;

        /// <summary>Feeds one hop's level.</summary>
        public void Add(long endSample, float levelDb, double sampleRate)
        {
            if (_candidate >= 0)
            {
                if (levelDb > _candidateDb)
                {
                    // Still rising (or rose again before falling far enough): the candidate moves up.
                    _candidate = endSample;
                    _candidateDb = levelDb;
                }
                else if (_candidateDb - levelDb >= ProminenceDb)
                {
                    Confirm(_candidate, _candidateDb, (long)(MinSpacingSeconds * sampleRate), (long)(CompanionSeconds * sampleRate));
                    _candidate = -1;
                    _minSincePeak = levelDb;
                }
            }
            else
            {
                _minSincePeak = Math.Min(_minSincePeak, levelDb);
                if (levelDb >= FloorDb && levelDb - _minSincePeak >= ProminenceDb)
                {
                    _candidate = endSample;
                    _candidateDb = levelDb;
                }
            }
            Trim(endSample, sampleRate);
        }

        void Confirm(long peak, float db, long minSpacing, long companion)
        {
            if (_peaks.Count > 0)
            {
                long gap = peak - _peaks[^1];
                bool merge = gap < minSpacing || (gap < companion && Math.Abs(db - _lastPeakDb) >= CompanionDb);
                if (merge)
                {
                    // One shot: keep the louder of the two.
                    if (db > _lastPeakDb) { _peaks[^1] = peak; _lastPeakDb = db; }
                    return;
                }
            }
            _peaks.Add(peak);
            _lastPeakDb = db;
        }

        void Trim(long now, double sampleRate)
        {
            long keep = (long)(HistorySeconds * sampleRate);
            int drop = 0;
            while (drop < _peaks.Count && now - _peaks[drop] > keep) drop++;
            if (drop > 0) _peaks.RemoveRange(0, drop);
        }

        /// <summary>Median spacing of the peaks in [from, to] (s); null with fewer than 3 peaks (2 spacings).</summary>
        public double? MedianSpacing(long from, long to, double sampleRate)
        {
            var inside = _peaks.Where(p => p >= from && p <= to).ToList();
            if (inside.Count < 3) return null;
            var gaps = inside.Zip(inside.Skip(1), (a, b) => (b - a) / sampleRate).OrderBy(g => g).ToList();
            int n = gaps.Count;
            return n % 2 == 1 ? gaps[n / 2] : 0.5 * (gaps[n / 2 - 1] + gaps[n / 2]);
        }

        /// <summary>
        /// True when the peaks from <paramref name="pastSeconds"/> before <paramref name="sample"/> to
        /// <paramref name="futureSeconds"/> after it come faster than <see cref="AutomaticBelowSeconds"/>; null when there are too
        /// few to tell. The look back is long because peaks sit on the hop grid: close shots that do not fall
        /// <see cref="ProminenceDb"/> between them lose peaks at some grid phases, and the gaps left are doubled. Measured on all
        /// five lab recordings at 16 hop phases (169 logged shots; only one burst, 30 AUG shots, is automatic): +-1 s called AUG
        /// shots single 17 times of 480 and other weapons automatic 38 times of 2224; 3 s back and 1 s ahead 0 and 24. The look
        /// ahead stays at 1 s because a label is made as soon as its clip is written.
        /// </summary>
        public bool? IsAutomatic(long sample, double sampleRate, double pastSeconds = 3.0, double futureSeconds = 1.0)
        {
            var median = MedianSpacing(sample - (long)(pastSeconds * sampleRate), sample + (long)(futureSeconds * sampleRate), sampleRate);
            return median is double m ? m < AutomaticBelowSeconds : null;
        }
    }
}
