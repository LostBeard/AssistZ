using AssistZ.Audio;
using System.Globalization;
using Xunit;

namespace AssistZ.Tests
{
    public class CadenceMeterTests
    {
        const double Rate = 48000;
        const int Hop = 1024;

        static CadenceMeter Feed(IEnumerable<(long end, float db)> levels)
        {
            var meter = new CadenceMeter();
            foreach (var (end, db) in levels) meter.Add(end, db, Rate);
            return meter;
        }

        static IEnumerable<(long, float)> RealLevels(string file = "run4_levels.csv")
        {
            foreach (var line in File.ReadLines(Path.Combine(AppContext.BaseDirectory, "Fixtures", file)))
            {
                if (line.StartsWith('#') || line.StartsWith("end_sample")) continue;
                var parts = line.Split(',');
                yield return (long.Parse(parts[0], CultureInfo.InvariantCulture), float.Parse(parts[1], CultureInfo.InvariantCulture));
            }
        }

        static long S(double seconds) => (long)(seconds * Rate);

        // Synthetic shots: a 25 dB jump over a -60 dB floor, decaying 3 dB per hop (an echo-ish tail), every period s.
        static IEnumerable<(long, float)> Shots(double period, double seconds)
        {
            for (long end = Hop; end < S(seconds); end += Hop)
            {
                double sinceShot = (end / Rate) % period;
                int hopsSince = (int)(sinceShot * Rate / Hop);
                yield return (end, MathF.Max(-60f, -35f - 3f * hopsSince));
            }
        }

        [Fact]
        public void SyntheticSingleShotsAreNotAutomatic()
        {
            var meter = Feed(Shots(0.52, 6));
            Assert.Equal(false, meter.IsAutomatic(S(3), Rate));
            Assert.InRange(meter.MedianSpacing(0, S(6), Rate)!.Value, 0.50, 0.54);
        }

        [Fact]
        public void SyntheticBurstIsAutomatic()
        {
            var meter = Feed(Shots(0.27, 6));
            Assert.Equal(true, meter.IsAutomatic(S(3), Rate));
            Assert.InRange(meter.MedianSpacing(0, S(6), Rate)!.Value, 0.25, 0.29);
        }

        [Fact]
        public void SilenceHasNoCadence()
        {
            var meter = Feed(Enumerable.Range(1, 300).Select(i => ((long)i * Hop, -65f)));
            Assert.Null(meter.IsAutomatic(S(3), Rate));
        }

        // Real DayZ audio (lab run 4) against the game's own shot log: Peter's SVD, 11 shots 0.33-0.63 s apart, and Naomi's
        // AUG, 30 shots 0.20-0.33 s apart.
        [Fact]
        public void RealSvdShotsAreSingleShots()
        {
            // Only this segment: the meter keeps HistorySeconds (10 s) of peaks, so feeding on to 129 s would trim these.
            var meter = Feed(RealLevels().Where(l => l.Item1 < S(40)));
            var svd = meter.Peaks.Count(p => p > S(31.5) && p < S(37.8));
            Assert.InRange(svd, 9, 13);
            Assert.InRange(meter.MedianSpacing(S(31.5), S(37.8), Rate)!.Value, 0.45, 0.62);
            foreach (var t in new[] { 32.5, 34.0, 35.5, 36.8 })
                Assert.Equal(false, meter.IsAutomatic(S(t), Rate));
        }

        // Lab run 5: the first SVD report has a quiet transient 150 ms before it (14 dB down). Counted as its own peak it made
        // the burst start read automatic (gaps 0.15 / 0.49 / 0.34 s); it must merge into the shot.
        [Fact]
        public void RealSvdBurstStartIsSingleShots()
        {
            var meter = Feed(RealLevels("run5_levels.csv"));
            foreach (var t in new[] { 30.6, 31.0, 31.5, 33.0, 35.0 })
                Assert.Equal(false, meter.IsAutomatic(S(t), Rate));
            Assert.InRange(meter.Peaks.Count(p => p > S(30.0) && p < S(36.5)), 10, 12);
        }

        // The AUG burst of lab run 4 with the stream started 256 samples later, which moves it against the hop grid. Peaks near
        // 123.14, 123.59 and 123.79 s do not clear the prominence at this phase. The old +-1 s rule (asserted so this test cannot
        // pass for the wrong reason) read those shots as single.
        [Fact]
        public void RealAugBurstIsAutomaticAtTheHardHopPhase()
        {
            var meter = Feed(RealLevels("run4_levels_skip256.csv"));
            foreach (var t in new[] { 123.59, 123.80 })
            {
                Assert.Equal(false, meter.IsAutomatic(S(t), Rate, pastSeconds: 1.0, futureSeconds: 1.0));
                Assert.Equal(true, meter.IsAutomatic(S(t), Rate));
            }
        }

        [Fact]
        public void RealAugBurstIsAutomatic()
        {
            var meter = Feed(RealLevels().Where(l => l.Item1 > S(110)));
            var aug = meter.Peaks.Count(p => p > S(119.4) && p < S(128.0));
            Assert.InRange(aug, 22, 34);
            Assert.InRange(meter.MedianSpacing(S(119.4), S(128.0), Rate)!.Value, 0.24, 0.34);
            foreach (var t in new[] { 121.0, 123.0, 125.0, 127.0 })
                Assert.Equal(true, meter.IsAutomatic(S(t), Rate));
        }
    }
}
