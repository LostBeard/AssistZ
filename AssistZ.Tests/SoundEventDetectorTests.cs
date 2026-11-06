using AssistZ.Audio;
using System.Globalization;
using Xunit;

namespace AssistZ.Tests
{
    public class SoundEventDetectorTests
    {
        const double Rate = 48000;
        const int Hop = 1024;

        static StereoFrame Frame(long end, float db)
        {
            float rms = MathF.Pow(10f, db / 20f);
            return new StereoFrame(end, rms, rms, 0, 1f, 1f, 0f, 0f, 0f);
        }

        static List<SoundEvent> Run(IEnumerable<(long end, float db)> levels)
        {
            var detector = new SoundEventDetector();
            var events = new List<SoundEvent>();
            foreach (var (end, db) in levels)
                if (detector.Process(Frame(end, db), Rate, DateTime.UnixEpoch) is SoundEvent e) events.Add(e);
            return events;
        }

        static IEnumerable<(long, float)> RealLevels()
        {
            foreach (var line in File.ReadLines(Path.Combine(AppContext.BaseDirectory, "Fixtures", "run4_levels.csv")))
            {
                if (line.StartsWith('#') || line.StartsWith("end_sample")) continue;
                var parts = line.Split(',');
                yield return (long.Parse(parts[0], CultureInfo.InvariantCulture), float.Parse(parts[1], CultureInfo.InvariantCulture));
            }
        }

        // Ambient at -45 dB from the first hop is not a sound. With the background starting at the floor it read as a 25 dB rise.
        [Fact]
        public void SteadyAmbientFromTheFirstHopIsNotAnEvent()
        {
            Assert.Empty(Run(Enumerable.Range(1, 300).Select(i => ((long)i * Hop, -45f))));
        }

        // The M0 test signal starts from digital silence: its first pulse must still count.
        [Fact]
        public void APulseAfterSilenceIsAnEvent()
        {
            var levels = Enumerable.Range(1, 100).Select(i => ((long)i * Hop, i < 50 ? -180f : -20f));
            Assert.Single(Run(levels.Where(l => l.Item1 <= 50 * Hop)));
        }

        // A sound after steady ambient still counts.
        [Fact]
        public void ARiseOverAmbientIsAnEvent()
        {
            var levels = Enumerable.Range(1, 200).Select(i => ((long)i * Hop, i == 150 ? -20f : -45f));
            Assert.Single(Run(levels));
        }

        // Real DayZ audio (lab run 4), from the start of the recording: nothing happens in its first 30 s, and the first shots are at about 32 s.
        [Fact]
        public void RealAmbientAtTheStartIsNotAnEvent()
        {
            var events = Run(RealLevels());
            Assert.DoesNotContain(events, e => e.EndSample < 30 * Rate);
            Assert.Contains(events, e => e.EndSample > 31.5 * Rate && e.EndSample < 38 * Rate);
        }
    }

    public class BurstVoteTests
    {
        const double Rate = 48000;
        static bool World(string w) => w is "gas strike" or "heli crash" or "sleigh crash";

        static List<string?> Feed(params (double t, string word)[] events)
        {
            var vote = new BurstVote(World);
            return events.Select(e => vote.Add((long)(e.t * Rate), e.word, Rate)).ToList();
        }

        // A real strike, lab run sniper_20261104_021109 from 181.4 s, in the order the probe named it live: it starts as "heli crash".
        [Fact]
        public void ARealStrikeEndsAsOneWord()
        {
            var words = Feed((181.42, "heli crash"), (181.59, "heli crash"), (181.76, "gas strike"), (181.93, "heli crash"), (182.29, "gas strike"), (182.46, "gas strike"), (182.63, "gas strike"),
                (182.80, "gas strike"), (182.97, "gas strike"), (183.15, "gas strike"), (183.32, "gas strike"), (183.49, "gas strike"), (183.66, "gas strike"), (183.98, "gas strike"), (185.60, "gas strike"));
            Assert.Equal("heli crash", words[0]);
            Assert.Equal("gas strike", words[^3]);
        }

        [Fact]
        public void ATieKeepsTheWordSeenFirst()
        {
            Assert.Equal("gas strike", Feed((1.0, "gas strike"), (1.2, "heli crash"))[1]);
        }

        [Fact]
        public void AGapStartsANewBurst()
        {
            var vote = new BurstVote(World);
            vote.Add((long)(1.0 * Rate), "gas strike", Rate);
            vote.Add((long)(1.2 * Rate), "gas strike", Rate);
            Assert.Equal("heli crash", vote.Add((long)(3.0 * Rate), "heli crash", Rate));
            Assert.Equal((long)(3.0 * Rate), vote.BurstStart);
        }

        static BurstVote Gun() => new BurstVote(w => w is "gas strike" or "heli crash" or "thunder", w => w is "gunshot" or "automatic fire");

        // A real one: lab run sniper_20261104_175558 at 172.8 s, a dry storm. The first event of the burst read gunshot, the rest thunder.
        [Fact]
        public void GunfireOutnumberedByThunderYields()
        {
            var vote = Gun();
            vote.Add((long)(172.8 * Rate), "gunshot", Rate);
            Assert.False(vote.GunfireYields);
            vote.Add((long)(173.0 * Rate), "thunder", Rate);
            Assert.False(vote.GunfireYields);   // a tie keeps the gunfire
            Assert.Equal("thunder", vote.Add((long)(173.2 * Rate), "thunder", Rate));
            Assert.True(vote.GunfireYields);
        }

        [Fact]
        public void RealFireKeepsItsWordAmongThunder()
        {
            var vote = Gun();
            foreach (double t in new[] { 1.0, 1.15, 1.3, 1.45 }) vote.Add((long)(t * Rate), "automatic fire", Rate);
            vote.Add((long)(1.6 * Rate), "thunder", Rate);
            vote.Add((long)(1.75 * Rate), "thunder", Rate);
            Assert.False(vote.GunfireYields);
        }

        [Fact]
        public void GunfireYieldsToNoOtherWord()
        {
            var vote = Gun();
            vote.Add((long)(1.0 * Rate), "gunshot", Rate);
            vote.Add((long)(1.2 * Rate), "gas strike", Rate);
            vote.Add((long)(1.4 * Rate), "gas strike", Rate);
            Assert.False(vote.GunfireYields);
            var heli = Gun();
            heli.Add((long)(1.0 * Rate), "automatic fire", Rate);
            heli.Add((long)(1.2 * Rate), "heli crash", Rate);
            heli.Add((long)(1.4 * Rate), "heli crash", Rate);
            Assert.False(heli.GunfireYields);
        }

        [Fact]
        public void ANewBurstForgetsItsGunfire()
        {
            var vote = Gun();
            vote.Add((long)(1.0 * Rate), "gunshot", Rate);
            vote.Add((long)(5.0 * Rate), "thunder", Rate);
            vote.Add((long)(5.2 * Rate), "thunder", Rate);
            Assert.False(vote.GunfireYields);
        }

        // Gunfire takes no part: it neither votes nor gets a word.
        [Fact]
        public void OtherWordsDoNotVote()
        {
            Assert.Null(Feed((1.0, "gunfire"), (1.2, "footsteps"))[1]);
            Assert.Equal("gas strike", Feed((1.0, "gunfire"), (1.2, "gunfire"), (1.3, "gas strike"))[2]);
        }
    }
}
