using AssistZ.Audio;
using Xunit;

namespace AssistZ.Tests
{
    public class AudioGapMeterTests
    {
        [Fact]
        public void ContinuousChunks_LoseNothing()
        {
            var m = new AudioGapMeter();
            for (int i = 0; i < 100; i++) Assert.Equal(0, m.Add(1_000_000 + i * 10_000L, 480, 48000), 9);
            Assert.Equal(0, m.LostSeconds, 9);
            Assert.Equal(0, m.Gaps);
        }

        [Fact]
        public void AMissingStretch_IsCounted()
        {
            var m = new AudioGapMeter();
            m.Add(0, 480, 48000);
            m.Add(10_000, 480, 48000);
            Assert.Equal(0.25, m.Add(270_000, 480, 48000), 9);      // due at 20 ms, came at 270 ms
            Assert.Equal(0.25, m.LostSeconds, 9);
            Assert.Equal(1, m.Gaps);
            Assert.Equal(0, m.Add(280_000, 480, 48000), 9);          // and it carries on from there
            Assert.Equal(0.1, m.Add(390_000, 480, 48000), 9);
            Assert.Equal(0.35, m.LostSeconds, 9);
            Assert.Equal(2, m.Gaps);
        }

        [Fact]
        public void Jitter_IsNotLoss()
        {
            var m = new AudioGapMeter();
            m.Add(0, 480, 48000);
            Assert.Equal(0, m.Add(13_000, 480, 48000), 9);          // 3 ms late
            Assert.Equal(0, m.Add(20_000, 480, 48000), 9);          // early
            Assert.Equal(0, m.LostSeconds, 9);
        }
    }
}
