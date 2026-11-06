using AssistZ.Audio;
using AssistZ.Services;
using Xunit;

namespace AssistZ.Tests
{
    public class SourceTrackerTests
    {
        static readonly DateTime T0 = new(2026, 11, 3, 12, 0, 0, DateTimeKind.Utc);

        [Theory]
        [InlineData(0, 0)]
        [InlineData(45, 45)]
        [InlineData(90, 90)]
        [InlineData(135, 45)]    // behind-right reads like front-right
        [InlineData(180, 0)]     // straight behind reads centre
        [InlineData(-135, -45)]
        [InlineData(200, -20)]   // 200 = -160: behind-left
        [InlineData(-30, -30)]
        [InlineData(390, 30)]
        public void Fold_MapsTheCircleOntoTheFrontHalf(double bearing, double expected)
            => Assert.Equal(expected, SourceTracker.Fold(bearing), 6);

        // A world-fixed source at head-relative bearing b0 (view heading 0 at start). The view sweeps 0 -> sweep degrees
        // and back; one event every step. Returns the last result.
        static TrackResult Run(SourceTracker tracker, double b0, double sweep, int steps = 24, double noise = 0, int seed = 1)
        {
            var rng = new Random(seed);
            TrackResult last = default;
            for (int i = 0; i < steps; i++)
            {
                double phase = (double)i / (steps - 1);
                double yaw = sweep * Math.Sin(phase * Math.PI);           // out and back
                double az = SourceTracker.Fold(b0 - yaw) + (rng.NextDouble() * 2 - 1) * noise;
                last = tracker.Add(az, yaw, T0.AddMilliseconds(300 * i));
            }
            return last;
        }

        [Theory]
        [InlineData(30)]
        [InlineData(-50)]
        [InlineData(0)]
        [InlineData(70)]
        public void FrontSource_ResolvesFront_WhenTheViewTurns(double bearing)
        {
            var r = Run(new SourceTracker(), bearing, sweep: 30);
            Assert.Equal(SourceSide.Front, r.Side);
            Assert.Equal(24, r.Events);   // one track, every event joined it
        }

        [Theory]
        [InlineData(150)]
        [InlineData(-120)]
        [InlineData(180)]
        [InlineData(110)]
        public void BackSource_ResolvesBack_WhenTheViewTurns(double bearing)
        {
            var r = Run(new SourceTracker(), bearing, sweep: 30);
            Assert.Equal(SourceSide.Back, r.Side);
            Assert.Equal(24, r.Events);
        }

        [Fact]
        public void WithoutTurning_TheSideStaysUnknown()
        {
            // Front at 30 and back at 150 read identically while the view is still: nothing can be known.
            Assert.Equal(SourceSide.Unknown, Run(new SourceTracker(), 30, sweep: 0).Side);
            Assert.Equal(SourceSide.Unknown, Run(new SourceTracker(), 150, sweep: 0).Side);
        }

        [Fact]
        public void SmallTurn_BelowTheThreshold_StaysUnknown()
        {
            var r = Run(new SourceTracker(), 150, sweep: 8);
            Assert.Equal(SourceSide.Unknown, r.Side);
            Assert.True(r.TurnedDegrees < 12);
        }

        [Theory]
        [InlineData(40, 1)]
        [InlineData(140, 2)]
        [InlineData(-40, 3)]
        [InlineData(-140, 4)]
        public void Resolves_WithThreeDegreesOfMeasurementNoise(double bearing, int seed)
        {
            var r = Run(new SourceTracker(), bearing, sweep: 35, noise: 3, seed: seed);
            Assert.Equal(Math.Abs(bearing) < 90 ? SourceSide.Front : SourceSide.Back, r.Side);
        }

        [Fact]
        public void TwoSources_OnDifferentSides_KeepSeparateTracks()
        {
            var tracker = new SourceTracker();
            TrackResult a = default, b = default;
            for (int i = 0; i < 24; i++)
            {
                double yaw = 30 * Math.Sin((double)i / 23 * Math.PI);
                var t = T0.AddMilliseconds(300 * i);
                a = tracker.Add(SourceTracker.Fold(-60 - yaw), yaw, t);                    // front-left
                b = tracker.Add(SourceTracker.Fold(130 - yaw), yaw, t.AddMilliseconds(150)); // behind-right
            }
            Assert.NotEqual(a.TrackId, b.TrackId);
            Assert.Equal(SourceSide.Front, a.Side);
            Assert.Equal(SourceSide.Back, b.Side);
        }

        [Fact]
        public void TwoSources_TwelveDegreesApart_KeepSeparateTracks()
        {
            // The lab case that merged at a 14 deg match window: alternating readings of +6.9 and -6.9, view still.
            var tracker = new SourceTracker();
            var ids = new HashSet<int>();
            for (int i = 0; i < 10; i++)
                ids.Add(tracker.Add(i % 2 == 0 ? 6.9 : -6.9, 0, T0.AddMilliseconds(1500 * i)).TrackId);
            Assert.Equal(2, ids.Count);
        }

        [Fact]
        public void Track_EndsAfterTheTimeout()
        {
            var tracker = new SourceTracker();
            var first = tracker.Add(20, 0, T0);
            var later = tracker.Add(20, 0, T0 + tracker.TrackTimeout + TimeSpan.FromSeconds(1));
            Assert.NotEqual(first.TrackId, later.TrackId);
            Assert.Equal(1, tracker.ActiveTracks);
        }

        [Fact]
        public void DayZProfile_UsesTheMeasuredCalibration()
        {
            var p = GameProfile.DayZ;
            Assert.Equal(45 / 0.91, p.Calibrate(45), 6);
            Assert.Equal(90, p.Calibrate(88));                       // clamped to the front half
            Assert.Equal(58.6, p.CalibrateGunfire(35), 6);           // gunfire reads compressed at the sides
            Assert.Equal(-58.6, p.CalibrateGunfire(-35), 6);         // mirrored
            Assert.Equal((29.8 + 58.6) / 2, p.CalibrateGunfire(30), 6);
            Assert.Equal(0, p.CalibrateGunfire(0), 6);
            Assert.Equal(90, p.CalibrateGunfire(120), 6);            // held at the side
            Assert.Equal(GameProfile.Generic.Calibrate(40), GameProfile.Generic.CalibrateGunfire(40), 6);   // measured on DayZ only
            Assert.Equal(10, p.ShiftToDegrees(88, 960), 6);           // 8.8 px per degree at 960 px
            Assert.Equal(10, p.ShiftToDegrees(44, 480), 6);           // scales with the frame width
        }

        [Fact]
        public void DistanceProbe_IsARidgeOnTheLogOfDistance()
        {
            const string near = "\"Near\":{\"Mean\":[1,0],\"Std\":[2,1],\"Weights\":[0.5,-1],\"Intercept\":5}";
            const string gate = "\"FarGate\":{\"Mean\":[0,0],\"Std\":[1,1],\"Weights\":[0,1],\"Intercept\":-10}";     // far only when the second value passes 10
            const string far = "\"Far\":{\"Mean\":[0,0],\"Std\":[1,1],\"Weights\":[1,0],\"Intercept\":8}";
            var probe = AssistZ.Audio.DistanceProbe.FromJson("{\"Checkpoint\":\"c\",\"CutMeters\":120," + near + "," + gate + "," + far + "}");
            Assert.Equal("c", probe.Checkpoint);
            Assert.Equal(32, probe.Estimate(new[] { 1f, 0f }), 6);          // at the mean: 2^5
            Assert.Equal(64, probe.Estimate(new[] { 5f, 0f }), 6);          // (5-1)/2 * 0.5 = +1
            Assert.Equal(16, probe.Estimate(new[] { 1f, 1f }), 6);          // -1
            Assert.Equal(512, probe.Estimate(new[] { 1f, 11f }), 6);        // past the gate, the far ridge: 2^(8+1)
            Assert.Equal(120, probe.Estimate(new[] { -5f, 11f }), 6);       // a far call never reads nearer than the cut
            Assert.Throws<ArgumentException>(() => { probe.Estimate(new[] { 1f }); });
        }

        [Theory]
        [InlineData(0, 0)]
        [InlineData(50, 0.3)]
        [InlineData(100, 0.6)]      // the inner ring
        [InlineData(800, 1)]        // the outer ring
        [InlineData(5000, 1)]       // held there
        public void Radar_ScaleIsLinearThenLogarithmic(double metres, double fraction) =>
            Assert.Equal(fraction, AssistZ.Audio.DistanceProbe.RadarFraction(metres), 6);

        [Fact]
        public void Radar_FarSitsBetweenTheRingsInOrder()
        {
            double at200 = AssistZ.Audio.DistanceProbe.RadarFraction(200), at400 = AssistZ.Audio.DistanceProbe.RadarFraction(400);
            Assert.True(0.6 < at200 && at200 < at400 && at400 < 1);
            Assert.Equal(0.6 + 0.4 / 3, at200, 6);                          // a third of the way in log: 100, 200, 400, 800
        }

        [Theory]
        [InlineData(63.4, "63 m")]
        [InlineData(99.4, "99 m")]
        [InlineData(100, "~100 m")]
        [InlineData(347, "~350 m")]
        [InlineData(799, "~800 m")]
        [InlineData(800, "800+ m")]
        public void Radar_ReadsNearInMetresAndFarRounded(double metres, string reading) =>
            Assert.Equal(reading, AssistZ.Audio.DistanceProbe.Describe(metres));
    }
}
