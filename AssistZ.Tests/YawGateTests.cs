using AssistZ.Video;
using Xunit;

namespace AssistZ.Tests
{
    public class YawGateTests
    {
        static float Sum(YawGate gate, params (float Shift, float Score)[] frames) => frames.Sum(f => gate.Next(f.Shift, f.Score));

        [Fact]
        public void IsolatedJump_IsNotATurn()
        {
            // The live run: still, one frame at about 200 px with a good score, still.
            Assert.Equal(0, Sum(new YawGate(), (0, 1), (0, 1), (-200, 0.94f), (0, 1), (0, 1)), 3);
        }

        [Fact]
        public void PeakAtTheSearchEdge_IsNotAMeasurement()
        {
            Assert.Equal(0, Sum(new YawGate(), (-245, 0.94f)), 3);
            Assert.Equal(0, new YawGate().Next(240, 0.99f), 3);
            // Mid-turn too: the frames around it still count.
            Assert.Equal(60, Sum(new YawGate(), (30, 0.9f), (245, 0.9f), (30, 0.9f)), 3);
        }

        [Fact]
        public void RealTurn_KeepsEveryFrame_OnceTheNextFrameConfirmsIt()
        {
            var gate = new YawGate();
            Assert.Equal(0, gate.Next(0, 1), 3);
            Assert.Equal(0, gate.Next(100, 0.9f), 3);       // from rest: held
            Assert.Equal(220, gate.Next(120, 0.9f), 3);     // moving: released with this frame
            Assert.Equal(80, gate.Next(80, 0.9f), 3);       // no longer from rest
            Assert.Equal(0, gate.Next(0, 1), 3);
        }

        [Fact]
        public void SlowTurn_IsNeverHeld()
        {
            var gate = new YawGate();
            Assert.Equal(0, gate.Next(0, 1), 3);
            Assert.Equal(3, gate.Next(3, 0.9f), 3);
            Assert.Equal(39, gate.Next(39, 0.9f), 3);
        }

        [Fact]
        public void LowScore_DoesNotCount()
        {
            Assert.Equal(0, new YawGate().Next(30, 0.5f), 3);
            Assert.Equal(30, new YawGate().Next(30, 0.51f), 3);
        }

        [Fact]
        public void JumpInto_ALowScoreFrame_StillSaysMoving()
        {
            // The next frame moved but did not line up: the held jump was real motion, that frame adds nothing.
            var gate = new YawGate();
            gate.Next(0, 1);
            gate.Next(100, 0.9f);
            Assert.Equal(100, gate.Next(120, 0.3f), 3);
        }
    }
}
