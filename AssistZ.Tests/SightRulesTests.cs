using AssistZ.Video;
using Xunit;

namespace AssistZ.Tests
{
    public class SightRulesTests
    {
        static SightBox Box(float x, float y = 400, float conf = 0.7f, string kind = "tile1", float h = 60) => new(kind, x, y, 30, h, conf);

        [Fact]
        public void ThePictureIsAFlatProjection()
        {
            Assert.Equal(0, SightRules.BearingOf(800), 3);
            // Not 14.67 px per degree all the way out: the edge of the picture is 43.6 degrees off, not 54.5.
            Assert.Equal(43.6, SightRules.BearingOf(1600), 1);
            Assert.Equal(-43.6, SightRules.BearingOf(0), 1);
        }

        [Fact]
        public void HeightGivesDistance() => Assert.Equal(21.1, SightRules.MetersOf(60), 1);

        [Theory]
        [InlineData("full", 0.25f, 400f, true)]
        [InlineData("full", 0.24f, 400f, false)]
        [InlineData("tile0", 0.35f, 400f, true)]
        [InlineData("tile0", 0.34f, 400f, false)]
        [InlineData("tile1", 0.9f, 120f, false)]
        [InlineData("tile1", 0.9f, 121.5f, false)]
        [InlineData("tile1", 0.9f, 122.5f, true)]
        [InlineData("full", 0.9f, 120f, true)]
        public void TheBarsAndTheTopEdge(string kind, float conf, float y, bool kept) =>
            Assert.Equal(kept, SightRules.Keep(new SightBox(kind, 100, y, 30, 60, conf)));

        [Fact]
        public void ScenerySaysNothing()
        {
            var tracker = new SightTracker();
            for (int i = 0; i < 40; i++)
                Assert.Empty(tracker.Update(i * 0.5, new[] { Box(500 + (i % 2) * 2, conf: 0.9f) }));
        }

        [Fact]
        public void AWalkingFigureIsSightedOnceItHasMoved()
        {
            var tracker = new SightTracker();
            Assert.Empty(tracker.Update(0, new[] { Box(500) }));
            Assert.Empty(tracker.Update(0.5, new[] { Box(505) }));
            var seen = tracker.Update(1, new[] { Box(511) });
            Assert.Single(seen);
            Assert.Equal(21.1, seen[0].Meters, 1);
        }

        [Fact]
        public void ItNeedsOneStrongBox()
        {
            var tracker = new SightTracker();
            for (int i = 0; i < 6; i++)
                Assert.Empty(tracker.Update(i * 0.5, new[] { Box(500 + i * 5, conf: 0.59f) }));
            Assert.Single(tracker.Update(3, new[] { Box(530, conf: 0.6f) }));
            // Having been strong, a weaker box on the same track still counts.
            Assert.Single(tracker.Update(3.5, new[] { Box(535, conf: 0.4f) }));
        }

        [Fact]
        public void ATrackEndsAfterFiveSeconds()
        {
            var tracker = new SightTracker();
            tracker.Update(0, new[] { Box(500, conf: 0.9f) });
            // Moved far enough, but too late: a new track, starting here, has not moved.
            Assert.Empty(tracker.Update(5.5, new[] { Box(520, conf: 0.9f) }));
        }

        [Fact]
        public void TwoFiguresAreTwoTracks()
        {
            var tracker = new SightTracker();
            tracker.Update(0, new[] { Box(200), Box(1000) });
            Assert.Equal(2, tracker.Update(1, new[] { Box(215), Box(1015) }).Count);
        }

        static byte[] Picture(byte r, byte g, byte b)
        {
            var a = new byte[SightRules.InventoryWidth * SightRules.InventoryHeight * 4];
            for (int i = 0; i < a.Length; i += 4) { a[i] = r; a[i + 1] = g; a[i + 2] = b; a[i + 3] = 255; }
            return a;
        }

        static void Paint(byte[] a, int x0, int x1, int y0, int y1, byte r, byte g, byte b)
        {
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++)
                {
                    int i = (y * SightRules.InventoryWidth + x) * 4;
                    a[i] = r; a[i + 1] = g; a[i + 2] = b;
                }
        }

        static byte[] Inventory()
        {
            var a = Picture(40, 60, 30);
            foreach (var (x0, x1, y0, y1) in new[] { (26, 128, 17, 24), (272, 374, 17, 24), (149, 248, 152, 159) })
                Paint(a, x0, x1, y0, y1, 118, 118, 120);
            return a;
        }

        [Fact]
        public void TheInventoryIsThreeGreyHeaders() => Assert.True(SightRules.IsInventory(Inventory()));

        [Fact]
        public void TheGameViewIsNot()
        {
            Assert.False(SightRules.IsInventory(Picture(40, 60, 30)));
            // A flat grey sky fills the top strips but not the HANDS header's.
            var sky = Picture(40, 60, 30);
            Paint(sky, 0, 400, 0, 100, 120, 120, 122);
            Assert.False(SightRules.IsInventory(sky));
        }

        [Fact]
        public void ATintedOrRoughStripIsNotAHeader()
        {
            var tinted = Inventory();
            Paint(tinted, 26, 56, 17, 24, 110, 140, 112);
            Assert.False(SightRules.IsInventory(tinted));
            var rough = Inventory();
            for (int x = 272; x < 302; x += 2)
                Paint(rough, x, x + 1, 17, 24, 140, 140, 140);
            for (int x = 273; x < 302; x += 2)
                Paint(rough, x, x + 1, 17, 24, 100, 100, 100);
            Assert.False(SightRules.IsInventory(rough));
        }
    }
}
