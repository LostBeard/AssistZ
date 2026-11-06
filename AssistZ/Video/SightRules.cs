namespace AssistZ.Video
{
    /// <summary>A person box from the detector, in the pixels of a 1600 wide picture. Kind is "full" or a tile.</summary>
    public readonly record struct SightBox(string Kind, float X, float Y, float W, float H, float Confidence);

    /// <summary>A person the rules believe in: where, how far, how sure.</summary>
    public readonly record struct Sighting(float X, float Y, float W, float H, float Confidence, double BearingDegrees, double Meters);

    /// <summary>
    /// Setup 4 in docs/PLAN.md, as measured on YOLOv8n in the lab. A new detector needs its own bars and <see cref="MetersConstant"/>: its confidences and box heights differ.
    /// </summary>
    public static class SightRules
    {
        public const float FullBar = 0.25f, TileBar = 0.35f;
        /// <summary>The tiles' top in a 1600 wide picture: a box on that edge was cut there, and is scenery far more often than a person.</summary>
        public const float TileTop = 120f;
        /// <summary>Distance is this over the box's height. Fixed from the first look: 75 to 94% within 25% on fresh runs.</summary>
        public const double MetersConstant = 1268;
        /// <summary>Focal length in px at 1600 wide: 8.8 px per degree at the centre at 960 wide. The picture is a flat projection, 87 degrees across.</summary>
        public const double FocalPixels = 840.34;

        public static bool Keep(SightBox b) =>
            b.Kind == "full" ? b.Confidence >= FullBar : b.Confidence >= TileBar && Math.Abs(b.Y - TileTop) >= 2;

        public static double BearingOf(float centreX) => Math.Atan((centreX - 800) / FocalPixels) * 180 / Math.PI;

        public static double MetersOf(float height) => MetersConstant / Math.Max(height, 1);

        // Either side of the VICINITY, EQUIPMENT and HANDS headers' text, in a 400 x 225 copy of the picture. Numbers, not game art. Tested at the default interface scale, 16:9.
        static readonly (int X0, int X1, int Y0, int Y1)[] Strips = { (26, 56, 17, 24), (98, 128, 17, 24), (272, 302, 17, 24), (344, 374, 17, 24), (149, 179, 152, 159), (218, 248, 152, 159) };

        public const int InventoryWidth = 400, InventoryHeight = 225;

        /// <summary>Is this the inventory screen? YOLOv8n calls its character preview a person at up to 0.86, and the movement rule lets it by. All 661 frames held open on fresh runs, none of about 37,000 others.</summary>
        public static bool IsInventory(byte[] rgba)
        {
            foreach (var (x0, x1, y0, y1) in Strips)
            {
                double r = 0, g = 0, b = 0, sum = 0, squares = 0;
                int n = 0;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                    {
                        int i = (y * InventoryWidth + x) * 4;
                        r += rgba[i]; g += rgba[i + 1]; b += rgba[i + 2];
                        sum += rgba[i] + rgba[i + 1] + rgba[i + 2];
                        squares += rgba[i] * rgba[i] + rgba[i + 1] * rgba[i + 1] + rgba[i + 2] * rgba[i + 2];
                        n++;
                    }
                r /= n; g /= n; b /= n;
                double low = Math.Min(r, Math.Min(g, b)), high = Math.Max(r, Math.Max(g, b));
                double mean = sum / (3.0 * n);
                double rough = Math.Sqrt(Math.Max(0, squares / (3.0 * n) - mean * mean));
                if (low < 110 || high > 142 || high - low > 25 || rough > 9)
                    return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Links boxes into tracks: up to 5 s apart and within 60 px. A track counts once it has moved 10 px from where it began and had one box at 0.6. Scenery stays put, and a calm infected rarely does (under 1 m in 2% of 10 s windows).
    /// </summary>
    public sealed class SightTracker
    {
        public const double GapSeconds = 5;
        public const float LinkPixels = 60, MovePixels = 10, StrongConfidence = 0.6f;

        sealed class Track { public float StartX, StartY, X, Y, Best; public double Last; public bool Moved; }

        readonly List<Track> _tracks = new();

        /// <summary>Takes one picture's boxes; returns those that count as sightings now.</summary>
        public List<Sighting> Update(double seconds, IEnumerable<SightBox> boxes)
        {
            var seen = new List<Sighting>();
            _tracks.RemoveAll(t => seconds - t.Last > GapSeconds);
            foreach (var b in boxes)
            {
                if (!SightRules.Keep(b)) continue;
                float cx = b.X + b.W / 2, foot = b.Y + b.H;
                var t = _tracks.Find(k => Math.Abs(k.X - cx) <= LinkPixels && Math.Abs(k.Y - foot) <= LinkPixels);
                if (t == null)
                {
                    t = new Track { StartX = cx, StartY = foot };
                    _tracks.Add(t);
                }
                t.X = cx;
                t.Y = foot;
                t.Last = seconds;
                t.Best = Math.Max(t.Best, b.Confidence);
                float dx = cx - t.StartX, dy = foot - t.StartY;
                if (dx * dx + dy * dy >= MovePixels * MovePixels) t.Moved = true;
                if (t.Moved && t.Best >= StrongConfidence)
                    seen.Add(new Sighting(b.X, b.Y, b.W, b.H, b.Confidence, SightRules.BearingOf(cx), SightRules.MetersOf(b.H)));
            }
            return seen;
        }
    }
}
