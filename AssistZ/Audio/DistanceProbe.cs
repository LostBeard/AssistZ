using System.Globalization;
using System.Text.Json;

namespace AssistZ.Audio
{
    /// <summary>
    /// How far away a gunshot was, from CLAP's audio embedding, in two steps fit on the game's shot logs (Lab/fit_distance_probe.py): is it past 120 m, then a ridge on the log of the distance for that side. Level alone says almost nothing here (a shot at 800 m is only about 8 dB below one at 150 m).
    /// One ridge over the whole range was worse on both sides: near gunfire in rain fell from 77% to 63% within 25%.
    /// Measured on the AKM, AUG, M4A1 and SVD. Near, 25-95 m, on runs it never trained on: median error about 5-6 m. Far, 150-800 m, is coarse: within a factor of 2 about 9 times in 10, so the radar shows it rounded. Far training is single shots. Not measured on anything but gunfire: use it only for gunfire.
    /// </summary>
    public sealed class DistanceProbe
    {
        sealed class Part
        {
            public required double[] Mean, Std, Weights;
            public double Intercept;

            public double Value(float[] embedding)
            {
                if (embedding.Length != Weights.Length) throw new ArgumentException($"embedding has {embedding.Length} values, the probe {Weights.Length}");
                double y = Intercept;
                for (int i = 0; i < embedding.Length; i++)
                    y += Weights[i] * ((embedding[i] - Mean[i]) / Std[i]);
                return y;
            }

            public static Part Read(JsonElement e)
            {
                static double[] Numbers(JsonElement a) => a.EnumerateArray().Select(x => x.GetDouble()).ToArray();
                var part = new Part { Mean = Numbers(e.GetProperty("Mean")), Std = Numbers(e.GetProperty("Std")), Weights = Numbers(e.GetProperty("Weights")), Intercept = e.GetProperty("Intercept").GetDouble() };
                if (part.Mean.Length != part.Std.Length || part.Mean.Length != part.Weights.Length) throw new FormatException("distance probe: Mean, Std and Weights differ in length");
                return part;
            }
        }

        readonly Part _near, _farGate, _far;
        readonly double _cutMeters;

        public string Checkpoint { get; }

        DistanceProbe(string checkpoint, Part near, Part farGate, Part far, double cutMeters)
        {
            Checkpoint = checkpoint;
            _near = near;
            _farGate = farGate;
            _far = far;
            _cutMeters = cutMeters;
        }

        public static DistanceProbe FromJson(string json)
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            return new DistanceProbe(root.GetProperty("Checkpoint").GetString() ?? "", Part.Read(root.GetProperty("Near")), Part.Read(root.GetProperty("FarGate")), Part.Read(root.GetProperty("Far")), root.GetProperty("CutMeters").GetDouble());
        }

        /// <summary>Estimated distance in metres.</summary>
        public double Estimate(float[] embedding) =>
            _farGate.Value(embedding) > 0 ? Math.Max(_cutMeters, Math.Pow(2, _far.Value(embedding))) : Math.Pow(2, _near.Value(embedding));

        /// <summary>Metres at the radar's inner ring. Inside it the scale is linear, since near estimates are good to a few metres.</summary>
        public const double InnerRingMeters = 100;
        /// <summary>Metres at the radar's outer ring: the farthest range measured.</summary>
        public const double OuterRingMeters = 800;
        /// <summary>Where the inner ring sits, as a share of the outer ring's radius.</summary>
        public const double InnerRingFraction = 0.6;

        /// <summary>Share of the radar's outer radius for a distance: linear to the inner ring, logarithmic past it, held at the outer ring beyond the measured range.</summary>
        public static double RadarFraction(double metres)
        {
            if (metres <= InnerRingMeters) return InnerRingFraction * Math.Max(metres, 0) / InnerRingMeters;
            double past = Math.Log(metres / InnerRingMeters) / Math.Log(OuterRingMeters / InnerRingMeters);
            return InnerRingFraction + (1 - InnerRingFraction) * Math.Min(past, 1);
        }

        /// <summary>A distance as the radar reads it: metres when near; rounded to 50 m past that, with "~", since a far estimate is only good to about a factor of 2; "800+ m" beyond the measured range.</summary>
        public static string Describe(double metres)
        {
            if (metres >= OuterRingMeters) return "800+ m";
            if (metres < InnerRingMeters) return metres.ToString("0", CultureInfo.InvariantCulture) + " m";
            return "~" + (Math.Round(metres / 50) * 50).ToString("0", CultureInfo.InvariantCulture) + " m";
        }
    }
}
