using System.Text.Json;

namespace AssistZ.Audio
{
    /// <summary>What the radar may say about a gun: its name, its family, or nothing beyond "gunshot".</summary>
    public readonly record struct GunCall(string? Name, string? Family, float Confidence);

    /// <summary>
    /// Which gun, from CLAP's audio embedding (TJ, 2026-11-05: the radar should say the gun type, "the whole point is info"). A multinomial fit on the game's shot logs (Lab/fit_gun_probe.py), ten guns. Each run held out, 83.8% right, and 91% right at the confidence bar here.
    /// A name is said only for a gun that was right at least 85% of the time when named, on runs never trained on; the rest say their family, and an unsure call says neither. Trained on single shots at range and bursts up close, so a gun only known from one of those has a thinner record. Not trained on any other gun: an unknown gun gets the nearest it knows.
    /// </summary>
    public sealed class GunProbe
    {
        /// <summary>How sure before anything is said.</summary>
        public const float Bar = 0.8f;

        /// <summary>Radar name and family. Null name: not said (when confident it was right under 85% of the time: Mosin 81%, MP5K 65%, FAL 33%).</summary>
        static readonly Dictionary<string, (string? Name, string Family)> Known = new()
        {
            ["AKM"] = ("AKM", "assault"),
            ["AugShort"] = ("AUG", "assault"),
            ["M4A1"] = ("M4A1", "assault"),
            ["SVD"] = ("SVD", "sniper"),
            ["Winchester70"] = ("Winchester", "sniper"),
            ["CZ75"] = ("CZ75", "pistol"),
            ["FNX45"] = ("FNX45", "pistol"),
            ["Mosin9130"] = (null, "sniper"),
            ["FAL"] = (null, "assault"),
            ["MP5K"] = (null, "smg"),
        };

        readonly string[] _classes;
        readonly double[] _mean, _std, _bias;
        readonly double[][] _weights;

        public string Checkpoint { get; }

        GunProbe(string checkpoint, string[] classes, double[] mean, double[] std, double[][] weights, double[] bias)
        {
            Checkpoint = checkpoint;
            _classes = classes;
            _mean = mean;
            _std = std;
            _weights = weights;
            _bias = bias;
        }

        public static GunProbe FromJson(string json)
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            static double[] Numbers(JsonElement a) => a.EnumerateArray().Select(x => x.GetDouble()).ToArray();
            var classes = root.GetProperty("Classes").EnumerateArray().Select(x => x.GetString() ?? "").ToArray();
            var weights = root.GetProperty("Weights").EnumerateArray().Select(Numbers).ToArray();
            var bias = Numbers(root.GetProperty("Bias"));
            if (weights.Length != classes.Length || bias.Length != classes.Length) throw new FormatException("gun probe: classes, weights and bias differ in length");
            return new GunProbe(root.GetProperty("Checkpoint").GetString() ?? "", classes, Numbers(root.GetProperty("Mean")), Numbers(root.GetProperty("Std")), weights, bias);
        }

        /// <summary>The probabilities over the probe's guns.</summary>
        public double[] Predict(float[] embedding)
        {
            if (embedding.Length != _mean.Length) throw new ArgumentException($"embedding has {embedding.Length} values, the probe {_mean.Length}");
            var scores = new double[_classes.Length];
            double best = double.NegativeInfinity;
            for (int c = 0; c < _classes.Length; c++)
            {
                double s = _bias[c];
                var w = _weights[c];
                for (int i = 0; i < embedding.Length; i++)
                    s += w[i] * ((embedding[i] - _mean[i]) / _std[i]);
                scores[c] = s;
                if (s > best) best = s;
            }
            double sum = 0;
            for (int c = 0; c < scores.Length; c++)
            {
                scores[c] = Math.Exp(scores[c] - best);
                sum += scores[c];
            }
            for (int c = 0; c < scores.Length; c++) scores[c] /= sum;
            return scores;
        }

        public GunCall Call(float[] embedding) => Decide(_classes, Predict(embedding));

        /// <summary>The name if the top gun is sure and may be named; else the family if its guns together are sure; else nothing.</summary>
        public static GunCall Decide(IReadOnlyList<string> classes, IReadOnlyList<double> probabilities)
        {
            int top = 0;
            for (int c = 1; c < probabilities.Count; c++)
                if (probabilities[c] > probabilities[top]) top = c;
            if (probabilities[top] >= Bar && Known.TryGetValue(classes[top], out var gun) && gun.Name != null)
                return new GunCall(gun.Name, gun.Family, (float)probabilities[top]);
            var families = new Dictionary<string, double>();
            for (int c = 0; c < probabilities.Count; c++)
                if (Known.TryGetValue(classes[c], out var known))
                    families[known.Family] = families.GetValueOrDefault(known.Family) + probabilities[c];
            foreach (var (family, p) in families.OrderByDescending(f => f.Value))
                return p >= Bar ? new GunCall(null, family, (float)p) : new GunCall(null, null, (float)p);
            return new GunCall(null, null, 0);
        }
    }
}
