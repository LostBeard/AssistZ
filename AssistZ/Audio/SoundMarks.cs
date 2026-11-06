using System.Globalization;

namespace AssistZ.Audio
{
    /// <summary>How much a sound matters on the radar. Colour says this; the icon says what it is.</summary>
    public enum ThreatLevel { Info, Low, Medium, High }

    /// <summary>How a named sound shows on the radar.</summary>
    /// <param name="Icon">Key into <see cref="SoundMarks.Icons"/>.</param>
    /// <param name="Measured">The name has been tested against the game's own logs. A word that only comes from CLAP's prompts has not: "zombie groaning" comes with an infected within 15 m just 32% of the time.</param>
    public sealed record SoundMark(string Icon, ThreatLevel Level, bool Measured, string Title);

    /// <summary>
    /// TJ, 2026-11-05: red and not red is too vague; small identifiable icons, and threat in levels, since a calm zombie is a small threat and an agroed one a bigger, different one (docs/PLAN.md). First cut of the levels, for TJ to change.
    /// </summary>
    public static class SoundMarks
    {
        /// <summary>What the list calls a sound: the mark's title where there is one, since a probe's word can be older than what was measured.</summary>
        public static string Name(string label)
        {
            if (OwnNames.TryGetValue(label, out var own)) return own;
            var mark = For(label);
            if (mark == null) return label;
            // A word that was never tested says so.
            return mark.Measured || mark.Title.StartsWith("Maybe") ? mark.Title : mark.Title + "?";
        }

        /// <summary>Only names tested against the game's logs get a place on the radar. TJ's first real play (2026-11-06): breaking a stick, swimming and walking showed as footsteps far out in front, from CLAP's closest word, a sound at centre, and no distance. The list still shows the guess, marked as one.</summary>
        public static bool PingsOnRadar(SoundMark mark) => mark.Measured;

        static readonly Dictionary<string, SoundMark> ByLabel = new()
        {
            // Trained and tested against the game's logs.
            ["gunshot"] = new("shot", ThreatLevel.High, true, "Gunshot"),
            ["automatic fire"] = new("burst", ThreatLevel.High, true, "Automatic fire"),
            // The probe's word is older than the measurement. By the game's mind states it fires on any agitated infected: 297 of 303 disturbed, alerted, chase or fight, against 6 of 67 calm, and at matching distances. It cannot tell chase from fight (25 of 42).
            ["infected attack"] = new("infected-agro", ThreatLevel.High, true, "Infected, agroed"),
            ["gas strike"] = new("gas", ThreatLevel.Medium, true, "Gas strike"),
            ["heli crash"] = new("heli-crash", ThreatLevel.Low, true, "Helicopter crash"),
            ["thunder"] = new("thunder", ThreatLevel.Info, true, "Thunder"),
            // CLAP's prompt words: never tested.
            // Not "calm": a calm infected at 15-40 m makes no more sound than none at all (3.2 events a minute against 4.4), so sound cannot find it. This word only says something sounded like one.
            ["zombie groaning"] = new("infected", ThreatLevel.Low, false, "Maybe infected"),
            // With nobody near and the character still, 16 of 26 events get this word: mostly a false alarm until it is trained.
            ["footsteps"] = new("steps", ThreatLevel.Low, false, "Footsteps"),
            ["speech"] = new("speech", ThreatLevel.Medium, false, "Voice"),
            ["car engine"] = new("car", ThreatLevel.Medium, false, "Vehicle"),
            ["helicopter"] = new("heli", ThreatLevel.Medium, false, "Helicopter"),
            ["explosion"] = new("explosion", ThreatLevel.High, false, "Explosion"),
            ["door closing"] = new("door", ThreatLevel.Medium, false, "Door"),
            ["metal clanking"] = new("metal", ThreatLevel.Low, false, "Metal"),
            ["wolf howling"] = new("wolf", ThreatLevel.Medium, false, "Wolf"),
            ["dog barking"] = new("dog", ThreatLevel.Low, false, "Dog"),
            ["birds chirping"] = new("bird", ThreatLevel.Info, false, "Birds"),
            ["music"] = new("music", ThreatLevel.Info, false, "Music"),
        };

        /// <summary>Weather, with no place on a map, and the player's own sounds: named in the list, not pinged.</summary>
        static readonly HashSet<string> Ambient = new() { "rain", "wind", "own move", "own shot" };

        /// <summary>What the list calls the player's own sounds.</summary>
        static readonly Dictionary<string, string> OwnNames = new() { ["own move"] = "You, moving", ["own shot"] = "Your shot" };

        /// <summary>Own sounds sit near centre: 99% of own movement and 93% of own shots within this. "Sounds like you" alone took 37% of another character's footsteps and 23% of far gunfire; with this, 2% and 1%.</summary>
        public const double OwnMaxAzimuth = 10;

        /// <summary>Losing a real shot is worse than pinging a footstep. Real gunfire in a rain storm went from 98 to 92 of 103 without this. All 15 real sounds called own over 2645 scored under 0.5; true own sounds have a median of 0.83.</summary>
        public const float OwnMinProbability = 0.55f;

        public static bool IsOwnClass(string probeClass) => probeClass is "own_move" or "own_shot";

        /// <summary>The probe's class to believe: the most likely, unless that says the player's own sound from away from centre or without confidence, in which case the most likely that does not.</summary>
        public static int PickClass(IReadOnlyList<string> classes, IReadOnlyList<float> scores, double azimuthDegrees)
        {
            int top = -1;
            bool ownAllowed = Math.Abs(azimuthDegrees) < OwnMaxAzimuth;
            for (int c = 0; c < classes.Count; c++)
            {
                if (IsOwnClass(classes[c]) && (!ownAllowed || scores[c] < OwnMinProbability)) continue;
                if (top < 0 || scores[c] > scores[top]) top = c;
            }
            return top;
        }

        /// <summary>The mark for a sound's name; null for a sound that does not ping, or a name with no mark yet.</summary>
        public static SoundMark? For(string? label) => label != null && ByLabel.TryGetValue(label, out var mark) ? mark : null;

        /// <summary>Gunfire past this ranks Medium, not High (TJ, 2026-11-05): a shot at 800 m is less of an immediate threat than one at 50 m. Except a sniper rifle: "Sniper shots from 300m are more dangerous than m4 or pistol at that range". Far estimates are good to about a factor of 2, so this is a broad line.</summary>
        public const double FarShotMeters = 300;

        /// <summary>The level of a sound as heard: its mark's, except that far gunfire ranks lower, unless from a sniper rifle.</summary>
        public static ThreatLevel LevelOf(SoundMark mark, double? meters, string? gunFamily = null) =>
            mark.Icon is "shot" or "burst" && meters >= FarShotMeters && gunFamily != "sniper" ? ThreatLevel.Medium : mark.Level;

        /// <summary>The icon for a sound: a single shot shows what fired it when that is known; a burst stays a burst.</summary>
        public static string IconOf(SoundMark mark, string? gunFamily) => mark.Icon == "shot" && gunFamily is "sniper" or "assault" or "smg" or "pistol" ? "gun-" + gunFamily : mark.Icon;

        /// <summary>What the list calls a sound with its gun: the name, then the family, then the sound's own word. A burst says so.</summary>
        public static string NameWithGun(string label, string? gunName, string? gunFamily)
        {
            string? gun = gunName ?? gunFamily switch { "sniper" => "Sniper rifle", "assault" => "Assault rifle", "smg" => "SMG", "pistol" => "Pistol", _ => null };
            if (gun == null) return Name(label);
            return label == "automatic fire" ? gun + ", auto" : gun;
        }

        public static bool IsAmbient(string? label) => label != null && Ambient.Contains(label);

        /// <summary>Every sound with a mark, for the legend.</summary>
        public static IEnumerable<SoundMark> All => ByLabel.Values;

        /// <summary>
        /// Which marks to draw, given marks newest first: a burst is many events on one spot, and a pile of the same icon says less than one. Returns the indexes kept: the newest of each kind within <paramref name="degrees"/> and <paramref name="radius"/> of a kept one.
        /// </summary>
        public static List<int> Keep(IReadOnlyList<(string Icon, double Azimuth, double Radius)> marks, double degrees = 15, double radius = 0.1)
        {
            var kept = new List<int>();
            for (int i = 0; i < marks.Count; i++)
            {
                bool covered = false;
                foreach (int k in kept)
                {
                    double d = Math.Abs(((marks[i].Azimuth - marks[k].Azimuth + 180) % 360 + 360) % 360 - 180);
                    if (marks[k].Icon == marks[i].Icon && d <= degrees && Math.Abs(marks[i].Radius - marks[k].Radius) <= radius)
                    {
                        covered = true;
                        break;
                    }
                }
                if (!covered) kept.Add(i);
            }
            return kept;
        }

        static string E(double cx, double cy, double rx, double ry)
        {
            string F(double v) => v.ToString("0.##", CultureInfo.InvariantCulture);
            return $"M{F(cx + rx)} {F(cy)}A{F(rx)} {F(ry)} 0 1 1 {F(cx - rx)} {F(cy)}A{F(rx)} {F(ry)} 0 1 1 {F(cx + rx)} {F(cy)}Z";
        }

        const string Head = "M-5-1A5 5.5 0 0 1 5-1V2.5L3 3.5V6.5H-3V3.5L-5 2.5Z";
        static readonly string Eyes = E(-2, 0, 1.2, 1.2) + E(2, 0, 1.2, 1.2);

        /// <summary>Icons drawn here, in a 16 unit box centred on 0, no game art: a line path and a filled path. Holes in a filled path use even-odd.</summary>
        public static readonly IReadOnlyDictionary<string, (string Line, string Fill)> Icons = new Dictionary<string, (string, string)>
        {
            ["shot"] = ("M0-7V-3M0 3V7M-7 0H-3M3 0H7" + E(0, 0, 5, 5), E(0, 0, 1.2, 1.2)),
            ["burst"] = ("", "M-7-6.8h9.5l3.5 1.3-3.5 1.3h-9.5zM-7-1.3h9.5l3.5 1.3-3.5 1.3h-9.5zM-7 4.2h9.5l3.5 1.3-3.5 1.3h-9.5z"),
            ["explosion"] = ("", "M0-7.5L2-2.8L6.8-4.5L4.2 0L7.5 2.5L2.8 2.8L3 7.5L0 3.8L-3 7.5L-2.8 2.8L-7.5 2.5L-4.2 0L-6.8-4.5L-2-2.8Z"),
            // Guns, by family, side on, muzzle to the right.
            ["gun-sniper"] = ("M-3-3.2H3", "M-8-.7H8V.7H-2L-4.2 4.5H-6.2L-5 .7H-8ZM-2.5-3.8H3.5V-1.8H-2.5Z"),
            ["gun-assault"] = ("", "M-8-2H6.5V0H1.5L2 3.5H.2L-.8 0H-2.5L-4.5 4H-6.5L-5.2 0H-8Z"),
            ["gun-smg"] = ("", "M-6.5-2.5H5.5V-.3H1V5.5H-1V-.3H-2.5L-3.8 2.5H-5.8L-4.8-.3H-6.5Z"),
            ["gun-pistol"] = ("", "M-6.5-3.5H6.5V-.3H.5L-1 5H-4.8L-3.5-.3H-6.5Z"),
            // Calm: an outline. Agroed: solid. Attacking: solid, with claw marks, drawn for when chase and fight can be told apart.
            ["infected"] = (Head + "M-1.2 4.5V6.5M1.2 4.5V6.5", Eyes),
            ["infected-agro"] = ("", Head + Eyes),
            ["infected-attack"] = ("M.8-8L4.2-3.6M3.6-8L7-3.6M4.2 0L7.6 4.4", "M-7.2-.5A4.2 4.6 0 0 1 1.2-.5V2.5L-.5 3.3V6H-5.5V3.3L-7.2 2.5Z" + E(-4.7, .3, 1, 1) + E(-1.3, .3, 1, 1)),
            ["steps"] = ("", E(-3, 2.2, 2.2, 3) + E(-3, 6.5, 1.4, 1) + E(3, -3.8, 2.2, 3) + E(3, .5, 1.4, 1)),
            ["speech"] = ("M-6.5-5.5H6.5V2.5H-1L-4 6V2.5H-6.5Z", ""),
            ["car"] = ("", "M-7.5 2.5V-.5L-4.5-1L-2.5-4.5H3L5.5-1L7.5-.5V2.5Z" + E(-4, 3.5, 1.9, 1.9) + E(4, 3.5, 1.9, 1.9)),
            ["heli"] = ("M-7-6H7M0-6V-3M3 0H7.5M7.5-2V2M-4.5 6H3M-3 3V6M1.5 3V6", E(-1, 0, 4.5, 3)),
            ["heli-crash"] = ("M-7.5-1.5L3-7M3.5-6.5L7-7.5", "M1-3.5L2.4 0L5.8-1.2L4.2 2L7 4.2L3.6 4.2L3.6 7.5L1 5.2L-1.6 7.5L-1.6 4.2L-5 4.2L-2.2 2L-3.8-1.2L-.4 0Z"),
            ["gas"] = ("M0-7L7.5 6H-7.5ZM0-2.5V2", E(0, 3.8, .9, .9)),
            ["thunder"] = ("", "M2-7.5L-4.5 1H-.5L-2 7.5L4.5-1.5H.5Z"),
            ["door"] = ("M-4.5-7.5H4.5V7.5H-4.5Z", E(1.8, 0, 1, 1)),
            ["metal"] = ("M0-7L6-3.5V3.5L0 7L-6 3.5V-3.5Z" + E(0, 0, 2.2, 2.2), ""),
            ["wolf"] = ("", "M-5.5-7L-2.5-2.5H2.5L5.5-7L6 1L0 7L-6 1Z"),
            ["dog"] = ("", E(0, 3, 3.8, 3) + E(-5, -.5, 1.5, 2) + E(-1.9, -4, 1.5, 2.2) + E(1.9, -4, 1.5, 2.2) + E(5, -.5, 1.5, 2)),
            ["bird"] = ("M-7.5 0Q-3.5-5 0 0Q3.5-5 7.5 0", ""),
            ["music"] = ("M-2.5 4.5V-5L5-6.5V3", E(-4.3, 4.5, 2, 1.6) + E(3.2, 3, 2, 1.6)),
        };
    }
}
