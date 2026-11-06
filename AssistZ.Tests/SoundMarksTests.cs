using AssistZ.Audio;
using Xunit;

namespace AssistZ.Tests
{
    public class SoundMarksTests
    {
        // SoundLabeler.Labels plus the probe's world event words and the cadence's word. Not linked here, since SoundLabeler needs the GPU.
        static readonly string[] Named =
        {
            "gunshot", "automatic fire", "explosion", "footsteps", "zombie groaning", "speech", "car engine", "helicopter",
            "wolf howling", "dog barking", "birds chirping", "thunder", "door closing", "metal clanking", "music",
            "gas strike", "heli crash", "infected attack",
        };

        [Fact]
        public void EverySoundTheAppCanName_HasAMarkWithADrawing()
        {
            foreach (string name in Named)
            {
                var mark = SoundMarks.For(name);
                Assert.True(mark != null, name);
                Assert.True(SoundMarks.Icons.ContainsKey(mark!.Icon), mark.Icon);
                var (line, fill) = SoundMarks.Icons[mark.Icon];
                Assert.False(line == "" && fill == "", mark.Icon);
            }
        }

        [Fact]
        public void Weather_DoesNotPing()
        {
            Assert.Null(SoundMarks.For("rain"));
            Assert.Null(SoundMarks.For("wind"));
            Assert.True(SoundMarks.IsAmbient("rain"));
            Assert.False(SoundMarks.IsAmbient("gunshot"));
            Assert.Null(SoundMarks.For(null));
        }

        [Fact]
        public void Levels_SayTheInfectedAreDifferentByState()
        {
            Assert.Equal(ThreatLevel.High, SoundMarks.For("infected attack")!.Level);
            Assert.Equal(ThreatLevel.Low, SoundMarks.For("zombie groaning")!.Level);
            Assert.Equal("Maybe infected", SoundMarks.For("zombie groaning")!.Title);       // never "calm": sound cannot tell
            Assert.NotEqual(SoundMarks.For("infected attack")!.Icon, SoundMarks.For("zombie groaning")!.Icon);
        }

        static readonly string[] Probe = { "other", "gunfire", "own_move", "own_shot" };

        [Fact]
        public void OwnSound_IsBelievedOnlyNearCentre()
        {
            float[] sounds = { 0.1f, 0.3f, 0.6f, 0f };
            Assert.Equal(2, SoundMarks.PickClass(Probe, sounds, 3));      // near centre: yours
            Assert.Equal(2, SoundMarks.PickClass(Probe, sounds, -9.9));
            Assert.Equal(1, SoundMarks.PickClass(Probe, sounds, 10));     // away from centre: the best that is not yours
            Assert.Equal(1, SoundMarks.PickClass(Probe, sounds, -45));
        }

        [Fact]
        public void OwnSound_WithoutConfidence_IsNotBelieved()
        {
            // Real gunfire in a rain storm read like this: own the most likely, but under the bar.
            float[] unsure = { 0.1f, 0.4f, 0.45f, 0.05f };
            Assert.Equal(1, SoundMarks.PickClass(Probe, unsure, 3));
            float[] sure = { 0.05f, 0.35f, 0.6f, 0f };
            Assert.Equal(2, SoundMarks.PickClass(Probe, sure, 3));
        }

        [Fact]
        public void OtherSounds_AreNotAffectedByTheRule()
        {
            float[] gun = { 0.1f, 0.7f, 0.1f, 0.1f };
            Assert.Equal(1, SoundMarks.PickClass(Probe, gun, 0));
            Assert.Equal(1, SoundMarks.PickClass(Probe, gun, 60));
        }

        [Fact]
        public void OwnSounds_DoNotPing_ButAreNamed()
        {
            Assert.Null(SoundMarks.For("own move"));
            Assert.True(SoundMarks.IsAmbient("own move"));
            Assert.True(SoundMarks.IsAmbient("own shot"));
            Assert.Equal("You, moving", SoundMarks.Name("own move"));
            Assert.Equal("Your shot", SoundMarks.Name("own shot"));
        }

        [Theory]
        [InlineData("gunshot", 50.0, ThreatLevel.High)]
        [InlineData("gunshot", 299.0, ThreatLevel.High)]
        [InlineData("gunshot", 300.0, ThreatLevel.Medium)]      // TJ: far shots rank lower
        [InlineData("automatic fire", 800.0, ThreatLevel.Medium)]
        [InlineData("infected attack", 800.0, ThreatLevel.High)] // only gunfire: no distance is measured for the rest
        public void FarGunfire_RanksLower(string label, double meters, ThreatLevel level) =>
            Assert.Equal(level, SoundMarks.LevelOf(SoundMarks.For(label)!, meters));

        [Fact]
        public void GunfireWithNoDistance_KeepsItsLevel() =>
            Assert.Equal(ThreatLevel.High, SoundMarks.LevelOf(SoundMarks.For("gunshot")!, null));

        [Fact]
        public void Sniper_StaysHigh_PastTheFarLine()
        {
            var shot = SoundMarks.For("gunshot")!;
            Assert.Equal(ThreatLevel.High, SoundMarks.LevelOf(shot, 600, "sniper"));
            Assert.Equal(ThreatLevel.Medium, SoundMarks.LevelOf(shot, 600, "assault"));
            Assert.Equal(ThreatLevel.Medium, SoundMarks.LevelOf(shot, 600, "pistol"));
            Assert.Equal(ThreatLevel.Medium, SoundMarks.LevelOf(shot, 600, null));
            Assert.Equal(ThreatLevel.High, SoundMarks.LevelOf(shot, 100, "assault"));
        }

        [Theory]
        [InlineData("gunshot", "AKM", "assault", "AKM")]
        [InlineData("automatic fire", "AKM", "assault", "AKM, auto")]
        [InlineData("gunshot", null, "sniper", "Sniper rifle")]
        [InlineData("gunshot", null, "smg", "SMG")]
        [InlineData("automatic fire", null, "assault", "Assault rifle, auto")]
        [InlineData("gunshot", null, null, "Gunshot")]
        public void TheListSaysTheGun(string label, string? name, string? family, string said) =>
            Assert.Equal(said, SoundMarks.NameWithGun(label, name, family));

        [Fact]
        public void IconsFollowTheFamily_ForSingleShots()
        {
            Assert.Equal("gun-sniper", SoundMarks.IconOf(SoundMarks.For("gunshot")!, "sniper"));
            Assert.Equal("shot", SoundMarks.IconOf(SoundMarks.For("gunshot")!, null));
            Assert.Equal("burst", SoundMarks.IconOf(SoundMarks.For("automatic fire")!, "assault"));
            foreach (var f in new[] { "sniper", "assault", "smg", "pistol" })
                Assert.True(SoundMarks.Icons.ContainsKey("gun-" + f), f);
        }

        static readonly string[] Guns = { "AKM", "AugShort", "CZ75", "FAL", "FNX45", "M4A1", "MP5K", "Mosin9130", "SVD", "Winchester70" };

        static double[] Sure(string gun, double p)
        {
            var v = new double[Guns.Length];
            double rest = (1 - p) / (Guns.Length - 1);
            for (int i = 0; i < v.Length; i++) v[i] = Guns[i] == gun ? p : rest;
            return v;
        }

        [Fact]
        public void Gun_IsNamedOnlyWhenSureAndAllowed()
        {
            Assert.Equal("AUG", GunProbe.Decide(Guns, Sure("AugShort", 0.9)).Name);
            Assert.Equal("assault", GunProbe.Decide(Guns, Sure("AugShort", 0.9)).Family);
            Assert.Null(GunProbe.Decide(Guns, Sure("AugShort", 0.5)).Name);
            // Right too rarely when named: the family, not the name.
            var fal = GunProbe.Decide(Guns, Sure("FAL", 0.95));
            Assert.Null(fal.Name);
            Assert.Equal("assault", fal.Family);
            Assert.Equal("sniper", GunProbe.Decide(Guns, Sure("Mosin9130", 0.9)).Family);
            Assert.Equal("smg", GunProbe.Decide(Guns, Sure("MP5K", 0.9)).Family);
        }

        [Fact]
        public void Family_CanBeSure_WhenNoOneGunIs()
        {
            var v = new double[Guns.Length];
            v[Array.IndexOf(Guns, "AKM")] = 0.45;
            v[Array.IndexOf(Guns, "M4A1")] = 0.45;
            v[Array.IndexOf(Guns, "SVD")] = 0.10;
            var call = GunProbe.Decide(Guns, v);
            Assert.Null(call.Name);
            Assert.Equal("assault", call.Family);
        }

        [Fact]
        public void UnsureOfBoth_SaysNeither()
        {
            var v = new double[Guns.Length];
            v[Array.IndexOf(Guns, "AKM")] = 0.4;
            v[Array.IndexOf(Guns, "SVD")] = 0.4;
            v[Array.IndexOf(Guns, "CZ75")] = 0.2;
            var call = GunProbe.Decide(Guns, v);
            Assert.Null(call.Name);
            Assert.Null(call.Family);
        }

        [Fact]
        public void GunProbe_ReadsItsJson_AndPredicts()
        {
            var probe = GunProbe.FromJson("{\"Checkpoint\":\"c\",\"Classes\":[\"AKM\",\"SVD\"],\"Mean\":[1,0],\"Std\":[2,1],\"Weights\":[[0,0],[0,3]],\"Bias\":[0,0]}");
            Assert.Equal("c", probe.Checkpoint);
            var p = probe.Predict(new[] { 1f, 1f });           // second gun's score 3: e^3 / (1 + e^3)
            Assert.Equal(Math.Exp(3) / (1 + Math.Exp(3)), p[1], 6);
            Assert.Equal("SVD", probe.Call(new[] { 1f, 1f }).Name);
            Assert.Throws<ArgumentException>(() => { probe.Predict(new[] { 1f }); });
        }

        [Fact]
        public void TheProbesAttackWord_ReadsAsAgroed()
        {
            Assert.Equal("Infected, agroed", SoundMarks.Name("infected attack"));
            Assert.Equal("infected-agro", SoundMarks.For("infected attack")!.Icon);
            Assert.Equal("rain", SoundMarks.Name("rain"));           // no mark: its own word
        }

        [Fact]
        public void OnlyTestedNames_ClaimToBeMeasured()
        {
            Assert.True(SoundMarks.For("gunshot")!.Measured);
            Assert.True(SoundMarks.For("infected attack")!.Measured);
            // Comes with an infected within 15 m only 32% of the time.
            Assert.False(SoundMarks.For("zombie groaning")!.Measured);
            Assert.False(SoundMarks.For("footsteps")!.Measured);
        }

        [Fact]
        public void Keep_MergesASpotButNotDifferentThingsOrPlaces()
        {
            var marks = new List<(string, double, double)>
            {
                ("shot", 30, 0.5),      // newest: kept
                ("shot", 35, 0.52),     // same spot: merged
                ("burst", 30, 0.5),     // another kind: kept
                ("shot", 80, 0.5),      // another bearing: kept
                ("shot", 30, 0.9),      // same bearing, much farther: kept
                ("shot", 25, 0.45),     // same spot again: merged
            };
            Assert.Equal(new List<int> { 0, 2, 3, 4 }, SoundMarks.Keep(marks));
        }

        [Fact]
        public void Keep_MergesAcrossTheBackOfTheRadar()
        {
            var marks = new List<(string, double, double)> { ("shot", 179, 1), ("shot", -179, 1) };
            Assert.Equal(new List<int> { 0 }, SoundMarks.Keep(marks));
        }
    }
}
