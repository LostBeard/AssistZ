namespace AssistZ.Services
{
    /// <summary>
    /// What AssistZ knows about how a particular game renders its audio and view, measured in the lab
    /// (docs/dayz-audio-findings.md). Generic = no calibration.
    /// </summary>
    public sealed record GameProfile(
        string Name,
        /// <summary>Measured / true azimuth: the stereo azimuth reads this fraction of the true (folded) angle.</summary>
        double PanScale,
        /// <summary>View-turn calibration: image shift in pixels per degree of turn at a 960 px wide frame (scales with width).</summary>
        double PixelsPerDegreeAt960)
    {
        /// <summary>
        /// DayZ 1.30 (lab, 2026-11-02): level-only panning reading 0.90-0.93x the true angle (beacon runs), and 8.8 px per
        /// degree of view turn at 960 px with the default field of view (mean of slow/medium/fast/sweep turns, coarse-to-fine
        /// tracker, lab run yaw_20261103_041604: spread 9.8%, fast turns +-25%).
        /// </summary>
        public static readonly GameProfile DayZ = new("DayZ", 0.91, 8.8);

        /// <summary>No game-specific calibration (a stream of an unknown game, the test signal).</summary>
        // Turn scale for an unknown game: the DayZ value (fields of view around 80-90 degrees are common).
        public static readonly GameProfile Generic = new("Generic", 1.0, 8.8);

        public static IReadOnlyList<GameProfile> All { get; } = new[] { DayZ, Generic };

        /// <summary>
        /// Gunfire reads compressed at the sides: AKM at 50 m all the way round, 90 deg reads about 60 (lab mode 25), where grenades read 0.91x. Measured azimuth to true, mirrored, DayZ only. On a run not used to fit (sniper_20261105_003257): median error 9.0 -> 5.3 deg, p90 32.9 -> 25.2, worst bearing 28.7 -> 15.5, left or right 99.0% both. One weapon at one distance; other weapons and ranges are not measured.
        /// </summary>
        static readonly double[] GunfireMeasured = { 0, 5, 15, 25, 35, 45, 55, 65, 75, 90 };
        static readonly double[] GunfireTrue = { 0, 3, 20.1, 29.8, 58.6, 61.2, 68.8, 77.2, 83.2, 90 };

        /// <summary>The true azimuth of a sound known to be gunfire; other games fall back to <see cref="Calibrate"/>.</summary>
        public double CalibrateGunfire(double rawAzimuthDegrees)
        {
            if (Name != "DayZ") return Calibrate(rawAzimuthDegrees);
            double a = Math.Min(Math.Abs(rawAzimuthDegrees), 90);
            int i = 1;
            while (i < GunfireMeasured.Length - 1 && a > GunfireMeasured[i]) i++;
            double t = (a - GunfireMeasured[i - 1]) / (GunfireMeasured[i] - GunfireMeasured[i - 1]);
            return Math.Sign(rawAzimuthDegrees) * (GunfireTrue[i - 1] + t * (GunfireTrue[i] - GunfireTrue[i - 1]));
        }

        /// <summary>The true azimuth for a measured stereo azimuth, clamped to the front half (-90..90).</summary>
        public double Calibrate(double rawAzimuthDegrees) => Math.Clamp(rawAzimuthDegrees / PanScale, -90, 90);

        /// <summary>Degrees of view turn for an image shift measured on a frame <paramref name="frameWidth"/> pixels wide.</summary>
        public double ShiftToDegrees(double shiftPixels, int frameWidth) => shiftPixels / (PixelsPerDegreeAt960 * frameWidth / 960.0);
    }
}
