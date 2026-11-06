namespace AssistZ.Audio
{
    /// <summary>One analysis window's stereo measurements (read back from the GPU: <see cref="StereoAnalyzer.ResultStride"/> floats).</summary>
    public readonly record struct StereoFrame(
        long EndSample,
        float RmsLeft,
        float RmsRight,
        int BestLagSamples,
        float PeakCorrelation,
        float ZeroLagCorrelation,
        float BandLevelDifferenceDb,
        float SpectralCentroid,
        float HighBandFraction)
    {
        /// <summary>
        /// Inter-channel level difference in dB, positive = louder on the right, clamped to +-60 dB: a hard pan leaves one
        /// channel at ~1e-17 (the panner's cos(pi/2) in float), which would otherwise read as a 300+ dB difference.
        /// </summary>
        public float LevelDifferenceDb
        {
            get
            {
                const float floor = 1e-9f;
                if (RmsLeft < floor && RmsRight < floor) return 0f;
                return Math.Clamp(20f * MathF.Log10(MathF.Max(RmsRight, floor) / MathF.Max(RmsLeft, floor)), -60f, 60f);
            }
        }

        /// <summary>
        /// Azimuth in degrees, -90 = hard left, +90 = hard right, inverting a constant-power pan law (L = cos t, R = sin t)
        /// on the broadband levels. Front and back cannot be told apart from stereo levels (measured: DayZ's mix carries no
        /// front/back cue); a game profile maps it to the game's true angle.
        /// </summary>
        public float PanAzimuthDegrees => RmsLeft + RmsRight > 0
            ? (MathF.Atan2(RmsRight, RmsLeft) / (MathF.PI / 2f)) * 180f - 90f
            : 0f;

        /// <summary>
        /// The same inversion applied to the energy-weighted per-band level difference (48 mel bands from
        /// SpawnDev.ILGPU.ML's SpectralFrontend): bands where the window has its energy decide, so a loud low rumble on one
        /// side does not drag a sharp sound on the other side toward the middle as the broadband levels do.
        /// </summary>
        public float SpectralAzimuthDegrees
        {
            get
            {
                // BandLevelDifferenceDb is left over right; the pan inverse wants R / L as an amplitude ratio.
                float ratio = MathF.Pow(10f, -BandLevelDifferenceDb / 20f);
                return (MathF.Atan(ratio) / (MathF.PI / 2f)) * 180f - 90f;
            }
        }

        public float Rms => MathF.Sqrt(0.5f * (RmsLeft * RmsLeft + RmsRight * RmsRight));
    }
}
