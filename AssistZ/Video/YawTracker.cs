using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace AssistZ.Video
{
    /// <summary>One frame-to-frame view-turn measurement.</summary>
    /// <param name="ShiftPixels">Horizontal image shift in preview pixels, + = content moved LEFT (the view turned right).</param>
    /// <param name="Score">Normalised correlation at the best shift (1 = identical profiles); low = unreliable.</param>
    public readonly record struct YawSample(double TimeSeconds, float ShiftPixels, float Score);

    /// <summary>
    /// Estimates how far the view turned between consecutive frames, entirely on the GPU. A first-person view turning
    /// about its vertical axis slides the whole image sideways (rotation has no parallax), so the horizontal offset that
    /// best aligns this frame's column profile with the previous one is the turn.
    /// <para>
    /// Coarse to fine, so fast turns are found without a huge search: a column-profile kernel (mean luminance of each
    /// column over the middle 80% of rows), a 4x decimated profile searched over +-<see cref="CoarseRange"/> coarse bins
    /// (+-240 px), its peak kept ON THE DEVICE, then a full-resolution search of +-<see cref="FineRange"/> px around it
    /// with a parabolic sub-pixel fit. One readback (2 floats) per frame.
    /// </para>
    /// Lab result (DayZ, 960 px preview): ~9.3 px per degree of view turn; a single +-120 px search failed above ~375
    /// deg/s (lab run yaw_20261103_040343), hence the coarse stage.
    /// </summary>
    public sealed class YawTracker : IDisposable
    {
        public const int Decimation = 4;
        public const int CoarseRange = 60;
        public const int FineRange = 6;
        const int CoarseCount = 2 * CoarseRange + 1;
        const int FineCount = 2 * FineRange + 1;
        /// <summary>Largest shift in full-resolution pixels the fine stage can reach (also its edge margin).</summary>
        public const int MaxShift = CoarseRange * Decimation + FineRange;
        /// <summary>Fewest columns the fine stage correlates over (the 960 px preview gives 468).</summary>
        public const int MinOverlap = 256;

        readonly Accelerator _accelerator;
        readonly Action<Index1D, ArrayView2D<int, Stride2D.DenseX>, ArrayView<float>> _profile;
        readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _decimate;
        readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int> _coarseNcc;
        readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _coarsePeak;
        readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int> _fineNcc;
        readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _finePeak;
        MemoryBuffer1D<float, Stride1D.Dense>? _prev, _cur, _prevC, _curC;
        readonly MemoryBuffer1D<float, Stride1D.Dense> _coarseScores, _fineScores, _center, _result;
        int _width;
        bool _havePrev;

        public YawTracker(Accelerator accelerator)
        {
            _accelerator = accelerator;
            _coarseScores = accelerator.Allocate1D<float>(CoarseCount);
            _fineScores = accelerator.Allocate1D<float>(FineCount);
            _center = accelerator.Allocate1D<float>(1);
            _result = accelerator.Allocate1D<float>(2);
            _profile = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<int, Stride2D.DenseX>, ArrayView<float>>(ProfileKernel);
            _decimate = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(DecimateKernel);
            _coarseNcc = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(CoarseNccKernel);
            _coarsePeak = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(CoarsePeakKernel);
            _fineNcc = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(FineNccKernel);
            _finePeak = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(FinePeakKernel);
        }

        /// <summary>
        /// Measures the shift between <paramref name="frame"/> and the frame passed last time. Returns null for the first
        /// frame, after a size change, and for frames too narrow to search (the fine stage compares columns
        /// [MaxShift, width - MaxShift), so the width must leave <see cref="MinOverlap"/> columns between the margins).
        /// </summary>
        public async Task<YawSample?> TrackAsync(MemoryBuffer2D<int, Stride2D.DenseX> frame, double timeSeconds)
        {
            int width = frame.IntExtent.X;
            if (width < 2 * MaxShift + MinOverlap) return null;
            if (width != _width || _prev == null || _cur == null || _prevC == null || _curC == null)
            {
                _prev?.Dispose(); _cur?.Dispose(); _prevC?.Dispose(); _curC?.Dispose();
                _prev = _accelerator.Allocate1D<float>(width);
                _cur = _accelerator.Allocate1D<float>(width);
                _prevC = _accelerator.Allocate1D<float>(width / Decimation);
                _curC = _accelerator.Allocate1D<float>(width / Decimation);
                _width = width;
                _havePrev = false;
            }
            _profile(width, frame.View, _cur.View);
            _decimate(width / Decimation, _cur.View, _curC.View);
            YawSample? sample = null;
            if (_havePrev)
            {
                _coarseNcc(CoarseCount, _prevC.View, _curC.View, _coarseScores.View, width / Decimation);
                _coarsePeak(1, _coarseScores.View, _center.View);
                _fineNcc(FineCount, _prev.View, _cur.View, _center.View, _fineScores.View, width);
                _finePeak(1, _fineScores.View, _center.View, _result.View);
                var r = await _result.View.CopyToHostAsync();
                sample = new YawSample(timeSeconds, r[0], r[1]);
                // Lab: what the picture did at a big jump. Character still, these should not exist.
                if (LogBigShifts && MathF.Abs(r[0]) >= 60)
                {
                    var before = await _prev.View.CopyToHostAsync();
                    var after = await _cur.View.CopyToHostAsync();
                    Console.WriteLine($"[AssistZ] yawprofile ts={timeSeconds:0.000} shift={r[0]:0.0} score={r[1]:0.00} before={Bins(before)} after={Bins(after)}");
                }
            }
            (_prev, _cur) = (_cur, _prev);
            (_prevC, _curC) = (_curC, _prevC);
            _havePrev = true;
            return sample;
        }

        /// <summary>Lab logging, on with ?lab=1.</summary>
        public bool LogBigShifts { get; set; }

        static string Bins(float[] profile)
        {
            const int count = 48;
            var parts = new string[count];
            for (int b = 0; b < count; b++)
            {
                int lo = b * profile.Length / count, hi = (b + 1) * profile.Length / count;
                float sum = 0f;
                for (int i = lo; i < hi; i++) sum += profile[i];
                parts[b] = (sum / (hi - lo)).ToString("0", System.Globalization.CultureInfo.InvariantCulture);
            }
            return string.Join(",", parts);
        }

        // Mean luminance (Rec.601 weights on packed RGBA8) of column x over rows [10%, 90%).
        static void ProfileKernel(Index1D x, ArrayView2D<int, Stride2D.DenseX> frame, ArrayView<float> profile)
        {
            int height = frame.IntExtent.Y;
            int y0 = height / 10, y1 = height - height / 10;
            float sum = 0f;
            for (int y = y0; y < y1; y++)
            {
                int c = frame[x, y];
                sum += 0.299f * (c & 0xFF) + 0.587f * ((c >> 8) & 0xFF) + 0.114f * ((c >> 16) & 0xFF);
            }
            profile[x] = sum / (y1 - y0);
        }

        static void DecimateKernel(Index1D i, ArrayView<float> profile, ArrayView<float> coarse)
        {
            float sum = 0f;
            for (int k = 0; k < Decimation; k++)
                sum += profile[i * Decimation + k];
            coarse[i] = sum / Decimation;
        }

        // Pearson correlation of prev[x] and cur[x - shift] over x in [x0, x1).
        static float Ncc(ArrayView<float> prev, ArrayView<float> cur, int shift, int x0, int x1)
        {
            int n = x1 - x0;
            float sa = 0f, sb = 0f, saa = 0f, sbb = 0f, sab = 0f;
            for (int x = x0; x < x1; x++)
            {
                float a = prev[x];
                float b = cur[x - shift];
                sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
            }
            float cov = sab - sa * sb / n;
            float va = saa - sa * sa / n;
            float vb = sbb - sb * sb / n;
            float den = MathF.Sqrt(va * vb);
            return den > 1e-6f ? cov / den : 0f;
        }

        static void CoarseNccKernel(Index1D s, ArrayView<float> prev, ArrayView<float> cur, ArrayView<float> scores, int coarseWidth)
        {
            scores[s] = Ncc(prev, cur, s - CoarseRange, CoarseRange, coarseWidth - CoarseRange);
        }

        // center[0] = best coarse shift in FULL-resolution pixels.
        static void CoarsePeakKernel(Index1D index, ArrayView<float> scores, ArrayView<float> center)
        {
            int best = CoarseRange;
            float bestScore = scores[CoarseRange];
            for (int s = 0; s < CoarseCount; s++)
            {
                if (scores[s] > bestScore)
                {
                    bestScore = scores[s];
                    best = s;
                }
            }
            center[0] = (best - CoarseRange) * Decimation;
        }

        static void FineNccKernel(Index1D k, ArrayView<float> prev, ArrayView<float> cur, ArrayView<float> center, ArrayView<float> scores, int width)
        {
            int shift = (int)center[0] + k - FineRange;
            scores[k] = Ncc(prev, cur, shift, MaxShift, width - MaxShift);
        }

        // result = [shiftPixels (sub-pixel), score].
        static void FinePeakKernel(Index1D index, ArrayView<float> scores, ArrayView<float> center, ArrayView<float> result)
        {
            int best = FineRange;
            float bestScore = scores[FineRange];
            for (int s = 0; s < FineCount; s++)
            {
                if (scores[s] > bestScore)
                {
                    bestScore = scores[s];
                    best = s;
                }
            }
            float offset = 0f;
            if (best > 0 && best < FineCount - 1)
            {
                float l = scores[best - 1], r = scores[best + 1];
                float den = l - 2f * bestScore + r;
                if (den < 0f) offset = 0.5f * (l - r) / den;
            }
            result[0] = center[0] + best - FineRange + offset;
            result[1] = bestScore;
        }

        public void Dispose()
        {
            _prev?.Dispose(); _cur?.Dispose(); _prevC?.Dispose(); _curC?.Dispose();
            _coarseScores.Dispose(); _fineScores.Dispose(); _center.Dispose(); _result.Dispose();
        }
    }
}
