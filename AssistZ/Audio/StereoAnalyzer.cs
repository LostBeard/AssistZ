using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.SpawnJS.JSObjects;

namespace AssistZ.Audio
{
    /// <summary>
    /// Keeps the captured stereo PCM in one planar GPU ring (<c>[2][RingSize]</c>: left then right) and measures, for every
    /// hop, on the device:
    /// <list type="bullet">
    /// <item>channel levels and the cross-correlation over +-<see cref="MaxLag"/> samples (a pure level-panned mix, which is
    /// what DayZ uses, peaks at lag 0);</item>
    /// <item>through SpawnDev.ILGPU.ML's <see cref="SpectralFrontend"/>: power spectra (radix-2 FFT, N = 2048, Hann), 48 mel
    /// bands per channel and the per-band level difference, reduced here to an energy-weighted level difference, the
    /// spectral centroid and the high-band energy fraction.</item>
    /// </list>
    /// Samples go AudioData -> JS Float32Array -> GPU (CopyFromJS) and never enter the .NET heap; only
    /// <see cref="ResultStride"/> floats per window are read back.
    /// </summary>
    public sealed class StereoAnalyzer : IDisposable
    {
        /// <summary>Ring size per channel, a power of two. 65536 samples = 1.37 s at 48 kHz.</summary>
        public const int RingSize = 1 << 16;
        const int RingMask = RingSize - 1;
        /// <summary>Analysis window in samples (42.7 ms at 48 kHz); also the FFT size.</summary>
        public const int Window = 2048;
        /// <summary>Largest lag searched, in samples. 48 samples = 1 ms at 48 kHz, wider than any human-head ITD (~0.7 ms).</summary>
        public const int MaxLag = 48;
        const int LagCount = 2 * MaxLag + 1;
        /// <summary>Most windows one dispatch measures. Chrome delivers tab audio in 2048-frame chunks, i.e. 2 hops of 1024.</summary>
        public const int MaxWindows = 16;
        /// <summary>Mel bands per channel.</summary>
        public const int Bands = 48;
        /// <summary>Floats per window in the readback: rmsL, rmsR, bestLag, peakNorm, zeroLagNorm, bandIld, centroid, highFraction.</summary>
        public const int ResultStride = 8;
        const int Bins = Window / 2 + 1;

        readonly Accelerator _accelerator;
        readonly MemoryBuffer1D<float, Stride1D.Dense> _ring, _corr, _result, _power, _bands, _ild;
        readonly IBrowserMemoryBuffer _ringJS;
        readonly Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> _xcorr;
        readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> _reduce;
        readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _spectralSummary;
        readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, float> _extract;
        readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> _peak;
        readonly MemoryBuffer1D<float, Stride1D.Dense> _peakResult;
        // Staging arrays by length: AudioData.copyTo needs a destination of the exact frame count, and chunk sizes repeat.
        readonly Dictionary<int, Float32Array> _staging = new();
        SpectralFrontend? _frontend;

        /// <summary>Total samples written per channel since start.</summary>
        public long WritePosition { get; private set; }
        /// <summary>Sample rate of the stream (from the first chunk).</summary>
        public int SampleRate { get; private set; }
        /// <summary>Run the SpectralFrontend path (on by default). Off only for A/B timing (?spectral=0): the spectral fields
        /// of the frames are then meaningless.</summary>
        public bool SpectralEnabled { get; set; } = true;

        ArrayView<float> Left => _ring.View.SubView(0, RingSize);
        ArrayView<float> Right => _ring.View.SubView(RingSize, RingSize);

        public StereoAnalyzer(Accelerator accelerator)
        {
            _accelerator = accelerator;
            _ring = accelerator.Allocate1D<float>(2L * RingSize);
            _ring.MemSetToZero();
            _ringJS = (IBrowserMemoryBuffer)_ring.Buffer;
            _corr = accelerator.Allocate1D<float>(LagCount * MaxWindows);
            _result = accelerator.Allocate1D<float>(ResultStride * MaxWindows);
            _power = accelerator.Allocate1D<float>((long)MaxWindows * 2 * Bins);
            _bands = accelerator.Allocate1D<float>((long)MaxWindows * 2 * Bands);
            _ild = accelerator.Allocate1D<float>((long)MaxWindows * Bands);
            _xcorr = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(XCorrKernel);
            _reduce = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(ReduceKernel);
            _spectralSummary = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(SpectralSummaryKernel);
            _extract = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, float>(ExtractMonoKernel);
            _peak = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(PeakKernel);
            _peakResult = accelerator.Allocate1D<float>(2);
        }

        /// <summary>
        /// Appends one AudioData chunk (f32-planar) to the ring. A mono chunk is written to both channels.
        /// </summary>
        public async Task WriteAsync(AudioData data)
        {
            SetSampleRate((int)data.SampleRate);
            int frames = data.NumberOfFrames;
            int channels = data.NumberOfChannels;
            int ringPos = (int)(WritePosition & RingMask);
            int first = Math.Min(frames, RingSize - ringPos);
            await WritePartAsync(data, 0, first, ringPos, channels);
            if (first < frames) await WritePartAsync(data, first, frames - first, 0, channels);
            WritePosition += frames;
        }

        async Task WritePartAsync(AudioData data, int frameOffset, int count, int ringPos, int channels)
        {
            var staging = Staging(count);
            long leftOffset = (long)ringPos * sizeof(float);
            long rightOffset = ((long)RingSize + ringPos) * sizeof(float);
            await data.CopyTo(staging, new AudioDataCopyToOptions { PlaneIndex = 0, FrameOffset = frameOffset, FrameCount = count });
            _ringJS.CopyFromJS(staging, leftOffset);
            if (channels > 1)
                await data.CopyTo(staging, new AudioDataCopyToOptions { PlaneIndex = 1, FrameOffset = frameOffset, FrameCount = count });
            _ringJS.CopyFromJS(staging, rightOffset);
        }

        Float32Array Staging(int length)
        {
            if (!_staging.TryGetValue(length, out var array))
                _staging[length] = array = new Float32Array(length);
            return array;
        }

        // The frontend's bands depend on the rate, so a stream that differs from the warmed-up rate rebuilds it.
        void SetSampleRate(int sampleRate)
        {
            if (SampleRate == sampleRate) return;
            SampleRate = sampleRate;
            _frontend?.Dispose();
            _frontend = null;
        }

        /// <summary>
        /// Appends planar samples that are already in JS arrays (a decoded recording in a lab replay): the same ring and the
        /// same JS -> GPU copy as <see cref="WriteAsync"/>, so the PCM never enters the .NET heap.
        /// </summary>
        public void Write(Float32Array left, Float32Array right, int frames, int sampleRate)
        {
            SetSampleRate(sampleRate);
            int ringPos = (int)(WritePosition & RingMask);
            int first = Math.Min(frames, RingSize - ringPos);
            WritePlanar(left, right, 0, first, ringPos);
            if (first < frames) WritePlanar(left, right, first, frames - first, 0);
            WritePosition += frames;
        }

        void WritePlanar(Float32Array left, Float32Array right, int offset, int count, int ringPos)
        {
            using var l = left.SubArray(offset, offset + count);
            using var r = right.SubArray(offset, offset + count);
            _ringJS.CopyFromJS(l, (long)ringPos * sizeof(float));
            _ringJS.CopyFromJS(r, ((long)RingSize + ringPos) * sizeof(float));
        }

        /// <summary>
        /// Runs every kernel once on the zeroed ring, before the first write. Kernels compile on their first dispatch: left
        /// to the first audio batch that stall measured 709.6 ms on an RX 590 (49.6 ms on an RTX 4070), during which the
        /// stream's windows are skipped and a sound can be lost.
        /// </summary>
        public async Task WarmUpAsync(int sampleRate)
        {
            if (WritePosition != 0) throw new InvalidOperationException("Warm up before the first write.");
            SetSampleRate(sampleRate);
            WritePosition = RingSize;
            try
            {
                using var mono = _accelerator.Allocate1D<float>(16);
                ExtractMono(0, 16, mono.View);
                await FindPeakAsync(0, 16);
                await AnalyzeAsync(RingSize - (MaxWindows - 1) * Window / 2, Window / 2, MaxWindows);
            }
            finally
            {
                WritePosition = 0;
            }
        }

        /// <summary>Sample position (exclusive end) of the oldest window still fully in the ring.</summary>
        public long OldestAnalyzableEnd => Math.Max(Window + 2L * MaxLag, WritePosition - RingSize + Window + 2L * MaxLag);

        SpectralFrontend Frontend()
        {
            if (_frontend != null) return _frontend;
            _frontend = new SpectralFrontend(_accelerator, n: Window, channels: 2, sampleRate: SampleRate, window: SpectralWindow.Hann);
            _frontend.SetBands(Bands, 40f, MathF.Min(20000f, SampleRate * 0.45f), BandScale.Mel);
            return _frontend;
        }

        /// <summary>
        /// Measures <paramref name="count"/> windows in one batch, ending at <paramref name="firstEnd"/>, firstEnd + hop, ...
        /// Every end must lie in [<see cref="OldestAnalyzableEnd"/>, <see cref="WritePosition"/>]. Window k covers samples
        /// [end_k - Window - MaxLag, end_k - MaxLag), so its largest positive lag reads end_k - 1 at most (never unwritten
        /// ring data); the spectral frame covers the same samples.
        /// </summary>
        public async Task<StereoFrame[]> AnalyzeAsync(long firstEnd, int hop, int count)
        {
            if (count < 1 || count > MaxWindows) throw new ArgumentOutOfRangeException(nameof(count));
            long lastEnd = firstEnd + (long)(count - 1) * hop;
            if (firstEnd < OldestAnalyzableEnd || lastEnd > WritePosition)
                throw new ArgumentOutOfRangeException(nameof(firstEnd), $"Windows {firstEnd}..{lastEnd} outside the ring's valid range {OldestAnalyzableEnd}..{WritePosition}.");
            int firstStart = (int)((firstEnd - Window - MaxLag) & RingMask);
            _xcorr(new Index2D(LagCount, count), Left, Right, _corr.View, firstStart, hop);
            _reduce(count, Left, Right, _corr.View, _result.View, firstStart, hop);

            if (SpectralEnabled)
            {
                var frontend = Frontend();
                frontend.Power(_ring.View, RingSize, WritePosition, firstEnd - MaxLag, hop, count, _power.View);
                frontend.PoolBands(_power.View, count, _bands.View, 0f, _ild.View);
                _spectralSummary(count, _bands.View, _ild.View, _result.View);
            }

            var r = await _result.View.SubView(0, ResultStride * count).CopyToHostAsync();
            var frames = new StereoFrame[count];
            for (int k = 0; k < count; k++)
            {
                int o = k * ResultStride;
                frames[k] = new StereoFrame(firstEnd + (long)k * hop, r[o], r[o + 1], (int)r[o + 2], r[o + 3], r[o + 4], r[o + 5], r[o + 6], r[o + 7]);
            }
            return frames;
        }

        /// <summary>True while samples [start, start + length) are all written and not yet overwritten.</summary>
        public bool InRing(long start, int length) => start >= 0 && start >= WritePosition - RingSize && start + length <= WritePosition;

        /// <summary>
        /// Copies samples [start, start + length) of the ring into <paramref name="mono"/> as (L + R) / 2, on the device
        /// (the PCM still never enters the .NET heap), multiplied by <paramref name="gain"/>. The span must be <see cref="InRing"/>.
        /// </summary>
        public void ExtractMono(long start, int length, ArrayView<float> mono, float gain = 1f)
        {
            if (!InRing(start, length)) throw new ArgumentOutOfRangeException(nameof(start), $"Samples {start}..{start + length} are not in the ring (written {WritePosition}).");
            if (mono.Length < length) throw new ArgumentException("Destination too short.", nameof(mono));
            _extract(length, Left, Right, mono, (int)(start & RingMask), gain);
        }

        /// <summary>
        /// The sample of samples [start, start + length) with the largest mono magnitude (the earliest on a tie), on the device;
        /// one int is read back. It belongs to the sound, not to the hop grid, so it is the same wherever the stream happened to
        /// start as long as the span holds the peak. Two floats are read back: the offset (exact, the span is far under 2^24) and the magnitude.
        /// </summary>
        public async Task<(long Position, float Magnitude)> FindPeakAsync(long start, int length)
        {
            if (!InRing(start, length)) throw new ArgumentOutOfRangeException(nameof(start), $"Samples {start}..{start + length} are not in the ring (written {WritePosition}).");
            _peak(1, Left, Right, _peakResult.View, (int)(start & RingMask), length);
            var peak = await _peakResult.View.CopyToHostAsync();
            return (start + (long)peak[0], peak[1]);
        }

        static void PeakKernel(Index1D thread, ArrayView<float> left, ArrayView<float> right, ArrayView<float> result, int start, int length)
        {
            int best = 0;
            float bestMagnitude = -1f;
            for (int i = 0; i < length; i++)
            {
                int k = (start + i) & RingMask;
                float magnitude = MathF.Abs(0.5f * (left[k] + right[k]));
                if (magnitude > bestMagnitude)
                {
                    bestMagnitude = magnitude;
                    best = i;
                }
            }
            result[0] = best;
            result[1] = bestMagnitude;
        }

        static void ExtractMonoKernel(Index1D i, ArrayView<float> left, ArrayView<float> right, ArrayView<float> mono, int start, float gain)
        {
            int k = (start + i) & RingMask;
            mono[i] = gain * 0.5f * (left[k] + right[k]);
        }

        // x = lagIndex in [0, 2*MaxLag] (lag = lagIndex - MaxLag), y = window. corr[y][lag] = sum_i L[s+i] * R[s+i+lag].
        static void XCorrKernel(Index2D index, ArrayView<float> left, ArrayView<float> right, ArrayView<float> corr, int firstStart, int hop)
        {
            int lagIndex = index.X, w = index.Y;
            int lag = lagIndex - MaxLag;
            int start = firstStart + w * hop;
            float sum = 0f;
            for (int i = 0; i < Window; i++)
            {
                int li = (start + i) & RingMask;
                int ri = (start + i + lag + RingSize) & RingMask;
                sum += left[li] * right[ri];
            }
            corr[w * LagCount + lagIndex] = sum;
        }

        // One thread per window: channel energies, the peak of the correlation and its normalised value.
        // result[w][0..4] = [rmsL, rmsR, bestLag, peakNorm, zeroLagNorm]
        static void ReduceKernel(Index1D w, ArrayView<float> left, ArrayView<float> right, ArrayView<float> corr, ArrayView<float> result, int firstStart, int hop)
        {
            int start = firstStart + w * hop;
            float eL = 0f, eR = 0f;
            for (int i = 0; i < Window; i++)
            {
                int k = (start + i) & RingMask;
                float l = left[k];
                float r = right[k];
                eL += l * l;
                eR += r * r;
            }
            int c0 = w * LagCount;
            int bestIndex = MaxLag;
            float best = corr[c0 + MaxLag];
            for (int k = 0; k < LagCount; k++)
            {
                float c = corr[c0 + k];
                if (c > best)
                {
                    best = c;
                    bestIndex = k;
                }
            }
            float norm = MathF.Sqrt(eL * eR);
            int o = w * ResultStride;
            result[o] = MathF.Sqrt(eL / Window);
            result[o + 1] = MathF.Sqrt(eR / Window);
            result[o + 2] = bestIndex - MaxLag;
            result[o + 3] = norm > 0f ? best / norm : 0f;
            result[o + 4] = norm > 0f ? corr[c0 + MaxLag] / norm : 0f;
        }

        // One thread per window over the frontend's bands [w][2][Bands] (linear) and ild [w][Bands] (dB, left over right):
        // result[w][5..7] = [energy-weighted ild dB, spectral centroid (0..1 of the band range), energy fraction in the top third].
        static void SpectralSummaryKernel(Index1D w, ArrayView<float> bands, ArrayView<float> ild, ArrayView<float> result)
        {
            int b0 = w * 2 * Bands;
            float total = 0f, weightedIld = 0f, moment = 0f, high = 0f;
            for (int b = 0; b < Bands; b++)
            {
                float e = bands[b0 + b] + bands[b0 + Bands + b];
                total += e;
                weightedIld += e * ild[w * Bands + b];
                moment += e * b;
                if (b >= 2 * Bands / 3) high += e;
            }
            int o = w * ResultStride;
            bool any = total > 1e-12f;
            result[o + 5] = any ? weightedIld / total : 0f;
            result[o + 6] = any ? moment / total / (Bands - 1) : 0f;
            result[o + 7] = any ? high / total : 0f;
        }

        public void Dispose()
        {
            foreach (var array in _staging.Values) array.Dispose();
            _staging.Clear();
            _frontend?.Dispose();
            _ring.Dispose();
            _corr.Dispose();
            _result.Dispose();
            _power.Dispose();
            _bands.Dispose();
            _ild.Dispose();
            _peakResult.Dispose();
        }
    }
}
