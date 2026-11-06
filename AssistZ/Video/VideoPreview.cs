using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.Rendering;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

namespace AssistZ.Video
{
    /// <summary>
    /// Captured VideoFrame -> GPU buffer (WebGPU copyExternalImageToTexture, no CPU readback) -> box-filter downsample
    /// kernel -> canvas. This is the front of every vision stage: later stages read <see cref="Frame"/> /
    /// <see cref="Preview"/> on the device instead of the canvas.
    /// </summary>
    public sealed class VideoPreview : IDisposable
    {
        public const int MaxPreviewWidth = 960;

        readonly Accelerator _accelerator;
        readonly IExternalImageCopier _copier;
        readonly ICanvasRenderer _renderer;
        readonly HTMLCanvasElement _canvas;
        readonly Action<Index2D, ArrayView<int>, int, int, ArrayView2D<int, Stride2D.DenseX>> _downsample;
        MemoryBuffer1D<int, Stride1D.Dense>? _frame;
        MemoryBuffer2D<int, Stride2D.DenseX>? _preview;

        public int SourceWidth { get; private set; }
        public int SourceHeight { get; private set; }
        /// <summary>The full-resolution frame, packed RGBA8, on the device.</summary>
        public MemoryBuffer1D<int, Stride1D.Dense>? Frame => _frame;
        /// <summary>The downsampled frame, packed RGBA8, on the device.</summary>
        public MemoryBuffer2D<int, Stride2D.DenseX>? Preview => _preview;

        /// <param name="canvasId">The page canvas to present into; this instance owns the handle.</param>
        public VideoPreview(Accelerator accelerator, SpawnJSRuntime js, string canvasId)
        {
            _accelerator = accelerator;
            _canvas = js.Call<string, HTMLCanvasElement>("document.getElementById", canvasId);
            _copier = ExternalImageCopier.Create(accelerator);
            _renderer = CanvasRendererFactory.Create(accelerator);
            _renderer.AttachCanvas(_canvas);
            _downsample = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<int>, int, int, ArrayView2D<int, Stride2D.DenseX>>(DownsampleKernel);
        }

        /// <summary>Copies the frame to the device, downsamples it and presents the preview. Does not close the frame.</summary>
        public async Task ProcessAsync(VideoFrame frame)
        {
            int width = frame.DisplayWidth, height = frame.DisplayHeight;
            if (width != SourceWidth || height != SourceHeight || _frame == null || _preview == null)
            {
                _frame?.Dispose();
                _preview?.Dispose();
                SourceWidth = width;
                SourceHeight = height;
                _frame = _accelerator.Allocate1D<int>((long)width * height);
                int previewWidth = Math.Min(MaxPreviewWidth, width);
                int previewHeight = Math.Max(1, (int)((long)height * previewWidth / width));
                _preview = _accelerator.Allocate2DDenseX<int>(new LongIndex2D(previewWidth, previewHeight));
                // SpawnDev.ILGPU 5.4.15+ sizes the attached canvas to the presented buffer (it used to keep the 300x150
                // default and show a corner crop); CSS fits the canvas to the stage.
            }
            _copier.CopyToBuffer(frame, width, height, _frame.View);
            _downsample(_preview.IntExtent, _frame.View, width, height, _preview.View);
            await _renderer.PresentAsync(_preview);
        }

        // Box filter: each preview pixel averages the source block it covers.
        static void DownsampleKernel(Index2D p, ArrayView<int> src, int srcWidth, int srcHeight, ArrayView2D<int, Stride2D.DenseX> dst)
        {
            int dstWidth = dst.IntExtent.X, dstHeight = dst.IntExtent.Y;
            int x0 = p.X * srcWidth / dstWidth, x1 = (p.X + 1) * srcWidth / dstWidth;
            int y0 = p.Y * srcHeight / dstHeight, y1 = (p.Y + 1) * srcHeight / dstHeight;
            if (x1 <= x0) x1 = x0 + 1;
            if (y1 <= y0) y1 = y0 + 1;
            int r = 0, g = 0, b = 0;
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    int c = src[y * srcWidth + x];
                    r += c & 0xFF;
                    g += (c >> 8) & 0xFF;
                    b += (c >> 16) & 0xFF;
                }
            }
            int n = (x1 - x0) * (y1 - y0);
            dst[p] = (r / n) | ((g / n) << 8) | ((b / n) << 16) | unchecked((int)0xFF000000);
        }

        public void Dispose()
        {
            _frame?.Dispose();
            _preview?.Dispose();
            _copier.Dispose();
            (_renderer as IDisposable)?.Dispose();
            _canvas.Dispose();
        }
    }
}
