using System.Diagnostics;
using System.Globalization;
using global::ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

namespace AssistZ.Video
{
    /// <summary>
    /// Lab only (?see=1): runs the video detector on live frames beside the sound work, logs what it finds and how long it takes, and shows nothing. For measuring whether it fits. YOLOv8n is a stand-in here, and its weights must not ship in the app (docs/PLAN.md): D-FINE-S replaces it once it runs on the library.
    /// </summary>
    public sealed class SightPass : IDisposable
    {
        // Under 500: a pass starts on the next video frame after this, so 500 gave 533 ms apart, 1.88 a second.
        const int MinimumIntervalMs = 470;
        const int TileSide = 640;
        static readonly int[] TileLefts = { 0, 480, 960 };

        readonly SpawnJSRuntime _js;
        readonly HttpClient _http;
        readonly Accelerator _accelerator;
        readonly SightTracker _tracker = new();
        readonly Stopwatch _clock = Stopwatch.StartNew();
        InferenceSession? _session;
        ObjectDetectionPipeline? _yolo;
        OffscreenCanvas? _canvas, _small;
        CanvasRenderingContext2D? _context, _smallContext;
        int _width, _height;
        string _share = "";
        long _lastStart = -MinimumIntervalMs;
        bool _busy;

        public SightPass(SpawnJSRuntime js, HttpClient http, Accelerator accelerator)
        {
            _js = js;
            _http = http;
            _accelerator = accelerator;
        }

        public bool Loaded => _yolo != null;
        public long Passes { get; private set; }

        public async Task LoadAsync()
        {
            if (_yolo != null) return;
            var clock = Stopwatch.StartNew();
            using var source = new HubModelSource(_js, _http);
            await using var stream = await source.OpenAsync(ModelHub.KnownModels.YOLOv8n, "yolov8n-onnx-web/yolov8n.onnx");
            _session = await InferenceSession.CreateFromStreamAsync(_accelerator, stream);
            _yolo = new ObjectDetectionPipeline(_session, _accelerator) { EnableGraphCapture = true };
            // Record the plan now, on a blank picture, before the live loops start: recording while other work runs on the GPU fails on the video copy, and took in 280 dispatches where the model has 250.
            var blank = new int[640 * 640];
            int runs = 0;
            while (runs < 8 && !_yolo.CaptureStatus.StartsWith("live"))
            {
                await _yolo.DetectAsync(blank, 640, 640);
                runs++;
            }
            Console.WriteLine($"[AssistZ] sight loaded ms={clock.ElapsedMilliseconds} warm={runs} capture={_yolo.CaptureStatus.Replace(' ', '_')}");
        }

        /// <summary>Time for another pass? At most one at a time, and no more than 2 a second.</summary>
        public bool Due => _yolo != null && !_busy && _clock.ElapsedMilliseconds - _lastStart >= MinimumIntervalMs;

        /// <summary>Copies the frame, which the caller closes, then runs the pass in the background.</summary>
        public Task StartAsync(VideoFrame frame)
        {
            // A shared window comes with its title bar and borders (1602 x 938 for a 1600 x 900 game), and the rules are in the game picture's pixels. A 16:9 share is taken as all game. Other shapes: the game is assumed 16:9 inside a 1 px border.
            int shareW = frame.DisplayWidth, shareH = frame.DisplayHeight, sx = 0, sy = 0, w = shareW, h = shareH;
            if (Math.Abs(shareH / (double)shareW - 9.0 / 16) > 0.005 && shareW > 2)
            {
                w = shareW - 2;
                h = (int)Math.Round(w * 9.0 / 16);
                if (h + 1 <= shareH)
                {
                    sx = 1;
                    sy = shareH - h - 1;
                }
                else
                {
                    w = shareW;
                    h = shareH;
                }
            }
            _share = $"{shareW}x{shareH}";
            if (_canvas == null || w != _width || h != _height)
            {
                _context?.Dispose();
                _canvas?.Dispose();
                _canvas = new OffscreenCanvas(w, h);
                _context = _canvas.Get2DContext(new CanvasRenderingContext2DSettings { WillReadFrequently = true });
                _width = w;
                _height = h;
            }
            if (_small == null)
            {
                _small = new OffscreenCanvas(SightRules.InventoryWidth, SightRules.InventoryHeight);
                _smallContext = _small.Get2DContext(new CanvasRenderingContext2DSettings { WillReadFrequently = true });
            }
            _context!.DrawImage(frame, sx, sy, w, h, 0, 0, w, h);
            _smallContext!.DrawImage(frame, sx, sy, w, h, 0, 0, SightRules.InventoryWidth, SightRules.InventoryHeight);
            _busy = true;
            long sinceLast = _clock.ElapsedMilliseconds - _lastStart;
            _lastStart = _clock.ElapsedMilliseconds;
            return RunAsync(sinceLast);
        }

        async Task RunAsync(long sinceLast)
        {
            try
            {
                var total = Stopwatch.StartNew();
                using var small = _smallContext!.GetImageData(0, 0, SightRules.InventoryWidth, SightRules.InventoryHeight);
                using var smallData = small.Data;
                bool inventory = SightRules.IsInventory(smallData.ToArray());
                double gateMs = total.Elapsed.TotalMilliseconds;
                var boxes = new List<SightBox>();
                if (!inventory)
                {
                    float scale = 1600f / _width;
                    await DetectAsync("full", 0, 0, _width, _height, scale, boxes);
                    int side = (int)(TileSide / scale), top = (int)(SightRules.TileTop / scale);
                    for (int t = 0; t < TileLefts.Length; t++)
                        await DetectAsync($"tile{t}", (int)(TileLefts[t] / scale), top, side, Math.Min(side, _height - top), scale, boxes);
                }
                var seen = _tracker.Update(_clock.Elapsed.TotalSeconds, boxes);
                Passes++;
                var culture = CultureInfo.InvariantCulture;
                string list = string.Join(";", seen.Select(s => string.Create(culture, $"{s.BearingDegrees:0.0}:{s.Meters:0}:{s.Confidence:0.00}")));
                Console.WriteLine(string.Create(culture, $"[AssistZ] sight utc={DateTime.UtcNow:HH:mm:ss.fff} frame={_width}x{_height} share={_share} since={sinceLast} ms={total.Elapsed.TotalMilliseconds:0.0} gate={gateMs:0.0} inventory={(inventory ? 1 : 0)} boxes={boxes.Count} seen={list} capture={_yolo!.CaptureStatus.Replace(' ', '_')}"));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AssistZ] sight failed {ex.GetType().Name}: {ex.Message}");
            }
            finally
            {
                _busy = false;
            }
        }

        async Task DetectAsync(string kind, int x, int y, int w, int h, float scale, List<SightBox> boxes)
        {
            using var image = _context!.GetImageData(x, y, w, h);
            using var data = image.Data;
            var result = await _yolo!.DetectAsync(data, w, h, confidenceThreshold: 0.1f);
            foreach (var o in result.Objects)
                if (o.Label == "person")
                    boxes.Add(new SightBox(kind, (o.X + x) * scale, (o.Y + y) * scale, o.Width * scale, o.Height * scale, o.Confidence));
        }

        public void Dispose()
        {
            _smallContext?.Dispose();
            _small?.Dispose();
            _context?.Dispose();
            _canvas?.Dispose();
            _yolo?.Dispose();
            _session?.Dispose();
        }
    }
}
