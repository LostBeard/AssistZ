using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;

namespace AssistZ.Services
{
    /// <summary>
    /// Owns the one ILGPU context and WebGPU accelerator the app shares. AssistZ requires WebGPU (Chrome/Edge):
    /// getDisplayMedia only shares tab/window audio there anyway, so there is no fallback backend to keep alive.
    /// </summary>
    public sealed class GpuHost : IAsyncDisposable
    {
        Context? _context;
        Accelerator? _accelerator;
        Task<Accelerator>? _init;

        /// <summary>The accelerator, created on first call. Throws <see cref="NotSupportedException"/> without WebGPU.</summary>
        public Task<Accelerator> GetAcceleratorAsync() => _init ??= CreateAsync();

        async Task<Accelerator> CreateAsync()
        {
            _context = await Context.CreateAsync(builder => builder.WebGPU());
            var devices = _context.GetWebGPUDevices();
            if (devices.Count == 0)
                throw new NotSupportedException("AssistZ needs WebGPU. Use a current Chrome or Edge with hardware acceleration on.");
            _accelerator = await devices[0].CreateAcceleratorAsync(_context);
            Console.WriteLine($"[AssistZ] GPU: {_accelerator.Name} ({_accelerator.AcceleratorType})");
            return _accelerator;
        }

        public ValueTask DisposeAsync()
        {
            _accelerator?.Dispose();
            _context?.Dispose();
            return ValueTask.CompletedTask;
        }
    }
}
