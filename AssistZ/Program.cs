using AssistZ;
using AssistZ.Services;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.SpawnJS;
using System.Reflection;

var buildTimestamp = typeof(App).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
    .FirstOrDefault(a => a.Key == "BuildTimestamp")?.Value;
Console.WriteLine($"[AssistZ] Build: {buildTimestamp}");

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddSpawnJSRuntime(out var JS);
JS.Verbose = false;

builder.Services.AddSingleton<GpuHost>();
builder.Services.AddSingleton<AssistZ.Audio.SoundLabeler>();
builder.Services.AddSingleton<CaptureSession>();
builder.Services.AddSingleton<OverlayWindow>();
builder.Services.AddSingleton(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().SpawnJSRunAsync();
