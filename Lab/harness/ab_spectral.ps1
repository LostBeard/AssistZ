# A/B of the audio batch latency with the SpectralFrontend path on vs off, on the test signal (same content both runs),
# alternating twice (on, off, on, off) to expose drift. Then the YouTube watch-along with the player-fill fix.
$ErrorActionPreference = "Stop"
# Work folder: publish output, lab Chrome profile, screenshots. Override with $env:ASSISTZ_LAB_WORK.
$S = if ($env:ASSISTZ_LAB_WORK) { $env:ASSISTZ_LAB_WORK } else { Join-Path $env:TEMP "assistz-lab" }
New-Item -ItemType Directory -Force $S | Out-Null
$H = $PSScriptRoot  # the harness scripts
$app = "D:\users\tj\Projects\AssistZ\AssistZ\AssistZ"
$out = "D:\users\tj\Projects\AssistZ\Lab\runs\ab_" + (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $app
dotnet publish -c Release -o "$S\publish" 2>&1 | Out-File "$out\publish.log"
$ok = $LASTEXITCODE -eq 0
Pop-Location
if (-not $ok) { "PUBLISH FAILED"; exit 1 }
"published"

$i = 0
foreach ($spectral in 1, 0, 1, 0) {
    $i++
    $chromeArgs = @("--user-data-dir=$S\chrome-profile", "--remote-debugging-port=9333", "--no-first-run", "--no-default-browser-check",
        "--autoplay-policy=no-user-gesture-required", "--auto-select-tab-capture-source-by-title=AssistZ Test Signal",
        "--new-window", "http://127.0.0.1:5390/testsignal")
    $p = Start-Process -FilePath "C:\Program Files\Google\Chrome\Application\chrome.exe" -ArgumentList $chromeArgs -PassThru
    try {
        for ($k = 0; $k -lt 40; $k++) { try { Invoke-WebRequest http://127.0.0.1:9333/json/version -UseBasicParsing -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Milliseconds 250 } }
        Push-Location $S
        python "$H\verify_m0.py" chrome.pid http://127.0.0.1:5390 25 "$out\shot_$i.png" "http://127.0.0.1:5390/?lab=1&spectral=$spectral" "$out\console_${i}_spectral$spectral.txt" | Select-String "RESULT|events in|spectral az"
        Pop-Location
    } finally {
        taskkill /PID $p.Id /T /F | Out-Null
    }
}
python "$H\ab_analyze.py" $out
"OUT $out"
