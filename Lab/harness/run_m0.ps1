# M0 end-to-end check (verify_m0.py) in a separate Chrome (own profile, port 9333). Kills only the Chrome PID it started.
# -Query "lab=1&warm=0" sets the app's query string; -ConsoleOut saves the whole console (analyze_m0.py reads it).
param([string]$Shot = "m0_shot.png", [int]$Seconds = 20, [string]$Query = "", [string]$ConsoleOut = "")
# Work folder: publish output, lab Chrome profile, screenshots. Override with $env:ASSISTZ_LAB_WORK.
$S = if ($env:ASSISTZ_LAB_WORK) { $env:ASSISTZ_LAB_WORK } else { Join-Path $env:TEMP "assistz-lab" }
New-Item -ItemType Directory -Force $S | Out-Null
$H = $PSScriptRoot  # the harness scripts
$chromeArgs = @("--user-data-dir=$S\chrome-profile", "--remote-debugging-port=9333", "--no-first-run", "--no-default-browser-check",
    "--autoplay-policy=no-user-gesture-required", "--auto-select-tab-capture-source-by-title=AssistZ Test Signal",
    "--new-window", "http://127.0.0.1:5390/testsignal")
$p = Start-Process -FilePath "C:\Program Files\Google\Chrome\Application\chrome.exe" -ArgumentList $chromeArgs -PassThru
"Chrome PID $($p.Id)"
try {
    for ($i = 0; $i -lt 40; $i++) {
        try { Invoke-WebRequest http://127.0.0.1:9333/json/version -UseBasicParsing -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Milliseconds 250 }
    }
    Push-Location $S
    $url = if ($Query) { "http://127.0.0.1:5390/?$Query" } else { "http://127.0.0.1:5390/" }
    python "$H\verify_m0.py" chrome.pid http://127.0.0.1:5390 $Seconds "$S\$Shot" $url $ConsoleOut
    Pop-Location
}
finally {
    # Only the PID this script started (and its children).
    taskkill /PID $p.Id /T /F | Out-Null
    "Killed Chrome PID $($p.Id)"
}
