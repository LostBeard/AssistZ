# CLAP sound naming on a lab recording with ground truth (verify_labels.py) in a separate Chrome (own profile, port 9333).
# The app must be published and served on :5390 (spa_serve.py). Kills only the Chrome PID it started.
# -Query "&clap=larger" picks the larger CLAP checkpoint; -Tag names the console file (labels_console<Tag>.txt).
param([Parameter(Mandatory)][string]$Run, [int]$Seconds = 150, [string]$Query = "", [string]$Tag = "")
$S = if ($env:ASSISTZ_LAB_WORK) { $env:ASSISTZ_LAB_WORK } else { Join-Path $env:TEMP "assistz-lab" }
New-Item -ItemType Directory -Force $S | Out-Null
$H = $PSScriptRoot
$chromeArgs = @("--user-data-dir=$S\chrome-profile", "--remote-debugging-port=9333", "--no-first-run", "--no-default-browser-check",
    "--autoplay-policy=no-user-gesture-required", "--new-window", "about:blank")
$p = Start-Process -FilePath "C:\Program Files\Google\Chrome\Application\chrome.exe" -ArgumentList $chromeArgs -PassThru
"Chrome PID $($p.Id)"
try {
    for ($i = 0; $i -lt 40; $i++) {
        try { Invoke-WebRequest http://127.0.0.1:9333/json/version -UseBasicParsing -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Milliseconds 250 }
    }
    python "$H\verify_labels.py" ($Run -replace '\\', '/') http://127.0.0.1:5390 $Seconds "$Run\labels_console$Tag.txt" $Query
}
finally {
    taskkill /PID $p.Id /T /F | Out-Null
    "Killed Chrome PID $($p.Id)"
}
