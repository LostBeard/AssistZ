# Front/back resolution run: DayZ lab mode 3 (two world-fixed repeating grenade beacons) + slow mouse sweeps, captured
# into AssistZ WITH AUDIO. Chrome's auto-select skips the picker's audio toggle, so picker_accept.ps1 picks the DayZ window and turns audio on, within
# $PickSeconds; then the driver takes the foreground and sweeps the mouse. Afterwards: analyze_frontback.py on the run folder.
param([int]$CaptureSeconds = 110, [int]$PickSeconds = 30)
$ErrorActionPreference = "Stop"
# Work folder: publish output, lab Chrome profile, screenshots. Override with $env:ASSISTZ_LAB_WORK.
$S = if ($env:ASSISTZ_LAB_WORK) { $env:ASSISTZ_LAB_WORK } else { Join-Path $env:TEMP "assistz-lab" }
New-Item -ItemType Directory -Force $S | Out-Null
$H = $PSScriptRoot
$Tools = Split-Path $H -Parent
$Lab = "D:\users\tj\Projects\AssistZ\Lab"
$out = "$Lab\runs\frontback_" + (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
New-Item -ItemType Directory -Force $out | Out-Null

Push-Location (Join-Path (Split-Path $Tools -Parent) "AssistZ")
dotnet publish -c Release -o "$S\publish" 2>&1 | Out-File "$out\publish.log"
$pubOk = $LASTEXITCODE -eq 0
Pop-Location
if (-not $pubOk) { "PUBLISH FAILED (see $out\publish.log)"; exit 1 }
"published"

& "$Tools\make_lab_missions.ps1" -Maps chernarusplus -Mode 3
python "$Tools\mod\build_mod.py" "$Lab\mods"
# The player is present only when a pose log CREATED BY THIS LAUNCH has rows: the previous run's log is still
# "recent" and once made a run start (and turn the view) while the client was still loading.
$launchTime = Get-Date
& "$Tools\launch_dayzdiag.ps1" -Map chernarusplus -LabMod
$chrome = $null
try {
    $deadline = (Get-Date).AddSeconds(240)
    $csv = $null
    while ((Get-Date) -lt $deadline) {
        $csv = Get-ChildItem "$Lab\profiles\server\assistz_pose_*.csv" | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending | Select-Object -First 1
        if ($csv -and $csv.CreationTime -gt $launchTime -and (Get-Content $csv.FullName | Measure-Object -Line).Lines -gt 20) { break }
        Start-Sleep -Seconds 3
    }
    if (-not $csv) { throw "no pose CSV" }
    "player present: $($csv.Name)"

    $chromeArgs = @("--user-data-dir=$S\chrome-profile", "--remote-debugging-port=9333", "--no-first-run", "--no-default-browser-check",
        "--new-window", "http://127.0.0.1:5390/?lab=1")
    $chrome = Start-Process -FilePath "C:\Program Files\Google\Chrome\Application\chrome.exe" -ArgumentList $chromeArgs -PassThru
    "Chrome PID $($chrome.Id)"
    for ($i = 0; $i -lt 40; $i++) { try { Invoke-WebRequest http://127.0.0.1:9333/json/version -UseBasicParsing -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Milliseconds 250 } }
    $cap = Start-Process -FilePath python -ArgumentList @("-u", "$H\capture_live.py", "http://127.0.0.1:5390", "$CaptureSeconds", "$out\shot", "$out\console.txt") `
        -RedirectStandardOutput "$out\capture.txt" -RedirectStandardError "$out\capture_err.txt" -PassThru -NoNewWindow
    # Only ask for the pick once the app has booted; a page that failed to load stops the run here.
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        $txt = Get-Content "$out\capture.txt" -Raw -ErrorAction SilentlyContinue
        if ($txt -match "APP READY") { $ready = $true; break }
        if ($txt -match "APP DID NOT BOOT" -or $cap.HasExited) { break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { Get-Content "$out\capture.txt" -ErrorAction SilentlyContinue; throw "AssistZ did not boot in the lab Chrome (see above)" }
    "APP READY: accepting the picker ($PickSeconds s)"
    Start-Process powershell -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$H\picker_accept.ps1", '-Match', 'DayZ') -NoNewWindow -RedirectStandardOutput "$out\picker.txt" | Out-Null
    Start-Sleep -Seconds $PickSeconds

    python "$Tools\drive_turns.py" "$Lab\dayzdiag.pids.json" "$out\turns.json" sweeps 2>&1 | Tee-Object "$out\drive.txt"

    $cap.WaitForExit(($CaptureSeconds + 60) * 1000) | Out-Null
    Copy-Item $csv.FullName "$out\pose.csv"
    $beacon = Get-ChildItem "$Lab\profiles\server\assistz_beacon_*.csv" | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending | Select-Object -First 1
    if ($beacon) { Copy-Item $beacon.FullName "$out\beacon.csv" }
    "capture done"
}
finally {
    if ($chrome) { taskkill /PID $chrome.Id /T /F | Out-Null; "Killed Chrome PID $($chrome.Id)" }
    & "$Tools\stop_dayzdiag.ps1"
}
python "$Tools\analyze_frontback.py" $out
"RUN DIR $out"
