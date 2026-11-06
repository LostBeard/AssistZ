param([int]$CaptureSeconds = 70, [ValidateSet("server", "mouse")][string]$Driver = "mouse")
$ErrorActionPreference = "Stop"
# Work folder: publish output, lab Chrome profile, screenshots. Override with $env:ASSISTZ_LAB_WORK.
$S = if ($env:ASSISTZ_LAB_WORK) { $env:ASSISTZ_LAB_WORK } else { Join-Path $env:TEMP "assistz-lab" }
New-Item -ItemType Directory -Force $S | Out-Null
$H = $PSScriptRoot  # the harness scripts
$Lab = "D:\users\tj\Projects\AssistZ\Lab"
$Tools = "D:\users\tj\Projects\AssistZ\AssistZ\Lab"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
$out = "$Lab\runs\yaw_$stamp"
New-Item -ItemType Directory -Force $out | Out-Null

# 1. Release publish of the app under test (fail = stop).
Push-Location D:\users\tj\Projects\AssistZ\AssistZ\AssistZ
dotnet publish -c Release -o "$S\publish" 2>&1 | Out-File "$out\publish.log"
$pubOk = $LASTEXITCODE -eq 0
Pop-Location
if (-not $pubOk) { "PUBLISH FAILED (see $out\publish.log)"; exit 1 }
"published"

# 2. DayZ lab (server + client).
$mode = if ($Driver -eq "server") { 2 } else { 0 }
& "$Tools\make_lab_missions.ps1" -Maps chernarusplus -Mode $mode
python "$Tools\mod\build_mod.py" "$Lab\mods"
# The player is present only when a pose log CREATED BY THIS LAUNCH has rows: the previous run's log is still
# "recent" and once made a run start (and turn the view) while the client was still loading.
$launchTime = Get-Date
& "$Tools\launch_dayzdiag.ps1" -Map chernarusplus -LabMod
$chrome = $null
try {
    # 3. Wait for the player to exist (pose rows in the newest CSV).
    $deadline = (Get-Date).AddSeconds(240)
    $csv = $null
    while ((Get-Date) -lt $deadline) {
        $csv = Get-ChildItem "$Lab\profiles\server\assistz_pose_*.csv" | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending | Select-Object -First 1
        if ($csv -and $csv.CreationTime -gt $launchTime -and (Get-Content $csv.FullName | Measure-Object -Line).Lines -gt 20) { break }
        Start-Sleep -Seconds 3
    }
    if (-not $csv) { throw "no pose CSV" }
    "player present: $($csv.Name)"
    Start-Sleep -Seconds 5

    # 4. Lab Chrome capturing the DayZ client window (video only: auto-select skips the audio toggle).
    $chromeArgs = @("--user-data-dir=$S\chrome-profile", "--remote-debugging-port=9333", "--no-first-run", "--no-default-browser-check",
        "--auto-select-desktop-capture-source=DayZ", "--new-window", "http://127.0.0.1:5390/?lab=1")
    $chrome = Start-Process -FilePath "C:\Program Files\Google\Chrome\Application\chrome.exe" -ArgumentList $chromeArgs -PassThru
    "Chrome PID $($chrome.Id)"
    for ($i = 0; $i -lt 40; $i++) { try { Invoke-WebRequest http://127.0.0.1:9333/json/version -UseBasicParsing -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Milliseconds 250 } }
    $cap = Start-Process -FilePath python -ArgumentList @("$H\capture_live.py", "http://127.0.0.1:5390", "$CaptureSeconds", "$out\shot", "$out\console.txt") `
        -RedirectStandardOutput "$out\capture.txt" -RedirectStandardError "$out\capture_err.txt" -PassThru -NoNewWindow
    Start-Sleep -Seconds 12

    # 5. Turn the view: lab mode 2 turns it from the server; otherwise the mouse driver (click to capture, then moves).
    if ($Driver -eq "mouse") {
        python "$Tools\drive_turns.py" "$Lab\dayzdiag.pids.json" "$out\turns.json" 2>&1 | Tee-Object "$out\drive.txt"
    }

    # 6. Let the capture finish.
    $cap.WaitForExit(($CaptureSeconds + 60) * 1000) | Out-Null
    Copy-Item $csv.FullName "$out\pose.csv"
    $turns = Get-ChildItem "$Lab\profiles\server\assistz_turns_*.csv" | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending | Select-Object -First 1
    # Only a server-driven run has a turns CSV of its own (an older one would be picked up otherwise).
    if ($turns -and $Driver -eq "server") { Copy-Item $turns.FullName "$out\turns.csv" }
    "capture done"
}
finally {
    if ($chrome) { taskkill /PID $chrome.Id /T /F | Out-Null; "Killed Chrome PID $($chrome.Id)" }
    & "$Tools\stop_dayzdiag.ps1"
}
"RUN DIR $out"
