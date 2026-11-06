# Unattended crack-to-report run: DayZ lab mode 4 (server-fired SVD at 50/100/200/400 m, per range 2 near misses at
# 2 m from the head and 2 wide misses at 25 m: the close-only bullet noise is a "being shot at" cue), the
# PC's audio recorded by WASAPI loopback (no browser, no picker), then analyze_shots.py. Ask every GPU peer first.
param([int]$RecordSeconds = 90, [switch]$RealShooter, [switch]$Expansion, [switch]$WideMiss, [switch]$LabelSet, [switch]$EngageTest, [switch]$Footsteps, [switch]$WorldEvents, [switch]$WeatherLab, [switch]$DirectionTest, [switch]$DistanceTest, [switch]$FarTest, [switch]$InfectedStates, [switch]$CalmInfected, [switch]$OwnSounds, [switch]$NewWeapons, [switch]$SeeInfected, [ValidateSet(0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27)][int]$StormLab = 0, [int]$PlayerWaitSeconds = 240, [ValidateSet("chernarusplus", "nasdara")][string]$Map = "chernarusplus", [ValidateSet(0, 1, 2, 3)][int]$Weather = 0, [string]$FrameDir = "", [double]$FramesPerSecond = 2, [int]$MapStartX = 0, [int]$Hour = -1)
# -MapStartX resumes a map survey from that column.
# -Weather 1|2|3 = weather over any mode: storm in heavy rain, dry storm, heavy rain with no storm. So -LabelSet is labelled gunfire in that weather.
# -Map nasdara: the held-out map. Modes that use their own spawn work there (12-18, 21); the survey lists (19, 20, 22, 23) are Chernarus coordinates.
# -StormLab 13|14 = thunderstorm with its minimum time between strikes set: 45 then 15 s, or 15 then 45 s, 290 s each, then the same rain with no storm; so -RecordSeconds 800. 15 = 90 s for 500 s, then 30 s for 300 s; so -RecordSeconds 1000. 16|17 = infected attacks at known times, two sets of types, then a storm; so -RecordSeconds 600. 18|21 = the same in heavy rain, -RecordSeconds 600. 19|20|22|23 = a survey of 12 sites, 55 s each, -RecordSeconds 700. 24 = thunderstorm with no rain, 15 then 30 s, then overcast with no storm, -RecordSeconds 800.
# -WeatherLab = lab mode 12 (clear, heavy rain, thunderstorm, clear, then infected at 10/25/50 m; so -RecordSeconds 800).
# -WorldEvents = lab mode 11 (heli and sleigh crashes 45 s apart, then gas strikes 220 s apart, at 300/1000/2500 m; a strike sounds minutes after its trigger, so -RecordSeconds 1000).
# -Footsteps = lab mode 10 (Expansion, an unarmed AI walks around the listener, no gunfire; 4 x 35 s, so -RecordSeconds 190).
# -EngageTest = lab mode 9 (Expansion, far shooters: noise and armed decoy, one at a time; 6 x 35 s, so -RecordSeconds 240).
# -StormLab 26|27 = map survey for the Vision Module; with -FrameDir the client window draws itself to JPEGs while recording; other windows on top do not matter. LOOK at the frames: a first version saved the operator's terminal for a whole run and every other check passed.
# -SeeInfected = lab mode 34 (one infected at a time in view, placed at 10 to 120 m; with -FrameDir, frames for the video work; -RecordSeconds 800). -Hour 12 sets noon and clear weather; add -Weather 3 for rain at that hour.
# -NewWeapons = lab mode 33 (the lab's own shooter: Mosin, Winchester, FAL, MP5K, CZ75, FNX45 at 50, 150, 300, 600 m, 4 shots each; about 650 s, so -RecordSeconds 690). Then Lab/far_to_shots.py <run>.
# -OwnSounds = lab mode 32 (the character alone with a loaded rifle; Lab/harness/drive_own.py drives it while this records).
# -CalmInfected = lab mode 31 (three infected at once at 15, 25, 40 m, 12 slots; env.csv logs the worst state of any; -RecordSeconds 1000).
# -InfectedStates = lab mode 30 (no gunfire: one infected at a time at 4, 10, 20, 40 m, 12 types; env.csv logs its mind state; -RecordSeconds 640).
# -FarTest = lab mode 29 (the lab's own shooter, no Expansion: AKM, AUG, M4A1, SVD at 150, 200, 300, 400, 600, 800 m, 4 shots each, round the listener; about 650 s, so -RecordSeconds 690). Then Lab/far_to_shots.py <run>.
# -DistanceTest = lab mode 28 (Expansion, three weapons at 30, 55, 80, 95 m, all the way round; 12 x 35 s, so -RecordSeconds 450).
# -DirectionTest = lab mode 25 (Expansion, one AKM at 50 m at 12 bearings all the way round; 12 x 35 s, so -RecordSeconds 450).
# -LabelSet = lab mode 8 (Expansion, three automatic weapons at three distances; 9 x 35 s, so -RecordSeconds 360).
# -WideMiss = lab mode 7 (Expansion, East shooter firing at a West decoy beside the listener; offsets: AzDecoyOffset).
$ErrorActionPreference = "Stop"
$H = $PSScriptRoot
$Tools = Split-Path $H -Parent
$Lab = "D:\users\tj\Projects\AssistZ\Lab"
$out = "$Lab\runs\sniper_" + (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
New-Item -ItemType Directory -Force $out | Out-Null

if ($WideMiss -or $LabelSet -or $EngageTest -or $Footsteps -or $DirectionTest -or $DistanceTest) { $Expansion = $true }
$mode = if ($SeeInfected) { 34 } elseif ($NewWeapons) { 33 } elseif ($OwnSounds) { 32 } elseif ($CalmInfected) { 31 } elseif ($InfectedStates) { 30 } elseif ($FarTest) { 29 } elseif ($DistanceTest) { 28 } elseif ($DirectionTest) { 25 } elseif ($StormLab -ne 0) { $StormLab } elseif ($WeatherLab) { 12 } elseif ($WorldEvents) { 11 } elseif ($Footsteps) { 10 } elseif ($EngageTest) { 9 } elseif ($LabelSet) { 8 } elseif ($WideMiss) { 7 } elseif ($Expansion) { 6 } elseif ($RealShooter) { 5 } else { 4 }
& "$Tools\make_lab_missions.ps1" -Maps $Map -Mode $mode -Weather $Weather -MapStartX $MapStartX -Hour $Hour
python "$Tools\mod\build_mod.py" "$Lab\mods"
# The player is present only when a pose log CREATED BY THIS LAUNCH has rows: the previous run's log is still
# "recent" and once made a run start (and turn the view) while the client was still loading.
$launchTime = Get-Date
if ($Expansion) { & "$Tools\launch_dayzdiag.ps1" -Map $Map -LabMod -Expansion }
elseif ($RealShooter) { & "$Tools\launch_dayzdiag.ps1" -Map $Map -LabMod -Shooter }
elseif ($FrameDir) { & "$Tools\launch_dayzdiag.ps1" -Map $Map -LabMod -Vision }
else { & "$Tools\launch_dayzdiag.ps1" -Map $Map -LabMod }
try {
    $deadline = (Get-Date).AddSeconds($PlayerWaitSeconds)
    $csv = $null
    $present = $false
    while ((Get-Date) -lt $deadline) {
        $csv = Get-ChildItem "$Lab\profiles\server\assistz_pose_*.csv" | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending | Select-Object -First 1
        if ($csv -and $csv.CreationTime -gt $launchTime -and (Get-Content $csv.FullName | Measure-Object -Line).Lines -gt 20) { $present = $true; break }
        Start-Sleep -Seconds 3
    }
    # A pose log with no rows means nobody joined. Recording anyway once produced 190 s of main menu audio that looked like a run.
    if (-not $present) { Remove-Item $out -Recurse -Force; throw "the player never appeared in $PlayerWaitSeconds s: no recording made" }
    "player present: $($csv.Name); recording $RecordSeconds s"
    $grabber = $null
    if ($FrameDir) { $grabber = Start-Process python -ArgumentList @("`"$Tools\grab_frames.py`"", "`"$FrameDir`"", "$RecordSeconds", "$FramesPerSecond") -PassThru -NoNewWindow -RedirectStandardOutput "$out\grab.txt" -RedirectStandardError "$out\grab_err.txt" }
    # A vision run uses -noPause, so its sound stops without focus, and it needs none: no recording and no audio check.
    if ($FrameDir) {
        # Ends on the grabber's timer, when the window goes, or when the player's pose log stops growing: a Steam error dropped the client to its main screen, which still draws.
        $audioOk = $true
        $seenLength = (Get-Item $csv.FullName).Length
        $seenAt = Get-Date
        while (-not $grabber.WaitForExit(10000)) {
            $length = (Get-Item $csv.FullName).Length
            if ($length -ne $seenLength) { $seenLength = $length; $seenAt = Get-Date }
            elseif (((Get-Date) - $seenAt).TotalSeconds -gt 120) {
                "player gone: the pose log has not grown for 120 s; stopping this run"
                Stop-Process -Id $grabber.Id -Force
                break
            }
        }
    }
    else { python "$Tools\record_loopback.py" $RecordSeconds "$out\audio.wav"; python "$Tools\check_audio.py" $out; $audioOk = $LASTEXITCODE -eq 0 }
    if ($grabber) { $grabber.WaitForExit(30000) | Out-Null; if (-not $grabber.HasExited) { Stop-Process -Id $grabber.Id -Force } }
    Copy-Item $csv.FullName "$out\pose.csv"
    $shots = Get-ChildItem "$Lab\profiles\server\assistz_shots_*.csv" | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending | Select-Object -First 1
    if ($shots) { Copy-Item $shots.FullName "$out\shots.csv" }
    $aipose = Get-ChildItem "$Lab\profiles\server\assistz_aipose_*.csv" -ErrorAction SilentlyContinue | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending | Select-Object -First 1
    if ($aipose) { Copy-Item $aipose.FullName "$out\aipose.csv" }
    $envLog = Get-ChildItem "$Lab\profiles\server\assistz_env_*.csv" -ErrorAction SilentlyContinue | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending | Select-Object -First 1
    if ($envLog) { Copy-Item $envLog.FullName "$out\env.csv" }
    $worldLog = Get-ChildItem "$Lab\profiles\server\assistz_world_*.csv" -ErrorAction SilentlyContinue | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending | Select-Object -First 1
    if ($worldLog) { Copy-Item $worldLog.FullName "$out\world.csv" }
    foreach ($side in "server", "client") {
        if (Test-Path "$Lab\profiles\$side\assistz_aishots.csv") { Move-Item "$Lab\profiles\$side\assistz_aishots.csv" "$out\aishots_$side.csv" -Force }
        if (Test-Path "$Lab\profiles\$side\assistz_impacts.csv") { Move-Item "$Lab\profiles\$side\assistz_impacts.csv" "$out\impacts_$side.csv" -Force }
    }
    $script = Get-ChildItem "$Lab\profiles\server\script_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($script) { Select-String -Path $script.FullName -Pattern "AssistZ|Can't compile|init.c\(" | Select-Object -First 30 | Out-File "$out\server_script.txt" }
    $shooterScript = Get-ChildItem "$Lab\profiles\shooter\script_*.log" -ErrorAction SilentlyContinue | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending | Select-Object -First 1
    if ($shooterScript) { Select-String -Path $shooterScript.FullName -Pattern "AssistZLab|Can't compile|\(E\)" | Select-Object -First 30 | Out-File "$out\shooter_script.txt" }
    $clientScript = Get-ChildItem "$Lab\profiles\client\script_*.log" | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending | Select-Object -First 1
    if ($clientScript) { Select-String -Path $clientScript.FullName -Pattern "AssistZLab|Can't compile|\(E\)" | Select-Object -First 30 | Out-File "$out\client_script.txt" }
}
finally {
    & "$Tools\stop_dayzdiag.ps1"
}
Get-Content "$out\server_script.txt" -ErrorAction SilentlyContinue | Select-Object -First 8
"--- shooter"
Get-Content "$out\shooter_script.txt" -ErrorAction SilentlyContinue | Select-Object -First 10
"--- client"
Get-Content "$out\client_script.txt" -ErrorAction SilentlyContinue | Select-Object -First 8
# A warning on stderr from this old analysis must not end a run whose recording is fine: with Stop, it did.
if (Test-Path "$out\shots.csv") { try { $ErrorActionPreference = "Continue"; python "$Tools\analyze_shots.py" "$out\audio.npy" "$out\shots.csv" "$out\pose.csv" 2>$null } catch { "analyze_shots failed: $_" } }
# After DayZ is stopped: a silent recording is renamed so no sniper_* glob treats it as data.
if (-not $audioOk) {
    $bad = Join-Path (Split-Path $out) ("invalid_silent_" + (Split-Path $out -Leaf))
    Rename-Item $out $bad
    throw "the recording went silent, run set aside as $bad"
}
if ($FrameDir) {
    $vision = Join-Path (Split-Path $out) ("vision_" + (Split-Path $out -Leaf))
    Rename-Item $out $vision
    $out = $vision
}
"RUN DIR $out"
