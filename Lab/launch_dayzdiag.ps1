# Starts the AssistZ lab: a DayZDiag SERVER running the lab mission (Bohemia's dayzOffline missions are server
# missions: a lone client with -mission just waits at "Player connect enabled") and a DayZDiag CLIENT that connects
# to it on 127.0.0.1. DayZDiag = the diagnostic build: no BattlEye, runs mission scripts, -filePatching allowed.
# Logs and the pose CSV go to Lab\profiles\server; client logs to Lab\profiles\client. Both PIDs are recorded so only
# these two processes are ever stopped (stop_dayzdiag.ps1).
param(
    [ValidateSet("chernarusplus", "nasdara")][string]$Map = "chernarusplus",
    # A vision run: keeps -noPause, because frames need no sound and an unfocused client without it opens the pause menu over the view.
    [switch]$Vision,
    # Empty = the first of the lab PCs' Steam libraries that holds DayZDiag.
    [string]$DayZ = "",
    [string]$LabRoot = "D:\users\tj\Projects\AssistZ\Lab",
    [int]$Port = 2302,
    [int]$ClientDelaySeconds = 25,
    # Load the AssistZ lab mod (Lab/mod, packed by Lab/mod/build_mod.py into Lab\mods\@AssistZLab) on both processes.
    [switch]$LabMod,
    # A second client named "Shooter" (lab mode 5), with -nosound so its own shots do not leak into a loopback recording.
    [switch]$Shooter,
    # DayZ-Expansion-AI (unmodified, from the Workshop folder; CC BY-NC-ND 4.0: used, never repacked or altered) for lab
    # mode 6, loaded with its dependencies before the lab mod.
    [switch]$Expansion
)
$ErrorActionPreference = "Stop"
if (-not $DayZ) {
    $DayZ = "E:\SteamLibrary\steamapps\common\DayZ", "L:\SteamLibrary\steamapps\common\DayZ" | Where-Object { Test-Path "$_\DayZDiag_x64.exe" } | Select-Object -First 1
    if (-not $DayZ) { throw "DayZDiag_x64.exe not found on E: or L:. Pass -DayZ." }
}
. (Join-Path $PSScriptRoot "lab_paths.ps1")
$mission = Join-Path (Get-LabMissionRoot $LabRoot) "assistzLab.$Map"
if (-not (Test-Path $mission)) { throw "Run make_lab_missions.ps1 first ($mission missing)." }
$serverProfiles = Join-Path $LabRoot "profiles\server"
$clientProfiles = Join-Path $LabRoot "profiles\client"
New-Item -ItemType Directory -Force $serverProfiles, $clientProfiles | Out-Null
$exe = Join-Path $DayZ "DayZDiag_x64.exe"
$cfg = Join-Path $PSScriptRoot "serverDZ.cfg"
$modArg = @()
$mods = @()
if ($Expansion) {
    $ws = Join-Path $DayZ "!Workshop"
    # Expansion Core requires Dabs Framework (CfgPatches DF_Scripts): a missing addon is a modal "Addon ... requires
    # addon ..." popup on TJ's screen (2026-11-03), so find the mod that provides it before starting anything.
    $dabs = Get-ChildItem $ws -Directory | Where-Object { $_.Name -match "dabs" } | Select-Object -First 1
    if (-not $dabs) { throw "Dabs Framework (DF_Scripts, required by DayZ-Expansion-Core) not found in $ws - subscribe to 'Dabs Framework' on the Workshop" }
    foreach ($m in "@CF", $dabs.Name, "@DayZ-Expansion-Core", "@DayZ-Expansion-AI") {
        if (-not (Test-Path (Join-Path $ws $m))) { throw "$m not found in $ws (subscribe on the Workshop)" }
        $mods += (Join-Path $ws $m)
    }
}
if ($LabMod) {
    $modDir = Join-Path $LabRoot "mods\@AssistZLab"
    if (-not (Test-Path "$modDir\addons\AssistZLab.pbo")) { throw "Run Lab\mod\build_mod.py $LabRoot\mods first" }
    $mods += $modDir
}
if ($mods.Count -gt 0) { $modArg = @("`"-mod=" + ($mods -join ";") + "`"") }

$serverArgs = @("-server", "-config=$cfg", "-port=$Port", "-mission=$mission", "-profiles=$serverProfiles",
    "-filePatching", "-doLogs", "-adminLog", "-freezeCheck") + $modArg
$server = Start-Process -FilePath $exe -WorkingDirectory $DayZ -ArgumentList $serverArgs -PassThru
"server PID $($server.Id)"

# Wait for the server's scripts: a compile error in the mission or the lab mod shows a dialog and crashes the
# process, and clients started anyway each show the same dialog (2026-11-03: three dialogs for one typo). Start the
# clients only after the server's script log shows the Mission module loaded; on "Can't compile" stop here.
$serverStart = Get-Date
$compiled = $false
for ($i = 0; $i -lt 120; $i++) {
    $log = Get-ChildItem "$serverProfiles\script_*.log" -ErrorAction SilentlyContinue | Where-Object { $_.CreationTime -gt $serverStart.AddSeconds(-2) } | Sort-Object CreationTime -Descending | Select-Object -First 1
    if ($log) {
        $text = Get-Content $log.FullName -Raw -ErrorAction SilentlyContinue
        if ($text -match "Can't compile") {
            $err = ($text -split "`n" | Where-Object { $_ -match "Can't compile|\.c\(\d+\)" } | Select-Object -First 3) -join " | "
            if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
            throw "server script compile error, clients not started: $err"
        }
        if ($text -match "Module: Mission") { $compiled = $true; break }
    }
    if ($server.HasExited) { throw "server exited before its scripts compiled" }
    Start-Sleep -Milliseconds 500
}
if (-not $compiled) { if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }; throw "server scripts did not finish compiling in 60 s" }
"server scripts compiled"
# Give the server time to load the world and the economy before the client tries to join.
Start-Sleep -Seconds $ClientDelaySeconds
# The server can stop on its own after its scripts compile (a failed economy database gives "termination in: 5"). A client started
# then only waits at the main menu, and a run records that menu's audio.
if ($server.HasExited) {
    $rpt = Get-ChildItem "$serverProfiles\*.RPT" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $why = if ($rpt) { (Select-String -Path $rpt.FullName -Pattern '!!! \[CE\]|Initialization failed|termination in: 5' | Select-Object -First 3 | ForEach-Object { $_.Line }) -join " | " } else { "" }
    throw "server exited before the client started: $why"
}
$clientArgs = @("-connect=127.0.0.1", "-port=$Port", "-name=AssistZLab", "-profiles=$clientProfiles",
    "-filePatching", "-window", "-nosplash", "-doLogs") + $modArg
# Keep the sound when the window loses focus, or a run goes digitally silent and is wasted. The menu's "Update In Background" is pauseMode: 2 = graphics and sound. -noPause must stay off this client: with it, sound stopped at 2 and at 0. Measured with Notepad taking focus for 60 s:
# 2 with -noPause, 0 with and without it, and -noPauseSound (which does nothing): silence for those 60 s. 2 without -noPause: no silence. The lab profile only, never the player's own.
Get-ChildItem (Join-Path $clientProfiles "Users\AssistZLab") -Filter '*_settings.DayZProfile' -ErrorAction SilentlyContinue | ForEach-Object {
    $settings = [System.IO.File]::ReadAllText($_.FullName)
    [System.IO.File]::WriteAllText($_.FullName, ($settings -replace 'pauseMode=\d+;', "pauseMode=2;"))
}
if ($Vision) { $clientArgs += "-noPause" }
# The window must fit the screen: this PC's desktop is 1680x1050, and the part of a 1920x1080 window off screen draws white in a capture. The lab client's own config only.
$screen = Join-Path $clientProfiles "Users\$env:USERNAME\DayZ.cfg"
if (Test-Path $screen) {
    $cfg2 = [System.IO.File]::ReadAllText($screen)
    [System.IO.File]::WriteAllText($screen, ($cfg2 -replace 'WindowWidth=\d+;', 'WindowWidth=1600;' -replace 'WindowHeight=\d+;', 'WindowHeight=900;' -replace 'WinX=-?\d+;', 'WinX=0;' -replace 'WinY=-?\d+;', 'WinY=0;'))
}
$client = Start-Process -FilePath $exe -WorkingDirectory $DayZ -ArgumentList $clientArgs -PassThru
"client PID $($client.Id)"

$pids = @{ server = $server.Id; client = $client.Id }
if ($Shooter) {
    Start-Sleep -Seconds 20
    $shooterProfiles = Join-Path $LabRoot "profiles\shooter"
    New-Item -ItemType Directory -Force $shooterProfiles | Out-Null
    $shooterArgs = @("-connect=127.0.0.1", "-port=$Port", "-name=Shooter", "-profiles=$shooterProfiles",
        "-filePatching", "-window", "-nosplash", "-noPause", "-nosound", "-doLogs") + $modArg
    $shooterProc = Start-Process -FilePath $exe -WorkingDirectory $DayZ -ArgumentList $shooterArgs -PassThru
    "shooter client PID $($shooterProc.Id)"
    $pids.shooter = $shooterProc.Id
}
$pids | ConvertTo-Json | Out-File (Join-Path $LabRoot "dayzdiag.pids.json")
