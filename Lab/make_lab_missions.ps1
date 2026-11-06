# Builds the AssistZ lab missions from Bohemia's offline missions (DayZ-Central-Economy, ADPL-SA: used locally,
# never committed here). Each lab mission = the stock dayzOffline.<map> with the pose logger injected into
# CustomMission. Output: <mission root>\assistzLab.<map> (lab_paths.ps1: under $env:ASSISTZ_LAB_WORK, on a local disk).
param(
    [string[]]$Maps = @("chernarusplus", "nasdara"),
    # 0 = pose log only, 1 = sound beacon (grenades at known bearings), 2 = scripted view turns, 3 = two world-fixed repeating beacons, 4 = lab sniper (server dummy, no client sound), 5 = real shooter client (same-SteamID rejected), 6 = Expansion AI shooter, 7 = Expansion wide misses (East shooter fires at a West decoy beside the listener), 8 = labelled set (three automatic weapons at three distances), 9 = why far shooters do not fire (noise and armed decoy, one at a time), 10 = real footsteps, an unarmed AI walking around the listener, 11 = world events at known times (gas strike, heli crash, sleigh crash), 12 = forced weather and infected at known times, 13, 14, 15 = thunderstorm with its minimum time between strikes set (45 then 15 s; 15 then 45 s; 90 then 30 s), 16 and 17 = infected attacks at known times (two sets of types), then a storm, 18 and 21 = the same in heavy rain, 19, 20, 22, 23 = listener moved across 12 sites (four lists sharing no site), 24 = thunderstorm with no rain, 25 = direction test (Expansion, one AKM at 50 m at 12 bearings all the way round), 26 and 27 = map survey for vision (index grid; offset grid), 28 = distance test (Expansion, three weapons at 30, 55, 80, 95 m all the way round)
    [ValidateSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34)][int]$Mode = 1,
    [string]$LabRoot = "D:\users\tj\Projects\AssistZ\Lab",
    # Weather over any mode: 0 none, 1 storm in heavy rain, 2 dry storm, 3 heavy rain with no storm.
    [ValidateSet(0, 1, 2, 3)][int]$Weather = 0,
    # A map survey resumes from this column (x in metres).
    [int]$MapStartX = 0,
    # Mode 34: the hour of day to set (with clear weather unless -Weather says otherwise); -1 leaves the mission's own.
    [int]$Hour = -1
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lab_paths.ps1")
$missionRoot = Get-LabMissionRoot $LabRoot
$source = Join-Path $LabRoot "DayZ-Central-Economy"
$logger = (Get-Content (Join-Path $PSScriptRoot "assistz_pose_logger.c") -Raw).Replace("AZ_LAB_MODE_VALUE", "$Mode").Replace("AZ_LAB_WEATHER_VALUE", "$Weather").Replace("AZ_MAP_START_X_VALUE", "$MapStartX").Replace("AZ_LAB_HOUR_VALUE", "$Hour")
if (-not (Test-Path $source)) { throw "Clone https://github.com/BohemiaInteractive/DayZ-Central-Economy into $source first." }

foreach ($map in $Maps) {
    $from = Join-Path $source "dayzOffline.$map"
    $to = Join-Path $missionRoot "assistzLab.$map"
    if (Test-Path $to) { Remove-Item $to -Recurse -Force }
    New-Item -ItemType Directory -Force (Split-Path $to) | Out-Null
    Copy-Item $from $to -Recurse
    $init = Join-Path $to "init.c"
    $text = Get-Content $init -Raw
    $anchor = [regex]::Match($text, "class CustomMission\s*:\s*MissionServer\s*\{")
    if (-not $anchor.Success) { throw "CustomMission not found in $init (mission layout changed?)" }
    $at = $anchor.Index + $anchor.Length
    $text = $text.Substring(0, $at) + "`r`n" + $logger + $text.Substring($at)
    Set-Content $init $text -NoNewline -Encoding ascii
    # Mode 6: Expansion's own map patrols off (they fired across our runs: 2026-11-03 run 4, an AUG patrol at 150 m), so
    # every shot is one of ours. The settings file is Expansion's generated one with Enabled = 0, kept beside the lab
    # (Lab\expansion_lab_settings, not committed: Expansion content); Expansion reads it from the mission folder.
    $labSettings = Join-Path $LabRoot "expansion_lab_settings"
    if (($Mode -ge 6) -and (Test-Path $labSettings)) {
        $dest = Join-Path $to "expansion\settings"
        New-Item -ItemType Directory -Force $dest | Out-Null
        Copy-Item (Join-Path $labSettings "*.json") $dest -Force
    }
    "built $to (mode $Mode)"
}
