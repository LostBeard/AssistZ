# Stops ONLY the two DayZDiag processes launch_dayzdiag.ps1 recorded (never by image name).
param([string]$LabRoot = "D:\users\tj\Projects\AssistZ\Lab")
$file = Join-Path $LabRoot "dayzdiag.pids.json"
if (-not (Test-Path $file)) { "no recorded lab PIDs"; return }
$pids = Get-Content $file -Raw | ConvertFrom-Json
foreach ($name in "shooter", "client", "server") {
    if (-not $pids.$name) { continue }
    $id = $pids.$name
    $p = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -eq "DayZDiag_x64") { Stop-Process -Id $id -Force; "stopped $name PID $id" }
    else { "$name PID $id not running (or reused by another program: left alone)" }
}
Remove-Item $file
