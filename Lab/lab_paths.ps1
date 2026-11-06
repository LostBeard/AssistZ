# Where the lab missions live. A mission must be on a LOCAL disk: DayZ keeps its Central Economy database in the mission's storage
# folder, and on a network share (NV1080's D: is \\Yoda\TData, where fsync fails) it dies at once with "[CE][offlineDB] :: Can't
# prepare data structure: disk I/O error", then "termination in: 5", and the client waits forever (2026-11-03).
function Get-LabMissionRoot([string]$LabRoot) {
    $root = if ($env:ASSISTZ_LAB_WORK) { Join-Path $env:ASSISTZ_LAB_WORK "missions" } else { Join-Path $LabRoot "missions" }
    $drive = (Split-Path -Qualifier $root)
    $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$drive'"
    if ($disk -and $disk.DriveType -eq 4) {
        throw "$root is on a network drive: DayZ's economy database fails there. Set `$env:ASSISTZ_LAB_WORK to a folder on a local disk."
    }
    $root
}
