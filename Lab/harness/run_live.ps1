param([string]$Source = "DayZ", [int]$Seconds = 40, [string]$Shot = "dayz", [string]$ConsoleOut = "", [switch]$NameSounds, [string]$Picker = "", [string]$Query = "", [string]$PickerAudio = "*audio*")
# -Picker <title match> uses Chrome's real share picker (picker_accept.ps1): the auto-select flag leaves audio off, so the app would hear nothing.
# Work folder: publish output, lab Chrome profile, screenshots. Override with $env:ASSISTZ_LAB_WORK.
$S = if ($env:ASSISTZ_LAB_WORK) { $env:ASSISTZ_LAB_WORK } else { Join-Path $env:TEMP "assistz-lab" }
New-Item -ItemType Directory -Force $S | Out-Null
$H = $PSScriptRoot  # the harness scripts
$chromeArgs = @("--user-data-dir=$S\chrome-profile", "--remote-debugging-port=9333", "--no-first-run", "--no-default-browser-check",
    "--new-window", $(if ($NameSounds) { "http://127.0.0.1:5390/?lab=1$Query" } else { "http://127.0.0.1:5390/" }))
if ($env:ASSISTZ_CHROME_FLAG) { $chromeArgs += $env:ASSISTZ_CHROME_FLAG }
if ($Source -and -not $Picker) { $chromeArgs += "--auto-select-desktop-capture-source=$Source" }
$p = Start-Process -FilePath "C:\Program Files\Google\Chrome\Application\chrome.exe" -ArgumentList $chromeArgs -PassThru
"Chrome PID $($p.Id)"
try {
    for ($i = 0; $i -lt 40; $i++) {
        try { Invoke-WebRequest http://127.0.0.1:9333/json/version -UseBasicParsing -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Milliseconds 250 }
    }
    Push-Location $S
    $accept = $null
    if ($Picker) { $accept = Start-Process powershell -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$H\picker_accept.ps1", '-Match', "`"$Picker`"", '-Audio', "`"$PickerAudio`"") -PassThru -NoNewWindow -RedirectStandardOutput "$S\picker_accept.log" -RedirectStandardError "$S\picker_accept_err.log" }
    if ($NameSounds) { python "$H\capture_live.py" http://127.0.0.1:5390 $Seconds "$S\$Shot" "$ConsoleOut" name }
    else { python "$H\capture_live.py" http://127.0.0.1:5390 $Seconds "$S\$Shot" }
    Pop-Location
}
finally {
    taskkill /PID $p.Id /T /F | Out-Null
    "Killed Chrome PID $($p.Id)"
}
