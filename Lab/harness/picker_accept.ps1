# Does what a person does in Chrome's share picker, through Windows UI Automation: the Window tab, the source whose title matches, audio on, Share.
# Chrome's --auto-select-desktop-capture-source flag picks a source but leaves audio off, so the app gets video and hears nothing (first live check, 2026-11-05).
# -Audio chooses the toggles to turn on, by label; every other audio toggle on screen is turned off. Windows 11 with windowAudio offers both "Also share app audio" and "Share with system audio", and system audio would hide whether app audio isolates.
param([string]$Match = 'DayZ', [int]$WaitSeconds = 180, [string]$Audio = '*audio*')
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$A = [System.Windows.Automation.AutomationElement]
$walker = [System.Windows.Automation.TreeWalker]::RawViewWalker
$Name = $A::NameProperty

function Find-All($e, $name, $cls) {
    $out = @()
    $k = $walker.GetFirstChild($e)
    while ($k) {
        try { if ((-not $name -or $k.Current.Name -like $name) -and (-not $cls -or $k.Current.ClassName -eq $cls)) { $out += $k } } catch {}
        $out += Find-All $k $name $cls
        $k = $walker.GetNextSibling($k)
    }
    return $out
}

# The picker: a child of the Chrome window on Windows 10 with Chrome 157; it may be its own top-level window elsewhere, so look at those and a few levels down.
function Find-Named($e, $depth) {
    try { if ($e.Current.Name -match 'Choose what to share') { return $e } } catch {}
    if ($depth -le 0) { return $null }
    $k = $walker.GetFirstChild($e)
    while ($k) {
        $hit = Find-Named $k ($depth - 1)
        if ($hit) { return $hit }
        $k = $walker.GetNextSibling($k)
    }
    return $null
}

function Find-Picker {
    $kid = $walker.GetFirstChild($A::RootElement)
    while ($kid) {
        try {
            if ($kid.Current.ClassName -match 'Chrome_WidgetWin') {
                $hit = Find-Named $kid 4
                if ($hit) { return $hit }
            }
        } catch {}
        $kid = $walker.GetNextSibling($kid)
    }
    return $null
}

# What was there, for when the picker is not found.
function Show-Chrome {
    $kid = $walker.GetFirstChild($A::RootElement)
    while ($kid) {
        try {
            if ($kid.Current.ClassName -match 'Chrome_WidgetWin') {
                "top '$($kid.Current.Name)' [$($kid.Current.ClassName)] offscreen=$($kid.Current.IsOffscreen) pid=$($kid.Current.ProcessId)"
                $w = $walker.GetFirstChild($kid)
                while ($w) {
                    "   child '$($w.Current.Name)' [$($w.Current.ClassName)]"
                    $w = $walker.GetNextSibling($w)
                }
            }
        } catch {}
        $kid = $walker.GetNextSibling($kid)
    }
}

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$picker = $null
while ((Get-Date) -lt $deadline -and -not $picker) {
    Start-Sleep -Milliseconds 500
    $picker = Find-Picker
}
if (-not $picker) { "PICKER NEVER APPEARED. Chrome's windows at the end:"; Show-Chrome; exit 1 }
"picker open"
$tab = Find-All $picker 'Window' 'TabbedPaneTab' | Select-Object -First 1
if (-not $tab) { "NO WINDOW TAB"; exit 1 }
$tab.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
Start-Sleep -Milliseconds 800
# The list fills in over time: a game window was missing on the first look at least once.
$source = $null
$until = (Get-Date).AddSeconds(30)
while (-not $source -and (Get-Date) -lt $until) {
    $sources = Find-All $picker $null 'DesktopMediaSourceView'
    $source = $sources | Where-Object { $_.Current.Name -match $Match } | Select-Object -First 1
    if (-not $source) { Start-Sleep -Milliseconds 800 }
}
"sources: " + (($sources | ForEach-Object { $_.Current.Name }) -join ' | ')
if (-not $source) { "NO SOURCE MATCHES $Match"; exit 1 }
"choosing '$($source.Current.Name)'"
$source.SetFocus()
$source.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
Start-Sleep -Milliseconds 800
# Each pane has its own toggle; the Window pane's is the one that is on screen now.
$toggles = Find-All $picker '*audio*' 'ToggleButton' | Where-Object { -not $_.Current.IsOffscreen }
"audio toggles on screen: $($toggles.Count)"
foreach ($t in $toggles) {
    $p = $t.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
    "  '$($t.Current.Name)' state $($p.Current.ToggleState)"
    $want = if ($t.Current.Name -like $Audio) { [System.Windows.Automation.ToggleState]::On } else { [System.Windows.Automation.ToggleState]::Off }
    if ($p.Current.ToggleState -ne $want) { $p.Toggle(); Start-Sleep -Milliseconds 300; "  now $($p.Current.ToggleState)" }
}
$buttons = Find-All $picker 'Share*' 'MdTextButton'
"share button: " + (($buttons | ForEach-Object { "'$($_.Current.Name)' enabled=$($_.Current.IsEnabled)" }) -join ' | ')
$share = $buttons | Select-Object -First 1
if (-not $share -or -not $share.Current.IsEnabled) { "SHARE NOT AVAILABLE"; exit 1 }
$share.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
"SHARED"
