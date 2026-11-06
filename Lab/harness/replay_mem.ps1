# One replay with Chrome's memory sampled once a second: GPU dedicated and shared memory by process, and private bytes. Read-only on everything but the replay's own output.
param([Parameter(Mandatory)][string]$Run, [Parameter(Mandatory)][string]$Tag, [string]$Query = "&embed=1")
$env:ASSISTZ_LAB_WORK = 'L:\AssistZLabWork'
$R = 'D:\users\tj\Projects\AssistZ\AssistZ'
$log = "L:\AssistZLabWork\memsample$Tag.csv"
$sampler = Start-Job -ArgumentList $log -ScriptBlock {
    param($log)
    "t,dedicated_mb,shared_mb,chrome_private_mb" | Out-File $log -Encoding ascii
    $t0 = Get-Date
    while ($true) {
        $ded = 0; $sh = 0
        try {
            $ids = (Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -match 'remote-debugging-port=9333|chrome-profile' }).ProcessId
            $all = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage', '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples
            foreach ($c in $all) {
                if ($c.InstanceName -match 'pid_(\d+)_' -and $ids -contains [int]$Matches[1]) {
                    if ($c.Path -match 'dedicated') { $ded += $c.CookedValue } else { $sh += $c.CookedValue }
                }
            }
            $priv = 0; foreach ($i in $ids) { $p = Get-Process -Id $i -ErrorAction SilentlyContinue; if ($p) { $priv += $p.PrivateMemorySize64 } }
            "$([int]((Get-Date)-$t0).TotalSeconds),$([int]($ded/1MB)),$([int]($sh/1MB)),$([int]($priv/1MB))" | Out-File $log -Append -Encoding ascii
        } catch {}
        Start-Sleep -Seconds 1
    }
}
$t0 = Get-Date
& "$R\Lab\harness\run_replay.ps1" -Run $Run -Tag $Tag -Query $Query *> "L:\AssistZLabWork\replay$Tag.log"
"replay wall $([int]((Get-Date)-$t0).TotalSeconds) s"
Stop-Job $sampler; Remove-Job $sampler -Force
Get-Content "L:\AssistZLabWork\replay$Tag.log" -Tail 2 | ForEach-Object { $_.Substring(0,[Math]::Min(150,$_.Length)) }
$rows = Import-Csv $log
"samples $($rows.Count); peak dedicated $(($rows.dedicated_mb | ForEach-Object {[int]$_} | Measure-Object -Maximum).Maximum) MB, peak shared $(($rows.shared_mb | ForEach-Object {[int]$_} | Measure-Object -Maximum).Maximum) MB, peak private $(($rows.chrome_private_mb | ForEach-Object {[int]$_} | Measure-Object -Maximum).Maximum) MB"
$half = $rows | Select-Object -Skip ([int]($rows.Count/2))
"second half median-ish: dedicated $(($half.dedicated_mb | ForEach-Object {[int]$_} | Sort-Object)[[int]($half.Count/2)]) MB, shared $(($half.shared_mb | ForEach-Object {[int]$_} | Sort-Object)[[int]($half.Count/2)]) MB"
