function Invoke-PythonLogged {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)][string]$PythonPath,
        [Parameter(Mandatory=$true)][string[]]$Arguments,
        [Parameter(Mandatory=$true)][string]$WorkingDirectory,
        [Parameter(Mandatory=$true)][string]$OutputDirectory,
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][ValidateRange(1,2147483647)][int]$TimeoutMs,
        [string]$ExpectedResultPath
    )

    $stdout = Join-Path $OutputDirectory "$Name.stdout.log"
    $stderr = Join-Path $OutputDirectory "$Name.stderr.log"
    $process = Start-Process -FilePath $PythonPath -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $script:activeProcess = $process
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $timedOut = $false
    try {
        while ($true) {
            $remaining = [Math]::Max(1, $TimeoutMs - [int]$watch.ElapsedMilliseconds)
            $waitMs = [Math]::Min(1000, $remaining)
            if ($process.WaitForExit($waitMs)) { break }
            # Progress belongs on the host/information stream, never the success
            # output captured as the process result by the caller.
            Write-Host "WAITING $Name elapsed_seconds=$([int]$watch.Elapsed.TotalSeconds)"
            if ($watch.ElapsedMilliseconds -ge $TimeoutMs) {
                $timedOut = $true
                $process.Kill($true)
                $process.WaitForExit()
                break
            }
        }
        $process.Refresh()
        $exited = [bool]$process.HasExited
        $exitCode = if ($exited) { [int]$process.ExitCode } else { $null }
        $resultPresent = if ($ExpectedResultPath) { [bool](Test-Path -LiteralPath $ExpectedResultPath -PathType Leaf) } else { $null }
        $status = if ($timedOut) { 'timed_out' }
                  elseif (-not $exited) { 'still_running' }
                  elseif ($exitCode -ne 0) { 'nonzero_exit' }
                  elseif ($ExpectedResultPath -and -not $resultPresent) { 'result_missing' }
                  else { 'success' }
        return [pscustomobject][ordered]@{
            name = $Name
            pid = [int]$process.Id
            started = $true
            exit_code = $exitCode
            exited = $exited
            timed_out = $timedOut
            status = $status
            result_present = $resultPresent
            stdout_path = $stdout
            stderr_path = $stderr
        }
    }
    finally {
        if (-not $process.HasExited) {
            $process.Kill($true)
            $process.WaitForExit()
        }
        $script:activeProcess = $null
    }
}
