$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
}
else {
    $pythonExe = (Get-Command python -ErrorAction Stop).Source
}

function Test-HttpReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    try {
        Invoke-WebRequest -UseBasicParsing $Url -TimeoutSec 3 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Wait-ForService {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [int]$Retries = 20
    )

    for ($attempt = 0; $attempt -lt $Retries; $attempt++) {
        if (Test-HttpReady -Url $Url) {
            Write-Host "$Name is ready: $Url"
            return
        }
        Start-Sleep -Seconds 1
    }

    Write-Warning "$Name did not respond at $Url yet."
}

function Start-PythonProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Arguments
    )

    $escapedRoot = $root.Replace("'", "''")
    $escapedPython = $pythonExe.Replace("'", "''")
    $command = "Set-Location -LiteralPath '$escapedRoot'; & '$escapedPython' $Arguments"

    $process = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-Command", $command) -WorkingDirectory $root -PassThru
    if (-not $process) {
        throw "Failed to start $Name."
    }

    Write-Host "$Name window started (PID $($process.Id))."
}

Write-Host "Training model..."
& $pythonExe train.py
if ($LASTEXITCODE -ne 0) {
    throw "Training failed."
}

$apiHealthUrl = "http://127.0.0.1:8000/health"
$streamlitHealthUrl = "http://127.0.0.1:8501/_stcore/health"

if (Test-HttpReady -Url $apiHealthUrl) {
    Write-Host "FastAPI is already running: $apiHealthUrl"
}
else {
    Start-PythonProcess -Name "FastAPI" -Arguments "-m uvicorn app.main:app --host 127.0.0.1 --port 8000"
    Wait-ForService -Name "FastAPI" -Url $apiHealthUrl -Retries 30
}

if (Test-HttpReady -Url $streamlitHealthUrl) {
    Write-Host "Streamlit is already running: http://127.0.0.1:8501"
}
else {
    Start-PythonProcess -Name "Streamlit" -Arguments "-m streamlit run ui/streamlit_app.py --server.headless true"
    Wait-ForService -Name "Streamlit" -Url $streamlitHealthUrl
}

Write-Host ""
Write-Host "FastAPI docs: http://127.0.0.1:8000/docs"
Write-Host "Streamlit UI: http://127.0.0.1:8501"
