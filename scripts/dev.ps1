# Launch BiasCure locally for testing. Usage: ./scripts/dev.ps1
# Opens http://localhost:8501 . Runs in DEMO MODE unless GROQ_API_KEY is set
# (env var, or .streamlit/secrets.toml which is gitignored).
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\streamlit.exe"
if (-not (Test-Path $py)) {
  Write-Host "No venv found. Creating one (Python 3.13)..."
  py -3.13 -m venv (Join-Path $root ".venv")
  & (Join-Path $root ".venv\Scripts\python.exe") -m pip install -q streamlit pypdf python-docx
}
& $py run (Join-Path $root "app.py")
