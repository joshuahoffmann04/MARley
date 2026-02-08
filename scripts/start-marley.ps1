param(
    [string]$BindAddress = "127.0.0.1",
    [int]$Port = 8010,
    [switch]$NoReload
)

$reloadArgs = @()
if (-not $NoReload) {
    $reloadArgs = @(
        "--reload",
        "--reload-dir", "MARley",
        "--reload-dir", "generator",
        "--reload-dir", "retrieval",
        "--reload-dir", "chunker",
        "--reload-dir", "pdf_extractor",
        "--reload-exclude", ".venv/*",
        "--reload-exclude", ".venv/**",
        "--reload-exclude", "data/**/databases/**"
    )
}

python -m uvicorn MARley.app:app --host $BindAddress --port $Port @reloadArgs
