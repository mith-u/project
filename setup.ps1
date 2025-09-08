# setup_structure.ps1

$folders = @(
    "data",
    "logs",
    "configs",
    "artifacts",
    "src",
    "src\simulate",
    "src\ingest",
    "src\features",
    "src\models",
    "src\eval",
    "src\app",
    "artifacts\models",
    "artifacts\reports"
)

foreach ($f in $folders) {
    if (-not (Test-Path $f)) {
        mkdir $f | Out-Null
    }
}

# Create empty __init__.py files for Python modules
$pyFolders = @("src\simulate","src\ingest","src\features","src\models","src\eval","src\app")

foreach ($f in $pyFolders) {
    $initFile = "$f\__init__.py"
    if (-not (Test-Path $initFile)) {
        ni $initFile -ItemType File | Out-Null
    }
}
