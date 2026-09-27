param(
    [switch]$Clean
)

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

$compiler = Get-Command latexmk -ErrorAction SilentlyContinue
if ($null -ne $compiler) {
    if ($Clean) {
        & latexmk -C main.tex
    }
    & latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    exit $LASTEXITCODE
}

$compiler = Get-Command pdflatex -ErrorAction SilentlyContinue
if ($null -eq $compiler) {
    $localPdflatex = Join-Path $env:LOCALAPPDATA 'Programs\MiKTeX\miktex\bin\x64\pdflatex.exe'
    if (Test-Path $localPdflatex) {
        $compiler = Get-Item $localPdflatex
    } else {
        Write-Error 'No LaTeX compiler found. Install MiKTeX or TeX Live, restart the terminal, and run this script again.'
    }
}

if ($Clean) {
    Remove-Item -ErrorAction SilentlyContinue *.aux, *.log, *.out, *.toc, *.synctex.gz
}

$pdflatex = $compiler.Source
if ([string]::IsNullOrWhiteSpace($pdflatex)) { $pdflatex = $compiler.FullName }
& $pdflatex -interaction=nonstopmode -halt-on-error main.tex
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$bibtex = Join-Path (Split-Path $pdflatex) 'bibtex.exe'
if (-not (Test-Path $bibtex)) { $bibtex = 'bibtex' }
& $bibtex main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pdflatex -interaction=nonstopmode -halt-on-error main.tex
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pdflatex -interaction=nonstopmode -halt-on-error main.tex
exit $LASTEXITCODE
