$count = 0
Get-ChildItem -Directory -Filter "OAS1_*_MR1" | ForEach-Object {
    $t88 = Join-Path $_.FullName "PROCESSED\MPRAGE\T88_111"
    if (Test-Path $t88) {
        $nii = Get-ChildItem -Path $t88 -Filter "*.nii.gz"
        if ($nii.Count -gt 0) {
            $count++
        }
    }
}
Write-Host "Subjects with .nii.gz files: $count"
