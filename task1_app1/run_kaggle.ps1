param(
    [Parameter(Mandatory=$true)]
    [string]$FolderName
)

# Execute the first kaggle command
Write-Host "Running: kaggle kernels output akshatatkaggle/task1-app1 -p $FolderName"
kaggle kernels output akshatatkaggle/task1-app1 -p $FolderName

# Execute the second kaggle command
Write-Host "Running: kaggle kernels pull akshatatkaggle/task1-app1 -p $FolderName"
kaggle kernels pull akshatatkaggle/task1-app1 -p $FolderName

Write-Host "Completed!"
