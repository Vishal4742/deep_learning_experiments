$ErrorActionPreference = "Continue"

# Clear potentially corrupted Python environment variables
Remove-Item Env:\PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue

# Create Output directory if it doesn't exist
if (!(Test-Path "Outputs")) {
    New-Item -ItemType Directory -Force -Path "Outputs" | Out-Null
}

function Run-Experiment {
    param (
        [string]$Script,
        [string]$OutputLog
    )
    $ScriptPath = "Programs\$Script"
    $OutputPath = "Outputs\$OutputLog"
    $TempLog = "$OutputPath.tmp"
    
    # Check if script exists
    if (!(Test-Path $ScriptPath)) {
        Write-Error "Script not found: $ScriptPath"
        return
    }

    Write-Host "Running $Script..."
    
    # Run python via cmd to capture raw output without PowerShell error wrapping
    # Note: We run from root, so path to script is Programs\script.py
    # Output images will be saved in root by default since cwd is root. We will move them to Outputs/ after execution.
    cmd /c "py -3.13 $ScriptPath > $TempLog 2>&1"
    
    # Filter out the benign warnings and save to final log
    Get-Content $TempLog | Where-Object { 
        $_ -notmatch "Could not find platform independent libraries" -and 
        $_ -notmatch "<prefix>" 
    } | Set-Content $OutputPath
    
    Remove-Item $TempLog
    
    # Move any generated .png files to Outputs directory
    Move-Item *.png Outputs/ -ErrorAction SilentlyContinue
    
    Write-Host "$Script Done. Output saved to $OutputPath"
}

Write-Host "Starting Deep Learning Experiments Execution (Clean Logs)..."

Run-Experiment -Script "exp1_ann_backprop.py" -OutputLog "exp1_output.txt"
Run-Experiment -Script "exp2_deep_ann.py" -OutputLog "exp2_output.txt"
Run-Experiment -Script "exp3_image_classification_ann.py" -OutputLog "exp3_output.txt"
Run-Experiment -Script "exp4_cnn_basic.py" -OutputLog "exp4_output.txt"
Run-Experiment -Script "exp5_data_augmentation.py" -OutputLog "exp5_output.txt"
Run-Experiment -Script "exp6_cnn_augmentation.py" -OutputLog "exp6_output.txt"
Run-Experiment -Script "exp7_lenet5.py" -OutputLog "exp7_output.txt"
Run-Experiment -Script "exp8_vgg.py" -OutputLog "exp8_output.txt"
Run-Experiment -Script "exp9_rnn_sentiment.py" -OutputLog "exp9_output.txt"
Run-Experiment -Script "exp10_bi_lstm_sentiment.py" -OutputLog "exp10_output.txt"

Write-Host "All experiments completed."
