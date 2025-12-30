$ErrorActionPreference = "Continue"

# Clear potentially corrupted Python environment variables
Remove-Item Env:\PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue


function Run-Experiment {
    param (
        [string]$Script,
        [string]$OutputLog
    )
    Write-Host "Running $Script..."
    $TempLog = "$OutputLog.tmp"
    
    # Run python via cmd to capture raw output without PowerShell error wrapping
    cmd /c "py -3.13 $Script > $TempLog 2>&1"
    
    # Filter out the benign warnings and save to final log
    Get-Content $TempLog | Where-Object { 
        $_ -notmatch "Could not find platform independent libraries" -and 
        $_ -notmatch "<prefix>" 
    } | Set-Content $OutputLog
    
    Remove-Item $TempLog
    Write-Host "$Script Done. Output saved to $OutputLog"
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
