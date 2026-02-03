# Run as Administrator
# This script creates a Windows Task to start the trader on boot

$Action = New-ScheduledTaskAction -Execute "D:\Project\AI\binance_auto_trader\start_trader.bat"
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive -RunLevel Highest

Register-ScheduledTask -TaskName "BinanceTrader" -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force

Write-Host "Task 'BinanceTrader' created successfully!"
Write-Host "The trader will start automatically on Windows boot."
