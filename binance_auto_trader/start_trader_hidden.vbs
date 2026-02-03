Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & "D:\Project\AI\binance_auto_trader\start_trader_background.bat" & Chr(34), 0
Set WshShell = Nothing
