@echo off
REM Kích hoạt môi trường nếu dùng virtualenv (bỏ dòng này nếu không dùng)
REM CALL D:\Project\trainning_bot\venv\Scripts\activate

REM Di chuyển tới thư mục dự án
cd /d D:\Project\trainning_bot

REM Tạo thư mục logs nếu chưa có
if not exist logs mkdir logs

REM Ghi log bắt đầu
echo 🕒 Training started at %date% %time% >> logs\daily_training_%date:~-4%-%date:~3,2%-%date:~0,2%.log

REM Chạy script train_dqn.py
python model\train_dqn.py >> logs\daily_training_%date:~-4%-%date:~3,2%-%date:~0,2%.log 2>&1

REM Ghi log hoàn thành
echo ✅ Training finished at %date% %time% >> logs\daily_training_%date:~-4%-%date:~3,2%-%date:~0,2%.log
