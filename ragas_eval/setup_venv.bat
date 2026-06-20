@echo off
REM setup_venv.bat — 在 ragas_eval\ 資料夾建立獨立虛擬環境並安裝依賴套件
REM 使用方式：在 meetGRAG 專案根目錄執行 ragas_eval\setup_venv.bat

setlocal

set VENV_DIR=ragas_eval\.venv

echo [1/3] 建立虛擬環境：%VENV_DIR%
python -m venv %VENV_DIR%
if errorlevel 1 (
    echo 錯誤：虛擬環境建立失敗，請確認 Python 3.10+ 已安裝
    exit /b 1
)

echo [2/3] 安裝依賴套件...
%VENV_DIR%\Scripts\pip install --upgrade pip --quiet
%VENV_DIR%\Scripts\pip install -r ragas_eval\requirements.txt
if errorlevel 1 (
    echo 錯誤：套件安裝失敗
    exit /b 1
)

echo [3/3] 完成！
echo.
echo 啟動虛擬環境：
echo   ragas_eval\.venv\Scripts\activate
echo.
echo 執行評估：
echo   python ragas_eval\run_ragas_eval.py --testset eval\testset\samples\sample_testset.json
echo.

endlocal
