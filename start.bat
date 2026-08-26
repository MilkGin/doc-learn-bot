@echo off
setlocal
title 技术文档学习机器人 - 一键启动
cd /d "%~dp0"

echo ============================================================
echo  技术文档学习机器人（多邻国式）
echo ============================================================

REM ---- 1. 启动 Ollama（本地离线模型后端需要） ----
REM 如果 .env 里 LLM_PROVIDER=deepseek，则使用线上 API，无需本地 Ollama。
set NEED_OLLAMA=1
findstr /i "LLM_PROVIDER=deepseek" .env >nul 2>&1 && set NEED_OLLAMA=0

if "%NEED_OLLAMA%"=="1" (
    echo [1/3] 正在准备 Ollama 本地服务...
    set "OL_PORT=11434"
    for /f "tokens=2 delims==" %%a in ('findstr /i "^OLLAMA_PORT" .env') do set "OL_PORT=%%a"
    if "%OL_PORT%"=="" set "OL_PORT=11434"

    powershell -NoProfile -Command "exit (Test-NetConnection -ComputerName 127.0.0.1 -Port %OL_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue)" >nul 2>&1
    if errorlevel 1 (
        echo   Ollama 未在 %OL_PORT% 端口运行，正在启动...
        if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
            start "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
        ) else if exist "%ProgramFiles%\Ollama\ollama.exe" (
            start "" "%ProgramFiles%\Ollama\ollama.exe" serve
        ) else (
            echo   [!] 未找到 Ollama，请先安装： https://ollama.com
            echo   然后手动运行 ollama serve
        )
    ) else (
        echo   Ollama 已在端口 %OL_PORT% 运行。
    )
    echo.
)

REM ---- 2. 可选：建立全局文档索引（仅当 docs/ 存在且 chroma_db 缺失时） ----
REM 注意：主程序基于章节内存检索，不依赖 chroma_db；此步骤失败不影响启动。
if exist "docs" (
    if not exist "chroma_db\chroma.sqlite3" (
        echo [2/3] 检测到 docs/，正在建立文档索引（可选）...
        call python ingest.py
        if errorlevel 1 (
            echo   [!] 索引建立失败，已跳过（不影响主程序运行）。
        )
    ) else (
        echo [2/3] 文档索引已存在。
    )
) else (
    echo [2/3] 未检测到 docs/ 目录，跳过全局索引（主程序仍可正常使用）。
)
echo.

REM ---- 3. 启动网页原型机 ----
echo [3/3] 正在启动网页界面...
echo   启动后请用浏览器打开： http://localhost:8501
echo   按 Ctrl+C 可停止。
echo.
start "" http://localhost:8501
python -m streamlit run app_learn.py --server.port 8501 --server.headless true

endlocal
