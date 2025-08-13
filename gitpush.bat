@echo off
setlocal enabledelayedexpansion

REM Script to commit and push with a random alpaca fact as the commit message
REM Usage: commit_with_alpaca_fact.bat

echo 🦙 Alpaquero Git Commit with Random Alpaca Fact
echo ===============================================

REM Check if we're in a git repository
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo Error: Not in a git repository
    exit /b 1
)

REM Check if alpaca_facts.txt exists
if not exist "alpaca_facts.txt" (
    echo Error: alpaca_facts.txt not found
    exit /b 1
)

REM Count lines in the facts file
for /f %%i in ('type alpaca_facts.txt ^| find /c /v ""') do set total_lines=%%i

REM Generate a random line number (1 to total_lines)
set /a random_line=%RANDOM% * %total_lines% / 32768 + 1

REM Get the random fact
set line_count=0
for /f "delims=" %%a in (alpaca_facts.txt) do (
    set /a line_count+=1
    if !line_count! equ %random_line% (
        set "random_fact=%%a"
    )
)

REM Check if there are any changes to commit
git diff --quiet >nul 2>&1
set diff_exit=%errorlevel%
git diff --staged --quiet >nul 2>&1
set staged_exit=%errorlevel%

if %diff_exit% equ 0 if %staged_exit% equ 0 (
    echo No changes to commit
    exit /b 0
)

REM Add all changes
echo Adding all changes...
git add .

REM Commit with the random alpaca fact
echo Committing with message: "🦙 !random_fact!"
git commit -m "🦙 !random_fact!"

if errorlevel 1 (
    echo ❌ Commit failed
    exit /b 1
)

echo Commit successful!

REM Push to origin main
echo Pushing to origin main...
git push origin main

if errorlevel 1 (
    echo ❌ Failed to push to origin main
    exit /b 1
)

echo ✅ Successfully pushed to origin main!
echo 📝 Commit message was: !random_fact!

endlocal
