@echo off
REM ============================================================================
REM Food Freshness Detection - CLI Interactive Menu
REM ============================================================================
REM Simply run this to launch the interactive menu

cd /d "%~dp0"
python food_cli.py --menu
pause
)
set fullpath=C:\Users\saket\SEM6-PROJECTS\DL\Project\Dataset\test\%imagename%
python food_cli.py predict --image "%fullpath%"
echo.
pause
goto menu

:batch_predict
cls
echo.
echo ============================================================================
echo        BATCH PREDICTION
echo ============================================================================
echo.
set /p imagedir="Enter image directory (e.g., C:\images): "
if "%imagedir%"=="" (
    echo Error: No directory provided.
    pause
    goto menu
)
set /p save="Save results to CSV? (y/n): "
if /i "%save%"=="y" (
    python food_cli.py predict --image "%imagedir%" --batch --save
) else (
    python food_cli.py predict --image "%imagedir%" --batch
)
echo.
pause
goto menu

:analyze
cls
echo.
echo ============================================================================
echo        DETAILED IMAGE ANALYSIS
echo ============================================================================
echo.
set /p image="Enter image name (e.g., fresh_apple_1.jpg): "
if "%image%"=="" (
    echo Error: No image name provided.
    pause
    goto menu
)
set fullpath=C:\Users\saket\SEM6-PROJECTS\DL\Project\Dataset\test\%image%
python food_cli.py analyze --image "%fullpath%"
echo.
pause
goto menu

:layers
cls
echo.
echo ============================================================================
echo        LAYER-BY-LAYER VISUALIZATION
echo ============================================================================
echo.
echo Shows how the image transforms through each neural network layer!
echo.
set /p image="Enter image name (e.g., fresh_apple_1.jpg): "
if "%image%"=="" (
    echo Error: No image name provided.
    pause
    goto menu
)
set fullpath=C:\Users\saket\SEM6-PROJECTS\DL\Project\Dataset\test\%image%
python food_cli.py layers --image "%fullpath%"
echo.
pause
goto menu

:model_info
cls
echo.
echo ============================================================================
echo        MODEL INFORMATION
echo ============================================================================
echo.
python food_cli.py info
echo.
pause
goto menu

:evaluate
cls
echo.
echo ============================================================================
echo        MODEL EVALUATION
echo ============================================================================
echo.
echo This may take a few minutes...
python food_cli.py evaluate
echo.
pause
goto menu
