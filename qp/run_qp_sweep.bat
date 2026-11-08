@echo off
setlocal enabledelayedexpansion
title Algorithm 3 (QP) - LIRR service instance - 5 seeds x 5 rates

rem ============================================================================
rem  run_qp_sweep.bat -- the buyer block of Algorithm 3 solved as ONE convex QP
rem  (node potentials kept as variables, every observed zero cell imposed as
rem  g_ia <= 0), in place of cycle projection + inactive-cell rows + hinge refit.
rem
rem  Everything else is the current pipeline: same sampling design, same seller
rem  block, Algorithm 2' forward solve (40 iterations), clearing solve for tau*,
rem  lambda_sigma = 1e-3, kappa = 1 x max tau0.
rem
rem  DWL / G(tau0) / wedge errors are written against TWO truths:
rem    *_paper  phi = Z theta_true (7 features) -- what the draft reports
rem    *_gen    phi = the generator's phi, with the 97 station ASCs -- correct
rem
rem  Folder layout: this file lives in qp\ inside the release; the release root
rem  (code\, data\lirr\) is the parent folder. Output, written here in qp\:
rem  qp_sweep.csv (one row per case, flushed as it goes) and
rem  qp_sweep_summary.csv (per-rate means).
rem
rem  Resumable: rerun and choose the same option; finished cases are skipped.
rem ============================================================================

set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
set "PARENT=%HERE%\.."
for %%i in ("%PARENT%") do set "PARENT=%%~fi"
set "REL=%PARENT%"
set "PYTHONIOENCODING=utf-8"

echo ============================================================================
echo   Algorithm 3 as a single QP   LIRR   5 seeds x 10-50%%
echo ============================================================================
echo.

rem ---- interpreter: pyexe.txt here or in the release root, else PATH ---------
set "PYEXE="
for %%P in ("%HERE%\pyexe.txt" "%PARENT%\pyexe.txt") do (
  if not defined PYEXE if exist %%P (
    set /p PYEXE=<%%P
    if not exist "!PYEXE!" set "PYEXE="
  )
)
if not defined PYEXE if exist "%USERPROFILE%\.conda\envs\mypython\python.exe" set "PYEXE=%USERPROFILE%\.conda\envs\mypython\python.exe"
if not defined PYEXE (
  where python >nul 2>&1 && for /f "delims=" %%i in ('where python') do if not defined PYEXE set "PYEXE=%%i"
)
if not defined PYEXE (
  echo [X] No Python interpreter found. Put its full path in pyexe.txt.
  goto :fail
)
set "CROOT=%PYEXE%\..\"
for %%i in ("%CROOT%.") do set "CROOT=%%~fi"
if exist "%CROOT%\Library\bin" set "PATH=%CROOT%;%CROOT%\Library\bin;%CROOT%\Library\mingw-w64\bin;%CROOT%\Scripts;%CROOT%\DLLs;%PATH%"
"%PYEXE%" -c "import numpy,pandas,scipy" >nul 2>&1
if errorlevel 1 (
  echo [X] %PYEXE% cannot import numpy/pandas/scipy.
  goto :fail
)
echo   interpreter : %PYEXE%

rem ---- cvxpy + CLARABEL (the only new dependency) ----------------------------
"%PYEXE%" -c "import cvxpy; assert 'CLARABEL' in cvxpy.installed_solvers()" >nul 2>&1
if errorlevel 1 (
  echo   installing cvxpy and clarabel ...
  "%PYEXE%" -m pip install --quiet cvxpy clarabel
  "%PYEXE%" -c "import cvxpy; assert 'CLARABEL' in cvxpy.installed_solvers()" >nul 2>&1
  if errorlevel 1 (
    echo [X] cvxpy/clarabel could not be installed. Try:  "%PYEXE%" -m pip install cvxpy clarabel
    goto :fail
  )
)
echo   cvxpy       : ok

rem ---- the release package (this folder's parent) ----------------------------
if not exist "%REL%\data\lirr\links_p2.csv" (
  echo [X] %REL%\data\lirr\links_p2.csv not found. Run this file from qp\ inside the release.
  goto :fail
)
echo   release     : %REL%
echo.
echo     [1] Full grid   5 seeds x 10/20/30/40/50%%  + step 9 (DWL, wedge)   ~2 h
echo     [2] Full grid   5 seeds x 10/20/30/40/50%%  no step 9               ~1.7 h
echo     [3] Test case   seed 0, 30%%, + step 9                              ~15 min
echo.
set /p CH=Choose [1-3]:

set "SEEDS=0 1 2 3 4"
set "RATES=0.1 0.2 0.3 0.4 0.5"
set "EXTRA="
set "OUT=%HERE%\qp_sweep.csv"
if "%CH%"=="2" set "EXTRA=--skip-step9"
if "%CH%"=="2" set "OUT=%HERE%\qp_sweep_nostep9.csv"
if "%CH%"=="3" (
  set "SEEDS=0"
  set "RATES=0.3"
  set "OUT=%HERE%\qp_sweep_test.csv"
)

echo.
echo   running... one row per case is appended to %OUT%
echo   (Ctrl-C is safe: rerun and pick the same option to resume.)
echo.
"%PYEXE%" -u "%HERE%\run_qp_sweep.py" --release "%REL%" --out "%OUT%" ^
    --seeds %SEEDS% --rates %RATES% %EXTRA% --resume ^
    --zero-cells hard --lambda-sigma 1e-3 --sigma-cap 1.0 ^
    --step7-max-iter 40 --clearing-max-iter 20000 --log "%OUT%.log"
if errorlevel 1 goto :fail

"%PYEXE%" "%HERE%\summarize_qp_sweep.py" "%OUT%"
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
