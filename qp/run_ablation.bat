@echo off
setlocal enabledelayedexpansion
title Algorithm 3 (QP) ablation - LIRR service instance

rem ============================================================================
rem  run_ablation.bat -- ablation and robustness arms for the QP Algorithm 3.
rem  Each arm is the full pipeline (QP, Algorithm 2' forward, clearing solve)
rem  over 5 seeds x 10-50%%, written to its own CSV:
rem
rem    A  ablation_nozero.csv   no zero-cell constraints (equalities only)
rem    B  ablation_Ronly.csv    zero cells limited to R (what the old rows reached)
rem    C  ablation_nocap.csv    no bound on sigma (kappa removed)
rem    D  ablation_nosigma.csv  sigma == 0 (no rationing prices)
rem    E  robust_hinge0.1.csv   squared hinge on zero cells, rho_V = 0.1
rem    F  robust_hinge1.csv     squared hinge, rho_V = 1
rem    G  robust_hinge10.csv    squared hinge, rho_V = 10
rem
rem  The main arm (hard zero cells, kappa = 1) is qp_sweep.csv from run_qp_sweep.bat.
rem  About 1.3 h per arm on this machine. Every arm is resumable: rerun and
rem  pick the same option. Arms write separate files, so two or three can run
rem  at once in separate windows:  run_ablation.bat A   /   run_ablation.bat C
rem  When done, summarize_ablation.py builds the new Table 2 (ablation_table.csv).
rem ============================================================================

set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
set "PARENT=%HERE%\.."
for %%i in ("%PARENT%") do set "PARENT=%%~fi"
set "REL=%PARENT%"
set "PYTHONIOENCODING=utf-8"

echo ============================================================================
echo   Algorithm 3 (QP) ablation   LIRR   5 seeds x 10-50%%
echo ============================================================================
echo.

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
"%PYEXE%" -c "import numpy,pandas,scipy,cvxpy" >nul 2>&1
if errorlevel 1 (
  echo [X] %PYEXE% cannot import numpy/pandas/scipy/cvxpy. Run run_qp_sweep.bat once first, it installs cvxpy and clarabel.
  goto :fail
)
if not exist "%REL%\data\lirr\links_p2.csv" (
  echo [X] %REL%\data\lirr\links_p2.csv not found. Run this file from qp\ inside the release.
  goto :fail
)
echo   interpreter : %PYEXE%
echo   release     : %REL%
echo.

set "CH=%~1"
if not defined CH (
  echo     [1] Ablation arms A-D, one after another                   ~5 h
  echo     [2] Robustness arms E-G ^(hinge^), one after another        ~4 h
  echo     [3] All seven arms                                          ~9 h
  echo     Several arms: type their letters separated by spaces, e.g.  A D C
  echo     [A]-[G] One arm only ^(see the header of this file^)       ~1.3 h
  echo     [S] Summarize what has finished so far
  echo.
  set /p CH=Choose:
)

if /i "%CH%"=="1" (set "LIST=A B C D") else if /i "%CH%"=="2" (set "LIST=E F G") else if /i "%CH%"=="3" (set "LIST=A B C D E F G") else if /i "%CH%"=="S" (set "LIST=") else (set "LIST=%CH%")

for %%X in (%LIST%) do call :arm %%X
if errorlevel 1 goto :fail

"%PYEXE%" "%HERE%\summarize_ablation.py" "%HERE%"
echo.
pause
exit /b 0

:arm
set "ARM=%~1"
set "FLAGS=" & set "OUT="
if /i "%ARM%"=="A" (set "FLAGS=--zero-cells none" & set "OUT=ablation_nozero.csv")
if /i "%ARM%"=="B" (set "FLAGS=--zero-set R" & set "OUT=ablation_Ronly.csv")
if /i "%ARM%"=="C" (set "FLAGS=--sigma-cap 0" & set "OUT=ablation_nocap.csv")
if /i "%ARM%"=="D" (set "FLAGS=--no-sigma" & set "OUT=ablation_nosigma.csv")
if /i "%ARM%"=="E" (set "FLAGS=--zero-cells hinge --rho-v 0.1" & set "OUT=robust_hinge0.1.csv")
if /i "%ARM%"=="F" (set "FLAGS=--zero-cells hinge --rho-v 1" & set "OUT=robust_hinge1.csv")
if /i "%ARM%"=="G" (set "FLAGS=--zero-cells hinge --rho-v 10" & set "OUT=robust_hinge10.csv")
if not defined OUT (
  echo [X] unknown arm "%ARM%"
  exit /b 1
)
title QP ablation - arm %ARM% - %OUT%
echo.
echo ---- arm %ARM%: %FLAGS%  -^>  %OUT%
"%PYEXE%" -u "%HERE%\run_qp_sweep.py" --release "%REL%" --out "%HERE%\%OUT%" ^
    --seeds 0 1 2 3 4 --rates 0.1 0.2 0.3 0.4 0.5 --resume ^
    --lambda-sigma 1e-3 --step7-max-iter 40 --clearing-max-iter 20000 ^
    %FLAGS% --log "%HERE%\%OUT%.log"
exit /b %errorlevel%

:fail
echo.
pause
exit /b 1
