@echo off
REM ============================================================================
REM SHARP main experiments -- sequential launcher (resumable).
REM
REM 4 scenarios x 2 learned algorithms x 3 seeds = 24 training runs.
REM A cell is considered complete if its summary.json already exists.
REM Re-running this script after a crash / restart will skip done cells.
REM ============================================================================

cd /d "%~dp0\.."
echo === SHARP main experiments started %DATE% %TIME% ===

for %%S in (S40_low S40_high S60_low S60_high) do (
  for %%A in (sharp lagrangian_ppo) do (
    for %%K in (0 1 2) do (
      if exist "results\main\%%S\%%A\seed_%%K\summary.json" (
        echo [skip]  scenario=%%S algo=%%A seed=%%K already done
      ) else (
        echo.
        echo --- scenario=%%S algo=%%A seed=%%K ---
        python scripts\run_one.py --scenario %%S --algo %%A --seed %%K --iters 100 --steps 256 --device cuda
        if errorlevel 1 (
          echo ERROR in scenario=%%S algo=%%A seed=%%K
          pause
          exit /b 1
        )
      )
    )
  )
)

echo.
echo === SHARP main experiments finished %DATE% %TIME% ===
echo Run scripts\run_all_baselines.bat next, then scripts\analyze.py.
pause
