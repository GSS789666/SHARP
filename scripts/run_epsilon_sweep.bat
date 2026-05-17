@echo off
REM ============================================================================
REM SHARP epsilon-sweep (resumable):
REM 4 epsilons x 2 scenarios x 1 seed = 8 runs.
REM ============================================================================

cd /d "%~dp0\.."
echo === SHARP epsilon sweep started %DATE% %TIME% ===

for %%E in (0.050 0.010 0.005 0.001) do (
  for %%S in (S40_low S60_high) do (
    if exist "results\eps_sweep\eps%%E\%%S\sharp\seed_0\summary.json" (
      echo [skip]  scenario=%%S epsilon=%%E already done
    ) else (
      echo.
      echo --- scenario=%%S epsilon=%%E ---
      python scripts\run_one.py --scenario %%S --algo sharp --seed 0 ^
        --iters 100 --steps 256 --epsilon %%E --device cuda ^
        --out-root results/eps_sweep/eps%%E
    )
  )
)

echo.
echo === SHARP epsilon sweep finished %DATE% %TIME% ===
echo Run  python scripts\analyze_epsilon.py  to produce the scaling figure.
pause
