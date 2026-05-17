@echo off
REM ============================================================================
REM SHARP ablation experiments: 3 ablations x 4 scenarios x 1 seed = 12 runs.
REM Resumable: cells with an existing summary.json are skipped.
REM ============================================================================

cd /d "%~dp0\.."
echo === SHARP ablations started %DATE% %TIME% ===

for %%S in (S40_low S40_high S60_low S60_high) do (
  for %%A in (no_cvar no_soft_bias no_dual) do (
    if exist "results\ablation\%%S\%%A\seed_0\summary.json" (
      echo [skip]  scenario=%%S ablation=%%A already done
    ) else (
      echo --- scenario=%%S ablation=%%A ---
      python scripts\run_one.py --scenario %%S --algo %%A --ablation ^
        --seed 0 --iters 100 --steps 256 --device cuda ^
        --out-root results/ablation
    )
  )
)

echo === SHARP ablations done %DATE% %TIME% ===
pause
