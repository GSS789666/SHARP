@echo off
REM ============================================================================
REM Non-learned baseline evaluations: greedy CAHT and MILP.
REM 4 scenarios x 2 baselines x 3 seeds = 24 evaluations (a few seconds each).
REM ============================================================================

cd /d "%~dp0\.."
echo === SHARP baseline eval started %DATE% %TIME% ===

for %%S in (S40_low S40_high S60_low S60_high) do (
  for %%P in (caht milp) do (
    for %%K in (0 1 2) do (
      echo --- scenario=%%S policy=%%P seed=%%K ---
      python scripts\eval_baseline.py --policy %%P ^
        --scenario %%S --seed %%K --episodes 1 ^
        --out results/main/%%S/%%P/seed_%%K/episode.json
    )
  )
)

echo === SHARP baseline eval done %DATE% %TIME% ===
pause
