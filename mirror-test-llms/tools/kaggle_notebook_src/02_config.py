# ============================== RUN SETTINGS ================================
# You normally never need to touch this cell.

# Stop starting/continuing work after this many hours, then package results.
# Kaggle GPU sessions allow ~9-12 h; 8.0 leaves a safe margin for packaging.
BUDGET_HOURS = 8.0

# Set True to only PRINT the plan/status without running any task
# (useful to check what a session would do).
DRY_RUN = False

# Notebook build fingerprint (filled in by the builder; do not edit).
NOTEBOOK_BUILD = "__BUILD_STAMP__"
print(f"[config] budget={BUDGET_HOURS}h dry_run={DRY_RUN} build={NOTEBOOK_BUILD}")
