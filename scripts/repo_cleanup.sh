#!/usr/bin/env bash
# Repository hygiene for FinSight.
#
#   bash scripts/repo_cleanup.sh           # report only, changes nothing   (make repo-check)
#   bash scripts/repo_cleanup.sh --apply   # untrack + delete caches         (make repo-clean)
#
# What it looks for:
#   1. Files Git still tracks although .gitignore now excludes them. --apply runs
#      `git rm --cached` on them: they leave the repository but STAY ON DISK.
#   2. Python / tool caches in the working tree (__pycache__, .pytest_cache, ...).
#      --apply deletes them; every one is rebuilt automatically when needed.
#   3. Tracked files over 1 MB that are neither in Git LFS nor allow-listed
#      (reported only: decide yourself, e.g. move them to DVC or LFS).
#   4. Stray files at the repository root that no tool reads (reported only).
#
# Never touched, whatever .gitignore says: docs/, tests/, dvc.yaml, dvc.lock,
# *.dvc, .dvc/config, params.yaml, secrets/.env.enc, reports/*.json, .gitkeep files.
set -euo pipefail

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

cd "$(git rev-parse --show-toplevel)"

PROTECTED='^(docs/|tests/|dvc\.yaml$|dvc\.lock$|.*\.dvc$|\.dvc/config$|params\.yaml$|secrets/\.env\.enc$|reports/[^/]+\.json$|.*\.gitkeep$)'

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# ---- 1. tracked but ignored -------------------------------------------------
say "1. Tracked files that .gitignore excludes"
mapfile -t ignored < <(git ls-files -ci --exclude-standard | grep -Ev "$PROTECTED" || true)
if ((${#ignored[@]} == 0)); then
  echo "   none"
else
  printf '   %s\n' "${ignored[@]}"
  if ((APPLY)); then
    printf '%s\0' "${ignored[@]}" | xargs -0 git rm --cached --quiet --
    echo "   -> untracked ${#ignored[@]} file(s); they are still on disk."
  fi
fi

# ---- 2. caches ----------------------------------------------------------------
say "2. Caches in the working tree"
mapfile -t caches < <(find . -path ./.git -prune -o -path ./.dvc/cache -prune -o \
  -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \
  -o -name .mypy_cache -o -name .ipynb_checkpoints -o -name htmlcov \
  -o -name spark-warehouse -o -name metastore_db \) -print)
if ((${#caches[@]} == 0)); then
  echo "   none"
else
  printf '   %s\n' "${caches[@]}"
  if ((APPLY)); then
    printf '%s\0' "${caches[@]}" | xargs -0 rm -rf --
    rm -f .coverage coverage.xml derby.log
    echo "   -> deleted ${#caches[@]} cache folder(s)."
  fi
fi

# ---- 3. big tracked files -------------------------------------------------------
say "3. Tracked files over 1 MB that are not in Git LFS"
lfs=$(git lfs ls-files -n 2>/dev/null || true)
found=0
while IFS= read -r -d '' f; do
  [[ -f "$f" ]] || continue
  size=$(wc -c <"$f")
  ((size > 1048576)) || continue
  grep -qxF "$f" <<<"$lfs" && continue
  [[ "$f" == *.ipynb ]] && continue   # the executed EDA notebook carries its figures
  printf '   %6d KB  %s\n' $((size / 1024)) "$f"
  found=1
done < <(git ls-files -z)
((found)) || echo "   none"

# ---- 4. stray root files ----------------------------------------------------------
say "4. Untracked or stray files at the repository root"
mapfile -t stray < <(git ls-files --others --exclude-standard --directory | grep -v '/.\+' || true)
if ((${#stray[@]} == 0)); then
  echo "   none"
else
  printf '   %s\n' "${stray[@]}"
  echo "   (not changed: add them to Git, to .gitignore, or delete them yourself)"
fi

if ((APPLY)); then
  say "Done. Review with 'git status', then commit:"
  echo '   git add .gitignore && git commit -m "chore: tidy repository (untrack ignored files)"'
else
  say "Nothing was changed. Run 'make repo-clean' to apply 1 and 2."
fi
