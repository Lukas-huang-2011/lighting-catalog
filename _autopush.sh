#!/bin/bash
# Auto-push any new changes to GitHub
# Run this from the "New lightning catalog" folder or via cron

cd "$(dirname "$0")"

# Check if there's anything to commit
if git diff --quiet && git diff --cached --quiet; then
  exit 0  # nothing changed
fi

git add lighting-catalog_*.py lighting-catalog_*.txt lighting-catalog_*.md lighting-catalog_*.sql lighting-catalog_*.toml 2>/dev/null
git add .gitignore _autopush.sh 2>/dev/null
git commit -m "Auto-push: $(date '+%Y-%m-%d %H:%M')"
git push origin master 2>&1
