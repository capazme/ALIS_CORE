#!/usr/bin/env bash
#
# Merge the current branch into main. Switches to main, merges, exits with 0
# on success and non-zero only when the merge fails (e.g. conflicts). No interactive prompts.
#

set -e

MAIN_BRANCH="${1:-main}"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" == "$MAIN_BRANCH" ]]; then
  echo "Already on $MAIN_BRANCH. Nothing to merge."
  exit 0
fi

git checkout "$MAIN_BRANCH"
git merge "$CURRENT_BRANCH" --no-edit

echo "Merged $CURRENT_BRANCH into $MAIN_BRANCH."
