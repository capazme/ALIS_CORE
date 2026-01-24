#!/usr/bin/env bash
# Rimuove tutte le repository .git nidificate in ALIS_CORE (mantiene la root .git).
# Esegui da Terminale di sistema: ./remove-nested-git-repos.sh

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPOS=(
  "merlt-models"
  "visualex-merlt"
  "merlt"
  "visualex-platform"
  "visualex-api"
  "Legacy/MERL-T_alpha"
  "Legacy/VisuaLexAPI"
)

for rel in "${REPOS[@]}"; do
  d="$ROOT/$rel/.git"
  if [ -d "$d" ]; then
    xattr -cr "$d" 2>/dev/null || true
    rm -rf "$d"
    echo "Rimosso: $rel/.git"
  fi
done

for rel in "${REPOS[@]}"; do
  d="$ROOT/$rel/.github"
  if [ -d "$d" ]; then
    xattr -cr "$d" 2>/dev/null || true
    rm -rf "$d"
    echo "Rimosso: $rel/.git"
  fi
done
echo "Fatto."
