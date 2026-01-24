#!/bin/bash
# Script helper per visualizzare i log di MERL-T API

MERLT_ROOT="/Users/gpuzio/Desktop/CODE/ALIS_CORE/Legacy/MERL-T_alpha"
LOG_FILE="$MERLT_ROOT/logs/merlt-api.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file non trovato: $LOG_FILE"
    exit 1
fi

echo "📋 Log di MERL-T API: $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Mostra ultime 50 righe
if [ "$1" = "-f" ] || [ "$1" = "--follow" ]; then
    echo "👀 Monitoraggio in tempo reale (Ctrl+C per uscire)..."
    echo ""
    tail -f "$LOG_FILE"
else
    echo "📄 Ultime 50 righe:"
    echo ""
    tail -50 "$LOG_FILE"
    echo ""
    echo "💡 Per monitoraggio in tempo reale: $0 -f"
fi
