#!/bin/bash

# Vai nella cartella del progetto dove hai avviato Claude
cd /Users/gpuzio/Desktop/CODE/ALIS_CORE

# Riprendi l'ultima conversazione e manda "continua"
claude -p "continua" --continue >> ~/claude_continua.log 2>&1
