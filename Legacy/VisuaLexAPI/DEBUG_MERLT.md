# Debug MERL-T API

## 📍 Dove sono i log

I log di MERL-T API si trovano in:
```
/Users/gpuzio/Desktop/CODE/ALIS_CORE/Legacy/MERL-T_alpha/logs/merlt-api.log
```

## 🔍 Come visualizzare i log

### Opzione 1: Script helper
```bash
cd /Users/gpuzio/Desktop/CODE/ALIS_CORE/Legacy/VisuaLexAPI
./show-merlt-logs.sh          # Ultime 50 righe
./show-merlt-logs.sh -f       # Monitoraggio in tempo reale
```

### Opzione 2: Comando diretto
```bash
# Ultime 50 righe
tail -50 /Users/gpuzio/Desktop/CODE/ALIS_CORE/Legacy/MERL-T_alpha/logs/merlt-api.log

# Monitoraggio in tempo reale
tail -f /Users/gpuzio/Desktop/CODE/ALIS_CORE/Legacy/MERL-T_alpha/logs/merlt-api.log
```

### Opzione 3: Durante l'avvio
Lo script `start.sh` ora mostra automaticamente gli ultimi 20 righe del log se MERL-T API non si avvia correttamente.

## ❌ Errori comuni e soluzioni

### 1. `ModuleNotFoundError: No module named 'falkordb'`
**Causa**: Dipendenza non installata nel venv di MERL-T

**Soluzione**:
```bash
cd /Users/gpuzio/Desktop/CODE/ALIS_CORE/Legacy/MERL-T_alpha
source .venv/bin/activate
pip install -e .
```

### 2. `ModuleNotFoundError: No module named 'visualex'`
**Causa**: Modulo visualex non trovato nel PYTHONPATH

**Soluzione**: Già risolto! Lo script ora aggiunge automaticamente il percorso corretto.

### 3. `ModuleNotFoundError: No module named 'uvicorn'`
**Causa**: uvicorn non installato

**Soluzione**:
```bash
cd /Users/gpuzio/Desktop/CODE/ALIS_CORE/Legacy/MERL-T_alpha
source .venv/bin/activate
pip install uvicorn[standard]
```

### 4. `Connection refused` o `ECONNREFUSED`
**Causa**: MERL-T API non è partita correttamente

**Soluzione**: 
1. Controlla i log: `./show-merlt-logs.sh`
2. Verifica che il processo sia in esecuzione: `ps aux | grep uvicorn`
3. Verifica che la porta 8000 sia libera: `lsof -i :8000`

## 🔧 Verifica rapida

```bash
# 1. Verifica che il venv esista
ls -la /Users/gpuzio/Desktop/CODE/ALIS_CORE/Legacy/MERL-T_alpha/.venv

# 2. Verifica dipendenze critiche
cd /Users/gpuzio/Desktop/CODE/ALIS_CORE/Legacy/MERL-T_alpha
source .venv/bin/activate
python -c "import uvicorn, falkordb, fastapi; print('✓ Tutte le dipendenze installate')"

# 3. Verifica che il processo sia in esecuzione
ps aux | grep "uvicorn.*merlt.api.visualex_bridge"

# 4. Verifica che la porta 8000 sia in ascolto
lsof -i :8000

# 5. Test connessione
curl http://localhost:8000/health
```

## 📝 Note

- Lo script `start.sh` ora verifica automaticamente le dipendenze critiche e le installa se necessario
- I log vengono mostrati automaticamente se MERL-T API non si avvia
- Il percorso del log viene sempre mostrato in caso di errore
