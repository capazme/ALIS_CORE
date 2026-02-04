# LLM Integration Test Results

**Date:** 2026-02-02 22:14:30
**Default Model:** google/gemini-2.5-flash
**Total Tests:** 12
**Total Cost:** $0.0095
**Total Tokens:** 3,153

## Summary

| Test | Model | Status | Latency | Tokens | Cost |
|------|-------|--------|---------|--------|------|
| Simple Query | google/gemini-2.5-flash | ✅ | 725ms | 12 | $0.0000 |
| Legal Query - Contratto | google/gemini-2.5-flash | ✅ | 6827ms | 88 | $0.0003 |
| Legal Query - Risoluzione | google/gemini-2.5-flash | ✅ | 1047ms | 78 | $0.0002 |
| LiteralExpert Query | google/gemini-2.5-flash | ✅ | 1810ms | 287 | $0.0009 |
| SystemicExpert Query | google/gemini-2.5-flash | ✅ | 3327ms | 454 | $0.0014 |
| PrinciplesExpert Query | google/gemini-2.5-flash | ✅ | 2472ms | 365 | $0.0011 |
| PrecedentExpert Query | google/gemini-2.5-flash | ✅ | 2885ms | 405 | $0.0012 |
| MoE Pipeline - Full | google/gemini-2.5-flash | ✅ | 17492ms | 1464 | $0.0044 |
| Failover Service | google/gemini-2.5-flash | ✅ | 0ms | N/A | N/A |
| Factory Create | openrouter | ✅ | 1666ms | N/A | N/A |
| Available Providers | API | ✅ | 0ms | N/A | N/A |
| List Models | API | ✅ | 0ms | N/A | N/A |

## Detailed Results

### Simple Query - google/gemini-2.5-flash

- **Status:** Success
- **Latency:** 725ms
- **Prompt Tokens:** 11
- **Completion Tokens:** 1
- **Total Tokens:** 12
- **Estimated Cost:** $0.0000

**Response Preview:**
```
OK
```

### Legal Query - Contratto - google/gemini-2.5-flash

- **Status:** Success
- **Latency:** 6827ms
- **Total Tokens:** 88
- **Estimated Cost:** $0.0003

**Response Preview:**
```
Secondo l'articolo 1321 del Codice Civile italiano, il contratto è l'accordo di due o più parti per costituire, regolare o estinguere tra loro un rapporto giuridico patrimoniale. In sintesi, è un negozio giuridico bilaterale o plurilaterale con contenuto patrimoniale.
```

### Legal Query - Risoluzione - google/gemini-2.5-flash

- **Status:** Success
- **Latency:** 1047ms
- **Total Tokens:** 78
- **Estimated Cost:** $0.0002

**Response Preview:**
```
Puoi risolvere un contratto quando una delle parti **non adempie in modo significativo** alle sue obbligazioni, e l'inadempimento non è di scarsa importanza.
```

### LiteralExpert Query - google/gemini-2.5-flash

- **Status:** Success
- **Latency:** 1810ms
- **Total Tokens:** 287
- **Estimated Cost:** $0.0009

**Response Preview:**
```
Analizziamo letteralmente l'articolo 1453 del Codice Civile per rispondere alla domanda.

Il testo recita: "Nei contratti con prestazioni corrispettive, quando uno dei contraenti non adempie le sue obbligazioni, l'altro può a sua scelta chiedere l'adempimento o la risoluzione del contratto, salvo, in ogni caso, il risarcimento del danno."

La frase chiave per rispondere alla domanda è: "**l'altro 
```

### SystemicExpert Query - google/gemini-2.5-flash

- **Status:** Success
- **Latency:** 3327ms
- **Total Tokens:** 454
- **Estimated Cost:** $0.0014

**Response Preview:**
```
Assolutamente, analizziamo le connessioni sistematiche tra gli articoli 1453, 1455 e 1456 del Codice Civile, che disciplinano la risoluzione del contratto per inadempimento.

La logica di fondo di questo sistema normativo è quella di **tutelare l'equilibrio sinallagmatico del contratto**, offrendo al contraente non inadempiente uno strumento per liberarsi dal vincolo contrattuale quando l'altra pa
```

### PrinciplesExpert Query - google/gemini-2.5-flash

- **Status:** Success
- **Latency:** 2472ms
- **Total Tokens:** 365
- **Estimated Cost:** $0.0011

**Response Preview:**
```
Come costituzionalista, posso indicare i principi costituzionali che tutelano il contraente debole nei contratti, tenendo conto degli articoli da te menzionati:

1.  **Art. 3 Cost. (Principio di Uguaglianza):**
    *   **Comma 1 (Uguaglianza formale):** Impone che la legge sia uguale per tutti, ma nel contesto contrattuale, ciò significa che non devono esserci discriminazioni arbitrarie. Tuttavia,
```

### PrecedentExpert Query - google/gemini-2.5-flash

- **Status:** Success
- **Latency:** 2885ms
- **Total Tokens:** 405
- **Estimated Cost:** $0.0012

**Response Preview:**
```
Certamente, analizziamo come la Cassazione ha interpretato il requisito della "non scarsa importanza" dell'inadempimento ai sensi dell'art. 1455 c.c., un elemento cruciale per la risoluzione del contratto.

**Art. 1455 c.c.: "Il contratto non si può risolvere se l'inadempimento di una delle parti ha scarsa importanza, avuto riguardo all'interesse dell'altra."**

Questo articolo stabilisce un princ
```

### MoE Pipeline - Full - google/gemini-2.5-flash

- **Status:** Success
- **Latency:** 17492ms
- **Total Tokens:** 1464
- **Estimated Cost:** $0.0044

**Response Preview:**
```

Experts: 4/4 success

SYNTHESIS:
È possibile risolvere un contratto per inadempimento quando una parte non esegue i propri doveri in un contratto con obblighi reciproci (art. 1453 c.c.). La parte lesa può scegliere tra chiedere l'adempimento o lo scioglimento del contratto.

La risoluzione può avvenire tramite:
*   **Diffida ad adempiere** (art. 1454 c.c.): intimazione scritta a adempiere entro un termine congruo.
*   **Clausola risolutiva espressa** (art. 1456 c.c.): clausola che prevede la risoluzione automatica in caso di sp...
```

### Failover Service - google/gemini-2.5-flash

- **Status:** Success
- **Latency:** 0ms

**Response Preview:**
```
Un contratto è un accordo legalmente vincolante tra due o più parti che crea obblighi reciproci.
```

### Factory Create - openrouter

- **Status:** Success
- **Latency:** 1666ms

**Response Preview:**
```
OK
```

### Available Providers - API

- **Status:** Success
- **Latency:** 0ms

**Response Preview:**
```
Providers: ['openai', 'anthropic', 'ollama', 'openrouter']
```

### List Models - API

- **Status:** Success
- **Latency:** 0ms

**Response Preview:**
```
Found 347 models. Sample: ['openrouter/free', 'stepfun/step-3.5-flash:free', 'arcee-ai/trinity-large-preview:free', 'moonshotai/kimi-k2.5', 'upstage/solar-pro-3:free']
```

## Cost Summary

- **Total Cost:** $0.0095
- **Total Tokens:** 3,153
- **Average Cost per Test:** $0.0008
