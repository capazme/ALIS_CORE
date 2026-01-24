# Sprint 1 - Task 7: Frontend QA Page Summary
## Data: 2026-01-03

---

## TASK COMPLETED ✅

**Obiettivo**: Creare QAPage.tsx con query input e useExpertQuery hook

**Status**: ✅ COMPLETATO

---

## FILE CREATI

### 1. **Types** (`/frontend/src/types/expert.ts`) - 102 lines

**Interface principali**:

```typescript
export interface ExpertQueryRequest {
  query: string;
  context?: Record<string, any>;
  max_experts?: number; // 1-4
}

export interface ExpertQueryResponse {
  trace_id: string;
  synthesis: string;
  mode: 'convergent' | 'divergent';
  alternatives?: AlternativeInterpretation[] | null;
  sources: SourceReference[];
  experts_used: string[];
  confidence: number; // 0-1
  execution_time_ms: number;
}

export interface InlineFeedbackRequest {
  trace_id: string;
  rating: number; // 1-5
}

export interface DetailedFeedbackRequest {
  trace_id: string;
  retrieval_score: number; // 0-1
  reasoning_score: number; // 0-1
  synthesis_score: number; // 0-1
  comment?: string;
}

export interface SourceFeedbackRequest {
  trace_id: string;
  source_id: string; // URN
  relevance: number; // 1-5
}

export interface RefineFeedbackRequest {
  trace_id: string;
  follow_up_query: string;
}
```

**UI State Types**:
```typescript
export type FeedbackType = 'inline' | 'detailed' | 'source' | 'refine' | null;

export interface ExpertQueryState {
  query: string;
  isQuerying: boolean;
  queryError: string | null;
  response: ExpertQueryResponse | null;
  isFeedbackSubmitting: boolean;
  feedbackError: string | null;
  showFeedback: boolean;
  // ... detailed state fields
}
```

---

### 2. **Service** (`/frontend/src/services/expertService.ts`) - 127 lines

**API Client Functions**:

```typescript
/**
 * Submit query to MultiExpertOrchestrator.
 */
export async function queryExperts(
  request: ExpertQueryRequest
): Promise<ExpertQueryResponse>

/**
 * Submit inline feedback (thumbs 1-5).
 */
export async function submitInlineFeedback(
  request: InlineFeedbackRequest
): Promise<FeedbackResponse>

/**
 * Submit detailed 3-dimension feedback.
 */
export async function submitDetailedFeedback(
  request: DetailedFeedbackRequest
): Promise<FeedbackResponse>

/**
 * Submit per-source feedback (1-5 stars).
 */
export async function submitSourceFeedback(
  request: SourceFeedbackRequest
): Promise<FeedbackResponse>

/**
 * Submit conversational refinement (follow-up query).
 */
export async function submitRefineFeedback(
  request: RefineFeedbackRequest
): Promise<ExpertQueryResponse>
```

**Pattern**:
- Usa axios client configurato (`api.ts`)
- Base URL: `/api/merlt/experts`
- Timeout: 120 seconds (inherited da api.ts)
- Auth automatica via interceptor

---

### 3. **Hook** (`/frontend/src/hooks/useExpertQuery.ts`) - 460 lines

**State Management Pattern** (seguendo `useLiveEnrichment.ts`):

```typescript
export function useExpertQuery() {
  const { user } = useAuth();
  const [state, setState] = useState<ExpertQueryState>(() => {
    // Load persisted state from localStorage
    const persisted = loadPersistedState();
    return persisted ? { ...initialState, ...persisted } : initialState;
  });

  // Auto-persist query history
  useEffect(() => {
    if (state.queryHistory.length > 0) {
      persistQueryHistory(state.queryHistory);
    }
  }, [state.queryHistory]);

  // Auto-show feedback after 3 seconds
  useEffect(() => {
    if (state.response && !state.showFeedback) {
      const timer = setTimeout(() => {
        setState((prev) => ({ ...prev, showFeedback: true }));
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [state.response, state.showFeedback]);

  // ... actions implementation
}
```

**Hook API**:

```typescript
const {
  // State
  isQuerying,
  queryError,
  response,
  showFeedback,
  queryHistory,
  isFeedbackSubmitting,
  feedbackError,

  // Loading helper
  isLoading,

  // Actions
  submitQuery,
  submitInlineFeedback,
  submitDetailedFeedback,
  submitSourceFeedback,
  submitRefinement,

  // Utilities
  clearErrors,
  resetQuery,
  clearHistory,
  loadFromHistory,
} = useExpertQuery();
```

**Features**:
- ✅ Local storage persistence (7 days TTL)
- ✅ Query history (max 10)
- ✅ Auto-show feedback after 3s
- ✅ Error handling graceful
- ✅ Loading states per operation
- ✅ Auth integration via `useAuth()`

---

### 4. **Component** (`/frontend/src/components/features/qa/QAPage.tsx`) - 386 lines

**UI Structure**:

```
QAPage
├── Header (title + description)
├── Query Input Card
│   ├── Textarea (query input)
│   ├── Character validation
│   ├── History button
│   └── Submit button (with loading state)
├── Error Display (if error)
├── History Dropdown (if showHistory)
│   └── List of previous queries
├── Response Display (if response)
│   ├── Synthesis Card
│   │   ├── Mode badge (convergent/divergent)
│   │   ├── Confidence score
│   │   ├── Execution time
│   │   ├── Synthesis text
│   │   ├── Experts used chips
│   │   ├── Sources list (with links)
│   │   └── Alternatives (if divergent)
│   ├── Feedback Card (if showFeedback)
│   │   ├── Thumbs up/down buttons
│   │   └── Detailed feedback button
│   └── New Query button
└── Empty State (if no query yet)
    └── Examples and instructions
```

**Key Features**:

1. **Query Input**:
   ```tsx
   <textarea
     value={query}
     onChange={(e) => setQuery(e.target.value)}
     placeholder="Es: Cos'è la legittima difesa secondo il codice penale?"
     rows={3}
     disabled={isQuerying}
   />
   ```

2. **Loading State**:
   ```tsx
   {isQuerying ? (
     <>
       <Loader2 className="w-4 h-4 mr-2 animate-spin" />
       Elaborazione...
     </>
   ) : (
     <>
       <Send className="w-4 h-4 mr-2" />
       Chiedi agli Expert
     </>
   )}
   ```

3. **Response Display**:
   - Mode badge (green for convergent, orange for divergent)
   - Confidence percentage
   - Execution time
   - Full synthesis text
   - Expert chips
   - Sources with links to articles
   - Alternatives (only for divergent mode)

4. **Feedback Buttons**:
   ```tsx
   <button onClick={handleThumbsUp}>
     <ThumbsUp className="w-4 h-4 mr-2" />
     Sì, utile
   </button>

   <button onClick={handleThumbsDown}>
     <ThumbsDown className="w-4 h-4 mr-2" />
     No, non utile
   </button>
   ```

5. **Query History**:
   - Dropdown with last 10 queries
   - Click to reload previous response
   - Shows synthesis preview, mode, and confidence

**Styling**:
- Tailwind CSS (matches existing codebase)
- Responsive design (max-w-4xl container)
- Color scheme:
  - Blue: primary actions
  - Green: positive (convergent, thumbs up)
  - Orange: warning (divergent)
  - Red: errors/negative
  - Gray: neutral/text

---

### 5. **Exports** (`/frontend/src/components/features/qa/index.ts`) - 8 lines

```typescript
export { QAPage } from './QAPage';
export { default } from './QAPage';
```

**Usage**:
```typescript
import { QAPage } from '@/components/features/qa';
// or
import QAPage from '@/components/features/qa';
```

---

## PATTERN IMPLEMENTATI

### 1. Service Layer Pattern

```
Component → Hook → Service → API (axios)
```

**Separation of concerns**:
- `QAPage.tsx`: UI rendering + user interactions
- `useExpertQuery.ts`: State management + business logic
- `expertService.ts`: API calls + request/response transformation
- `api.ts`: Axios configuration + interceptors

### 2. State Management

**Local State** (component level):
```tsx
const [query, setQuery] = useState('');
const [showHistory, setShowHistory] = useState(false);
```

**Global State** (hook level):
```tsx
const [state, setState] = useState<ExpertQueryState>(() => {
  const persisted = loadPersistedState();
  return persisted ? { ...initialState, ...persisted } : initialState;
});
```

**Persistence**:
- LocalStorage key: `merlt_expert_query_history`
- TTL: 7 days
- Max history: 10 queries
- Auto-persist on state change

### 3. Error Handling

**Hook level**:
```typescript
try {
  const response = await expertService.queryExperts(request);
  setState((prev) => ({ ...prev, isQuerying: false, response }));
  return response;
} catch (error: any) {
  setState((prev) => ({
    ...prev,
    isQuerying: false,
    queryError: error.response?.data?.error || 'Errore durante la query',
  }));
  return null;
}
```

**Component level**:
```tsx
{queryError && (
  <Card className="mb-6 bg-red-50 border-red-200">
    <div className="p-4 flex items-start">
      <AlertCircle className="w-5 h-5 text-red-600 mr-3" />
      <div>
        <h3 className="text-sm font-medium text-red-800">
          Errore durante la query
        </h3>
        <p className="text-sm text-red-700 mt-1">{queryError}</p>
      </div>
    </div>
  </Card>
)}
```

### 4. Loading States

**Query submission**:
```tsx
<button disabled={isQuerying || query.trim().length < 5}>
  {isQuerying ? 'Elaborazione...' : 'Chiedi agli Expert'}
</button>
```

**Feedback submission**:
```tsx
const handleThumbsUp = async () => {
  await submitInlineFeedback(5);
  // isFeedbackSubmitting handled internally by hook
};
```

---

## USER FLOW

### 1. Query Submission

```
1. User types query (min 5 chars)
2. Click "Chiedi agli Expert"
3. Loading state (isQuerying=true)
4. API call to /api/merlt/experts/query
5. Response displayed (isQuerying=false)
6. After 3s, feedback buttons appear
```

### 2. Inline Feedback

```
1. User sees response
2. After 3s, feedback card shows
3. Click "Sì, utile" (thumbs up) → submitInlineFeedback(5)
4. Or "No, non utile" (thumbs down) → submitInlineFeedback(1)
5. Feedback saved in database (both VisuaLex + MERL-T)
```

### 3. Query History

```
1. Click "Cronologia (N)"
2. Dropdown shows last 10 queries
3. Click on a query → loadFromHistory(trace_id)
4. Response re-displayed
5. Feedback can be given again (separate feedback record)
```

---

## INTEGRATION POINTS

### Backend Integration

**VisuaLex Backend** (proxy):
```typescript
POST http://localhost:3001/api/merlt/experts/query
POST http://localhost:3001/api/merlt/experts/feedback/inline
```

**MERL-T Backend** (direct):
```typescript
POST http://localhost:8000/api/experts/query
POST http://localhost:8000/api/experts/feedback/inline
```

**Flow**:
```
Frontend → VisuaLex Proxy → MERL-T API → PostgreSQL
              ↓
      merltFeedback table
      (local tracking)
```

### Auth Integration

```typescript
const { user } = useAuth();

// Hook checks user before API calls
if (!user) {
  setState((prev) => ({
    ...prev,
    queryError: 'Devi essere autenticato per fare domande',
  }));
  return null;
}
```

**Token management**: Automatic via axios interceptors in `api.ts`

---

## PROSSIMI PASSI (Tasks Rimanenti)

### Task 8-10: Componenti Feedback Dettagliati

**Da creare** (optional - già funziona con inline):

1. **ExpertResponseCard.tsx** (extract da QAPage)
   - Componente separato per response display
   - Riusabile in history/search
   - Props: `response: ExpertQueryResponse`

2. **FeedbackInline.tsx** (extract da QAPage)
   - Componente separato per thumbs buttons
   - Props: `onFeedback: (rating: number) => void`

3. **FeedbackDetailedForm.tsx** (NEW)
   - Modal/Drawer con 3 sliders
   - Retrieval (0-1), Reasoning (0-1), Synthesis (0-1)
   - Comment textarea
   - Submit button

**Pattern**:
```tsx
// In QAPage.tsx
import { FeedbackDetailedForm } from './FeedbackDetailedForm';

<FeedbackDetailedForm
  isOpen={showDetailedForm}
  onClose={() => setShowDetailedForm(false)}
  onSubmit={async (scores) => {
    await submitDetailedFeedback(
      scores.retrieval,
      scores.reasoning,
      scores.synthesis,
      scores.comment
    );
    setShowDetailedForm(false);
  }}
  isSubmitting={isFeedbackSubmitting}
/>
```

### Task 11: Routing

**File da modificare**: `/frontend/src/App.tsx` o routing config

```tsx
import { QAPage } from '@/components/features/qa';

// Add route
<Route path="/qa" element={<QAPage />} />
// or
<Route path="/experts" element={<QAPage />} />
```

**Navigation link** (in sidebar/menu):
```tsx
<NavLink to="/qa">
  <MessageSquare className="w-5 h-5" />
  Chiedi agli Expert
</NavLink>
```

### Task 12: E2E Testing

**Test scenarios**:
1. ✅ Submit query → get response
2. ✅ Give thumbs up feedback
3. ✅ Give thumbs down feedback
4. ✅ Load query from history
5. ✅ Handle error (401, 500)
6. ✅ Persistence (reload page)
7. ⏳ Detailed feedback modal
8. ⏳ Per-source feedback
9. ⏳ Refinement (follow-up)

---

## TESTING CHECKLIST

### Manual Testing

- [ ] Start VisuaLex Backend (`npm run dev`)
- [ ] Start MERL-T API (`uvicorn merlt.api.visualex_bridge:app --port 8000`)
- [ ] Start Frontend (`npm run dev`)
- [ ] Navigate to `/qa` (or integrate into app)
- [ ] Submit query "Cos'è la legittima difesa?"
- [ ] Verify loading state shows
- [ ] Verify response displays correctly
- [ ] Wait 3 seconds, verify feedback buttons appear
- [ ] Click "Sì, utile" → verify no error
- [ ] Check VisuaLex database: `SELECT * FROM merltFeedback WHERE interactionType='expert_query';`
- [ ] Check MERL-T database: `SELECT * FROM qa_traces;`
- [ ] Check MERL-T database: `SELECT * FROM qa_feedback;`
- [ ] Submit another query
- [ ] Click "Cronologia" → verify history shows
- [ ] Click on history item → verify response reloads
- [ ] Refresh page → verify history persists

### Integration Testing

```bash
# 1. Start all services
docker-compose -f docker-compose.dev.yml up -d
cd /Users/gpuzio/Desktop/CODE/VisuaLexAPI/backend && npm run dev &
cd /Users/gpuzio/Desktop/CODE/MERL-T_alpha && source .venv/bin/activate && uvicorn merlt.api.visualex_bridge:app --port 8000 &
cd /Users/gpuzio/Desktop/CODE/VisuaLexAPI/frontend && npm run dev &

# 2. Test API directly
curl -X POST http://localhost:3001/api/merlt/experts/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"Cos è la legittima difesa?"}' | jq

# 3. Test feedback
curl -X POST http://localhost:3001/api/merlt/experts/feedback/inline \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"trace_id":"trace_abc123","rating":5}' | jq
```

---

## FILE STRUCTURE (Final)

```
frontend/src/
├── types/
│   └── expert.ts (NEW - 102 lines)
├── services/
│   └── expertService.ts (NEW - 127 lines)
├── hooks/
│   └── useExpertQuery.ts (NEW - 460 lines)
└── components/features/qa/
    ├── QAPage.tsx (NEW - 386 lines)
    ├── index.ts (NEW - 8 lines)
    ├── ExpertResponseCard.tsx (TODO - Task 8)
    ├── FeedbackInline.tsx (TODO - Task 9)
    └── FeedbackDetailedForm.tsx (TODO - Task 10)
```

**Total lines created**: ~1,083 lines

---

## COMPLETION STATUS

✅ **Task 7 COMPLETATO con successo**

**Deliverables**:
- [x] Types definition (expert.ts)
- [x] Service layer (expertService.ts)
- [x] Custom hook (useExpertQuery.ts)
- [x] QA Page component (QAPage.tsx)
- [x] Export index (index.ts)
- [x] Local storage persistence
- [x] Query history (last 10)
- [x] Error handling graceful
- [x] Loading states
- [x] Auth integration
- [x] Inline feedback (thumbs)

**Ready for**:
- Task 8-10: Optional component extraction
- Task 11: Routing integration
- Task 12: E2E testing

---

## METRICHE DI SUCCESSO

**Sprint 1 Progress**: 7/12 tasks completati (58%)

**Backend completo** ✅:
1. Database schema
2. MERL-T API endpoints
3. FastAPI integration
4. VisuaLex proxy
5. Test suite

**Frontend base completo** ✅:
6. Types, service, hook
7. **QA Page component**

**Frontend advanced** ⏳:
8. ExpertResponseCard (optional)
9. FeedbackInline (optional)
10. FeedbackDetailedForm (optional)
11. Routing integration
12. E2E testing

---

*Documento generato il 2026-01-03 durante Sprint 1 - MVP Q&A Foundation*
