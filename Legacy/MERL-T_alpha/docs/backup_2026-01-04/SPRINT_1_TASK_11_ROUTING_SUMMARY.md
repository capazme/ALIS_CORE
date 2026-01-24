# Sprint 1 - Task 11: Add QAPage to App Routing

**Status**: ✅ COMPLETED
**Date**: 2026-01-03
**Task**: Integrate Q&A Expert System page into application routing and navigation

---

## Overview

Completed integration of the Expert Q&A page into the VisuaLex frontend application, making it accessible via routing and adding navigation link in the sidebar.

---

## Changes Made

### 1. App Routing (`App.tsx`)

**File**: `/Users/gpuzio/Desktop/CODE/VisuaLexAPI/frontend/src/App.tsx`

**Changes**:
- Added import for QAPage component
- Added route at `/qa` path
- Route is protected (requires authentication)
- Route uses shared Layout with sidebar

```typescript
// Import
import { QAPage } from './components/features/qa/QAPage';

// Route definition
<Route path="qa" element={<QAPage />} />
```

**Route URL**: `http://localhost:5173/qa`

---

### 2. Sidebar Navigation (`Sidebar.tsx`)

**File**: `/Users/gpuzio/Desktop/CODE/VisuaLexAPI/frontend/src/components/layout/Sidebar.tsx`

**Changes**:
- Added `MessageSquare` icon import from lucide-react
- Added navigation item for Q&A page
- Positioned second in sidebar (after Ricerca, before Dossier)

```typescript
// Icon import
import { ..., MessageSquare } from 'lucide-react';

// Navigation item
<NavItem to="/qa" icon={MessageSquare} label="Chiedi agli Expert" onClick={closeMobile} />
```

**Navigation Order**:
1. 🔍 Ricerca (Search)
2. 💬 Chiedi agli Expert (Q&A) ← NEW
3. 📁 Dossier
4. 🌐 Ambienti (Environments)
5. 👥 Bacheca (Bulletin Board)
6. 🕒 Cronologia (History)

---

### 3. Bug Fix (`useExpertQuery.ts`)

**File**: `/Users/gpuzio/Desktop/CODE/VisuaLexAPI/frontend/src/hooks/useExpertQuery.ts:330`

**Issue**: Syntax error - extra closing parenthesis in `submitSourceFeedback` function

**Fix**:
```typescript
// BEFORE (error)
await expertService.submitSourceFeedback({
  trace_id: state.response.trace_id,
  source_id: sourceId,
  relevance,
})); // ❌ Two closing parens

// AFTER (fixed)
await expertService.submitSourceFeedback({
  trace_id: state.response.trace_id,
  source_id: sourceId,
  relevance,
}); // ✅ One closing paren
```

**Error**: `Expected ";" but found ")"`
**Resolution**: Removed extra closing parenthesis

---

## Integration Details

### Route Protection

The Q&A page is protected by the existing `ProtectedRoute` component:
- Requires user authentication
- Redirects to `/login` if not authenticated
- Access to user context via `useAuth()` hook

### Layout Integration

The Q&A page uses the shared `Layout` component:
- ✅ Sidebar navigation available
- ✅ Theme switching (light/dark)
- ✅ Focus mode support
- ✅ Keyboard shortcuts active
- ✅ Mobile responsive

### Navigation UX

**Desktop**:
- Sidebar always visible (unless focus mode)
- Click MessageSquare icon to navigate
- Tooltip shows "Chiedi agli Expert" on hover
- Active indicator (blue dot) when on Q&A page

**Mobile**:
- Sidebar hidden by default
- Hamburger menu to open sidebar
- Click navigation item (closes sidebar automatically)
- Larger touch targets (44px min)

---

## User Flow

### Accessing Q&A Page

1. **Login Required**:
   - User must be authenticated
   - If not logged in, redirected to `/login`

2. **Navigation**:
   - Click MessageSquare icon (💬) in sidebar
   - Direct URL: `/qa`
   - Keyboard shortcut: None (could add `Cmd+Q` in future)

3. **Page Load**:
   - QAPage component mounts
   - useExpertQuery hook initializes
   - Loads query history from localStorage (if exists)
   - Auto-focuses query input

4. **Query Submission**:
   - Enter query (min 5 characters)
   - Click "Chiedi agli Expert" button
   - Loading state shown
   - Response displayed when ready

5. **Feedback**:
   - Auto-show feedback section after 3 seconds
   - Thumbs up/down for inline feedback
   - Detailed feedback button (TODO: Task 10)

---

## Testing Checklist

### ✅ Manual Testing Performed

- [x] Route accessible at `/qa`
- [x] Navigation link visible in sidebar
- [x] Click navigation → page loads
- [x] Page requires authentication
- [x] Sidebar active indicator works
- [x] Tooltip shows on hover (desktop)
- [x] Mobile sidebar opens/closes correctly
- [x] Theme switching works (light/dark)
- [x] Focus mode hides sidebar
- [x] Syntax error fixed (Vite compiles)

### ⏳ Pending E2E Testing (Task 12)

- [ ] Submit query → get response
- [ ] Inline feedback → saves to database
- [ ] Query history → persists in localStorage
- [ ] Multiple queries → history updates
- [ ] Error states → displays correctly

---

## Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `frontend/src/App.tsx` | +1 import, +1 route | Added QAPage import and route |
| `frontend/src/components/layout/Sidebar.tsx` | +1 icon, +1 nav item | Added MessageSquare icon and navigation |
| `frontend/src/hooks/useExpertQuery.ts` | 1 fix (line 330) | Removed extra closing paren |

---

## Related Tasks

| Task | Status | Description |
|------|--------|-------------|
| Task 7 | ✅ Complete | Created QAPage.tsx component |
| Task 8 | ✅ Complete | Response display (integrated in QAPage) |
| Task 9 | ✅ Complete | Inline feedback (integrated in QAPage) |
| **Task 11** | **✅ Complete** | **Add QAPage to routing (this task)** |
| Task 10 | ⏳ Pending | Detailed feedback modal (3 sliders) |
| Task 12 | ⏳ Pending | End-to-end testing |

---

## Next Steps

### Immediate (Task 10)

Create **FeedbackDetailedForm** component:
- Modal/drawer UI
- 3 range sliders (retrieval, reasoning, synthesis 0-1)
- Comment textarea (optional)
- Submit button
- Integration with `submitDetailedFeedback()` from useExpertQuery

**File to create**: `/frontend/src/components/features/qa/FeedbackDetailedForm.tsx`

### After Task 10 (Task 12)

**End-to-End Testing**:
1. Start all services (MERL-T API, VisuaLex backend, frontend)
2. Test complete flow:
   - Login → Navigate to Q&A
   - Submit query → Verify response
   - Submit inline feedback → Check database
   - Submit detailed feedback → Check database
   - Query history → Verify localStorage
3. Test error cases:
   - Network errors
   - Invalid inputs
   - Empty responses
4. Performance testing:
   - Query response time < 5s
   - Feedback submission < 500ms
   - localStorage size limits

---

## Success Metrics (Sprint 1 Goal)

**Target**: Q&A Accuracy as primary metric

### User Engagement Metrics

- **Query Volume**: Track number of queries per day
- **Feedback Rate**: % of queries that receive feedback
- **Inline Feedback**: Avg rating (target: >= 4.0 / 5.0)
- **Detailed Feedback**: Avg scores (target: >= 0.7 / 1.0)
- **Refinement Rate**: % of queries with follow-up (target: >= 20%)

### Technical Metrics

- **Response Time**: < 5 seconds for 95th percentile
- **Error Rate**: < 5% of queries fail
- **Convergent Mode**: >= 70% of responses convergent
- **Expert Coverage**: All 4 experts used regularly

### Database Tracking

```sql
-- Query count
SELECT COUNT(*) FROM qa_traces WHERE created_at > NOW() - INTERVAL '24 hours';

-- Avg inline rating
SELECT AVG(inline_rating) FROM qa_feedback WHERE inline_rating IS NOT NULL;

-- Feedback participation rate
SELECT
  COUNT(DISTINCT trace_id) * 100.0 / (SELECT COUNT(*) FROM qa_traces) as feedback_rate
FROM qa_feedback;
```

---

## Known Issues

None at this time.

---

## Changelog

**2026-01-03**:
- ✅ Added QAPage route to App.tsx
- ✅ Added navigation link to Sidebar.tsx
- ✅ Fixed syntax error in useExpertQuery.ts:330
- ✅ Verified Vite compilation successful
- ✅ Updated todo list to mark Task 11 complete

---

*Sprint 1 - Task 11 completed successfully. Ready for Task 10 (Detailed Feedback Modal) or Task 12 (E2E Testing).*
