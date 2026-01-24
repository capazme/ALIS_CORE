# PluginProvider Setup - Implementation Summary

## Changes Made

### 1. Created AuthContext (`src/contexts/AuthContext.tsx`)

A centralized authentication context that:
- Wraps the existing `useAuth` hook
- Provides auth state to the entire app
- Exposes `getAuthToken()` async function for plugins
- Maintains backward compatibility with existing code

**Key exports:**
- `AuthProvider` - Context provider component
- `useAuthContext()` - Hook to access auth state

### 2. Created PluginProviderWrapper (`src/components/PluginProviderWrapper.tsx`)

A wrapper component that:
- Consumes `AuthContext`
- Maps user data to plugin context format
- Extracts feature flags (`merlt`, `admin`) from user
- Provides API base URL from environment variables
- Passes all required props to `PluginProvider`

### 3. Updated main.tsx (`src/main.tsx`)

The application entry point now includes:
```tsx
<AuthProvider>
  <PluginProviderWrapper plugins={plugins}>
    <App />
  </PluginProviderWrapper>
</AuthProvider>
```

**Plugin configuration:**
```tsx
const plugins = [
  {
    id: 'merlt',
    enabled: true,
    loader: () => import('@visualex/merlt-frontend/plugin'),
  },
];
```

## Architecture

```
main.tsx
  └─ AuthProvider (new)
      └─ PluginProviderWrapper (new)
          └─ PluginProvider (existing)
              └─ App
```

## Data Flow

1. **User Authentication**
   - `useAuth` hook loads user from API
   - `AuthProvider` wraps auth state
   - `AuthContext` exposes user + `getAuthToken()`

2. **Plugin Initialization**
   - `PluginProviderWrapper` reads auth state
   - Maps user to plugin format:
     ```ts
     {
       id: user.id,
       features: ['merlt', 'admin'], // based on user flags
     }
     ```
   - Passes to `PluginProvider`

3. **Plugin Loading**
   - `PluginProvider` checks user features
   - If user has `merlt` feature → loads MERLT plugin
   - Plugin receives:
     - `user` (with id and features)
     - `apiBaseUrl` (from VITE_API_URL)
     - `getAuthToken` (async function)
     - `emit` (event bus function)

## Required Environment Variables

```env
VITE_API_URL=http://localhost:3001/api
```

## MERLT Plugin Requirements

The MERLT plugin will load if:
1. User is authenticated
2. User has `is_merlt_enabled: true` in database
3. Plugin package `@visualex/merlt-frontend` is installed
4. Plugin file at `@visualex/merlt-frontend/plugin` exports default plugin

## Testing Checklist

- [ ] App renders without errors
- [ ] AuthContext provides user data
- [ ] PluginProvider initializes
- [ ] MERLT plugin loads for enabled users
- [ ] Console shows: `[MERLT] Initializing plugin...`
- [ ] Console shows: `[MERLT] Plugin initialized`
- [ ] PluginSlot components render plugin content
- [ ] User without MERLT feature → plugin doesn't load
- [ ] Logout → plugin unloads properly

## Verification Steps

1. **Check console on app load:**
   ```
   [MERLT] Initializing plugin...
   [MERLT] Plugin initialized
   ```

2. **Check React DevTools:**
   - Should see `AuthProvider` → `PluginProviderWrapper` → `PluginProvider` hierarchy

3. **Check user features:**
   - User with `is_merlt_enabled: true` → plugin loads
   - User with `is_merlt_enabled: false` → plugin doesn't load

4. **Test PluginSlot rendering:**
   - Navigate to article page
   - Check if `article-sidebar` slot renders MERLT panel
   - Check if `article-toolbar` slot renders MERLT toolbar

## Backward Compatibility

- Existing code using `useAuth` hook → **still works**
- New code can use `useAuthContext` → **optional**
- All existing components → **no changes required**

## Next Steps

1. Build MERLT plugin package:
   ```bash
   cd visualex-merlt/frontend
   npm run build:lib
   ```

2. Start development server:
   ```bash
   cd visualex-platform/frontend
   npm run dev
   ```

3. Login as user with MERLT enabled

4. Check browser console for plugin initialization

5. Navigate to article page to test slots

## Troubleshooting

### Plugin doesn't load
- Check: User has `is_merlt_enabled: true`
- Check: MERLT package built (`visualex-merlt/frontend/dist/`)
- Check: Console for errors

### TypeScript errors
- Check: All imports are correct
- Check: `@visualex/merlt-frontend` package.json exports
- Run: `npm run typecheck`

### Runtime errors
- Check: VITE_API_URL is set
- Check: AuthProvider wraps app
- Check: User is authenticated

## Files Modified

1. `/src/main.tsx` - Added providers
2. `/src/contexts/AuthContext.tsx` - New file
3. `/src/components/PluginProviderWrapper.tsx` - New file

## Files Referenced

- `/src/lib/plugins/PluginProvider.tsx`
- `/src/lib/plugins/types.ts`
- `/src/hooks/useAuth.ts`
- `/src/services/authService.ts`
- `/visualex-merlt/frontend/src/plugin/index.ts`
