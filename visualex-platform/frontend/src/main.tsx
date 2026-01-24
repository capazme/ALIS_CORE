import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { AuthProvider } from './contexts/AuthContext'
import { PluginProviderWrapper } from './components/PluginProviderWrapper'

// Configure plugins
const merltEnabled = import.meta.env.VITE_MERLT_ENABLED === 'true'
const plugins = [
  {
    id: 'merlt',
    enabled: merltEnabled,
    loader: () => import('@visualex/merlt-frontend/plugin'),
  },
]

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AuthProvider>
      <PluginProviderWrapper plugins={plugins}>
        <App />
      </PluginProviderWrapper>
    </AuthProvider>
  </StrictMode>,
)
