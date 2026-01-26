import path from 'path'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env from root directory (parent of frontend)
  const rootDir = path.resolve(__dirname, '..')
  const env = loadEnv(mode, rootDir, 'VITE_')
  const merltEnabled = env.VITE_MERLT_ENABLED === 'true'

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, 'src'),
        ...(merltEnabled
          ? {}
          : {
              '@visualex/merlt-frontend/plugin': path.resolve(
                __dirname,
                'src/plugins/merltPluginStub.ts'
              ),
            }),
      },
    },
    server: {
      proxy: {
        // Python API routes (port 5000)
        '/fetch_norma_data': 'http://localhost:5000',
        '/fetch_article_text': 'http://localhost:5000',
        '/stream_article_text': 'http://localhost:5000',
        '/fetch_brocardi_info': 'http://localhost:5000',
        '/fetch_all_data': 'http://localhost:5000',
        '/fetch_tree': 'http://localhost:5000',
        '/export_pdf': 'http://localhost:5000',
        '/version': 'http://localhost:5000',
        '/health': 'http://localhost:5000',
        // MERL-T API routes (port 8000) - DEVE essere prima di /api generico
        '/api/merlt': {
          target: 'http://localhost:8000',
          changeOrigin: true,
          ws: true,  // Abilita supporto WebSocket per Pipeline monitoring
          rewrite: (path) => path.replace(/^\/api\/merlt/, '/api/v1'),
        },
        // Node.js backend routes (port 3001)
        '/api': 'http://localhost:3001',
      }
    }
  }
})
