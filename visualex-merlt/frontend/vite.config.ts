import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig(({ mode }) => {
  // Plugin build mode
  if (mode === 'plugin') {
    return {
      plugins: [react()],
      build: {
        lib: {
          entry: path.resolve(__dirname, 'src/plugin/index.ts'),
          name: 'MerltPlugin',
          formats: ['es'],
          fileName: () => 'merlt-plugin.js',
        },
        rollupOptions: {
          // Externalize peer dependencies
          external: [
            'react',
            'react-dom',
            /^react\//,
            /^react-dom\//,
            // Also externalize react/jsx-runtime to avoid bundling it
            'react/jsx-runtime',
          ],
          output: {
            // Preserve dynamic imports for code splitting
            preserveModules: false,
            // Global names for UMD build (if needed)
            globals: {
              react: 'React',
              'react-dom': 'ReactDOM',
              'react/jsx-runtime': 'jsxRuntime',
            },
            // Ensure CSS is injected inline
            inlineDynamicImports: true,
            // Use ES modules
            format: 'es',
          },
        },
        outDir: 'dist',
        emptyOutDir: false,
        sourcemap: true,
        // Production optimizations
        minify: 'esbuild',
        // CSS handling - inline in JS for single-file plugin
        cssCodeSplit: false,
        // Target modern browsers
        target: 'es2020',
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, 'src'),
        },
      },
      // Optimize dependencies
      optimizeDeps: {
        include: [
          '@tanstack/react-query',
          'zustand',
          'recharts',
          'framer-motion',
          'lucide-react',
          'reagraph',
        ],
      },
    };
  }

  // Regular dev/build mode
  return {
    plugins: [react()],
    server: {
      port: 5174,
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, 'src'),
      },
    },
  };
});
