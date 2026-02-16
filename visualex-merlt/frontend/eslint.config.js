import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';
import globals from 'globals';

export default tseslint.config(
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['src/**/*.{ts,tsx}'],
    plugins: {
      'react-hooks': reactHooks,
    },
    languageOptions: {
      globals: {
        ...globals.browser,
      },
    },
    rules: {
      // React hooks
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',

      // No any
      '@typescript-eslint/no-explicit-any': 'warn',

      // No console in production
      'no-console': ['warn', { allow: ['warn', 'error'] }],

      // Allow unused vars with _ prefix
      '@typescript-eslint/no-unused-vars': ['warn', {
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
      }],
    },
  },
  {
    ignores: ['dist/**', 'node_modules/**', '*.config.*'],
  },
);
