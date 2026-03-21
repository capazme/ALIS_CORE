/// <reference types="vite/client" />

// Ambient declaration for tsup DTS build (vite/client types not available in tsup)
declare interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_MERLT_API_URL: string;
  readonly [key: string]: string | undefined;
}

declare interface ImportMeta {
  readonly env: ImportMetaEnv;
}
