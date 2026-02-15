/**
 * MERLT Plugin Initialization
 *
 * Sets up the MERLT backend connection when the plugin is loaded.
 */

interface MerltConfig {
  apiBaseUrl: string;
  getAuthToken: () => Promise<string | null>;
  userId?: string;
}

let config: MerltConfig | null = null;
let websocket: WebSocket | null = null;

/**
 * Initialize MERLT services
 */
export async function initializeMerltServices(cfg: MerltConfig): Promise<void> {
  config = cfg;

  // Connect to MERLT WebSocket for real-time updates
  const merltWsUrl = cfg.apiBaseUrl.replace(/^http/, 'ws') + '/merlt/ws';

  try {
    const token = await cfg.getAuthToken();
    if (token) {
      websocket = new WebSocket(`${merltWsUrl}?token=${token}`);

      websocket.onopen = () => {
        console.log('[MERLT] WebSocket connected');
      };

      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMerltMessage(data);
      };

      websocket.onerror = (error) => {
        console.warn('[MERLT] WebSocket error:', error);
      };

      websocket.onclose = () => {
        console.log('[MERLT] WebSocket closed');
      };
    }
  } catch (error) {
    console.warn('[MERLT] WebSocket connection failed, using polling:', error);
  }
}

/**
 * Shutdown MERLT services
 */
export function shutdownMerltServices(): void {
  if (websocket) {
    websocket.close();
    websocket = null;
  }
  config = null;
}

/**
 * Get current MERLT configuration
 */
export function getMerltConfig(): MerltConfig | null {
  return config;
}

/**
 * Get the current user ID from MERLT config or auth token.
 *
 * Priority: 1) config.userId, 2) JWT sub claim from access_token, 3) "anonymous"
 */
export function getCurrentUserId(): string {
  if (config?.userId) return config.userId;

  const token = localStorage.getItem('access_token');
  if (token) {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      if (payload.sub) return payload.sub as string;
    } catch {
      // malformed token, fall through
    }
  }

  return 'anonymous';
}

/**
 * Handle incoming MERLT WebSocket messages
 */
function handleMerltMessage(data: { type: string; payload: unknown }): void {
  switch (data.type) {
    case 'enrichment:started':
      console.log('[MERLT] Enrichment started:', data.payload);
      break;
    case 'enrichment:completed':
      console.log('[MERLT] Enrichment completed:', data.payload);
      break;
    case 'validation:assigned':
      console.log('[MERLT] New validation assigned:', data.payload);
      break;
    case 'system:connected':
    case 'keepalive':
      // Expected system messages, no action needed
      break;
    default:
      console.log('[MERLT] Unknown message type:', data.type);
  }
}
