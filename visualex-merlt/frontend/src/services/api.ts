/**
 * Centralized API client using fetch with auth handling.
 */
type RequestMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';
const DEFAULT_TIMEOUT_MS = 30000;

async function refreshAccessToken(): Promise<string | null> {
  const refreshToken = localStorage.getItem('refresh_token');
  if (!refreshToken) {
    return null;
  }

  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });

  if (!response.ok) {
    return null;
  }

  const data = await response.json() as { access_token?: string; refresh_token?: string };
  if (!data.access_token || !data.refresh_token) {
    return null;
  }

  localStorage.setItem('access_token', data.access_token);
  localStorage.setItem('refresh_token', data.refresh_token);
  return data.access_token;
}

async function request<T>(
  method: RequestMethod,
  url: string,
  data?: unknown,
  options?: { timeout?: number },
  retryOnAuth: boolean = true
): Promise<T> {
  const token = localStorage.getItem('access_token');
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), options?.timeout ?? DEFAULT_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_BASE_URL}${url}`, {
      method,
      headers,
      body: data !== undefined ? JSON.stringify(data) : undefined,
      signal: controller.signal,
    });

    if (response.status === 401 && retryOnAuth) {
      const newToken = await refreshAccessToken();
      if (!newToken) {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
        throw new Error('Unauthorized');
      }

      return request<T>(method, url, data, options, false);
    }

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      const errorMessage = (errorPayload as { detail?: string }).detail || response.statusText;
      throw {
        status: response.status,
        message: errorMessage,
        data: errorPayload,
      };
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return await response.json() as T;
  } finally {
    window.clearTimeout(timeout);
  }
}

export const get = async <T = unknown>(url: string, params?: Record<string, unknown>): Promise<T> => {
  if (!params) {
    return request<T>('GET', url);
  }

  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null) {
      return;
    }
    searchParams.append(key, String(value));
  });

  const query = searchParams.toString();
  return request<T>('GET', query ? `${url}?${query}` : url);
};

export const post = async <T = unknown>(
  url: string,
  data?: unknown,
  options?: { timeout?: number }
): Promise<T> => request<T>('POST', url, data, options);

export const put = async <T = unknown>(url: string, data?: unknown): Promise<T> =>
  request<T>('PUT', url, data);

export const patch = async <T = unknown>(url: string, data?: unknown): Promise<T> =>
  request<T>('PATCH', url, data);

export const del = async <T = unknown>(url: string): Promise<T> =>
  request<T>('DELETE', url);

export default {
  get,
  post,
  put,
  patch,
  del,
};
