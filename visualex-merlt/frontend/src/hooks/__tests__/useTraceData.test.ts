import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';

const mockGetTraceWithSources = vi.fn();
const mockGetTraceValidity = vi.fn();

vi.mock('../../services/traceService', () => ({
  getTraceWithSources: (...args: unknown[]) => mockGetTraceWithSources(...args),
  getTraceValidity: (...args: unknown[]) => mockGetTraceValidity(...args),
}));

import { useTraceData } from '../useTraceData';

const mockTraceWithSources = {
  trace: {
    id: 'trace-123',
    query: 'Test query',
    timestamp: '2026-01-01T00:00:00Z',
    synthesis: 'Test synthesis text',
    confidence: 0.85,
    experts: [
      {
        expertId: 'literal',
        displayName: 'Letterale',
        interpretation: 'Literal interpretation',
        confidence: 0.9,
        weight: 0.4,
        sources: [],
      },
    ],
  },
  sourcesResponse: {
    traceId: 'trace-123',
    sources: [
      {
        sourceId: 'src-1',
        urn: 'urn:nir:stato:codice.civile:1942;262~art1218',
        label: 'Art. 1218 c.c.',
        chunkText: 'Test chunk',
        score: 0.95,
        expertId: 'literal',
      },
    ],
  },
};

const mockValidity = {
  traceId: 'trace-123',
  validity: {
    total: 1,
    vigente: 1,
    abrogato: 0,
    modificato: 0,
    unknown: 0,
    results: [
      {
        urn: 'urn:nir:stato:codice.civile:1942;262~art1218',
        status: 'vigente' as const,
      },
    ],
  },
};

describe('useTraceData', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('[P2] should return initial empty state when traceId is null', () => {
    const { result } = renderHook(() => useTraceData(null));

    expect(result.current.trace).toBeNull();
    expect(result.current.sources).toEqual([]);
    expect(result.current.validity).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('[P2] should fetch trace and sources on mount when traceId is provided', async () => {
    mockGetTraceWithSources.mockResolvedValue(mockTraceWithSources);
    mockGetTraceValidity.mockResolvedValue(mockValidity);

    const { result } = renderHook(() => useTraceData('trace-123'));

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.trace).toEqual(mockTraceWithSources.trace);
    expect(result.current.sources).toEqual(mockTraceWithSources.sourcesResponse.sources);
    expect(result.current.validity).toEqual(mockValidity.validity);
    expect(result.current.error).toBeNull();
  });

  it('[P2] should set loading state during fetch', async () => {
    let resolveTrace: (val: unknown) => void;
    mockGetTraceWithSources.mockReturnValue(
      new Promise((resolve) => { resolveTrace = resolve; }),
    );
    mockGetTraceValidity.mockResolvedValue(mockValidity);

    const { result } = renderHook(() => useTraceData('trace-123'));

    expect(result.current.isLoading).toBe(true);

    await act(async () => {
      resolveTrace!(mockTraceWithSources);
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
  });

  it('[P2] should set error state when trace fetch fails', async () => {
    mockGetTraceWithSources.mockRejectedValue(new Error('Network error'));
    mockGetTraceValidity.mockResolvedValue(mockValidity);

    const { result } = renderHook(() => useTraceData('trace-123'));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBe('Errore nel caricamento del trace');
    expect(result.current.trace).toBeNull();
  });

  it('[P2] should still set validity when trace fails but validity succeeds', async () => {
    mockGetTraceWithSources.mockRejectedValue(new Error('fail'));
    mockGetTraceValidity.mockResolvedValue(mockValidity);

    const { result } = renderHook(() => useTraceData('trace-123'));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.validity).toEqual(mockValidity.validity);
    expect(result.current.error).toBe('Errore nel caricamento del trace');
  });

  it('[P2] should still set trace when validity fails but trace succeeds', async () => {
    mockGetTraceWithSources.mockResolvedValue(mockTraceWithSources);
    mockGetTraceValidity.mockRejectedValue(new Error('fail'));

    const { result } = renderHook(() => useTraceData('trace-123'));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.trace).toEqual(mockTraceWithSources.trace);
    expect(result.current.validity).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('[P2] should refetch when refetch is called', async () => {
    mockGetTraceWithSources.mockResolvedValue(mockTraceWithSources);
    mockGetTraceValidity.mockResolvedValue(mockValidity);

    const { result } = renderHook(() => useTraceData('trace-123'));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockGetTraceWithSources).toHaveBeenCalledTimes(1);

    act(() => {
      result.current.refetch();
    });

    await waitFor(() => {
      expect(mockGetTraceWithSources).toHaveBeenCalledTimes(2);
    });
  });

  it('[P2] should reset state when traceId changes to null', async () => {
    mockGetTraceWithSources.mockResolvedValue(mockTraceWithSources);
    mockGetTraceValidity.mockResolvedValue(mockValidity);

    const { result, rerender } = renderHook(
      ({ id }) => useTraceData(id),
      { initialProps: { id: 'trace-123' as string | null } },
    );

    await waitFor(() => {
      expect(result.current.trace).not.toBeNull();
    });

    rerender({ id: null });

    expect(result.current.trace).toBeNull();
    expect(result.current.sources).toEqual([]);
    expect(result.current.validity).toBeNull();
  });

  it('[P2] should fetch new data when traceId changes', async () => {
    mockGetTraceWithSources.mockResolvedValue(mockTraceWithSources);
    mockGetTraceValidity.mockResolvedValue(mockValidity);

    const { result, rerender } = renderHook(
      ({ id }) => useTraceData(id),
      { initialProps: { id: 'trace-123' } },
    );

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const secondTrace = {
      ...mockTraceWithSources,
      trace: { ...mockTraceWithSources.trace, id: 'trace-456', query: 'Second query' },
    };
    mockGetTraceWithSources.mockResolvedValue(secondTrace);

    rerender({ id: 'trace-456' });

    await waitFor(() => {
      expect(result.current.trace?.id).toBe('trace-456');
    });

    expect(mockGetTraceWithSources).toHaveBeenCalledTimes(2);
  });
});
