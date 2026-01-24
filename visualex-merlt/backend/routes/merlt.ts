/**
 * MERL-T Proxy Routes
 * ===================
 *
 * Proxy layer che forwarda richieste al backend MERL-T.
 * Tutte le richieste a /api/merlt/* vengono inoltrate a MERLT_API_URL.
 *
 * Endpoints proxied:
 * - /api/merlt/api/experts/*     -> Expert System Q&A
 * - /api/merlt/enrichment/*      -> Enrichment & Validation
 * - /api/merlt/authority/*       -> User Authority
 * - /api/merlt/graph/*           -> Graph Queries
 */

import { Router, Request, Response, NextFunction } from 'express';
import { config } from '../config';

const router = Router();

// Proxy middleware
async function proxyToMerlt(req: Request, res: Response, _next: NextFunction) {

  if (!config.merlt.enabled) {
    return res.status(503).json({
      error: 'MERL-T integration disabled',
      message: 'Knowledge Graph features are not available',
    });
  }

  try {
    // Build target URL
    // Frontend sends: /api/merlt/enrichment/check-article
    // We need to map to: /api/v1/enrichment/check-article
    // Special case: /api/merlt/api/experts/* -> /api/v1/experts/*
    let targetPath = req.originalUrl.replace('/api/merlt', '');

    // Handle expert system routes (frontend uses /api/experts, MERL-T uses /api/v1/experts)
    if (targetPath.startsWith('/api/experts')) {
      targetPath = targetPath.replace('/api/experts', '/api/v1/experts');
    } else if (!targetPath.startsWith('/api/v1')) {
      // Add /api/v1 prefix for other routes
      targetPath = '/api/v1' + targetPath;
    }

    const targetUrl = `${config.merlt.apiUrl}${targetPath}`;

    console.log(`[MERL-T Proxy] ${req.method} ${targetUrl}`);

    // Forward request
    const fetchOptions: RequestInit = {
      method: req.method,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        // Forward auth header if present
        ...(req.headers.authorization && {
          'Authorization': req.headers.authorization,
        }),
        // Forward user ID from JWT if available
        ...(req.user?.id && {
          'X-User-ID': req.user.id,
        }),
      },
    };

    // Add body for POST/PUT/PATCH
    if (['POST', 'PUT', 'PATCH'].includes(req.method) && req.body) {
      fetchOptions.body = JSON.stringify(req.body);
    }

    const response = await fetch(targetUrl, fetchOptions);

    // Handle response
    const contentType = response.headers.get('content-type');

    if (contentType?.includes('application/json')) {
      const data = await response.json();
      return res.status(response.status).json(data);
    } else {
      const text = await response.text();
      return res.status(response.status).send(text);
    }
  } catch (error) {
    console.error('[MERL-T Proxy] Error:', error);

    // Check if MERL-T is unreachable
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return res.status(503).json({
        error: 'MERL-T service unavailable',
        message: 'Knowledge Graph backend is not responding. Please ensure MERL-T is running.',
        details: config.nodeEnv === 'development' ? String(error) : undefined,
      });
    }

    return res.status(500).json({
      error: 'Proxy error',
      message: 'Failed to communicate with MERL-T backend',
      details: config.nodeEnv === 'development' ? String(error) : undefined,
    });
  }
}

// =============================================================================
// EXPERT SYSTEM (Q&A)
// =============================================================================

// Query experts
router.post('/api/experts/query', proxyToMerlt);

// Feedback endpoints
router.post('/api/experts/feedback/inline', proxyToMerlt);
router.post('/api/experts/feedback/detailed', proxyToMerlt);
router.post('/api/experts/feedback/source', proxyToMerlt);

// =============================================================================
// ENRICHMENT & VALIDATION
// =============================================================================

// Check if article is in graph
router.get('/enrichment/check-article', proxyToMerlt);

// Live enrichment
router.post('/enrichment/live', proxyToMerlt);

// Pending queue
router.get('/enrichment/pending', proxyToMerlt);

// Validate entity/relation
router.post('/enrichment/validate-entity', proxyToMerlt);
router.post('/enrichment/validate-relation', proxyToMerlt);

// Propose new entity/relation
router.post('/enrichment/propose-entity', proxyToMerlt);
router.post('/enrichment/propose-relation', proxyToMerlt);

// =============================================================================
// AUTHORITY
// =============================================================================

router.get('/authority/:userId', proxyToMerlt);

// =============================================================================
// GRAPH QUERIES
// =============================================================================

router.get('/graph/check-article', proxyToMerlt);
router.get('/graph/node/:nodeId', proxyToMerlt);
router.get('/graph/subgraph', proxyToMerlt);

// Entity search for autocomplete (R3)
router.get('/graph/entities/search', proxyToMerlt);

// Norm resolver (R5) - risolve citazioni in linguaggio naturale
router.post('/graph/resolve-norm', proxyToMerlt);

// =============================================================================
// PROFILE
// =============================================================================

router.get('/profile/full', proxyToMerlt);
router.get('/profile/authority/domains', proxyToMerlt);
router.get('/profile/stats/detailed', proxyToMerlt);
router.patch('/profile/qualification', proxyToMerlt);
router.patch('/profile/notifications', proxyToMerlt);

// =============================================================================
// CATCH-ALL PROXY
// =============================================================================

// Fallback: proxy any other MERL-T request
router.all('/*', proxyToMerlt);

export default router;
