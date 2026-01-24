import { Router, type Request, type Response, type NextFunction } from 'express';

export interface MerltProxyConfig {
  apiUrl: string;
  enabled: boolean;
}

type RequestWithUser = Request & { user?: { id?: string; isAdmin?: boolean; isMerltEnabled?: boolean } };

export function createMerltRouter(config: MerltProxyConfig): Router {
  const router = Router();

  async function proxyToMerlt(req: Request, res: Response, _next: NextFunction) {
    if (!config.enabled) {
      return res.status(503).json({
        error: 'MERL-T integration disabled',
        message: 'Knowledge Graph features are not available',
      });
    }

    const requestWithUser = req as RequestWithUser;
    if (!requestWithUser.user?.id) {
      return res.status(401).json({
        error: 'Authentication required',
        message: 'MERL-T access requires authentication',
      });
    }

    if (!requestWithUser.user.isAdmin && !requestWithUser.user.isMerltEnabled) {
      return res.status(403).json({
        error: 'MERL-T access denied',
        message: 'MERL-T access is not enabled for this user',
      });
    }

    try {
      let targetPath = req.originalUrl.replace('/api/merlt', '');

      if (targetPath.startsWith('/api/experts')) {
        targetPath = targetPath.replace('/api/experts', '/api/v1/experts');
      } else if (!targetPath.startsWith('/api/v1')) {
        targetPath = '/api/v1' + targetPath;
      }

      const targetUrl = `${config.apiUrl}${targetPath}`;

      const fetchOptions: RequestInit = {
        method: req.method,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          ...(req.headers.authorization && {
            'Authorization': req.headers.authorization,
          }),
          ...(requestWithUser.user?.id && {
            'X-User-ID': requestWithUser.user.id,
          }),
        },
      };

      if (['POST', 'PUT', 'PATCH'].includes(req.method) && req.body) {
        fetchOptions.body = JSON.stringify(req.body);
      }

      const response = await fetch(targetUrl, fetchOptions);
      const contentType = response.headers.get('content-type');

      if (contentType?.includes('application/json')) {
        const data = await response.json();
        return res.status(response.status).json(data);
      }

      const text = await response.text();
      return res.status(response.status).send(text);
    } catch (error) {
      console.error('[MERL-T Proxy] Error:', error);

      if (error instanceof TypeError && error.message.includes('fetch')) {
        return res.status(503).json({
          error: 'MERL-T service unavailable',
          message: 'Knowledge Graph backend is not responding. Please ensure MERL-T is running.',
          details: process.env.NODE_ENV === 'development' ? String(error) : undefined,
        });
      }

      return res.status(500).json({
        error: 'Proxy error',
        message: 'Failed to communicate with MERL-T backend',
        details: process.env.NODE_ENV === 'development' ? String(error) : undefined,
      });
    }
  }

  router.post('/api/experts/query', proxyToMerlt);
  router.post('/api/experts/feedback/inline', proxyToMerlt);
  router.post('/api/experts/feedback/detailed', proxyToMerlt);
  router.post('/api/experts/feedback/source', proxyToMerlt);

  router.get('/enrichment/check-article', proxyToMerlt);
  router.post('/enrichment/live', proxyToMerlt);
  router.get('/enrichment/pending', proxyToMerlt);
  router.post('/enrichment/validate-entity', proxyToMerlt);
  router.post('/enrichment/validate-relation', proxyToMerlt);
  router.post('/enrichment/propose-entity', proxyToMerlt);
  router.post('/enrichment/propose-relation', proxyToMerlt);

  router.get('/authority/:userId', proxyToMerlt);

  router.get('/graph/check-article', proxyToMerlt);
  router.get('/graph/node/:nodeId', proxyToMerlt);
  router.get('/graph/subgraph', proxyToMerlt);
  router.get('/graph/entities/search', proxyToMerlt);
  router.post('/graph/resolve-norm', proxyToMerlt);

  router.get('/profile/full', proxyToMerlt);
  router.get('/profile/authority/domains', proxyToMerlt);
  router.get('/profile/stats/detailed', proxyToMerlt);
  router.patch('/profile/qualification', proxyToMerlt);
  router.patch('/profile/notifications', proxyToMerlt);

  router.all('/*', proxyToMerlt);

  return router;
}
