import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import 'express-async-errors';
import { config } from './config';
import { errorHandler } from './middleware/errorHandler';
import authRoutes from './routes/auth';
import profileRoutes from './routes/profile';
import adminRoutes from './routes/admin';
import folderRoutes from './routes/folders';
import bookmarkRoutes from './routes/bookmarks';
import highlightRoutes from './routes/highlights';
import annotationRoutes from './routes/annotations';
import dossierRoutes from './routes/dossiers';
import feedbackRoutes from './routes/feedback';
import historyRoutes from './routes/history';
import sharedEnvironmentRoutes from './routes/sharedEnvironments';
import consentRoutes from './routes/consent';
import authorityRoutes from './routes/authority';
import privacyRoutes from './routes/privacy';

const app = express();

// Trust proxy for rate limiting and correct client IPs behind nginx
app.set('trust proxy', 1);

// Security headers
app.use(helmet({
  contentSecurityPolicy: config.nodeEnv === 'production' ? undefined : false, // Disable CSP in dev for hot reload
  crossOriginEmbedderPolicy: false, // Allow embedding resources
}));

// Global rate limiting: 100 requests per minute per IP (disabled in test mode)
const isTestEnv = process.env.NODE_ENV === 'test' || process.env.E2E_TEST === 'true';
const globalLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: isTestEnv ? 0 : 100, // 0 = disabled in test mode
  message: { detail: 'Too many requests, please try again later' },
  standardHeaders: true,
  legacyHeaders: false,
  skip: () => isTestEnv, // Skip rate limiting entirely in test mode
});
app.use(globalLimiter);

// CORS
app.use(cors({
  origin: config.cors.origins,
  credentials: true,
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Health check
app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    environment: config.nodeEnv,
  });
});

// Routes
// MERL-T routes are loaded dynamically only if enabled and package exists
if (config.merlt.enabled) {
  import('@visualex/merlt-backend')
    .then(({ createMerltRouter }) => {
      app.use('/api/merlt', createMerltRouter({
        apiUrl: config.merlt.apiUrl,
        enabled: config.merlt.enabled,
      }));
      console.log('[MERL-T] Routes enabled');
    })
    .catch(() => {
      console.log('[MERL-T] Package not available, skipping routes');
    });
}
app.use('/api', authRoutes);
app.use('/api', profileRoutes);
app.use('/api', adminRoutes);
app.use('/api', folderRoutes);
app.use('/api', bookmarkRoutes);
app.use('/api', highlightRoutes);
app.use('/api', annotationRoutes);
app.use('/api', dossierRoutes);
app.use('/api', feedbackRoutes);
app.use('/api/history', historyRoutes);
app.use('/api', sharedEnvironmentRoutes);
app.use('/api', consentRoutes);
app.use('/api', authorityRoutes);
app.use('/api', privacyRoutes);

// 404 handler
app.use((_req, res) => {
  res.status(404).json({ detail: 'Not Found' });
});

// Error handler (must be last)
app.use(errorHandler);

// Start server if not in test mode
if (process.env.NODE_ENV !== 'test') {
  app.listen(config.port, () => {
    console.log(`
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║  VisuaLex Platform Backend                               ║
║                                                           ║
║  Status: Running                                          ║
║  Port: ${config.port}                                             ║
║  Environment: ${config.nodeEnv}                            ║
║                                                           ║
║  API Endpoints:                                           ║
║  - Health: http://localhost:${config.port}/api/health            ║
║  - Auth: http://localhost:${config.port}/api/auth/*              ║
║  - Admin: http://localhost:${config.port}/api/admin/*            ║
║  - Folders: http://localhost:${config.port}/api/folders/*        ║
║  - Bookmarks: http://localhost:${config.port}/api/bookmarks/*    ║
║  - Highlights: http://localhost:${config.port}/api/highlights/*  ║
║  - Annotations: http://localhost:${config.port}/api/annotations/*║
║  - Feedback: http://localhost:${config.port}/api/feedback/*      ║
║  - History: http://localhost:${config.port}/api/history/*        ║
║  - Bulletin: http://localhost:${config.port}/api/shared-environments/*║
╚═══════════════════════════════════════════════════════════╝
    `);
  });
}

export default app;
