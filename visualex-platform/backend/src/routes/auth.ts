import { Router } from 'express';
import rateLimit from 'express-rate-limit';
import * as authController from '../controllers/authController';
import { authenticate } from '../middleware/auth';

const router = Router();

// Strict rate limiting for auth endpoints (brute force protection)
// Disabled in test mode to allow E2E tests to run smoothly
const isTestEnv = process.env.NODE_ENV === 'test' || process.env.E2E_TEST === 'true';
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: isTestEnv ? 0 : 5, // 0 = disabled in test mode, 5 in production
  message: { detail: 'Too many authentication attempts, please try again in 15 minutes' },
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: true, // Don't count successful logins
  skip: () => isTestEnv, // Skip rate limiting entirely in test mode
});

// Public routes (with stricter rate limiting)
router.post('/auth/register', authLimiter, authController.register);
router.post('/auth/login', authLimiter, authController.login);
router.post('/auth/refresh', authController.refresh);

// Email verification routes
router.get('/auth/verify-email/:token', authController.verifyEmail);
router.post('/auth/resend-verification', authLimiter, authController.resendVerificationEmail);

// Protected routes
router.get('/auth/me', authenticate, authController.getCurrentUser);
router.put('/auth/change-password', authenticate, authController.changePassword);

export default router;
