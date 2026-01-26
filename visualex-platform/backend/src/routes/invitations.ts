import { Router } from 'express';
import rateLimit from 'express-rate-limit';
import * as invitationController from '../controllers/invitationController';
import { authenticate } from '../middleware/auth';

const router = Router();

// Rate limiting for invitation creation (prevent spam)
const invitationLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 10, // 10 invitations per hour
  message: { detail: 'Too many invitations created, please try again later' },
  standardHeaders: true,
  legacyHeaders: false,
});

// Public route - validate invitation token
router.get('/invitations/:token/validate', invitationController.validateInvitation);

// Protected routes
router.post('/invitations', authenticate, invitationLimiter, invitationController.createInvitation);
router.get('/invitations', authenticate, invitationController.listMyInvitations);
router.delete('/invitations/:id', authenticate, invitationController.revokeInvitation);

export default router;
