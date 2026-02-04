import { Router } from 'express';
import * as consentController from '../controllers/consentController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All consent routes require authentication
router.get('/consent', authenticate, consentController.getConsent);
router.put('/consent', authenticate, consentController.updateConsent);
router.get('/consent/history', authenticate, consentController.getConsentHistory);

export default router;
