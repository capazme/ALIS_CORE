import { Router } from 'express';
import * as privacyController from '../controllers/privacyController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All privacy routes require authentication
router.post('/privacy/export', authenticate, privacyController.exportData);
router.post('/privacy/delete-account', authenticate, privacyController.requestDeletion);
router.post('/privacy/cancel-deletion', authenticate, privacyController.cancelDeletion);
router.get('/privacy/status', authenticate, privacyController.getPrivacyStatus);

export default router;
