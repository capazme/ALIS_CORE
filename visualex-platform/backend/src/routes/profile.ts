import { Router } from 'express';
import * as profileController from '../controllers/profileController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All profile routes require authentication
router.get('/profile', authenticate, profileController.getProfile);
router.put('/profile', authenticate, profileController.updateProfile);
router.put('/profile/preferences', authenticate, profileController.updatePreferences);

export default router;
