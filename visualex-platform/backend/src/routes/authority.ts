import { Router } from 'express';
import * as authorityController from '../controllers/authorityController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All authority routes require authentication
router.get('/authority', authenticate, authorityController.getAuthority);

export default router;
