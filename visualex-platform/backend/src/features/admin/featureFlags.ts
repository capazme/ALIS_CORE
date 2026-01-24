/**
 * Feature Flags Management
 *
 * Admin API for managing user feature flags.
 * Allows enabling/disabling MERLT and other opt-in features per user.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { PrismaClient } from '@prisma/client';

const router = Router();
const prisma = new PrismaClient();

/**
 * Available feature flags
 */
const AVAILABLE_FEATURES = [
  {
    id: 'merlt',
    name: 'MERLT Research',
    description: 'Enable MERLT research features (entity extraction, validation)',
    requiresConsent: true,
  },
  {
    id: 'merlt_contribution',
    name: 'MERLT Contribution',
    description: 'Allow user to propose new entities and relations',
    requiresConsent: true,
    parent: 'merlt',
  },
  {
    id: 'merlt_validation',
    name: 'MERLT Validation',
    description: 'Allow user to validate proposed entities and relations',
    requiresConsent: true,
    parent: 'merlt',
  },
  {
    id: 'beta_features',
    name: 'Beta Features',
    description: 'Access to beta features before general release',
    requiresConsent: false,
  },
] as const;

/**
 * Middleware to check admin role
 */
function requireAdmin(req: Request, res: Response, next: NextFunction): void {
  const user = (req as { user?: { role: string } }).user;
  if (!user || user.role !== 'admin') {
    res.status(403).json({ error: 'Admin access required' });
    return;
  }
  next();
}

/**
 * GET /api/admin/features
 * List all available feature flags
 */
router.get('/features', requireAdmin, (_req: Request, res: Response) => {
  res.json({ features: AVAILABLE_FEATURES });
});

/**
 * GET /api/admin/users/:userId/features
 * Get feature flags for a specific user
 */
router.get(
  '/users/:userId/features',
  requireAdmin,
  async (req: Request, res: Response): Promise<void> => {
    const { userId } = req.params;

    try {
      const userFeatures = await prisma.userFeature.findMany({
        where: { userId },
      });

      const features = AVAILABLE_FEATURES.map((feature) => {
        const userFeature = userFeatures.find((uf) => uf.featureId === feature.id);
        return {
          ...feature,
          enabled: userFeature?.enabled ?? false,
          consentGiven: userFeature?.consentGiven ?? false,
          consentDate: userFeature?.consentDate ?? null,
          enabledBy: userFeature?.enabledBy ?? null,
          enabledAt: userFeature?.enabledAt ?? null,
        };
      });

      res.json({ userId, features });
    } catch (error) {
      console.error('Error fetching user features:', error);
      res.status(500).json({ error: 'Failed to fetch user features' });
    }
  }
);

/**
 * PUT /api/admin/users/:userId/features/:featureId
 * Enable or disable a feature for a user
 */
router.put(
  '/users/:userId/features/:featureId',
  requireAdmin,
  async (req: Request, res: Response): Promise<void> => {
    const { userId, featureId } = req.params;
    const { enabled } = req.body;
    const adminUser = (req as { user?: { id: string } }).user;

    // Validate feature exists
    const feature = AVAILABLE_FEATURES.find((f) => f.id === featureId);
    if (!feature) {
      res.status(400).json({ error: `Unknown feature: ${featureId}` });
      return;
    }

    // Check if parent feature is required
    if ('parent' in feature && feature.parent && enabled) {
      const parentFeature = await prisma.userFeature.findFirst({
        where: { userId, featureId: feature.parent, enabled: true },
      });

      if (!parentFeature) {
        res.status(400).json({
          error: `Parent feature "${feature.parent}" must be enabled first`,
        });
        return;
      }
    }

    try {
      const result = await prisma.userFeature.upsert({
        where: {
          userId_featureId: { userId, featureId },
        },
        create: {
          userId,
          featureId,
          enabled,
          enabledBy: adminUser?.id ?? 'system',
          enabledAt: enabled ? new Date() : null,
          consentGiven: false, // User must give consent separately
        },
        update: {
          enabled,
          enabledBy: adminUser?.id ?? 'system',
          enabledAt: enabled ? new Date() : null,
        },
      });

      // If disabling a parent feature, disable children too
      if (!enabled && !('parent' in feature)) {
        const childFeatures = AVAILABLE_FEATURES.filter(
          (f) => 'parent' in f && f.parent === featureId
        );
        if (childFeatures.length > 0) {
          await prisma.userFeature.updateMany({
            where: {
              userId,
              featureId: { in: childFeatures.map((f) => f.id) },
            },
            data: {
              enabled: false,
              enabledBy: adminUser?.id ?? 'system',
              enabledAt: null,
            },
          });
        }
      }

      res.json({
        success: true,
        feature: {
          ...feature,
          enabled: result.enabled,
          enabledBy: result.enabledBy,
          enabledAt: result.enabledAt,
        },
      });
    } catch (error) {
      console.error('Error updating user feature:', error);
      res.status(500).json({ error: 'Failed to update user feature' });
    }
  }
);

/**
 * POST /api/users/me/features/:featureId/consent
 * User gives consent for a feature (non-admin endpoint)
 */
router.post(
  '/users/me/features/:featureId/consent',
  async (req: Request, res: Response): Promise<void> => {
    const { featureId } = req.params;
    const { consent } = req.body;
    const user = (req as { user?: { id: string } }).user;

    if (!user) {
      res.status(401).json({ error: 'Authentication required' });
      return;
    }

    const feature = AVAILABLE_FEATURES.find((f) => f.id === featureId);
    if (!feature) {
      res.status(400).json({ error: `Unknown feature: ${featureId}` });
      return;
    }

    if (!feature.requiresConsent) {
      res.status(400).json({ error: 'This feature does not require consent' });
      return;
    }

    try {
      const result = await prisma.userFeature.upsert({
        where: {
          userId_featureId: { userId: user.id, featureId },
        },
        create: {
          userId: user.id,
          featureId,
          enabled: false, // Admin must enable separately
          consentGiven: consent,
          consentDate: consent ? new Date() : null,
        },
        update: {
          consentGiven: consent,
          consentDate: consent ? new Date() : null,
        },
      });

      res.json({
        success: true,
        feature: {
          ...feature,
          consentGiven: result.consentGiven,
          consentDate: result.consentDate,
        },
      });
    } catch (error) {
      console.error('Error updating consent:', error);
      res.status(500).json({ error: 'Failed to update consent' });
    }
  }
);

/**
 * GET /api/users/me/features
 * Get current user's enabled features (returns only IDs for frontend)
 */
router.get('/users/me/features', async (req: Request, res: Response): Promise<void> => {
  const user = (req as { user?: { id: string } }).user;

  if (!user) {
    res.status(401).json({ error: 'Authentication required' });
    return;
  }

  try {
    const userFeatures = await prisma.userFeature.findMany({
      where: {
        userId: user.id,
        enabled: true,
        consentGiven: true, // Only return features with consent
      },
    });

    res.json({
      features: userFeatures.map((f) => f.featureId),
    });
  } catch (error) {
    console.error('Error fetching user features:', error);
    res.status(500).json({ error: 'Failed to fetch features' });
  }
});

export default router;
