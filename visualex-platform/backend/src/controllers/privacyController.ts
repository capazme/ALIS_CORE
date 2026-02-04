/**
 * Privacy Controller
 * ==================
 *
 * Handles GDPR privacy rights for users.
 * - Art. 20: Data portability (export)
 * - Art. 17: Right to erasure (deletion)
 *
 * Endpoints:
 * - POST /api/privacy/export - Export user data
 * - POST /api/privacy/delete-account - Request account deletion
 * - POST /api/privacy/cancel-deletion - Cancel pending deletion
 * - GET /api/privacy/status - Get deletion status
 */
import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { z } from 'zod';
import { AppError } from '../middleware/errorHandler';
import { verifyPassword } from '../utils/password';

const prisma = new PrismaClient();

// Validation schemas
const deleteAccountSchema = z.object({
  password: z.string().min(1, 'Password richiesta'),
  reason: z.string().optional(),
});

/**
 * Export user data (GDPR Art. 20 - Data Portability)
 */
export const exportData = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  // Fetch all user data
  const user = await prisma.user.findUnique({
    where: { id: req.user.id },
    include: {
      preferences: true,
      consent: true,
      authority: true,
    },
  });

  if (!user) {
    throw new AppError(404, 'User not found');
  }

  // Fetch consent history
  const consentHistory = await prisma.consentAuditLog.findMany({
    where: { userId: req.user.id },
    orderBy: { changedAt: 'desc' },
  });

  // Build export object
  const exportData = {
    exported_at: new Date().toISOString(),
    gdpr_reference: 'Art. 20 GDPR - Right to data portability',
    user: {
      id: user.id,
      email: user.email,
      username: user.username,
      profile_type: user.profileType,
      is_verified: user.isVerified,
      created_at: user.createdAt.toISOString(),
      last_login_at: user.lastLoginAt?.toISOString() || null,
      login_count: user.loginCount,
    },
    preferences: user.preferences ? {
      theme: user.preferences.theme,
      language: user.preferences.language,
      notifications_enabled: user.preferences.notificationsEnabled,
    } : {
      theme: 'system',
      language: 'it',
      notifications_enabled: true,
    },
    consent: {
      current_level: user.consent?.consentLevel || 'basic',
      granted_at: user.consent?.grantedAt?.toISOString() || null,
      history: consentHistory.map(entry => ({
        previous_level: entry.previousLevel,
        new_level: entry.newLevel,
        changed_at: entry.changedAt.toISOString(),
      })),
    },
    authority: user.authority ? {
      score: user.authority.computedScore,
      baseline: user.authority.baselineScore,
      track_record: user.authority.trackRecordScore,
      recent_performance: user.authority.recentPerformance,
      feedback_count: user.authority.feedbackCount,
      updated_at: user.authority.updatedAt.toISOString(),
    } : {
      score: 0,
      baseline: 0,
      track_record: 0,
      recent_performance: 0,
      feedback_count: 0,
      updated_at: null,
    },
  };

  // Set headers for file download
  res.setHeader('Content-Type', 'application/json');
  res.setHeader(
    'Content-Disposition',
    `attachment; filename="visualex-data-export-${user.username}-${new Date().toISOString().split('T')[0]}.json"`
  );

  res.json(exportData);
};

/**
 * Request account deletion (GDPR Art. 17 - Right to Erasure)
 */
export const requestDeletion = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  const { password, reason } = deleteAccountSchema.parse(req.body);

  // Fetch user with password
  const user = await prisma.user.findUnique({
    where: { id: req.user.id },
  });

  if (!user) {
    throw new AppError(404, 'User not found');
  }

  // Check if already pending deletion
  if (user.deletionRequestedAt) {
    throw new AppError(400, 'La richiesta di cancellazione è già in corso');
  }

  // Verify password
  const isValid = await verifyPassword(password, user.password);
  if (!isValid) {
    throw new AppError(401, 'Password non corretta');
  }

  // Mark account for deletion
  await prisma.user.update({
    where: { id: req.user.id },
    data: {
      deletionRequestedAt: new Date(),
      deletionReason: reason || null,
      isActive: false, // Deactivate immediately
    },
  });

  // Revoke all refresh tokens
  await prisma.refreshToken.updateMany({
    where: { userId: req.user.id },
    data: { revoked: true },
  });

  res.json({
    message: 'Richiesta di cancellazione account ricevuta',
    deletion_requested_at: new Date().toISOString(),
    grace_period_days: 30,
    warning: 'Il tuo account sarà eliminato definitivamente dopo 30 giorni. ' +
      'Puoi annullare la richiesta effettuando il login entro questo periodo.',
  });
};

/**
 * Cancel pending account deletion
 */
export const cancelDeletion = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  // Fetch user
  const user = await prisma.user.findUnique({
    where: { id: req.user.id },
  });

  if (!user) {
    throw new AppError(404, 'User not found');
  }

  // Check if deletion is pending
  if (!user.deletionRequestedAt) {
    throw new AppError(400, 'Nessuna richiesta di cancellazione in corso');
  }

  // Cancel deletion
  await prisma.user.update({
    where: { id: req.user.id },
    data: {
      deletionRequestedAt: null,
      deletionReason: null,
      isActive: true,
    },
  });

  res.json({
    message: 'Richiesta di cancellazione annullata',
    account_status: 'active',
  });
};

/**
 * Get current privacy/deletion status
 */
export const getPrivacyStatus = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  // Fetch user
  const user = await prisma.user.findUnique({
    where: { id: req.user.id },
    select: {
      deletionRequestedAt: true,
      deletionReason: true,
      isActive: true,
      consent: {
        select: {
          consentLevel: true,
          grantedAt: true,
        },
      },
    },
  });

  if (!user) {
    throw new AppError(404, 'User not found');
  }

  // Calculate days remaining if deletion pending
  let daysRemaining: number | null = null;
  if (user.deletionRequestedAt) {
    const deletionDate = new Date(user.deletionRequestedAt);
    deletionDate.setDate(deletionDate.getDate() + 30);
    const now = new Date();
    daysRemaining = Math.max(0, Math.ceil((deletionDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)));
  }

  res.json({
    deletion_pending: !!user.deletionRequestedAt,
    deletion_requested_at: user.deletionRequestedAt?.toISOString() || null,
    deletion_reason: user.deletionReason,
    days_remaining: daysRemaining,
    account_active: user.isActive,
    consent_level: user.consent?.consentLevel || 'basic',
  });
};
