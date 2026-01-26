import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { z } from 'zod';
import crypto from 'crypto';
import { AppError } from '../middleware/errorHandler';

const prisma = new PrismaClient();

// Validation schemas
const createInvitationSchema = z.object({
  email: z.string().email().optional(),
});

const validateInvitationSchema = z.object({
  token: z.string().uuid(),
});

// Constants
const INVITATION_EXPIRY_DAYS = 7;

/**
 * Create a new invitation
 * POST /api/invitations
 * Requires authentication
 */
export const createInvitation = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  const { email } = createInvitationSchema.parse(req.body);

  // Generate secure token
  const token = crypto.randomUUID();

  // Calculate expiration date (7 days from now)
  const expiresAt = new Date();
  expiresAt.setDate(expiresAt.getDate() + INVITATION_EXPIRY_DAYS);

  // Create invitation
  const invitation = await prisma.invitation.create({
    data: {
      email: email || null,
      token,
      expiresAt,
      inviterId: req.user.id,
    },
  });

  // Build invitation URL (use environment variable or default)
  const baseUrl = process.env.FRONTEND_URL || 'http://localhost:5173';
  const invitationUrl = `${baseUrl}/register?token=${token}`;

  res.status(201).json({
    invitation_url: invitationUrl,
    token: invitation.token,
    expires_at: invitation.expiresAt.toISOString(),
    email: invitation.email,
  });
};

/**
 * Validate an invitation token
 * GET /api/invitations/:token/validate
 * Public endpoint
 */
export const validateInvitation = async (req: Request, res: Response) => {
  const { token } = validateInvitationSchema.parse({ token: req.params.token });

  const invitation = await prisma.invitation.findUnique({
    where: { token },
    include: {
      inviter: {
        select: {
          username: true,
          email: true,
        },
      },
    },
  });

  if (!invitation) {
    throw new AppError(404, 'Invitation not found');
  }

  // Check if already used
  if (invitation.usedAt) {
    throw new AppError(400, 'This invitation has already been used');
  }

  // Check if expired
  if (new Date() > invitation.expiresAt) {
    throw new AppError(400, 'This invitation has expired');
  }

  res.json({
    valid: true,
    email: invitation.email,
    expires_at: invitation.expiresAt.toISOString(),
    invited_by: invitation.inviter.username,
  });
};

/**
 * List invitations sent by the current user
 * GET /api/invitations
 * Requires authentication
 */
export const listMyInvitations = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  const invitations = await prisma.invitation.findMany({
    where: { inviterId: req.user.id },
    orderBy: { createdAt: 'desc' },
    select: {
      id: true,
      email: true,
      token: true,
      expiresAt: true,
      usedAt: true,
      createdAt: true,
    },
  });

  const result = invitations.map((inv) => ({
    id: inv.id,
    email: inv.email,
    token: inv.token,
    expires_at: inv.expiresAt.toISOString(),
    used_at: inv.usedAt?.toISOString() || null,
    created_at: inv.createdAt.toISOString(),
    status: inv.usedAt ? 'used' : new Date() > inv.expiresAt ? 'expired' : 'pending',
  }));

  res.json({ invitations: result });
};

/**
 * Revoke an invitation
 * DELETE /api/invitations/:id
 * Requires authentication, must be the inviter
 */
export const revokeInvitation = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  const { id } = req.params;

  const invitation = await prisma.invitation.findUnique({
    where: { id },
  });

  if (!invitation) {
    throw new AppError(404, 'Invitation not found');
  }

  if (invitation.inviterId !== req.user.id && !req.user.isAdmin) {
    throw new AppError(403, 'Not authorized to revoke this invitation');
  }

  if (invitation.usedAt) {
    throw new AppError(400, 'Cannot revoke an already used invitation');
  }

  await prisma.invitation.delete({
    where: { id },
  });

  res.json({ message: 'Invitation revoked successfully' });
};
