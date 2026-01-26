import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { z } from 'zod';
import crypto from 'crypto';
import { hashPassword, verifyPassword } from '../utils/password';
import { generateAccessToken, generateRefreshToken, verifyToken, verifyTokenType } from '../utils/jwt';
import { AppError } from '../middleware/errorHandler';
import { sendVerificationEmail } from '../services/emailService';

const prisma = new PrismaClient();

// Password validation regex: min 8 chars, at least 1 uppercase, 1 lowercase, 1 number
const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/;
const passwordErrorMessage = 'Password must be at least 8 characters with 1 uppercase, 1 lowercase, and 1 number';

// Validation schemas
const registerSchema = z.object({
  invitation_token: z.string().uuid(),
  email: z.string().email(),
  username: z.string().min(3).max(50),
  password: z.string().min(8).regex(passwordRegex, passwordErrorMessage),
  name: z.string().min(1).max(100).optional(),
  role: z.enum(['member', 'researcher']).default('member'),
});

// Email verification expiry: 24 hours
const EMAIL_VERIFICATION_EXPIRY_HOURS = 24;

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string(),
});

const refreshSchema = z.object({
  refresh_token: z.string(),
});

const changePasswordSchema = z.object({
  current_password: z.string(),
  new_password: z.string().min(8).regex(passwordRegex, passwordErrorMessage),
});

// Register - requires valid invitation token
export const register = async (req: Request, res: Response) => {
  const { invitation_token, email, username, password, name, role } = registerSchema.parse(req.body);

  // Validate invitation token
  const invitation = await prisma.invitation.findUnique({
    where: { token: invitation_token },
  });

  if (!invitation) {
    throw new AppError(400, 'Registration requires an invitation from an existing member');
  }

  // Check if invitation already used
  if (invitation.usedAt) {
    throw new AppError(400, 'This invitation has already been used');
  }

  // Check if invitation expired
  if (new Date() > invitation.expiresAt) {
    throw new AppError(400, 'This invitation has expired');
  }

  // If invitation was for specific email, verify it matches
  if (invitation.email && invitation.email.toLowerCase() !== email.toLowerCase()) {
    throw new AppError(400, 'This invitation was sent to a different email address');
  }

  // Check if user already exists
  const existingUser = await prisma.user.findFirst({
    where: {
      OR: [{ email }, { username }],
    },
  });

  if (existingUser) {
    if (existingUser.email === email) {
      throw new AppError(400, 'This email is already registered');
    }
    throw new AppError(400, 'This username is already taken');
  }

  // Hash password
  const hashedPassword = await hashPassword(password);

  // Generate email verification token
  const verificationToken = crypto.randomBytes(32).toString('hex');
  const verificationExpiry = new Date();
  verificationExpiry.setHours(verificationExpiry.getHours() + EMAIL_VERIFICATION_EXPIRY_HOURS);

  // Create user and mark invitation as used in a transaction
  const user = await prisma.$transaction(async (tx) => {
    // Create user with isActive: false (pending email verification)
    const newUser = await tx.user.create({
      data: {
        email,
        username,
        password: hashedPassword,
        name: name || null,
        role,
        isActive: false,
        isVerified: false,
      },
    });

    // Mark invitation as used
    await tx.invitation.update({
      where: { id: invitation.id },
      data: { usedAt: new Date() },
    });

    // Create email verification record
    await tx.emailVerification.create({
      data: {
        userId: newUser.id,
        token: verificationToken,
        expiresAt: verificationExpiry,
      },
    });

    return newUser;
  });

  // Send verification email (async, don't block response)
  sendVerificationEmail(user.email, verificationToken).catch((err) => {
    console.error('Failed to send verification email:', err);
  });

  res.status(201).json({
    message: 'Registration successful. Please check your email to verify your account.',
    verification_pending: true,
  });
};

// Verify email address
export const verifyEmail = async (req: Request, res: Response) => {
  const { token } = req.params;

  const verification = await prisma.emailVerification.findUnique({
    where: { token },
    include: { user: true },
  });

  if (!verification) {
    throw new AppError(400, 'Invalid verification token');
  }

  if (new Date() > verification.expiresAt) {
    throw new AppError(400, 'Verification token has expired. Please request a new one.');
  }

  // Update user as verified and active
  await prisma.$transaction(async (tx) => {
    await tx.user.update({
      where: { id: verification.userId },
      data: {
        isVerified: true,
        isActive: true, // Activate user on email verification
      },
    });

    // Delete the verification record
    await tx.emailVerification.delete({
      where: { id: verification.id },
    });
  });

  res.json({
    message: 'Email verified successfully. You can now log in.',
    verified: true,
  });
};

// Resend verification email
export const resendVerificationEmail = async (req: Request, res: Response) => {
  const { email } = z.object({ email: z.string().email() }).parse(req.body);

  const user = await prisma.user.findUnique({
    where: { email },
    include: { emailVerification: true },
  });

  if (!user) {
    // Don't reveal if user exists
    res.json({ message: 'If the email exists, a verification link has been sent.' });
    return;
  }

  if (user.isVerified) {
    throw new AppError(400, 'Email is already verified');
  }

  // Generate new verification token
  const verificationToken = crypto.randomBytes(32).toString('hex');
  const verificationExpiry = new Date();
  verificationExpiry.setHours(verificationExpiry.getHours() + EMAIL_VERIFICATION_EXPIRY_HOURS);

  // Upsert verification record
  await prisma.emailVerification.upsert({
    where: { userId: user.id },
    update: {
      token: verificationToken,
      expiresAt: verificationExpiry,
    },
    create: {
      userId: user.id,
      token: verificationToken,
      expiresAt: verificationExpiry,
    },
  });

  // Send verification email
  await sendVerificationEmail(user.email, verificationToken);

  res.json({ message: 'Verification email sent. Please check your inbox.' });
};

// Login
export const login = async (req: Request, res: Response) => {
  const { email, password } = loginSchema.parse(req.body);

  // Find user
  const user = await prisma.user.findUnique({
    where: { email },
  });

  if (!user) {
    throw new AppError(401, 'Invalid credentials');
  }

  // Verify password
  const isPasswordValid = await verifyPassword(password, user.password);

  if (!isPasswordValid) {
    throw new AppError(401, 'Invalid credentials');
  }

  if (!user.isActive) {
    throw new AppError(403, 'Il tuo account è in attesa di approvazione. Contatta un amministratore.');
  }

  // Update login stats
  const updatedUser = await prisma.user.update({
    where: { id: user.id },
    data: {
      loginCount: { increment: 1 },
      lastLoginAt: new Date(),
    },
  });

  // Generate tokens
  const accessToken = generateAccessToken(updatedUser.id, updatedUser.email);
  const refreshToken = generateRefreshToken(updatedUser.id, updatedUser.email);

  res.json({
    access_token: accessToken,
    refresh_token: refreshToken,
    token_type: 'Bearer',
    user: {
      id: updatedUser.id,
      email: updatedUser.email,
      username: updatedUser.username,
      is_active: updatedUser.isActive,
      is_verified: updatedUser.isVerified,
      is_admin: updatedUser.isAdmin,
      is_merlt_enabled: updatedUser.isMerltEnabled,
      created_at: updatedUser.createdAt,
      login_count: updatedUser.loginCount,
      last_login_at: updatedUser.lastLoginAt,
    },
  });
};

// Refresh token
export const refresh = async (req: Request, res: Response) => {
  const { refresh_token } = refreshSchema.parse(req.body);

  const payload = verifyToken(refresh_token);

  if (!payload || !verifyTokenType(payload, 'refresh')) {
    throw new AppError(401, 'Invalid refresh token');
  }

  // Verify user still exists and is active
  const user = await prisma.user.findUnique({
    where: { id: payload.userId },
  });

  if (!user || !user.isActive) {
    throw new AppError(401, 'User not found or inactive');
  }

  // Generate new tokens
  const accessToken = generateAccessToken(user.id, user.email);
  const refreshToken = generateRefreshToken(user.id, user.email);

  res.json({
    access_token: accessToken,
    refresh_token: refreshToken,
    token_type: 'Bearer',
  });
};

// Get current user
export const getCurrentUser = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  res.json({
    id: req.user.id,
    email: req.user.email,
    username: req.user.username,
    is_active: req.user.isActive,
    is_verified: req.user.isVerified,
    is_admin: req.user.isAdmin,
    is_merlt_enabled: req.user.isMerltEnabled,
    created_at: req.user.createdAt,
    login_count: req.user.loginCount,
    last_login_at: req.user.lastLoginAt,
  });
};

// Change password
export const changePassword = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  const { current_password, new_password } = changePasswordSchema.parse(req.body);

  // Verify current password
  const isPasswordValid = await verifyPassword(current_password, req.user.password);

  if (!isPasswordValid) {
    throw new AppError(400, 'Current password is incorrect');
  }

  // Hash new password
  const hashedPassword = await hashPassword(new_password);

  // Update password
  await prisma.user.update({
    where: { id: req.user.id },
    data: { password: hashedPassword },
  });

  res.json({ message: 'Password changed successfully' });
};
