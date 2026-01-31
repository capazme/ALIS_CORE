/**
 * Consent Controller
 * ==================
 *
 * Handles GDPR-compliant consent management for RLCF data participation.
 * Implements Art. 6 (lawful basis) and Art. 7 (consent conditions).
 *
 * Endpoints:
 * - GET /api/consent - Get current consent level
 * - PUT /api/consent - Update consent level (with audit logging)
 */
import { Request, Response } from 'express';
import { PrismaClient, ConsentLevel } from '@prisma/client';
import { z } from 'zod';
import crypto from 'crypto';
import { AppError } from '../middleware/errorHandler';
import { config } from '../config';

const prisma = new PrismaClient();

// Salt for IP hashing (separate from JWT for security isolation)
const IP_HASH_SALT = config.consentIpSalt;

// Consent level descriptions for UI
export const CONSENT_DESCRIPTIONS = {
  basic: {
    emoji: '🔒',
    name: 'Base',
    description: 'Nessun dato raccolto oltre la sessione. Solo uso del sistema.',
    dataCollected: ['Nessuno'],
  },
  learning: {
    emoji: '📊',
    name: 'Apprendimento',
    description: 'Query anonimizzate + feedback usati per il training RLCF.',
    dataCollected: ['Query anonimizzate', 'Feedback', 'Interazioni di ricerca'],
  },
  research: {
    emoji: '🔬',
    name: 'Ricerca',
    description: 'Dati aggregati disponibili per analisi accademica.',
    dataCollected: ['Query anonimizzate', 'Feedback', 'Pattern di utilizzo', 'Dati aggregati per ricerca'],
  },
};

// Consent level hierarchy (for downgrade detection)
const CONSENT_HIERARCHY: Record<ConsentLevel, number> = {
  basic: 0,
  learning: 1,
  research: 2,
};

// Validation schemas
const updateConsentSchema = z.object({
  consent_level: z.nativeEnum(ConsentLevel),
});

/**
 * Hash IP address for privacy-preserving audit logging
 */
const hashIP = (ip: string | undefined): string | null => {
  if (!ip) return null;
  return crypto.createHash('sha256').update(ip + IP_HASH_SALT).digest('hex').substring(0, 32);
};

/**
 * Get client IP from request (handles proxies)
 */
const getClientIP = (req: Request): string | undefined => {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string') {
    return forwarded.split(',')[0].trim();
  }
  return req.ip || req.socket.remoteAddress;
};

/**
 * Get current user's consent level and details
 */
export const getConsent = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  // Get or create consent record
  let consent = await prisma.userConsent.findUnique({
    where: { userId: req.user.id },
  });

  // If no consent record exists, create default (basic)
  if (!consent) {
    consent = await prisma.userConsent.create({
      data: {
        userId: req.user.id,
        consentLevel: ConsentLevel.basic,
        ipHash: hashIP(getClientIP(req)),
      },
    });

    // Log initial consent
    await prisma.consentAuditLog.create({
      data: {
        userId: req.user.id,
        previousLevel: null,
        newLevel: ConsentLevel.basic,
        ipHash: hashIP(getClientIP(req)),
        userAgent: req.headers['user-agent']?.substring(0, 500),
      },
    });
  }

  res.json({
    consent_level: consent.consentLevel,
    granted_at: consent.grantedAt,
    available_levels: Object.entries(CONSENT_DESCRIPTIONS).map(([key, value]) => ({
      level: key,
      ...value,
    })),
  });
};

/**
 * Update user's consent level
 */
export const updateConsent = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  const { consent_level } = updateConsentSchema.parse(req.body);

  // Get current consent (if exists)
  const currentConsent = await prisma.userConsent.findUnique({
    where: { userId: req.user.id },
  });

  const previousLevel = currentConsent?.consentLevel || null;
  const isDowngrade = previousLevel !== null &&
    CONSENT_HIERARCHY[consent_level] < CONSENT_HIERARCHY[previousLevel];

  // Update or create consent record
  const ipHash = hashIP(getClientIP(req));
  const userAgent = req.headers['user-agent']?.substring(0, 500);

  const updatedConsent = await prisma.userConsent.upsert({
    where: { userId: req.user.id },
    update: {
      consentLevel: consent_level,
      grantedAt: new Date(),
      ipHash,
    },
    create: {
      userId: req.user.id,
      consentLevel: consent_level,
      ipHash,
    },
  });

  // Create immutable audit log entry
  await prisma.consentAuditLog.create({
    data: {
      userId: req.user.id,
      previousLevel,
      newLevel: consent_level,
      ipHash,
      userAgent,
    },
  });

  // Build response message
  let message = 'Consenso aggiornato con successo';
  let warning: string | undefined;

  if (isDowngrade) {
    warning = 'Hai ridotto il tuo livello di consenso. I dati precedentemente raccolti ' +
      '(se presenti) rimarranno fino a quando non richiederai la cancellazione. ' +
      'Puoi richiedere la cancellazione nelle impostazioni privacy.';
  }

  res.json({
    message,
    warning,
    consent_level: updatedConsent.consentLevel,
    granted_at: updatedConsent.grantedAt,
    is_downgrade: isDowngrade,
  });
};

/**
 * Get consent audit history for current user
 */
export const getConsentHistory = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  const history = await prisma.consentAuditLog.findMany({
    where: { userId: req.user.id },
    orderBy: { changedAt: 'desc' },
    take: 50, // Limit to last 50 changes
  });

  res.json({
    history: history.map((entry) => ({
      id: entry.id,
      previous_level: entry.previousLevel,
      new_level: entry.newLevel,
      changed_at: entry.changedAt,
    })),
  });
};
