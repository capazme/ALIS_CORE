/**
 * Authority Controller
 * ====================
 *
 * Manages user authority scores for RLCF participation.
 * Authority determines influence on system learning.
 *
 * Formula: A_u(t) = 0.3·B_u + 0.5·T_u(t) + 0.2·P_u(t)
 * - B: Baseline credentials (from profile type)
 * - T: Track record (historical feedback accuracy)
 * - P: Recent performance (last N feedback quality)
 *
 * Endpoints:
 * - GET /api/authority - Get current user authority breakdown
 */
import { Request, Response } from 'express';
import { PrismaClient, ProfileType } from '@prisma/client';
import { AppError } from '../middleware/errorHandler';

const prisma = new PrismaClient();

// Authority score component weights (must sum to 1.0)
const WEIGHTS = {
  baseline: 0.3,          // α: Baseline credentials
  trackRecord: 0.5,       // β: Historical feedback accuracy
  recentPerformance: 0.2, // γ: Last N feedback quality
};

// Baseline score mapping based on profile type
const BASELINE_BY_PROFILE: Record<ProfileType, number> = {
  quick_consultation: 0.1,   // Casual user - minimal baseline
  assisted_research: 0.2,    // Student/researcher - moderate baseline
  expert_analysis: 0.4,      // Professional - higher baseline
  active_contributor: 0.5,   // Expert contributor - highest baseline
};

// Component descriptions for tooltips (Italian)
export const COMPONENT_DESCRIPTIONS = {
  baseline: {
    name: 'Credenziali Base',
    description: 'Punteggio basato sul tuo profilo e livello di esperienza. ' +
      'Seleziona un profilo più avanzato per aumentare questo valore.',
    weight: WEIGHTS.baseline,
    icon: '🎓',
  },
  trackRecord: {
    name: 'Storico Feedback',
    description: 'Misura l\'accuratezza storica dei tuoi feedback. ' +
      'Fornisci feedback precisi e utili per aumentare questo punteggio.',
    weight: WEIGHTS.trackRecord,
    icon: '📊',
  },
  recentPerformance: {
    name: 'Performance Recente',
    description: 'Qualità dei tuoi ultimi feedback. ' +
      'Feedback recenti di alta qualità aumentano rapidamente questo valore.',
    weight: WEIGHTS.recentPerformance,
    icon: '⚡',
  },
};

/**
 * Calculate the weighted authority score
 */
const calculateComputedScore = (
  baseline: number,
  trackRecord: number,
  recentPerformance: number
): number => {
  const score =
    WEIGHTS.baseline * baseline +
    WEIGHTS.trackRecord * trackRecord +
    WEIGHTS.recentPerformance * recentPerformance;

  // Clamp to 0.0 - 1.0 range
  return Math.max(0, Math.min(1, score));
};

/**
 * Get the baseline score for a given profile type
 */
const getBaselineForProfile = (profileType: ProfileType): number => {
  return BASELINE_BY_PROFILE[profileType] || 0.1;
};

/**
 * Get current user's authority score breakdown
 */
export const getAuthority = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  // Get user with profile type
  const user = await prisma.user.findUnique({
    where: { id: req.user.id },
    select: {
      profileType: true,
      authority: true,
    },
  });

  if (!user) {
    throw new AppError(404, 'User not found');
  }

  let authority = user.authority;

  // Lazy initialization: create authority record if it doesn't exist
  if (!authority) {
    const baselineScore = getBaselineForProfile(user.profileType);
    const computedScore = calculateComputedScore(baselineScore, 0, 0);

    authority = await prisma.userAuthority.create({
      data: {
        userId: req.user.id,
        baselineScore,
        trackRecordScore: 0,
        recentPerformance: 0,
        computedScore,
        feedbackCount: 0,
      },
    });

    // Also update the User.authorityScore field for consistency
    await prisma.user.update({
      where: { id: req.user.id },
      data: { authorityScore: computedScore },
    });
  }

  // Build response with breakdown
  res.json({
    authority_score: authority.computedScore,
    feedback_count: authority.feedbackCount,
    updated_at: authority.updatedAt,
    components: {
      baseline: {
        score: authority.baselineScore,
        weighted: authority.baselineScore * WEIGHTS.baseline,
        ...COMPONENT_DESCRIPTIONS.baseline,
      },
      track_record: {
        score: authority.trackRecordScore,
        weighted: authority.trackRecordScore * WEIGHTS.trackRecord,
        ...COMPONENT_DESCRIPTIONS.trackRecord,
      },
      recent_performance: {
        score: authority.recentPerformance,
        weighted: authority.recentPerformance * WEIGHTS.recentPerformance,
        ...COMPONENT_DESCRIPTIONS.recentPerformance,
      },
    },
    // Show message for new users with no feedback
    message: authority.feedbackCount === 0
      ? 'Contribuisci feedback per aumentare la tua autorità'
      : undefined,
  });
};

/**
 * Recalculate authority score when profile type changes
 * Called internally when user updates their profile
 */
export const recalculateBaseline = async (userId: string, profileType: ProfileType) => {
  const authority = await prisma.userAuthority.findUnique({
    where: { userId },
  });

  if (!authority) {
    // Will be created on next getAuthority call
    return;
  }

  const newBaseline = getBaselineForProfile(profileType);
  const computedScore = calculateComputedScore(
    newBaseline,
    authority.trackRecordScore,
    authority.recentPerformance
  );

  await prisma.userAuthority.update({
    where: { userId },
    data: {
      baselineScore: newBaseline,
      computedScore,
    },
  });

  // Also update User.authorityScore for consistency
  await prisma.user.update({
    where: { id: userId },
    data: { authorityScore: computedScore },
  });
};
