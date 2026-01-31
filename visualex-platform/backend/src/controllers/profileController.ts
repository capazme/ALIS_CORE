import { Request, Response } from 'express';
import { PrismaClient, ProfileType } from '@prisma/client';
import { z } from 'zod';
import { AppError } from '../middleware/errorHandler';
import { recalculateBaseline } from './authorityController';

const prisma = new PrismaClient();

// Minimum authority score required for Active Contributor profile
const CONTRIBUTOR_MIN_AUTHORITY = 0.5;

// Profile type descriptions for UI
export const PROFILE_DESCRIPTIONS = {
  quick_consultation: {
    emoji: '⚡',
    name: 'Consultazione Rapida',
    description: 'Risposte veloci, minima interazione',
  },
  assisted_research: {
    emoji: '📖',
    name: 'Ricerca Assistita',
    description: 'Esplorazione guidata con suggerimenti',
  },
  expert_analysis: {
    emoji: '🔍',
    name: 'Analisi Esperta',
    description: 'Accesso completo a Expert trace e feedback',
  },
  active_contributor: {
    emoji: '🎓',
    name: 'Contributore Attivo',
    description: 'Feedback granulare, impatto sul training',
    requiresAuthority: CONTRIBUTOR_MIN_AUTHORITY,
  },
};

// Validation schemas
const updateProfileSchema = z.object({
  profile_type: z.nativeEnum(ProfileType),
});

const updatePreferencesSchema = z.object({
  theme: z.enum(['light', 'dark', 'system']).optional(),
  language: z.enum(['it', 'en']).optional(),
  notifications_enabled: z.boolean().optional(),
});

/**
 * Get current user's profile and preferences
 */
export const getProfile = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  // Get user with preferences
  const user = await prisma.user.findUnique({
    where: { id: req.user.id },
    include: { preferences: true },
  });

  if (!user) {
    throw new AppError(404, 'User not found');
  }

  // If no preferences exist, return defaults
  const preferences = user.preferences || {
    theme: 'system',
    language: 'it',
    notificationsEnabled: true,
  };

  res.json({
    profile_type: user.profileType,
    authority_score: user.authorityScore,
    preferences: {
      theme: preferences.theme,
      language: preferences.language,
      notifications_enabled: preferences.notificationsEnabled,
    },
    available_profiles: Object.entries(PROFILE_DESCRIPTIONS).map(([key, value]) => ({
      type: key,
      ...value,
      available: key !== 'active_contributor' || user.authorityScore >= CONTRIBUTOR_MIN_AUTHORITY,
    })),
  });
};

/**
 * Update user's profile type
 */
export const updateProfile = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  const { profile_type } = updateProfileSchema.parse(req.body);

  // Get current user to check authority score
  const user = await prisma.user.findUnique({
    where: { id: req.user.id },
  });

  if (!user) {
    throw new AppError(404, 'User not found');
  }

  // Validate contributor profile requires sufficient authority
  if (profile_type === ProfileType.active_contributor && user.authorityScore < CONTRIBUTOR_MIN_AUTHORITY) {
    throw new AppError(
      403,
      `Il profilo Contributore Attivo richiede un punteggio autorità di almeno ${CONTRIBUTOR_MIN_AUTHORITY}. ` +
        `Il tuo punteggio attuale è ${user.authorityScore.toFixed(2)}. ` +
        `Ti consigliamo il profilo Analisi Esperta.`
    );
  }

  // Update profile
  const updatedUser = await prisma.user.update({
    where: { id: req.user.id },
    data: { profileType: profile_type },
  });

  // Recalculate authority baseline when profile changes
  await recalculateBaseline(req.user.id, profile_type);

  res.json({
    message: 'Profilo aggiornato con successo',
    profile_type: updatedUser.profileType,
  });
};

/**
 * Update user's preferences
 */
export const updatePreferences = async (req: Request, res: Response) => {
  if (!req.user) {
    throw new AppError(401, 'Not authenticated');
  }

  const data = updatePreferencesSchema.parse(req.body);

  // Build update object with proper field names
  const updateData: Record<string, unknown> = {};
  if (data.theme !== undefined) updateData.theme = data.theme;
  if (data.language !== undefined) updateData.language = data.language;
  if (data.notifications_enabled !== undefined) updateData.notificationsEnabled = data.notifications_enabled;

  // Upsert preferences (create if not exist, update if exist)
  const preferences = await prisma.userPreferences.upsert({
    where: { userId: req.user.id },
    update: updateData,
    create: {
      userId: req.user.id,
      theme: data.theme || 'system',
      language: data.language || 'it',
      notificationsEnabled: data.notifications_enabled ?? true,
    },
  });

  res.json({
    message: 'Preferenze aggiornate con successo',
    preferences: {
      theme: preferences.theme,
      language: preferences.language,
      notifications_enabled: preferences.notificationsEnabled,
    },
  });
};
