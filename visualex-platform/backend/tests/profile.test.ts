/**
 * Profile API endpoint tests
 */
import request from 'supertest';
import app from '../src/index';
import { PrismaClient } from '@prisma/client';
import { hashPassword } from '../src/utils/password';
import { generateAccessToken } from '../src/utils/jwt';

const prisma = new PrismaClient();

describe('Profile API', () => {
  let testUser: { id: string; email: string };
  let accessToken: string;
  let contributorUser: { id: string; email: string };
  let contributorToken: string;

  beforeAll(async () => {
    // Create test user with low authority
    const hashedPassword = await hashPassword('TestPassword123');
    testUser = await prisma.user.create({
      data: {
        email: 'profile-test@example.com',
        username: 'profiletest',
        password: hashedPassword,
        isActive: true,
        authorityScore: 0.3,
        profileType: 'assisted_research',
      },
    });
    accessToken = generateAccessToken(testUser.id, testUser.email);

    // Create contributor user with high authority
    contributorUser = await prisma.user.create({
      data: {
        email: 'contributor-test@example.com',
        username: 'contributortest',
        password: hashedPassword,
        isActive: true,
        authorityScore: 0.7,
        profileType: 'assisted_research',
      },
    });
    contributorToken = generateAccessToken(contributorUser.id, contributorUser.email);
  });

  afterAll(async () => {
    // Cleanup test users
    await prisma.userPreferences.deleteMany({
      where: { userId: { in: [testUser.id, contributorUser.id] } },
    });
    await prisma.user.deleteMany({
      where: { id: { in: [testUser.id, contributorUser.id] } },
    });
    await prisma.$disconnect();
  });

  describe('GET /api/profile', () => {
    it('returns 401 without authentication', async () => {
      const response = await request(app).get('/api/profile');
      expect(response.status).toBe(401);
    });

    it('returns user profile and available profiles', async () => {
      const response = await request(app)
        .get('/api/profile')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('profile_type', 'assisted_research');
      expect(response.body).toHaveProperty('authority_score', 0.3);
      expect(response.body).toHaveProperty('preferences');
      expect(response.body).toHaveProperty('available_profiles');
      expect(response.body.available_profiles).toHaveLength(4);
    });

    it('returns default preferences if none exist', async () => {
      const response = await request(app)
        .get('/api/profile')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.body.preferences).toEqual({
        theme: 'system',
        language: 'it',
        notifications_enabled: true,
      });
    });

    it('marks active_contributor as unavailable for low authority users', async () => {
      const response = await request(app)
        .get('/api/profile')
        .set('Authorization', `Bearer ${accessToken}`);

      const contributorProfile = response.body.available_profiles.find(
        (p: { type: string }) => p.type === 'active_contributor'
      );
      expect(contributorProfile.available).toBe(false);
    });

    it('marks active_contributor as available for high authority users', async () => {
      const response = await request(app)
        .get('/api/profile')
        .set('Authorization', `Bearer ${contributorToken}`);

      const contributorProfile = response.body.available_profiles.find(
        (p: { type: string }) => p.type === 'active_contributor'
      );
      expect(contributorProfile.available).toBe(true);
    });
  });

  describe('PUT /api/profile', () => {
    it('returns 401 without authentication', async () => {
      const response = await request(app)
        .put('/api/profile')
        .send({ profile_type: 'expert_analysis' });
      expect(response.status).toBe(401);
    });

    it('updates profile type to expert_analysis', async () => {
      const response = await request(app)
        .put('/api/profile')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ profile_type: 'expert_analysis' });

      expect(response.status).toBe(200);
      expect(response.body.profile_type).toBe('expert_analysis');
      expect(response.body.message).toBe('Profilo aggiornato con successo');
    });

    it('updates profile type to quick_consultation', async () => {
      const response = await request(app)
        .put('/api/profile')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ profile_type: 'quick_consultation' });

      expect(response.status).toBe(200);
      expect(response.body.profile_type).toBe('quick_consultation');
    });

    it('rejects active_contributor for low authority user', async () => {
      const response = await request(app)
        .put('/api/profile')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ profile_type: 'active_contributor' });

      expect(response.status).toBe(403);
      expect(response.body.detail).toContain('punteggio autorità');
    });

    it('allows active_contributor for high authority user', async () => {
      const response = await request(app)
        .put('/api/profile')
        .set('Authorization', `Bearer ${contributorToken}`)
        .send({ profile_type: 'active_contributor' });

      expect(response.status).toBe(200);
      expect(response.body.profile_type).toBe('active_contributor');
    });

    it('rejects invalid profile type', async () => {
      const response = await request(app)
        .put('/api/profile')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ profile_type: 'invalid_type' });

      expect(response.status).toBe(400);
    });
  });

  describe('PUT /api/profile/preferences', () => {
    it('returns 401 without authentication', async () => {
      const response = await request(app)
        .put('/api/profile/preferences')
        .send({ theme: 'dark' });
      expect(response.status).toBe(401);
    });

    it('updates theme preference', async () => {
      const response = await request(app)
        .put('/api/profile/preferences')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ theme: 'dark' });

      expect(response.status).toBe(200);
      expect(response.body.preferences.theme).toBe('dark');
    });

    it('updates language preference', async () => {
      const response = await request(app)
        .put('/api/profile/preferences')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ language: 'en' });

      expect(response.status).toBe(200);
      expect(response.body.preferences.language).toBe('en');
    });

    it('updates notifications preference', async () => {
      const response = await request(app)
        .put('/api/profile/preferences')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ notifications_enabled: false });

      expect(response.status).toBe(200);
      expect(response.body.preferences.notifications_enabled).toBe(false);
    });

    it('updates multiple preferences at once', async () => {
      const response = await request(app)
        .put('/api/profile/preferences')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({
          theme: 'light',
          language: 'it',
          notifications_enabled: true,
        });

      expect(response.status).toBe(200);
      expect(response.body.preferences).toEqual({
        theme: 'light',
        language: 'it',
        notifications_enabled: true,
      });
    });

    it('rejects invalid theme value', async () => {
      const response = await request(app)
        .put('/api/profile/preferences')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ theme: 'invalid' });

      expect(response.status).toBe(400);
    });

    it('rejects invalid language value', async () => {
      const response = await request(app)
        .put('/api/profile/preferences')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ language: 'fr' });

      expect(response.status).toBe(400);
    });
  });
});
