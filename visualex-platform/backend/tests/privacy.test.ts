/**
 * Privacy API endpoint tests (GDPR Art. 17, Art. 20)
 */
import request from 'supertest';
import app from '../src/index';
import { PrismaClient, ProfileType } from '@prisma/client';
import { hashPassword } from '../src/utils/password';
import { generateAccessToken } from '../src/utils/jwt';

const prisma = new PrismaClient();

describe('Privacy API', () => {
  let testUser: { id: string; email: string };
  let accessToken: string;
  const testPassword = 'TestPassword123';

  beforeAll(async () => {
    // Create test user
    const hashedPassword = await hashPassword(testPassword);
    testUser = await prisma.user.create({
      data: {
        email: 'privacy-test@example.com',
        username: 'privacytest',
        password: hashedPassword,
        isActive: true,
        profileType: ProfileType.assisted_research,
      },
    });
    accessToken = generateAccessToken(testUser.id, testUser.email);

    // Create related data for export test
    await prisma.userPreferences.create({
      data: {
        userId: testUser.id,
        theme: 'dark',
        language: 'it',
        notificationsEnabled: true,
      },
    });

    await prisma.userConsent.create({
      data: {
        userId: testUser.id,
        consentLevel: 'learning',
      },
    });

    await prisma.userAuthority.create({
      data: {
        userId: testUser.id,
        baselineScore: 0.2,
        trackRecordScore: 0.1,
        recentPerformance: 0.15,
        computedScore: 0.13,
        feedbackCount: 5,
      },
    });
  });

  afterAll(async () => {
    // Cleanup test data
    await prisma.userAuthority.deleteMany({ where: { userId: testUser.id } });
    await prisma.consentAuditLog.deleteMany({ where: { userId: testUser.id } });
    await prisma.userConsent.deleteMany({ where: { userId: testUser.id } });
    await prisma.userPreferences.deleteMany({ where: { userId: testUser.id } });
    await prisma.refreshToken.deleteMany({ where: { userId: testUser.id } });
    await prisma.user.delete({ where: { id: testUser.id } });
    await prisma.$disconnect();
  });

  describe('GET /api/privacy/status', () => {
    it('returns 401 without authentication', async () => {
      const response = await request(app).get('/api/privacy/status');
      expect(response.status).toBe(401);
    });

    it('returns privacy status for authenticated user', async () => {
      const response = await request(app)
        .get('/api/privacy/status')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('deletion_pending', false);
      expect(response.body).toHaveProperty('account_active', true);
      expect(response.body).toHaveProperty('consent_level');
    });

    it('shows no days remaining when deletion not pending', async () => {
      const response = await request(app)
        .get('/api/privacy/status')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body.days_remaining).toBeNull();
    });
  });

  describe('POST /api/privacy/export', () => {
    it('returns 401 without authentication', async () => {
      const response = await request(app).post('/api/privacy/export');
      expect(response.status).toBe(401);
    });

    it('exports user data as JSON', async () => {
      const response = await request(app)
        .post('/api/privacy/export')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.headers['content-type']).toContain('application/json');
    });

    it('export contains required sections', async () => {
      const response = await request(app)
        .post('/api/privacy/export')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('exported_at');
      expect(response.body).toHaveProperty('gdpr_reference');
      expect(response.body).toHaveProperty('user');
      expect(response.body).toHaveProperty('preferences');
      expect(response.body).toHaveProperty('consent');
      expect(response.body).toHaveProperty('authority');
    });

    it('export contains correct user data', async () => {
      const response = await request(app)
        .post('/api/privacy/export')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body.user.email).toBe('privacy-test@example.com');
      expect(response.body.user.username).toBe('privacytest');
      expect(response.body.user.profile_type).toBe('assisted_research');
    });

    it('export contains preferences', async () => {
      const response = await request(app)
        .post('/api/privacy/export')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body.preferences.theme).toBe('dark');
      expect(response.body.preferences.language).toBe('it');
    });

    it('export contains authority data', async () => {
      const response = await request(app)
        .post('/api/privacy/export')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body.authority.score).toBeCloseTo(0.13);
      expect(response.body.authority.feedback_count).toBe(5);
    });
  });

  describe('POST /api/privacy/delete-account', () => {
    it('returns 401 without authentication', async () => {
      const response = await request(app)
        .post('/api/privacy/delete-account')
        .send({ password: testPassword });
      expect(response.status).toBe(401);
    });

    it('returns 401 with wrong password', async () => {
      const response = await request(app)
        .post('/api/privacy/delete-account')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ password: 'wrongpassword' });

      expect(response.status).toBe(401);
    });

    it('requests deletion with correct password', async () => {
      const response = await request(app)
        .post('/api/privacy/delete-account')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ password: testPassword });

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('message');
      expect(response.body).toHaveProperty('grace_period_days', 30);
      expect(response.body).toHaveProperty('warning');
    });

    it('marks user as deletion pending', async () => {
      const user = await prisma.user.findUnique({
        where: { id: testUser.id },
      });

      expect(user?.deletionRequestedAt).not.toBeNull();
      expect(user?.isActive).toBe(false);
    });

    it('rejects duplicate deletion request', async () => {
      const response = await request(app)
        .post('/api/privacy/delete-account')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ password: testPassword });

      expect(response.status).toBe(400);
    });
  });

  describe('POST /api/privacy/cancel-deletion', () => {
    it('returns 401 without authentication', async () => {
      const response = await request(app).post('/api/privacy/cancel-deletion');
      expect(response.status).toBe(401);
    });

    it('cancels pending deletion', async () => {
      const response = await request(app)
        .post('/api/privacy/cancel-deletion')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('message');
      expect(response.body).toHaveProperty('account_status', 'active');
    });

    it('restores user to active state', async () => {
      const user = await prisma.user.findUnique({
        where: { id: testUser.id },
      });

      expect(user?.deletionRequestedAt).toBeNull();
      expect(user?.isActive).toBe(true);
    });

    it('rejects cancellation when no deletion pending', async () => {
      const response = await request(app)
        .post('/api/privacy/cancel-deletion')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(400);
    });
  });

  describe('Privacy status after deletion request', () => {
    beforeAll(async () => {
      // Request deletion again for this test
      await request(app)
        .post('/api/privacy/delete-account')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ password: testPassword, reason: 'Test reason' });
    });

    afterAll(async () => {
      // Cancel to clean up
      await request(app)
        .post('/api/privacy/cancel-deletion')
        .set('Authorization', `Bearer ${accessToken}`);
    });

    it('shows deletion pending in status', async () => {
      const response = await request(app)
        .get('/api/privacy/status')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body.deletion_pending).toBe(true);
      expect(response.body.days_remaining).toBeLessThanOrEqual(30);
      expect(response.body.deletion_reason).toBe('Test reason');
    });
  });
});
