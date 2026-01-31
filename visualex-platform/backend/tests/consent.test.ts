/**
 * Consent API endpoint tests
 */
import request from 'supertest';
import app from '../src/index';
import { PrismaClient, ConsentLevel } from '@prisma/client';
import { hashPassword } from '../src/utils/password';
import { generateAccessToken } from '../src/utils/jwt';

const prisma = new PrismaClient();

describe('Consent API', () => {
  let testUser: { id: string; email: string };
  let accessToken: string;

  beforeAll(async () => {
    // Create test user
    const hashedPassword = await hashPassword('TestPassword123');
    testUser = await prisma.user.create({
      data: {
        email: 'consent-test@example.com',
        username: 'consenttest',
        password: hashedPassword,
        isActive: true,
      },
    });
    accessToken = generateAccessToken(testUser.id, testUser.email);
  });

  afterAll(async () => {
    // Cleanup test data
    await prisma.consentAuditLog.deleteMany({
      where: { userId: testUser.id },
    });
    await prisma.userConsent.deleteMany({
      where: { userId: testUser.id },
    });
    await prisma.user.delete({
      where: { id: testUser.id },
    });
    await prisma.$disconnect();
  });

  describe('GET /api/consent', () => {
    it('returns 401 without authentication', async () => {
      const response = await request(app).get('/api/consent');
      expect(response.status).toBe(401);
    });

    it('returns default consent level (basic) for new user', async () => {
      const response = await request(app)
        .get('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('consent_level', 'basic');
      expect(response.body).toHaveProperty('granted_at');
      expect(response.body).toHaveProperty('available_levels');
      expect(response.body.available_levels).toHaveLength(3);
    });

    it('returns all three consent level options', async () => {
      const response = await request(app)
        .get('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`);

      const levels = response.body.available_levels.map((l: { level: string }) => l.level);
      expect(levels).toContain('basic');
      expect(levels).toContain('learning');
      expect(levels).toContain('research');
    });

    it('includes descriptions for each consent level', async () => {
      const response = await request(app)
        .get('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`);

      response.body.available_levels.forEach((level: { name: string; description: string; dataCollected: string[] }) => {
        expect(level).toHaveProperty('name');
        expect(level).toHaveProperty('description');
        expect(level).toHaveProperty('dataCollected');
        expect(Array.isArray(level.dataCollected)).toBe(true);
      });
    });
  });

  describe('PUT /api/consent', () => {
    it('returns 401 without authentication', async () => {
      const response = await request(app)
        .put('/api/consent')
        .send({ consent_level: 'learning' });
      expect(response.status).toBe(401);
    });

    it('updates consent level to learning', async () => {
      const response = await request(app)
        .put('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ consent_level: 'learning' });

      expect(response.status).toBe(200);
      expect(response.body.consent_level).toBe('learning');
      expect(response.body.message).toBe('Consenso aggiornato con successo');
      expect(response.body.is_downgrade).toBe(false);
    });

    it('updates consent level to research', async () => {
      const response = await request(app)
        .put('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ consent_level: 'research' });

      expect(response.status).toBe(200);
      expect(response.body.consent_level).toBe('research');
      expect(response.body.is_downgrade).toBe(false);
    });

    it('detects downgrade and includes warning', async () => {
      // First upgrade to research
      await request(app)
        .put('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ consent_level: 'research' });

      // Then downgrade to basic
      const response = await request(app)
        .put('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ consent_level: 'basic' });

      expect(response.status).toBe(200);
      expect(response.body.consent_level).toBe('basic');
      expect(response.body.is_downgrade).toBe(true);
      expect(response.body.warning).toBeDefined();
      expect(response.body.warning).toContain('ridotto');
    });

    it('rejects invalid consent level', async () => {
      const response = await request(app)
        .put('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ consent_level: 'invalid' });

      expect(response.status).toBe(400);
    });

    it('creates audit log entry on consent change', async () => {
      // Update consent
      await request(app)
        .put('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ consent_level: 'learning' });

      // Check audit log
      const auditEntries = await prisma.consentAuditLog.findMany({
        where: { userId: testUser.id },
        orderBy: { changedAt: 'desc' },
        take: 1,
      });

      expect(auditEntries.length).toBeGreaterThan(0);
      expect(auditEntries[0].newLevel).toBe('learning');
    });
  });

  describe('GET /api/consent/history', () => {
    it('returns 401 without authentication', async () => {
      const response = await request(app).get('/api/consent/history');
      expect(response.status).toBe(401);
    });

    it('returns consent change history', async () => {
      const response = await request(app)
        .get('/api/consent/history')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('history');
      expect(Array.isArray(response.body.history)).toBe(true);
    });

    it('history entries have required fields', async () => {
      const response = await request(app)
        .get('/api/consent/history')
        .set('Authorization', `Bearer ${accessToken}`);

      if (response.body.history.length > 0) {
        const entry = response.body.history[0];
        expect(entry).toHaveProperty('id');
        expect(entry).toHaveProperty('new_level');
        expect(entry).toHaveProperty('changed_at');
      }
    });

    it('history is ordered by most recent first', async () => {
      const response = await request(app)
        .get('/api/consent/history')
        .set('Authorization', `Bearer ${accessToken}`);

      if (response.body.history.length > 1) {
        const dates = response.body.history.map((e: { changed_at: string }) => new Date(e.changed_at));
        for (let i = 1; i < dates.length; i++) {
          expect(dates[i - 1].getTime()).toBeGreaterThanOrEqual(dates[i].getTime());
        }
      }
    });
  });

  describe('Audit Log Immutability', () => {
    it('audit log entries are preserved after multiple changes', async () => {
      // Make several consent changes
      await request(app)
        .put('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ consent_level: 'basic' });

      await request(app)
        .put('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ consent_level: 'learning' });

      await request(app)
        .put('/api/consent')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ consent_level: 'research' });

      // Check all entries exist
      const auditEntries = await prisma.consentAuditLog.findMany({
        where: { userId: testUser.id },
      });

      // Should have at least 3 entries from this test
      expect(auditEntries.length).toBeGreaterThanOrEqual(3);
    });
  });
});
