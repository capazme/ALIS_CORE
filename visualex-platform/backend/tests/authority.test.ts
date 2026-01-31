/**
 * Authority API endpoint tests
 */
import request from 'supertest';
import app from '../src/index';
import { PrismaClient, ProfileType } from '@prisma/client';
import { hashPassword } from '../src/utils/password';
import { generateAccessToken } from '../src/utils/jwt';

const prisma = new PrismaClient();

describe('Authority API', () => {
  let testUser: { id: string; email: string };
  let accessToken: string;

  beforeAll(async () => {
    // Create test user
    const hashedPassword = await hashPassword('TestPassword123');
    testUser = await prisma.user.create({
      data: {
        email: 'authority-test@example.com',
        username: 'authoritytest',
        password: hashedPassword,
        isActive: true,
        profileType: ProfileType.assisted_research,
      },
    });
    accessToken = generateAccessToken(testUser.id, testUser.email);
  });

  afterAll(async () => {
    // Cleanup test data
    await prisma.userAuthority.deleteMany({
      where: { userId: testUser.id },
    });
    await prisma.user.delete({
      where: { id: testUser.id },
    });
    await prisma.$disconnect();
  });

  describe('GET /api/authority', () => {
    it('returns 401 without authentication', async () => {
      const response = await request(app).get('/api/authority');
      expect(response.status).toBe(401);
    });

    it('creates authority record on first access (lazy initialization)', async () => {
      const response = await request(app)
        .get('/api/authority')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('authority_score');
      expect(response.body).toHaveProperty('feedback_count', 0);
      expect(response.body).toHaveProperty('components');
    });

    it('returns correct component structure', async () => {
      const response = await request(app)
        .get('/api/authority')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);

      // Check baseline component
      expect(response.body.components.baseline).toHaveProperty('score');
      expect(response.body.components.baseline).toHaveProperty('weighted');
      expect(response.body.components.baseline).toHaveProperty('name');
      expect(response.body.components.baseline).toHaveProperty('description');
      expect(response.body.components.baseline).toHaveProperty('weight', 0.3);

      // Check track record component
      expect(response.body.components.track_record).toHaveProperty('score');
      expect(response.body.components.track_record).toHaveProperty('weight', 0.5);

      // Check recent performance component
      expect(response.body.components.recent_performance).toHaveProperty('score');
      expect(response.body.components.recent_performance).toHaveProperty('weight', 0.2);
    });

    it('returns baseline based on profile type (assisted_research = 0.2)', async () => {
      const response = await request(app)
        .get('/api/authority')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body.components.baseline.score).toBe(0.2);
    });

    it('shows message for new users with no feedback', async () => {
      const response = await request(app)
        .get('/api/authority')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body.message).toBe('Contribuisci feedback per aumentare la tua autorità');
    });

    it('calculates computed score correctly', async () => {
      const response = await request(app)
        .get('/api/authority')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);

      const baseline = response.body.components.baseline;
      const trackRecord = response.body.components.track_record;
      const recentPerformance = response.body.components.recent_performance;

      // Computed = 0.3 * baseline + 0.5 * trackRecord + 0.2 * recentPerformance
      const expectedScore =
        baseline.weight * baseline.score +
        trackRecord.weight * trackRecord.score +
        recentPerformance.weight * recentPerformance.score;

      expect(response.body.authority_score).toBeCloseTo(expectedScore, 4);
    });

    it('returns updated_at timestamp', async () => {
      const response = await request(app)
        .get('/api/authority')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('updated_at');
      expect(new Date(response.body.updated_at)).toBeInstanceOf(Date);
    });
  });

  describe('Baseline recalculation on profile change', () => {
    it('updates baseline when profile type changes', async () => {
      // Get initial baseline
      const initialResponse = await request(app)
        .get('/api/authority')
        .set('Authorization', `Bearer ${accessToken}`);

      const initialBaseline = initialResponse.body.components.baseline.score;

      // Change profile to expert_analysis
      await request(app)
        .put('/api/profile')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ profile_type: 'expert_analysis' });

      // Get updated authority
      const updatedResponse = await request(app)
        .get('/api/authority')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(updatedResponse.body.components.baseline.score).toBe(0.4);
      expect(updatedResponse.body.components.baseline.score).toBeGreaterThan(initialBaseline);

      // Reset profile type
      await request(app)
        .put('/api/profile')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({ profile_type: 'assisted_research' });
    });
  });

  describe('Authority Score Computation', () => {
    it('authority score is between 0 and 1', async () => {
      const response = await request(app)
        .get('/api/authority')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);
      expect(response.body.authority_score).toBeGreaterThanOrEqual(0);
      expect(response.body.authority_score).toBeLessThanOrEqual(1);
    });

    it('component weights sum to 1.0', async () => {
      const response = await request(app)
        .get('/api/authority')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(response.status).toBe(200);

      const totalWeight =
        response.body.components.baseline.weight +
        response.body.components.track_record.weight +
        response.body.components.recent_performance.weight;

      expect(totalWeight).toBeCloseTo(1.0, 4);
    });
  });
});
