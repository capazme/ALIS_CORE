/**
 * Integration tests for auth routes
 */
import { describe, it, expect, beforeAll, afterAll, beforeEach } from '@jest/globals';
import request from 'supertest';

// Mock Prisma before importing app
jest.mock('@prisma/client', () => ({
  PrismaClient: jest.fn().mockImplementation(() => ({
    user: {
      findUnique: jest.fn(),
      findFirst: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
    },
    $connect: jest.fn(),
    $disconnect: jest.fn(),
  })),
}));

describe('Auth Routes', () => {
  let app: Express.Application;
  let mockPrisma: {
    user: {
      findUnique: jest.Mock;
      findFirst: jest.Mock;
      create: jest.Mock;
      update: jest.Mock;
    };
  };

  beforeAll(async () => {
    const appModule = await import('../../src/index');
    app = appModule.app;

    const { PrismaClient } = await import('@prisma/client');
    mockPrisma = new PrismaClient() as unknown as typeof mockPrisma;
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('POST /api/auth/register', () => {
    it('registers a new user successfully', async () => {
      mockPrisma.user.findFirst.mockResolvedValue(null);
      mockPrisma.user.create.mockResolvedValue({
        id: '1',
        email: 'newuser@example.com',
        name: 'New User',
        createdAt: new Date(),
      });

      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'newuser@example.com',
          password: 'StrongP@ss123',
          name: 'New User',
        });

      expect(response.status).toBe(201);
      expect(response.body).toHaveProperty('user');
      expect(response.body).toHaveProperty('token');
    });

    it('returns 400 for invalid email', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'invalid-email',
          password: 'StrongP@ss123',
          name: 'Test User',
        });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('returns 400 for weak password', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'test@example.com',
          password: '123',
          name: 'Test User',
        });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('returns 409 for existing email', async () => {
      mockPrisma.user.findFirst.mockResolvedValue({
        id: '1',
        email: 'existing@example.com',
      });

      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'existing@example.com',
          password: 'StrongP@ss123',
          name: 'Test User',
        });

      expect(response.status).toBe(409);
    });
  });

  describe('POST /api/auth/login', () => {
    it('logs in with valid credentials', async () => {
      const { hashPassword } = await import('../../src/utils/password');
      const hashedPassword = await hashPassword('correctPassword123');

      mockPrisma.user.findUnique.mockResolvedValue({
        id: '1',
        email: 'user@example.com',
        password: hashedPassword,
        name: 'Test User',
      });

      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'user@example.com',
          password: 'correctPassword123',
        });

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('token');
      expect(response.body).toHaveProperty('user');
    });

    it('returns 401 for invalid password', async () => {
      const { hashPassword } = await import('../../src/utils/password');
      const hashedPassword = await hashPassword('correctPassword123');

      mockPrisma.user.findUnique.mockResolvedValue({
        id: '1',
        email: 'user@example.com',
        password: hashedPassword,
      });

      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'user@example.com',
          password: 'wrongPassword',
        });

      expect(response.status).toBe(401);
    });

    it('returns 401 for non-existent user', async () => {
      mockPrisma.user.findUnique.mockResolvedValue(null);

      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'nonexistent@example.com',
          password: 'anyPassword123',
        });

      expect(response.status).toBe(401);
    });
  });

  describe('GET /api/auth/me', () => {
    it('returns user data with valid token', async () => {
      const { generateToken } = await import('../../src/utils/jwt');
      const token = generateToken({ userId: '1' });

      mockPrisma.user.findUnique.mockResolvedValue({
        id: '1',
        email: 'user@example.com',
        name: 'Test User',
      });

      const response = await request(app)
        .get('/api/auth/me')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('id', '1');
      expect(response.body).toHaveProperty('email', 'user@example.com');
    });

    it('returns 401 without token', async () => {
      const response = await request(app).get('/api/auth/me');

      expect(response.status).toBe(401);
    });

    it('returns 401 with invalid token', async () => {
      const response = await request(app)
        .get('/api/auth/me')
        .set('Authorization', 'Bearer invalid-token');

      expect(response.status).toBe(401);
    });
  });
});
