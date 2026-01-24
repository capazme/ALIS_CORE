/**
 * Integration tests for bookmark routes
 */
import { describe, it, expect, beforeAll, beforeEach } from '@jest/globals';
import request from 'supertest';

// Mock Prisma
jest.mock('@prisma/client', () => ({
  PrismaClient: jest.fn().mockImplementation(() => ({
    bookmark: {
      findMany: jest.fn(),
      findUnique: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
      count: jest.fn(),
    },
    user: {
      findUnique: jest.fn(),
    },
    $connect: jest.fn(),
    $disconnect: jest.fn(),
  })),
}));

describe('Bookmark Routes', () => {
  let app: Express.Application;
  let authToken: string;
  let mockPrisma: {
    bookmark: {
      findMany: jest.Mock;
      findUnique: jest.Mock;
      create: jest.Mock;
      update: jest.Mock;
      delete: jest.Mock;
      count: jest.Mock;
    };
    user: {
      findUnique: jest.Mock;
    };
  };

  beforeAll(async () => {
    const appModule = await import('../../src/index');
    app = appModule.app;

    const { generateToken } = await import('../../src/utils/jwt');
    authToken = generateToken({ userId: 'user-1' });

    const { PrismaClient } = await import('@prisma/client');
    mockPrisma = new PrismaClient() as unknown as typeof mockPrisma;

    // Mock authenticated user
    mockPrisma.user.findUnique.mockResolvedValue({
      id: 'user-1',
      email: 'test@example.com',
      name: 'Test User',
    });
  });

  beforeEach(() => {
    jest.clearAllMocks();

    // Re-setup user mock after clearing
    mockPrisma.user.findUnique.mockResolvedValue({
      id: 'user-1',
      email: 'test@example.com',
      name: 'Test User',
    });
  });

  describe('GET /api/bookmarks', () => {
    it('returns user bookmarks', async () => {
      mockPrisma.bookmark.findMany.mockResolvedValue([
        {
          id: 'bm-1',
          userId: 'user-1',
          articleUrn: 'urn:nir:stato:codice.civile~art1453',
          title: 'Art. 1453 c.c.',
          createdAt: new Date(),
        },
        {
          id: 'bm-2',
          userId: 'user-1',
          articleUrn: 'urn:nir:stato:codice.civile~art1454',
          title: 'Art. 1454 c.c.',
          createdAt: new Date(),
        },
      ]);

      const response = await request(app)
        .get('/api/bookmarks')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveLength(2);
      expect(response.body[0]).toHaveProperty('articleUrn');
    });

    it('returns empty array for user with no bookmarks', async () => {
      mockPrisma.bookmark.findMany.mockResolvedValue([]);

      const response = await request(app)
        .get('/api/bookmarks')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toEqual([]);
    });

    it('returns 401 without authentication', async () => {
      const response = await request(app).get('/api/bookmarks');

      expect(response.status).toBe(401);
    });
  });

  describe('POST /api/bookmarks', () => {
    it('creates a new bookmark', async () => {
      mockPrisma.bookmark.create.mockResolvedValue({
        id: 'bm-new',
        userId: 'user-1',
        articleUrn: 'urn:nir:stato:codice.civile~art1453',
        title: 'Art. 1453 c.c.',
        notes: 'Risoluzione del contratto',
        createdAt: new Date(),
      });

      const response = await request(app)
        .post('/api/bookmarks')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          articleUrn: 'urn:nir:stato:codice.civile~art1453',
          title: 'Art. 1453 c.c.',
          notes: 'Risoluzione del contratto',
        });

      expect(response.status).toBe(201);
      expect(response.body).toHaveProperty('id');
      expect(response.body.articleUrn).toBe('urn:nir:stato:codice.civile~art1453');
    });

    it('returns 400 for missing required fields', async () => {
      const response = await request(app)
        .post('/api/bookmarks')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          // Missing articleUrn
          title: 'Art. 1453',
        });

      expect(response.status).toBe(400);
    });
  });

  describe('DELETE /api/bookmarks/:id', () => {
    it('deletes a bookmark', async () => {
      mockPrisma.bookmark.findUnique.mockResolvedValue({
        id: 'bm-1',
        userId: 'user-1',
      });
      mockPrisma.bookmark.delete.mockResolvedValue({
        id: 'bm-1',
      });

      const response = await request(app)
        .delete('/api/bookmarks/bm-1')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
    });

    it('returns 404 for non-existent bookmark', async () => {
      mockPrisma.bookmark.findUnique.mockResolvedValue(null);

      const response = await request(app)
        .delete('/api/bookmarks/non-existent')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(404);
    });

    it('returns 403 when deleting another user bookmark', async () => {
      mockPrisma.bookmark.findUnique.mockResolvedValue({
        id: 'bm-1',
        userId: 'other-user', // Different user
      });

      const response = await request(app)
        .delete('/api/bookmarks/bm-1')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(403);
    });
  });
});
