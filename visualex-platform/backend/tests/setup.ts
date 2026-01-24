/**
 * Jest setup for backend tests
 */
import { jest } from '@jest/globals';

// Set test environment
process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test-secret-key-for-testing';
process.env.DATABASE_URL = 'postgresql://test:test@localhost:5432/test_db';

// Mock Prisma client
jest.mock('@prisma/client', () => ({
  PrismaClient: jest.fn().mockImplementation(() => ({
    user: {
      findUnique: jest.fn(),
      findMany: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
    },
    bookmark: {
      findMany: jest.fn(),
      create: jest.fn(),
      delete: jest.fn(),
    },
    highlight: {
      findMany: jest.fn(),
      create: jest.fn(),
      delete: jest.fn(),
    },
    dossier: {
      findMany: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
    },
    $connect: jest.fn(),
    $disconnect: jest.fn(),
  })),
}));

// Global test timeout
jest.setTimeout(10000);
