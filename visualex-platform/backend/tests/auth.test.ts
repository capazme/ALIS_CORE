import request from 'supertest';
import { PrismaClient } from '@prisma/client';
import app from '../src/index';

const prisma = new PrismaClient();

describe('Auth Integration Tests', () => {
    const testUser = {
        email: 'test@example.com',
        username: 'testuser',
        password: 'Password123',
    };

    beforeAll(async () => {
        // Clean up DB before tests
        await prisma.refreshToken.deleteMany();
        await prisma.user.deleteMany();
    });

    afterAll(async () => {
        // Clean up DB after tests
        await prisma.refreshToken.deleteMany();
        await prisma.user.deleteMany();
        await prisma.$disconnect();
    });

    describe('POST /api/auth/register', () => {
        it('should register a new user', async () => {
            const res = await request(app)
                .post('/api/auth/register')
                .send(testUser);

            expect(res.status).toBe(201);
            expect(res.body).toHaveProperty('message');
            expect(res.body.pending_approval).toBe(true);

            // Verify user in DB
            const user = await prisma.user.findUnique({
                where: { email: testUser.email },
            });
            expect(user).toBeTruthy();
            expect(user?.isActive).toBe(false);
        });

        it('should prevent duplicate registration', async () => {
            const res = await request(app)
                .post('/api/auth/register')
                .send(testUser);

            expect(res.status).toBe(400);
        });
    });

    describe('POST /api/auth/login', () => {
        beforeAll(async () => {
            // Manually activate the user for login test
            await prisma.user.update({
                where: { email: testUser.email },
                data: { isActive: true },
            });
        });

        it('should login with valid credentials', async () => {
            const res = await request(app)
                .post('/api/auth/login')
                .send({
                    email: testUser.email,
                    password: testUser.password,
                });

            expect(res.status).toBe(200);
            expect(res.body).toHaveProperty('access_token');
            expect(res.body).toHaveProperty('refresh_token');
            expect(res.body.user.email).toBe(testUser.email);
        });

        it('should reject invalid password', async () => {
            const res = await request(app)
                .post('/api/auth/login')
                .send({
                    email: testUser.email,
                    password: 'WrongPassword123',
                });

            expect(res.status).toBe(401);
        });
    });

    describe('Rate Limiting', () => {
        it('should block after 5 failed login attempts', async () => {
            // Make 5 failed attempts
            for (let i = 0; i < 5; i++) {
                await request(app)
                    .post('/api/auth/login')
                    .send({
                        email: 'nonexistent@example.com',
                        password: 'WrongPassword123',
                    });
            }

            // 6th attempt should be rate limited
            const res = await request(app)
                .post('/api/auth/login')
                .send({
                    email: 'nonexistent@example.com',
                    password: 'WrongPassword123',
                });

            expect(res.status).toBe(429);
            expect(res.body.detail).toContain('Too many');
        });
    });

    describe('POST /api/auth/refresh', () => {
        let refreshToken = '';

        beforeAll(async () => {
            // Get a fresh token
            const res = await request(app)
                .post('/api/auth/login')
                .send({
                    email: testUser.email,
                    password: testUser.password,
                });
            refreshToken = res.body.refresh_token;
        });

        it('should refresh access token with valid refresh token', async () => {
            const res = await request(app)
                .post('/api/auth/refresh')
                .send({ refresh_token: refreshToken });

            expect(res.status).toBe(200);
            expect(res.body).toHaveProperty('access_token');
            expect(res.body).toHaveProperty('refresh_token');
            expect(res.body.refresh_token).not.toBe(refreshToken); // Token rotation
        });

        it('should reject reused (old) refresh token', async () => {
            // Try to use the old token again
            const res = await request(app)
                .post('/api/auth/refresh')
                .send({ refresh_token: refreshToken });

            expect(res.status).toBe(401); // Should be rejected due to rotation
        });
    });
});
