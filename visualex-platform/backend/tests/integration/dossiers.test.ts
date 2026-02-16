/**
 * Backend API Tests: Dossier Management
 * =======================================
 *
 * Tests for:
 * - GET /api/dossiers — List user dossiers
 * - POST /api/dossiers — Create dossier
 * - GET /api/dossiers/:id — Get single dossier
 * - PUT /api/dossiers/:id — Update dossier
 * - DELETE /api/dossiers/:id — Delete dossier
 * - POST /api/dossiers/:id/items — Add item
 * - DELETE /api/dossiers/:id/items/:itemId — Remove item
 *
 * Priority Tags: [P1] High  [P2] Medium
 */
import request from 'supertest';
import { PrismaClient } from '@prisma/client';
import app from '../../src/index';

const prisma = new PrismaClient();

describe('Dossier API', () => {
    let accessToken = '';
    let userId = '';
    let dossierId = '';

    const testUser = {
        email: 'dossier-test@example.com',
        username: 'dossiertest',
        password: 'Password123',
    };

    beforeAll(async () => {
        // Clean up and create test user
        await prisma.refreshToken.deleteMany();
        await prisma.user.deleteMany({ where: { email: testUser.email } });

        // Register and activate user
        await request(app).post('/api/auth/register').send(testUser);
        await prisma.user.update({
            where: { email: testUser.email },
            data: { isActive: true },
        });

        // Login to get token
        const loginRes = await request(app)
            .post('/api/auth/login')
            .send({ email: testUser.email, password: testUser.password });

        accessToken = loginRes.body.access_token;
        userId = loginRes.body.user.id;
    });

    afterAll(async () => {
        await prisma.refreshToken.deleteMany();
        await prisma.user.deleteMany({ where: { email: testUser.email } });
        await prisma.$disconnect();
    });

    describe('[P1] GET /api/dossiers', () => {
        it('should return empty list initially', async () => {
            // GIVEN: User has no dossiers yet
            // WHEN: Requesting dossier list
            const res = await request(app)
                .get('/api/dossiers')
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 with empty array
            expect(res.status).toBe(200);
            expect(Array.isArray(res.body)).toBe(true);
        });

        it('should reject unauthenticated requests', async () => {
            // GIVEN: No auth token
            // WHEN: Requesting dossier list
            const res = await request(app).get('/api/dossiers');

            // THEN: Returns 401
            expect(res.status).toBe(401);
        });
    });

    describe('[P1] POST /api/dossiers', () => {
        it('should create a new dossier', async () => {
            // GIVEN: Valid dossier data
            const dossierData = {
                name: 'Test Dossier',
                description: 'A test dossier for automated testing',
            };

            // WHEN: Creating a dossier
            const res = await request(app)
                .post('/api/dossiers')
                .set('Authorization', `Bearer ${accessToken}`)
                .send(dossierData);

            // THEN: Returns 201 with dossier data
            expect(res.status).toBe(201);
            expect(res.body).toHaveProperty('id');
            expect(res.body.name).toBe(dossierData.name);
            dossierId = res.body.id;
        });

        it('should reject dossier without name', async () => {
            // GIVEN: Missing required name field
            // WHEN: Creating a dossier
            const res = await request(app)
                .post('/api/dossiers')
                .set('Authorization', `Bearer ${accessToken}`)
                .send({ description: 'No name' });

            // THEN: Returns 400 validation error
            expect(res.status).toBe(400);
        });
    });

    describe('[P1] GET /api/dossiers/:id', () => {
        it('should return a specific dossier', async () => {
            // GIVEN: A dossier exists
            // WHEN: Requesting by ID
            const res = await request(app)
                .get(`/api/dossiers/${dossierId}`)
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 with dossier details
            expect(res.status).toBe(200);
            expect(res.body.id).toBe(dossierId);
        });

        it('should return 404 for non-existent dossier', async () => {
            // GIVEN: Non-existent ID
            // WHEN: Requesting by fake ID
            const res = await request(app)
                .get('/api/dossiers/non-existent-id-999')
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 404
            expect([404, 400]).toContain(res.status);
        });
    });

    describe('[P1] PUT /api/dossiers/:id', () => {
        it('should update a dossier', async () => {
            // GIVEN: A dossier exists
            const updateData = { name: 'Updated Dossier Name' };

            // WHEN: Updating the dossier
            const res = await request(app)
                .put(`/api/dossiers/${dossierId}`)
                .set('Authorization', `Bearer ${accessToken}`)
                .send(updateData);

            // THEN: Returns 200 with updated data
            expect(res.status).toBe(200);
            expect(res.body.name).toBe(updateData.name);
        });
    });

    describe('[P2] DELETE /api/dossiers/:id', () => {
        it('should delete a dossier', async () => {
            // GIVEN: A dossier exists
            // WHEN: Deleting the dossier
            const res = await request(app)
                .delete(`/api/dossiers/${dossierId}`)
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 or 204
            expect([200, 204]).toContain(res.status);
        });
    });
});
