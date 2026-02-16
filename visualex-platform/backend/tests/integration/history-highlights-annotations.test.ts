/**
 * Backend API Tests: History & Highlights & Annotations
 * ======================================================
 *
 * Tests for:
 * - GET /api/history — List search history
 * - POST /api/history — Add history entry
 * - DELETE /api/history/:id — Delete history entry
 * - DELETE /api/history — Clear all history
 * - POST /api/highlights — Create highlight
 * - GET /api/highlights — Get highlights
 * - DELETE /api/highlights/:id — Delete highlight
 * - POST /api/annotations — Create annotation
 * - GET /api/annotations — Get annotations
 * - DELETE /api/annotations/:id — Delete annotation
 *
 * Priority Tags: [P1] High  [P2] Medium
 */
import request from 'supertest';
import { PrismaClient } from '@prisma/client';
import app from '../../src/index';

const prisma = new PrismaClient();

describe('History API', () => {
    let accessToken = '';

    const testUser = {
        email: 'history-test@example.com',
        username: 'historytest',
        password: 'Password123',
    };

    beforeAll(async () => {
        await prisma.refreshToken.deleteMany();
        await prisma.user.deleteMany({ where: { email: testUser.email } });

        await request(app).post('/api/auth/register').send(testUser);
        await prisma.user.update({
            where: { email: testUser.email },
            data: { isActive: true },
        });

        const loginRes = await request(app)
            .post('/api/auth/login')
            .send({ email: testUser.email, password: testUser.password });

        accessToken = loginRes.body.access_token;
    });

    afterAll(async () => {
        await prisma.refreshToken.deleteMany();
        await prisma.user.deleteMany({ where: { email: testUser.email } });
        await prisma.$disconnect();
    });

    describe('[P1] History CRUD', () => {
        let historyId = '';

        it('should return empty history initially', async () => {
            // GIVEN: User has no history
            // WHEN: Requesting history
            const res = await request(app)
                .get('/api/history')
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 with empty array
            expect(res.status).toBe(200);
            expect(Array.isArray(res.body)).toBe(true);
        });

        it('should add a history entry', async () => {
            // GIVEN: Valid history data
            const historyData = {
                query: 'codice civile art. 1',
                type: 'keyword',
            };

            // WHEN: Adding history entry
            const res = await request(app)
                .post('/api/history')
                .set('Authorization', `Bearer ${accessToken}`)
                .send(historyData);

            // THEN: Returns 201 with entry data
            expect(res.status).toBe(201);
            expect(res.body).toHaveProperty('id');
            historyId = res.body.id;
        });

        it('should delete a single history entry', async () => {
            // GIVEN: A history entry exists
            // WHEN: Deleting the entry
            const res = await request(app)
                .delete(`/api/history/${historyId}`)
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 or 204
            expect([200, 204]).toContain(res.status);
        });

        it('should reject unauthenticated requests', async () => {
            // GIVEN: No auth token
            // WHEN: Requesting history
            const res = await request(app).get('/api/history');

            // THEN: Returns 401
            expect(res.status).toBe(401);
        });
    });
});

describe('Highlights API', () => {
    let accessToken = '';
    let highlightId = '';

    const testUser = {
        email: 'highlight-test@example.com',
        username: 'highlighttest',
        password: 'Password123',
    };

    beforeAll(async () => {
        await prisma.refreshToken.deleteMany();
        await prisma.user.deleteMany({ where: { email: testUser.email } });

        await request(app).post('/api/auth/register').send(testUser);
        await prisma.user.update({
            where: { email: testUser.email },
            data: { isActive: true },
        });

        const loginRes = await request(app)
            .post('/api/auth/login')
            .send({ email: testUser.email, password: testUser.password });

        accessToken = loginRes.body.access_token;
    });

    afterAll(async () => {
        await prisma.refreshToken.deleteMany();
        await prisma.user.deleteMany({ where: { email: testUser.email } });
        await prisma.$disconnect();
    });

    describe('[P1] Highlights CRUD', () => {
        it('should create a highlight', async () => {
            // GIVEN: Valid highlight data
            const highlightData = {
                normaKey: 'cc-art-1',
                text: 'Test highlighted text',
                color: '#FFFF00',
                startOffset: 0,
                endOffset: 20,
            };

            // WHEN: Creating highlight
            const res = await request(app)
                .post('/api/highlights')
                .set('Authorization', `Bearer ${accessToken}`)
                .send(highlightData);

            // THEN: Returns 201 with highlight data
            expect(res.status).toBe(201);
            expect(res.body).toHaveProperty('id');
            highlightId = res.body.id;
        });

        it('should get highlights by normaKey', async () => {
            // GIVEN: Highlights exist for a norma
            // WHEN: Querying by normaKey
            const res = await request(app)
                .get('/api/highlights?normaKey=cc-art-1')
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 with array
            expect(res.status).toBe(200);
            expect(Array.isArray(res.body)).toBe(true);
        });

        it('should delete a highlight', async () => {
            // GIVEN: A highlight exists
            // WHEN: Deleting the highlight
            const res = await request(app)
                .delete(`/api/highlights/${highlightId}`)
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 or 204
            expect([200, 204]).toContain(res.status);
        });
    });
});

describe('Annotations API', () => {
    let accessToken = '';
    let annotationId = '';

    const testUser = {
        email: 'annotation-test@example.com',
        username: 'annotationtest',
        password: 'Password123',
    };

    beforeAll(async () => {
        await prisma.refreshToken.deleteMany();
        await prisma.user.deleteMany({ where: { email: testUser.email } });

        await request(app).post('/api/auth/register').send(testUser);
        await prisma.user.update({
            where: { email: testUser.email },
            data: { isActive: true },
        });

        const loginRes = await request(app)
            .post('/api/auth/login')
            .send({ email: testUser.email, password: testUser.password });

        accessToken = loginRes.body.access_token;
    });

    afterAll(async () => {
        await prisma.refreshToken.deleteMany();
        await prisma.user.deleteMany({ where: { email: testUser.email } });
        await prisma.$disconnect();
    });

    describe('[P1] Annotations CRUD', () => {
        it('should create an annotation', async () => {
            // GIVEN: Valid annotation data
            const annotationData = {
                normaKey: 'cc-art-1',
                text: 'This is a test note',
                type: 'note',
            };

            // WHEN: Creating annotation
            const res = await request(app)
                .post('/api/annotations')
                .set('Authorization', `Bearer ${accessToken}`)
                .send(annotationData);

            // THEN: Returns 201 with annotation data
            expect(res.status).toBe(201);
            expect(res.body).toHaveProperty('id');
            annotationId = res.body.id;
        });

        it('should get annotations by normaKey', async () => {
            // GIVEN: Annotations exist for a norma
            // WHEN: Querying by normaKey
            const res = await request(app)
                .get('/api/annotations?normaKey=cc-art-1')
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 with array
            expect(res.status).toBe(200);
            expect(Array.isArray(res.body)).toBe(true);
        });

        it('should delete an annotation', async () => {
            // GIVEN: An annotation exists
            // WHEN: Deleting the annotation
            const res = await request(app)
                .delete(`/api/annotations/${annotationId}`)
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 or 204
            expect([200, 204]).toContain(res.status);
        });
    });
});
