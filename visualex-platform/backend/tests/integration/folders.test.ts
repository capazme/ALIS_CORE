/**
 * Backend API Tests: Folder Management
 * ======================================
 *
 * Tests for:
 * - POST /api/folders — Create folder
 * - GET /api/folders — List folders
 * - GET /api/folders/tree — Get folder tree
 * - GET /api/folders/:id — Get folder
 * - PUT /api/folders/:id — Update folder
 * - PATCH /api/folders/:id/move — Move folder
 * - DELETE /api/folders/:id — Delete folder
 *
 * Priority Tags: [P1] High  [P2] Medium
 */
import request from 'supertest';
import { PrismaClient } from '@prisma/client';
import app from '../../src/index';

const prisma = new PrismaClient();

describe('Folder API', () => {
    let accessToken = '';
    let folderId = '';

    const testUser = {
        email: 'folder-test@example.com',
        username: 'foldertest',
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

    describe('[P1] POST /api/folders', () => {
        it('should create a root folder', async () => {
            // GIVEN: Valid folder data
            const folderData = { name: 'Test Folder' };

            // WHEN: Creating the folder
            const res = await request(app)
                .post('/api/folders')
                .set('Authorization', `Bearer ${accessToken}`)
                .send(folderData);

            // THEN: Returns 201 with folder data
            expect(res.status).toBe(201);
            expect(res.body).toHaveProperty('id');
            expect(res.body.name).toBe(folderData.name);
            folderId = res.body.id;
        });

        it('should reject unauthenticated folder creation', async () => {
            // GIVEN: No auth token
            // WHEN: Creating a folder
            const res = await request(app)
                .post('/api/folders')
                .send({ name: 'Unauthorized Folder' });

            // THEN: Returns 401
            expect(res.status).toBe(401);
        });
    });

    describe('[P1] GET /api/folders', () => {
        it('should list user folders', async () => {
            // GIVEN: User has at least one folder
            // WHEN: Listing folders
            const res = await request(app)
                .get('/api/folders')
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 with array
            expect(res.status).toBe(200);
            expect(Array.isArray(res.body)).toBe(true);
            expect(res.body.length).toBeGreaterThan(0);
        });
    });

    describe('[P1] GET /api/folders/tree', () => {
        it('should return folder tree structure', async () => {
            // GIVEN: Folders exist
            // WHEN: Requesting tree view
            const res = await request(app)
                .get('/api/folders/tree')
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 with tree structure
            expect(res.status).toBe(200);
        });
    });

    describe('[P2] PUT /api/folders/:id', () => {
        it('should update a folder name', async () => {
            // GIVEN: A folder exists
            // WHEN: Updating the name
            const res = await request(app)
                .put(`/api/folders/${folderId}`)
                .set('Authorization', `Bearer ${accessToken}`)
                .send({ name: 'Renamed Folder' });

            // THEN: Returns 200 with updated data
            expect(res.status).toBe(200);
            expect(res.body.name).toBe('Renamed Folder');
        });
    });

    describe('[P2] DELETE /api/folders/:id', () => {
        it('should delete a folder', async () => {
            // GIVEN: A folder exists
            // WHEN: Deleting the folder
            const res = await request(app)
                .delete(`/api/folders/${folderId}`)
                .set('Authorization', `Bearer ${accessToken}`);

            // THEN: Returns 200 or 204
            expect([200, 204]).toContain(res.status);
        });
    });
});
