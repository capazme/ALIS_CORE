/**
 * Unit Tests: Bookmark Service
 * =============================
 *
 * Tests for:
 * - CRUD operations (create, getAll, getById, update, delete)
 * - Bulk operations (bulkDelete, bulkMove)
 * - API call correctness
 *
 * Priority Tags: [P1] High  [P2] Medium
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { bookmarkService } from '../../services/bookmarkService';

// Mock the API client
vi.mock('../../services/api', () => ({
    apiClient: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        patch: vi.fn(),
        delete: vi.fn(),
    },
}));

import { apiClient } from '../../services/api';

const mockGet = vi.mocked(apiClient.get);
const mockPost = vi.mocked(apiClient.post);
const mockPut = vi.mocked(apiClient.put);
const mockPatch = vi.mocked(apiClient.patch);
const mockDelete = vi.mocked(apiClient.delete);

describe('Bookmark Service', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('[P1] create', () => {
        it('should call POST /bookmarks with correct data', async () => {
            // GIVEN: Valid bookmark data
            const bookmarkData = { normaId: '123', title: 'Test Bookmark' };
            mockPost.mockResolvedValue({ data: { id: '1', ...bookmarkData } } as never);

            // WHEN: Creating a bookmark
            const result = await bookmarkService.create(bookmarkData as never);

            // THEN: API is called correctly
            expect(mockPost).toHaveBeenCalledWith('/bookmarks', bookmarkData);
            expect(result).toHaveProperty('id');
        });
    });

    describe('[P1] getAll', () => {
        it('should call GET /bookmarks with no params', async () => {
            // GIVEN: No filter params
            mockGet.mockResolvedValue({ data: [] } as never);

            // WHEN: Getting all bookmarks
            const result = await bookmarkService.getAll();

            // THEN: API is called correctly
            expect(mockGet).toHaveBeenCalledWith('/bookmarks', { params: undefined });
            expect(result).toEqual([]);
        });

        it('should call GET /bookmarks with folderId filter', async () => {
            // GIVEN: A folder filter
            mockGet.mockResolvedValue({ data: [] } as never);

            // WHEN: Getting bookmarks by folder
            await bookmarkService.getAll({ folderId: 'folder-1' });

            // THEN: API is called with params
            expect(mockGet).toHaveBeenCalledWith('/bookmarks', { params: { folderId: 'folder-1' } });
        });
    });

    describe('[P1] getById', () => {
        it('should call GET /bookmarks/:id', async () => {
            // GIVEN: A bookmark ID
            mockGet.mockResolvedValue({ data: { id: '1', title: 'Test' } } as never);

            // WHEN: Getting a bookmark by ID
            const result = await bookmarkService.getById('1');

            // THEN: API is called with the ID
            expect(mockGet).toHaveBeenCalledWith('/bookmarks/1');
            expect(result.id).toBe('1');
        });
    });

    describe('[P1] update', () => {
        it('should call PUT /bookmarks/:id with update data', async () => {
            // GIVEN: Update data
            const updateData = { title: 'Updated Bookmark' };
            mockPut.mockResolvedValue({ data: { id: '1', ...updateData } } as never);

            // WHEN: Updating a bookmark
            await bookmarkService.update('1', updateData as never);

            // THEN: API is called correctly
            expect(mockPut).toHaveBeenCalledWith('/bookmarks/1', updateData);
        });
    });

    describe('[P1] move', () => {
        it('should call PATCH /bookmarks/:id/move', async () => {
            // GIVEN: Move parameters
            mockPatch.mockResolvedValue({ data: { id: '1' } } as never);

            // WHEN: Moving a bookmark to a folder
            await bookmarkService.move('1', 'folder-2');

            // THEN: API is called correctly
            expect(mockPatch).toHaveBeenCalledWith('/bookmarks/1/move', { folderId: 'folder-2' });
        });
    });

    describe('[P2] delete', () => {
        it('should call DELETE /bookmarks/:id', async () => {
            // GIVEN: A bookmark to delete
            mockDelete.mockResolvedValue({ data: undefined } as never);

            // WHEN: Deleting a bookmark
            await bookmarkService.delete('1');

            // THEN: API is called correctly
            expect(mockDelete).toHaveBeenCalledWith('/bookmarks/1');
        });
    });

    describe('[P2] bulkDelete', () => {
        it('should call POST /bookmarks/bulk/delete', async () => {
            // GIVEN: Multiple bookmark IDs
            const bookmarkIds = ['1', '2', '3'];
            mockPost.mockResolvedValue({ data: { deleted_count: 3 } } as never);

            // WHEN: Bulk deleting
            const result = await bookmarkService.bulkDelete(bookmarkIds);

            // THEN: API is called correctly
            expect(mockPost).toHaveBeenCalledWith('/bookmarks/bulk/delete', { bookmarkIds });
            expect(result.deleted_count).toBe(3);
        });
    });

    describe('[P2] bulkMove', () => {
        it('should call POST /bookmarks/bulk/move', async () => {
            // GIVEN: Multiple bookmark IDs and target folder
            const bookmarkIds = ['1', '2'];
            mockPost.mockResolvedValue({ data: { updated_count: 2 } } as never);

            // WHEN: Bulk moving
            const result = await bookmarkService.bulkMove(bookmarkIds, 'folder-1');

            // THEN: API is called correctly
            expect(mockPost).toHaveBeenCalledWith('/bookmarks/bulk/move', { bookmarkIds, folderId: 'folder-1' });
            expect(result.updated_count).toBe(2);
        });
    });
});
