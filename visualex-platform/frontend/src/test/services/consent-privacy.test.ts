/**
 * Unit Tests: Consent & Privacy Services
 * ========================================
 *
 * Tests for:
 * - Consent: getConsent, updateConsent, getConsentHistory
 * - Privacy: getPrivacyStatus, exportData, requestDeletion, cancelDeletion
 *
 * Priority Tags: [P1] High  [P2] Medium
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the API module
vi.mock('../../services/api', () => ({
    get: vi.fn(),
    put: vi.fn(),
    post: vi.fn(),
}));

import { get, put, post } from '../../services/api';

const mockGet = vi.mocked(get);
const mockPut = vi.mocked(put);
const mockPost = vi.mocked(post);

describe('Consent Service', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('[P1] getConsent', () => {
        it('should call GET /consent', async () => {
            // GIVEN: API returns consent data
            mockGet.mockResolvedValue({
                consent_level: 'base',
                available_levels: ['base', 'learning', 'research'],
            });

            // WHEN: Getting consent
            const { getConsent } = await import('../../services/consentService');
            const result = await getConsent();

            // THEN: Correct endpoint is called
            expect(mockGet).toHaveBeenCalledWith('/consent');
            expect(result).toHaveProperty('consent_level');
        });
    });

    describe('[P1] updateConsent', () => {
        it('should call PUT /consent with consent level', async () => {
            // GIVEN: A new consent level
            mockPut.mockResolvedValue({ success: true, consent_level: 'learning' });

            // WHEN: Updating consent
            const { updateConsent } = await import('../../services/consentService');
            await updateConsent('learning' as never);

            // THEN: Correct endpoint and data are sent
            expect(mockPut).toHaveBeenCalledWith('/consent', { consent_level: 'learning' });
        });
    });

    describe('[P2] getConsentHistory', () => {
        it('should call GET /consent/history', async () => {
            // GIVEN: API returns history
            mockGet.mockResolvedValue({ history: [] });

            // WHEN: Getting consent history
            const { getConsentHistory } = await import('../../services/consentService');
            await getConsentHistory();

            // THEN: Correct endpoint is called
            expect(mockGet).toHaveBeenCalledWith('/consent/history');
        });
    });
});

describe('Privacy Service', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('[P1] getPrivacyStatus', () => {
        it('should call GET /privacy/status', async () => {
            // GIVEN: API returns privacy status
            mockGet.mockResolvedValue({ deletion_pending: false });

            // WHEN: Getting privacy status
            const { getPrivacyStatus } = await import('../../services/privacyService');
            const result = await getPrivacyStatus();

            // THEN: Correct endpoint is called
            expect(mockGet).toHaveBeenCalledWith('/privacy/status');
            expect(result).toHaveProperty('deletion_pending');
        });
    });

    describe('[P1] exportData', () => {
        it('should call POST /privacy/export', async () => {
            // GIVEN: API returns exported data
            mockPost.mockResolvedValue({
                user: { email: 'test@example.com' },
                export_date: '2026-02-16',
            });

            // WHEN: Exporting data
            const { exportData } = await import('../../services/privacyService');
            const result = await exportData();

            // THEN: Correct endpoint is called
            expect(mockPost).toHaveBeenCalledWith('/privacy/export', {});
            expect(result).toHaveProperty('user');
        });
    });

    describe('[P1] requestDeletion', () => {
        it('should call POST /privacy/delete-account with password', async () => {
            // GIVEN: Password and reason
            mockPost.mockResolvedValue({ success: true, scheduled_for: '2026-02-23' });

            // WHEN: Requesting account deletion
            const { requestDeletion } = await import('../../services/privacyService');
            await requestDeletion('MyPassword123', 'No longer needed');

            // THEN: Correct endpoint and data are sent
            expect(mockPost).toHaveBeenCalledWith('/privacy/delete-account', {
                password: 'MyPassword123',
                reason: 'No longer needed',
            });
        });

        it('should work without a reason', async () => {
            // GIVEN: Password only
            mockPost.mockResolvedValue({ success: true });

            // WHEN: Requesting deletion without reason
            const { requestDeletion } = await import('../../services/privacyService');
            await requestDeletion('MyPassword123');

            // THEN: Reason is undefined
            expect(mockPost).toHaveBeenCalledWith('/privacy/delete-account', {
                password: 'MyPassword123',
                reason: undefined,
            });
        });
    });

    describe('[P2] cancelDeletion', () => {
        it('should call POST /privacy/cancel-deletion', async () => {
            // GIVEN: API returns cancellation success
            mockPost.mockResolvedValue({ success: true });

            // WHEN: Cancelling deletion
            const { cancelDeletion } = await import('../../services/privacyService');
            await cancelDeletion();

            // THEN: Correct endpoint is called
            expect(mockPost).toHaveBeenCalledWith('/privacy/cancel-deletion', {});
        });
    });
});
