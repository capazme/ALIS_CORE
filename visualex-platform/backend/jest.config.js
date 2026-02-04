/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
    preset: 'ts-jest',
    testEnvironment: 'node',
    moduleNameMapper: {
        '@visualex/merlt-backend': '<rootDir>/tests/mocks/merltBackend.ts'
    },
    transform: {
        '^.+\\.tsx?$': ['ts-jest', { diagnostics: false }]
    },
    testMatch: [
        '<rootDir>/tests/**/*.test.ts',
        '<rootDir>/tests/**/*.spec.ts'
    ],
    testPathIgnorePatterns: [
        '/node_modules/',
        '/dist/'
    ],
    setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
    verbose: true,
    forceExit: true,
    clearMocks: true,
    resetMocks: true,
    restoreMocks: true,
    collectCoverageFrom: [
        'src/**/*.ts',
        '!src/**/*.d.ts',
        '!src/index.ts'
    ],
    coverageDirectory: 'coverage',
    coverageReporters: ['text', 'lcov', 'html'],
};
