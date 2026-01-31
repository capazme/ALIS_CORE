import dotenv from 'dotenv';

dotenv.config();

// Validate required environment variables
const requiredEnvVars = ['JWT_SECRET', 'DATABASE_URL'] as const;

for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    throw new Error(`Missing required environment variable: ${envVar}. Check your .env file.`);
  }
}

export const config = {
  port: parseInt(process.env.PORT || '3001', 10),
  nodeEnv: process.env.NODE_ENV || 'development',

  jwt: {
    secret: process.env.JWT_SECRET!, // Validated above
    accessExpiry: process.env.JWT_ACCESS_EXPIRY || '1h', // AC1: 1 hour expiry
    refreshExpiry: process.env.JWT_REFRESH_EXPIRY || '7d',
  },

  cors: {
    origins: (process.env.ALLOWED_ORIGINS || 'http://localhost:5173,http://localhost:3001')
      .split(',')
      .map(origin => origin.trim()),
  },

  database: {
    url: process.env.DATABASE_URL!, // Validated above
  },

  merlt: {
    apiUrl: process.env.MERLT_API_URL || 'http://merlt-api:8000',
    enabled: process.env.MERLT_ENABLED === 'true',
  },

  // Consent IP hashing (GDPR compliance)
  consentIpSalt: process.env.CONSENT_IP_SALT || 'consent-ip-salt-change-in-production',

  // Legacy alias for backwards compatibility
  jwtSecret: process.env.JWT_SECRET!,
};
