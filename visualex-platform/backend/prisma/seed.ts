/**
 * Database Seed Script
 *
 * Creates test users with pre-verified accounts for E2E testing.
 * Run with: npx prisma db seed
 */
import { PrismaClient } from '@prisma/client';
import { hashPassword } from '../src/utils/password';

const prisma = new PrismaClient();

async function main() {
  console.log('Seeding database...');

  // Create test user for E2E tests (pre-verified)
  const testUserPassword = await hashPassword('TestPassword123!');

  const testUser = await prisma.user.upsert({
    where: { email: 'e2e-test@visualex.it' },
    update: {
      password: testUserPassword,
      isVerified: true,
      isActive: true,
    },
    create: {
      email: 'e2e-test@visualex.it',
      username: 'e2e_test_user',
      password: testUserPassword,
      isVerified: true,
      isActive: true,
      profileType: 'assisted_research',
      authorityScore: 0.5,
    },
  });
  console.log(`Created/Updated test user: ${testUser.email}`);

  // Create admin user for testing
  const adminPassword = await hashPassword('AdminPassword123!');

  const adminUser = await prisma.user.upsert({
    where: { email: 'admin@visualex.it' },
    update: {
      password: adminPassword,
      isVerified: true,
      isActive: true,
      isAdmin: true,
    },
    create: {
      email: 'admin@visualex.it',
      username: 'admin',
      password: adminPassword,
      isVerified: true,
      isActive: true,
      isAdmin: true,
      profileType: 'expert_analysis',
      authorityScore: 1.0,
    },
  });
  console.log(`Created/Updated admin user: ${adminUser.email}`);

  // Create preferences for test user
  await prisma.userPreferences.upsert({
    where: { userId: testUser.id },
    update: {},
    create: {
      userId: testUser.id,
      theme: 'light',
      language: 'it',
    },
  });

  // Create consent for test user
  await prisma.userConsent.upsert({
    where: { userId: testUser.id },
    update: {},
    create: {
      userId: testUser.id,
      consentLevel: 'learning',
    },
  });

  console.log('Database seeded successfully!');
}

main()
  .catch((e) => {
    console.error('Seed error:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
