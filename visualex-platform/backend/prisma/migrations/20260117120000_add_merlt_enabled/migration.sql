-- Add MERL-T opt-in flag for users
ALTER TABLE "users"
ADD COLUMN "is_merlt_enabled" BOOLEAN NOT NULL DEFAULT false;
