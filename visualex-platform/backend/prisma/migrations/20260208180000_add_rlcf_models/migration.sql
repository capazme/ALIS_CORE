-- CreateEnum
CREATE TYPE "ProfileType" AS ENUM ('quick_consultation', 'assisted_research', 'expert_analysis', 'active_contributor');
CREATE TYPE "ConsentLevel" AS ENUM ('basic', 'learning', 'research');

-- AlterTable (User)
ALTER TABLE "users" ADD COLUMN "profile_type" "ProfileType" DEFAULT 'quick_consultation';
ALTER TABLE "users" ADD COLUMN "authority_score" DOUBLE PRECISION DEFAULT 0;
ALTER TABLE "users" ADD COLUMN "deletion_requested_at" TIMESTAMP(3);
ALTER TABLE "users" ADD COLUMN "deletion_reason" TEXT;

-- CreateTable
CREATE TABLE "user_consents" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "consent_level" "ConsentLevel" NOT NULL DEFAULT 'basic',
    "granted_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "ip_hash" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "user_consents_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "consent_audit_logs" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "previous_level" TEXT,
    "new_level" TEXT NOT NULL,
    "ip_hash" TEXT,
    "user_agent" TEXT,
    "changed_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "consent_audit_logs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "user_authorities" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "baseline_score" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "track_record_score" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "recent_performance" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "computed_score" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "feedback_count" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "user_authorities_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "user_preferences" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "theme" TEXT NOT NULL DEFAULT 'system',
    "language" TEXT NOT NULL DEFAULT 'it',
    "notifications_enabled" BOOLEAN NOT NULL DEFAULT true,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "user_preferences_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "user_consents_user_id_key" ON "user_consents"("user_id");
CREATE INDEX "user_consents_user_id_idx" ON "user_consents"("user_id");

CREATE INDEX "consent_audit_logs_user_id_idx" ON "consent_audit_logs"("user_id");
CREATE INDEX "consent_audit_logs_changed_at_idx" ON "consent_audit_logs"("changed_at");

CREATE UNIQUE INDEX "user_authorities_user_id_key" ON "user_authorities"("user_id");
CREATE INDEX "user_authorities_user_id_idx" ON "user_authorities"("user_id");

CREATE UNIQUE INDEX "user_preferences_user_id_key" ON "user_preferences"("user_id");
CREATE INDEX "user_preferences_user_id_idx" ON "user_preferences"("user_id");

-- AddForeignKey
ALTER TABLE "user_consents" ADD CONSTRAINT "user_consents_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "user_authorities" ADD CONSTRAINT "user_authorities_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "user_preferences" ADD CONSTRAINT "user_preferences_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
