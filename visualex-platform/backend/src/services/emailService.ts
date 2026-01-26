/**
 * Email Service - MVP Stub Implementation
 *
 * This is a stub implementation for MVP that logs emails to console.
 * In production, replace with actual email provider (SendGrid, AWS SES, etc.)
 */

const BASE_URL = process.env.FRONTEND_URL || 'http://localhost:5173';

interface EmailOptions {
  to: string;
  subject: string;
  text: string;
  html?: string;
}

/**
 * Send email - MVP stub that logs to console
 * TODO: Replace with actual email provider in production
 */
async function sendEmail(options: EmailOptions): Promise<void> {
  if (process.env.NODE_ENV === 'production' && process.env.EMAIL_PROVIDER) {
    // Production: send actual email
    // Implement based on EMAIL_PROVIDER (sendgrid, ses, etc.)
    console.log('[EmailService] Production email sending not configured yet');
    console.log('[EmailService] Would send to:', options.to);
    return;
  }

  // Development: log email to console
  console.log('\n========================================');
  console.log('📧 EMAIL SERVICE (Development Mode)');
  console.log('========================================');
  console.log(`To: ${options.to}`);
  console.log(`Subject: ${options.subject}`);
  console.log('----------------------------------------');
  console.log(options.text);
  console.log('========================================\n');
}

/**
 * Send email verification link to user
 */
export async function sendVerificationEmail(email: string, token: string): Promise<void> {
  const verificationUrl = `${BASE_URL}/verify-email?token=${token}`;

  await sendEmail({
    to: email,
    subject: 'Verifica il tuo account ALIS',
    text: `
Benvenuto su ALIS!

Per completare la registrazione, verifica il tuo indirizzo email cliccando sul link seguente:

${verificationUrl}

Il link scadrà tra 24 ore.

Se non hai richiesto questa registrazione, ignora questa email.

---
ALIS - Artificial Legal Intelligence System
    `.trim(),
    html: `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }
    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { background: #1e3a5f; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
    .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }
    .button { display: inline-block; background: #1e3a5f; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; margin: 20px 0; }
    .footer { margin-top: 20px; font-size: 12px; color: #666; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>ALIS</h1>
      <p>Artificial Legal Intelligence System</p>
    </div>
    <div class="content">
      <h2>Benvenuto su ALIS!</h2>
      <p>Per completare la registrazione, verifica il tuo indirizzo email cliccando sul pulsante seguente:</p>
      <p style="text-align: center;">
        <a href="${verificationUrl}" class="button">Verifica Email</a>
      </p>
      <p>Oppure copia e incolla questo link nel browser:</p>
      <p style="word-break: break-all; background: #eee; padding: 10px; border-radius: 4px;">
        ${verificationUrl}
      </p>
      <p><strong>Il link scadrà tra 24 ore.</strong></p>
      <p class="footer">
        Se non hai richiesto questa registrazione, ignora questa email.<br>
        &copy; ${new Date().getFullYear()} ALIS - Artificial Legal Intelligence System
      </p>
    </div>
  </div>
</body>
</html>
    `.trim(),
  });
}

/**
 * Send password reset email
 * TODO: Implement when password reset feature is added
 */
export async function sendPasswordResetEmail(email: string, token: string): Promise<void> {
  const resetUrl = `${BASE_URL}/reset-password?token=${token}`;

  await sendEmail({
    to: email,
    subject: 'Reimposta la tua password ALIS',
    text: `
Hai richiesto di reimpostare la tua password ALIS.

Clicca sul link seguente per reimpostare la password:

${resetUrl}

Il link scadrà tra 1 ora.

Se non hai richiesto questa operazione, ignora questa email.

---
ALIS - Artificial Legal Intelligence System
    `.trim(),
  });
}
