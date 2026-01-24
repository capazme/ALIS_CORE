import { Request, Response, NextFunction } from 'express';
import { ZodError } from 'zod';

export class AppError extends Error {
  constructor(
    public statusCode: number,
    public message: string,
  ) {
    super(message);
    this.name = this.constructor.name;
    Error.captureStackTrace(this, this.constructor);
  }
}

export const errorHandler = (
  err: Error | AppError,
  _req: Request,
  res: Response,
  _next: NextFunction
) => {
  // Handle AppError
  if (err instanceof AppError) {
    return res.status(err.statusCode).json({
      detail: err.message,
    });
  }

  // Handle Zod validation errors
  if (err instanceof ZodError) {
    const messages = err.errors.map(e => `${e.path.join('.')}: ${e.message}`).join(', ');
    return res.status(400).json({
      detail: `Validation error: ${messages}`,
    });
  }

  // Log error safely (avoid circular references)
  console.error('Unexpected error:', err.message || String(err));
  if (err.stack) {
    console.error(err.stack);
  }

  return res.status(500).json({
    detail: 'Internal server error',
  });
};
