import { useState, useCallback, useRef, useEffect } from 'react';

export interface UndoableActionOptions<T> {
  action: () => T | Promise<T>;
  undo: (result: T) => void | Promise<void>;
  message: string;
  duration?: number;
  onComplete?: () => void;
  onUndo?: () => void;
}

export interface UndoableActionState {
  isActive: boolean;
  message: string;
  timeRemaining: number;
}

export interface UseUndoableActionReturn {
  execute: () => Promise<void>;
  undoAction: () => Promise<void>;
  dismiss: () => void;
  state: UndoableActionState;
}

export function useUndoableAction<T>({
  action,
  undo,
  message,
  duration = 5000,
  onComplete,
  onUndo,
}: UndoableActionOptions<T>): UseUndoableActionReturn {
  const [isActive, setIsActive] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [currentMessage, setCurrentMessage] = useState('');

  const resultRef = useRef<T | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const clearTimers = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const execute = useCallback(async () => {
    clearTimers();

    const result = await action();
    resultRef.current = result;

    setCurrentMessage(message);
    setIsActive(true);
    setTimeRemaining(duration);

    intervalRef.current = setInterval(() => {
      setTimeRemaining((prev) => {
        const next = prev - 100;
        if (next <= 0) {
          if (intervalRef.current) clearInterval(intervalRef.current);
          return 0;
        }
        return next;
      });
    }, 100);

    timerRef.current = setTimeout(() => {
      setIsActive(false);
      setTimeRemaining(0);
      resultRef.current = null;
      onComplete?.();
      clearTimers();
    }, duration);
  }, [action, message, duration, onComplete, clearTimers]);

  const undoAction = useCallback(async () => {
    if (!isActive || resultRef.current === null) return;

    clearTimers();

    await undo(resultRef.current);

    setIsActive(false);
    setTimeRemaining(0);
    resultRef.current = null;
    onUndo?.();
  }, [isActive, undo, onUndo, clearTimers]);

  const dismiss = useCallback(() => {
    if (!isActive) return;

    clearTimers();
    setIsActive(false);
    setTimeRemaining(0);
    resultRef.current = null;
    onComplete?.();
  }, [isActive, onComplete, clearTimers]);

  return {
    execute,
    undoAction,
    dismiss,
    state: {
      isActive,
      message: currentMessage,
      timeRemaining,
    },
  };
}

type ToastCallback = (toast: UndoToast | null) => void;

export interface UndoToast {
  message: string;
  timeRemaining: number;
  onUndo: () => void;
  onDismiss: () => void;
}

let toastListener: ToastCallback | null = null;

export function registerUndoToastListener(callback: ToastCallback): () => void {
  toastListener = callback;
  return () => {
    toastListener = null;
  };
}

export function showUndoToast<T>({
  action,
  undo,
  message,
  duration = 5000,
}: {
  action: () => T | Promise<T>;
  undo: (result: T) => void | Promise<void>;
  message: string;
  duration?: number;
}): Promise<boolean> {
  return new Promise(async (resolve) => {
    const result = await action();

    let resolved = false;

    const handleUndo = async () => {
      if (resolved) return;
      resolved = true;
      await undo(result);
      toastListener?.(null);
      resolve(false);
    };

    const handleDismiss = () => {
      if (resolved) return;
      resolved = true;
      toastListener?.(null);
      resolve(true);
    };

    let timeRemaining = duration;

    const updateToast = () => {
      toastListener?.({
        message,
        timeRemaining,
        onUndo: handleUndo,
        onDismiss: handleDismiss,
      });
    };

    updateToast();

    const interval = setInterval(() => {
      timeRemaining -= 100;
      if (timeRemaining <= 0) {
        clearInterval(interval);
        handleDismiss();
      } else {
        updateToast();
      }
    }, 100);
  });
}
