type ClassValue =
  | string
  | number
  | null
  | undefined
  | boolean
  | ClassValue[]
  | { [key: string]: boolean };

function toClassName(value: ClassValue): string[] {
  if (!value) {
    return [];
  }

  if (typeof value === 'string' || typeof value === 'number') {
    return [String(value)];
  }

  if (Array.isArray(value)) {
    return value.flatMap(toClassName);
  }

  if (typeof value === 'object') {
    return Object.entries(value)
      .filter(([, enabled]) => Boolean(enabled))
      .map(([className]) => className);
  }

  return [];
}

export function cn(...inputs: ClassValue[]) {
  return inputs.flatMap(toClassName).join(' ');
}
