/**
 * PluginSlot Component
 *
 * Renders plugin components that have registered for a specific slot.
 * This is how visualex-platform provides extension points for plugins.
 */

import { useMemo } from 'react';
import type { PluginSlotName, SlotProps } from './types';
import { PluginRegistry } from './PluginRegistry';

export interface PluginSlotProps<T extends PluginSlotName> {
  name: T;
  props: SlotProps[T];
  /**
   * Wrapper component for each plugin component
   */
  wrapper?: React.ComponentType<{ children: React.ReactNode }>;
  /**
   * Fallback when no plugins are rendered
   */
  fallback?: React.ReactNode;
  /**
   * Class name for the container
   */
  className?: string;
}

export function PluginSlot<T extends PluginSlotName>({
  name,
  props,
  wrapper: Wrapper,
  fallback = null,
  className,
}: PluginSlotProps<T>): React.ReactElement | null {
  const components = useMemo(() => PluginRegistry.getSlotComponents(name), [name]);

  if (components.length === 0) {
    return fallback as React.ReactElement | null;
  }

  const rendered = components.map((slotComponent, index) => {
    const Component = slotComponent.component;
    const element = <Component key={index} {...props} />;

    if (Wrapper) {
      return <Wrapper key={index}>{element}</Wrapper>;
    }

    return element;
  });

  if (className) {
    return <div className={className}>{rendered}</div>;
  }

  return <>{rendered}</>;
}
