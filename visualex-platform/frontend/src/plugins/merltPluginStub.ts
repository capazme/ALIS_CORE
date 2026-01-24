import type { Plugin } from '../lib/plugins/types';

const merltPluginStub: Plugin = {
  manifest: {
    id: 'merlt',
    name: 'MERL-T (disabled)',
    version: '0.0.0',
    description: 'MERL-T plugin disabled for this build',
    requiredFeatures: [],
    subscribedEvents: [],
    contributedSlots: [],
  },
  async initialize() {
    return () => {};
  },
  getSlotComponents() {
    return [];
  },
  getEventHandlers() {
    return {};
  },
};

export default merltPluginStub;
