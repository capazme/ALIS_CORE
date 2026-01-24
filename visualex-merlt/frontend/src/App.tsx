import { useState } from 'react';
import { AcademicDashboard } from './components/merlt/dashboard/AcademicDashboard';
import { PipelineMonitoringDashboard } from './components/merlt/monitoring/PipelineMonitoringDashboard';

type TabId = 'dashboard' | 'pipeline';

const TABS: Array<{ id: TabId; label: string }> = [
  { id: 'dashboard', label: 'Academic Dashboard' },
  { id: 'pipeline', label: 'Pipeline Monitoring' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('dashboard');

  return (
    <div style={{ minHeight: '100vh' }}>
      <header style={{ padding: '16px 24px', borderBottom: '1px solid #1f2937' }}>
        <h1 style={{ margin: 0, fontSize: 20 }}>VisuaLex MERL-T</h1>
        <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                background: activeTab === tab.id ? '#2563eb' : '#111827',
                color: '#e5e7eb',
                border: '1px solid #1f2937',
                borderRadius: 8,
                padding: '6px 12px',
                cursor: 'pointer',
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </header>

      <main>
        {activeTab === 'dashboard' ? (
          <AcademicDashboard />
        ) : (
          <PipelineMonitoringDashboard />
        )}
      </main>
    </div>
  );
}
