/**
 * Policy Evolution Chart
 *
 * Visualizes RLCF policy evolution over time with 3 tabs:
 * - Confidence/Reward time series (LineChart)
 * - Expert usage evolution (stacked AreaChart)
 * - Aggregation disagreement trends (LineChart)
 */

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@components/ui/Card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@components/ui/Tabs';
import { QueryWrapper } from '@components/shared/QueryWrapper';
import {
  useTimeSeries,
  useExpertEvolution,
  useAggregationHistory,
} from '@hooks/usePolicyEvolution';
import { TrendingUp, Users, AlertTriangle } from 'lucide-react';

const EXPERT_COLORS: Record<string, string> = {
  literal: '#8884d8',
  systemic: '#82ca9d',
  principles: '#ffc658',
  precedent: '#ff8042',
};

function formatTimestamp(ts: string): string {
  const d = new Date(ts);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

export function PolicyEvolutionChart() {
  const timeSeriesQuery = useTimeSeries('confidence', 50);
  const expertEvolutionQuery = useExpertEvolution(30);
  const aggregationQuery = useAggregationHistory(30);

  const formattedTimeSeries = useMemo(() => {
    if (!timeSeriesQuery.data) return [];
    return timeSeriesQuery.data.map((p) => ({
      ...p,
      label: formatTimestamp(p.timestamp),
    }));
  }, [timeSeriesQuery.data]);

  const formattedExpertEvolution = useMemo(() => {
    if (!expertEvolutionQuery.data) return [];
    return expertEvolutionQuery.data.map((p) => ({
      ...p,
      label: formatTimestamp(p.timestamp),
    }));
  }, [expertEvolutionQuery.data]);

  const formattedAggregation = useMemo(() => {
    if (!aggregationQuery.data) return [];
    return aggregationQuery.data.map((p) => ({
      ...p,
      label: formatTimestamp(p.timestamp),
    }));
  }, [aggregationQuery.data]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Policy Evolution</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="confidence">
          <TabsList>
            <TabsTrigger value="confidence">
              <span className="flex items-center gap-1.5">
                <TrendingUp className="h-3.5 w-3.5" aria-hidden="true" />
                Confidence
              </span>
            </TabsTrigger>
            <TabsTrigger value="experts">
              <span className="flex items-center gap-1.5">
                <Users className="h-3.5 w-3.5" aria-hidden="true" />
                Expert Usage
              </span>
            </TabsTrigger>
            <TabsTrigger value="disagreement">
              <span className="flex items-center gap-1.5">
                <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
                Disagreement
              </span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="confidence">
            <QueryWrapper
              query={timeSeriesQuery}
              loadingMessage="Loading time series..."
              emptyMessage="No time series data yet"
              minHeight="300px"
            >
              {() => (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={formattedTimeSeries}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="label" stroke="#94a3b8" fontSize={12} />
                    <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #475569',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="confidence"
                      stroke="#8b5cf6"
                      strokeWidth={2}
                      dot={false}
                      name="Confidence"
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="reward"
                      stroke="#22d3ee"
                      strokeWidth={2}
                      dot={false}
                      name="Reward"
                      connectNulls
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </QueryWrapper>
          </TabsContent>

          <TabsContent value="experts">
            <QueryWrapper
              query={expertEvolutionQuery}
              loadingMessage="Loading expert evolution..."
              emptyMessage="No expert evolution data yet"
              minHeight="300px"
            >
              {() => (
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={formattedExpertEvolution}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="label" stroke="#94a3b8" fontSize={12} />
                    <YAxis stroke="#94a3b8" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #475569',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    {Object.entries(EXPERT_COLORS).map(([key, color]) => (
                      <Area
                        key={key}
                        type="monotone"
                        dataKey={key}
                        stackId="experts"
                        stroke={color}
                        fill={color}
                        fillOpacity={0.6}
                        name={key.charAt(0).toUpperCase() + key.slice(1)}
                      />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </QueryWrapper>
          </TabsContent>

          <TabsContent value="disagreement">
            <QueryWrapper
              query={aggregationQuery}
              loadingMessage="Loading aggregation history..."
              emptyMessage="No aggregation data yet"
              minHeight="300px"
            >
              {() => (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={formattedAggregation}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="label" stroke="#94a3b8" fontSize={12} />
                    <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #475569',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="avg_rating"
                      stroke="#82ca9d"
                      strokeWidth={2}
                      dot={false}
                      name="Avg Rating"
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="disagreement_score"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                      name="Disagreement"
                      connectNulls
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </QueryWrapper>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
