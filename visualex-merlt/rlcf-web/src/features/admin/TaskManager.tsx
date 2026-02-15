import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Badge } from '../../components/ui/Badge';
import { CsvUpload } from './CsvUpload';
import { apiClient } from '../../lib/api';
import type { TaskFilters, TaskStatus } from '../../types/index';

interface Task {
  id: number;
  task_type: string;
  status: string;
  created_at: string;
  input_data: any;
  ground_truth_data?: any;
}

const STATUS_COLORS = {
  OPEN: 'bg-blue-100 text-blue-800',
  BLIND_EVALUATION: 'bg-yellow-100 text-yellow-800',
  AGGREGATED: 'bg-green-100 text-green-800',
  CLOSED: 'bg-slate-100 text-slate-800',
};

const TASK_TYPE_LABELS = {
  STATUTORY_RULE_QA: 'Statutory Rule Q&A',
  QA: 'Question Answering',
  CLASSIFICATION: 'Classification',
  SUMMARIZATION: 'Summarization',
  PREDICTION: 'Prediction',
  NLI: 'Natural Language Inference',
  NER: 'Named Entity Recognition',
  DRAFTING: 'Legal Drafting',
  RISK_SPOTTING: 'Risk Spotting',
  DOCTRINE_APPLICATION: 'Doctrine Application',
};

export const TaskManager: React.FC = () => {
  const [filters, setFilters] = useState<TaskFilters>({ limit: 50 });
  const [selectedTasks, setSelectedTasks] = useState<number[]>([]);
  const [showUpload, setShowUpload] = useState(false);
  
  const queryClient = useQueryClient();

  // Fetch tasks
  const { data: tasks = [], isLoading, error } = useQuery({
    queryKey: ['tasks', filters],
    queryFn: () => apiClient.tasks.list(filters),
  });

  // Bulk delete mutation
  const deleteMutation = useMutation({
    mutationFn: (taskIds: number[]) => apiClient.tasks.bulkDelete(taskIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      setSelectedTasks([]);
    },
  });

  // Bulk status update mutation
  const updateStatusMutation = useMutation({
    mutationFn: ({ taskIds, status }: { taskIds: number[], status: string }) => 
      apiClient.tasks.bulkUpdateStatus(taskIds, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      setSelectedTasks([]);
    },
  });

  const handleSelectAll = () => {
    if (selectedTasks.length === tasks.length) {
      setSelectedTasks([]);
    } else {
      setSelectedTasks(tasks.map((task: Task) => task.id));
    }
  };

  const handleTaskSelect = (taskId: number) => {
    setSelectedTasks(prev => 
      prev.includes(taskId) 
        ? prev.filter(id => id !== taskId)
        : [...prev, taskId]
    );
  };

  const getTaskTitle = (task: Task) => {
    if (task.input_data?.question) {
      return task.input_data.question.slice(0, 100) + (task.input_data.question.length > 100 ? '...' : '');
    }
    if (task.input_data?.text) {
      return task.input_data.text.slice(0, 100) + (task.input_data.text.length > 100 ? '...' : '');
    }
    if (task.input_data?.document) {
      return task.input_data.document.slice(0, 100) + (task.input_data.document.length > 100 ? '...' : '');
    }
    return `Task #${task.id}`;
  };

  if (showUpload) {
    return (
      <div className="space-y-6 p-4 md:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <h2 className="text-2xl font-bold text-white">Upload Dataset</h2>
          <Button
            variant="outline"
            onClick={() => setShowUpload(false)}
          >
            Back to Task List
          </Button>
        </div>
        
        <CsvUpload
          onUploadComplete={(tasks) => {
            setShowUpload(false);
            queryClient.invalidateQueries({ queryKey: ['tasks'] });
          }}
        />
      </div>
    );
  }

  return (
    <div className="space-y-6 p-4 md:p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Task Management</h2>
          <p className="text-sm text-slate-400 mt-1">
            Manage tasks, upload datasets, and monitor progress
          </p>
        </div>
        <Button onClick={() => setShowUpload(true)} className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
          Upload CSV Dataset
        </Button>
      </div>

      {/* Filters */}
      <Card className="p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label htmlFor="filter-status" className="block text-sm font-medium text-slate-300 mb-2">
              Status
            </label>
            <select
              id="filter-status"
              value={filters.status || ''}
              onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value || undefined }))}
              className="w-full p-2 bg-slate-800 border border-slate-600 rounded-md text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              <option value="">All Statuses</option>
              <option value="OPEN">Open</option>
              <option value="BLIND_EVALUATION">Blind Evaluation</option>
              <option value="AGGREGATED">Aggregated</option>
              <option value="CLOSED">Closed</option>
            </select>
          </div>

          <div>
            <label htmlFor="filter-task-type" className="block text-sm font-medium text-slate-300 mb-2">
              Task Type
            </label>
            <select
              id="filter-task-type"
              value={filters.task_type || ''}
              onChange={(e) => setFilters(prev => ({ ...prev, task_type: e.target.value || undefined }))}
              className="w-full p-2 bg-slate-800 border border-slate-600 rounded-md text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              <option value="">All Types</option>
              {Object.entries(TASK_TYPE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="filter-limit" className="block text-sm font-medium text-slate-300 mb-2">
              Limit
            </label>
            <select
              id="filter-limit"
              value={filters.limit || ''}
              onChange={(e) => setFilters(prev => ({ ...prev, limit: e.target.value ? parseInt(e.target.value) : undefined }))}
              className="w-full p-2 bg-slate-800 border border-slate-600 rounded-md text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              <option value="25">25 tasks</option>
              <option value="50">50 tasks</option>
              <option value="100">100 tasks</option>
              <option value="500">500 tasks</option>
            </select>
          </div>

          <div className="flex items-end">
            <Button
              variant="outline"
              onClick={() => setFilters({ limit: 50 })}
              className="w-full"
            >
              Clear Filters
            </Button>
          </div>
        </div>
      </Card>

      {/* Bulk Actions */}
      {selectedTasks.length > 0 && (
        <Card className="p-4 bg-blue-950/20 border-blue-600">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
            <span className="text-sm font-medium text-blue-300">
              {selectedTasks.length} task{selectedTasks.length !== 1 ? 's' : ''} selected
            </span>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => updateStatusMutation.mutate({ taskIds: selectedTasks, status: 'OPEN' })}
                disabled={updateStatusMutation.isPending}
              >
                Mark Open
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => updateStatusMutation.mutate({ taskIds: selectedTasks, status: 'CLOSED' })}
                disabled={updateStatusMutation.isPending}
              >
                Mark Closed
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => deleteMutation.mutate(selectedTasks)}
                disabled={deleteMutation.isPending}
                className="text-red-600 hover:text-red-700"
              >
                Delete
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Task List */}
      <Card>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-700">
            <thead className="bg-slate-800">
              <tr>
                <th className="px-4 md:px-6 py-3 text-left" scope="col">
                  <input
                    type="checkbox"
                    checked={tasks.length > 0 && selectedTasks.length === tasks.length}
                    onChange={handleSelectAll}
                    className="rounded border-slate-600 focus-visible:ring-2 focus-visible:ring-blue-500"
                    aria-label="Select all tasks"
                  />
                </th>
                <th className="px-4 md:px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider" scope="col">
                  Task
                </th>
                <th className="px-4 md:px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider" scope="col">
                  Type
                </th>
                <th className="px-4 md:px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider" scope="col">
                  Status
                </th>
                <th className="px-4 md:px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider hidden md:table-cell" scope="col">
                  Created
                </th>
                <th className="px-4 md:px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider" scope="col">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700">
              {isLoading ? (
                <tr>
                  <td colSpan={6} className="px-6 py-12 text-center text-slate-500">
                    <div role="status" className="flex items-center justify-center gap-2">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-400" aria-hidden="true"></div>
                      <span>Loading tasks...</span>
                      <span className="sr-only">Loading tasks</span>
                    </div>
                  </td>
                </tr>
              ) : tasks.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-12 text-center text-slate-500">
                    No tasks found. Upload a CSV file to get started.
                  </td>
                </tr>
              ) : (
                tasks.map((task: Task) => (
                  <tr key={task.id} className="hover:bg-slate-800/50">
                    <td className="px-4 md:px-6 py-4">
                      <input
                        type="checkbox"
                        checked={selectedTasks.includes(task.id)}
                        onChange={() => handleTaskSelect(task.id)}
                        className="rounded border-slate-600 focus-visible:ring-2 focus-visible:ring-blue-500"
                        aria-label={`Select task ${task.id}`}
                      />
                    </td>
                    <td className="px-4 md:px-6 py-4">
                      <div>
                        <p className="text-sm font-medium text-slate-200">
                          {getTaskTitle(task)}
                        </p>
                        <p className="text-xs text-slate-500">ID: {task.id}</p>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <Badge variant="outline">
                        {TASK_TYPE_LABELS[task.task_type as keyof typeof TASK_TYPE_LABELS] || task.task_type}
                      </Badge>
                    </td>
                    <td className="px-6 py-4">
                      <Badge className={STATUS_COLORS[task.status as keyof typeof STATUS_COLORS]}>
                        {task.status.replace('_', ' ')}
                      </Badge>
                    </td>
                    <td className="px-4 md:px-6 py-4 text-sm text-slate-400 hidden md:table-cell">
                      {new Date(task.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-4 md:px-6 py-4">
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline">
                          View
                        </Button>
                        <Button size="sm" variant="outline">
                          Edit
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="text-2xl font-bold text-white">{tasks.length}</div>
          <div className="text-sm text-slate-400">Total Tasks</div>
        </Card>
        <Card className="p-4">
          <div className="text-2xl font-bold text-yellow-400">
            {tasks.filter((t: Task) => t.status === 'BLIND_EVALUATION').length}
          </div>
          <div className="text-sm text-slate-400">In Evaluation</div>
        </Card>
        <Card className="p-4">
          <div className="text-2xl font-bold text-green-400">
            {tasks.filter((t: Task) => t.status === 'AGGREGATED').length}
          </div>
          <div className="text-sm text-slate-400">Aggregated</div>
        </Card>
        <Card className="p-4">
          <div className="text-2xl font-bold text-slate-300">
            {tasks.filter((t: Task) => t.status === 'CLOSED').length}
          </div>
          <div className="text-sm text-slate-400">Closed</div>
        </Card>
      </div>
    </div>
  );
};