import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Badge } from '../../components/ui/Badge';
import { apiClient } from '../../lib/api';
import { toast } from 'sonner';
import JsonView from '@uiw/react-json-view';
import { darkTheme } from '@uiw/react-json-view/dark';

interface ModelConfig {
  authority_weights: {
    baseline_credentials: number;
    track_record: number;
    recent_performance: number;
  };
  thresholds: {
    uncertainty_threshold: number;
    min_evaluators: number;
    devils_advocate_probability: number;
  };
  scoring_functions: {
    academic_degree: Record<string, number>;
    professional_experience: {
      base: number;
      sqrt_multiplier: number;
    };
    publications: {
      base: number;
      multiplier: number;
      max_value: number;
    };
  };
}

interface TaskConfig {
  [taskType: string]: {
    input_data: Record<string, string>;
    ground_truth_keys: string[];
    validation_schema: Record<string, any>;
  };
}

export function ConfigurationManager() {
  const [activeTab, setActiveTab] = useState<'model' | 'tasks' | 'preview'>('model');
  const [modelConfigText, setModelConfigText] = useState('');
  const [taskConfigText, setTaskConfigText] = useState('');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [showSyntaxPreview, setShowSyntaxPreview] = useState(false);
  const [showSaveConfirmation, setShowSaveConfirmation] = useState(false);
  const [pendingSaveType, setPendingSaveType] = useState<'model' | 'tasks' | null>(null);

  const queryClient = useQueryClient();

  // Fetch current model configuration
  const { data: currentModelConfig, isLoading: loadingModelConfig } = useQuery({
    queryKey: ['config-model'],
    queryFn: () => apiClient.config.getModel(),
  });

  // Fetch current task configuration
  const { data: currentTaskConfig, isLoading: loadingTaskConfig } = useQuery({
    queryKey: ['config-tasks'],
    queryFn: () => apiClient.config.getTasks(),
  });

  // Update model config mutation
  const updateModelConfigMutation = useMutation({
    mutationFn: (config: ModelConfig) => apiClient.config.updateModel(config),
    onMutate: () => {
      toast.loading('Saving model configuration...', { id: 'save-model-config' });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config-model'] });
      setHasUnsavedChanges(false);
      setValidationErrors([]);
      toast.success('Model configuration updated successfully! ✅', {
        id: 'save-model-config',
        description: 'Authority scoring parameters have been updated.'
      });
    },
    onError: (error: any) => {
      toast.error('Failed to update model configuration', {
        id: 'save-model-config',
        description: error.message || 'Please check your configuration and try again.'
      });
    },
  });

  // Update task config mutation
  const updateTaskConfigMutation = useMutation({
    mutationFn: (config: TaskConfig) => apiClient.config.updateTasks(config),
    onMutate: () => {
      toast.loading('Saving task configuration...', { id: 'save-task-config' });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config-tasks'] });
      setHasUnsavedChanges(false);
      setValidationErrors([]);
      toast.success('Task configuration updated successfully! ✅', {
        id: 'save-task-config',
        description: 'Task type schemas have been updated.'
      });
    },
    onError: (error: any) => {
      toast.error('Failed to update task configuration', {
        id: 'save-task-config',
        description: error.message || 'Please check your configuration and try again.'
      });
    },
  });

  // Initialize text areas when data loads
  useEffect(() => {
    if (currentModelConfig && !modelConfigText) {
      setModelConfigText(JSON.stringify(currentModelConfig, null, 2));
    }
  }, [currentModelConfig, modelConfigText]);

  useEffect(() => {
    if (currentTaskConfig && !taskConfigText) {
      setTaskConfigText(JSON.stringify(currentTaskConfig, null, 2));
    }
  }, [currentTaskConfig, taskConfigText]);

  // Validate JSON configuration
  const validateConfig = (configText: string, type: 'model' | 'tasks'): string[] => {
    const errors: string[] = [];
    
    try {
      const config = JSON.parse(configText);
      
      if (type === 'model') {
        // Validate model config structure
        if (!config.authority_weights) {
          errors.push('Missing authority_weights section');
        } else {
          const weights = config.authority_weights;
          const sum = (weights.baseline_credentials || 0) + 
                     (weights.track_record || 0) + 
                     (weights.recent_performance || 0);
          if (Math.abs(sum - 1.0) > 0.001) {
            errors.push(`Authority weights must sum to 1.0, got ${sum}`);
          }
        }
        
        if (!config.thresholds) {
          errors.push('Missing thresholds section');
        } else {
          if (config.thresholds.uncertainty_threshold < 0 || config.thresholds.uncertainty_threshold > 1) {
            errors.push('Uncertainty threshold must be between 0 and 1');
          }
        }
      } else {
        // Validate task config structure
        Object.keys(config).forEach(taskType => {
          const taskDef = config[taskType];
          if (!taskDef.input_data) {
            errors.push(`Task ${taskType} missing input_data`);
          }
          if (!taskDef.ground_truth_keys) {
            errors.push(`Task ${taskType} missing ground_truth_keys`);
          }
        });
      }
    } catch (e) {
      errors.push(`Invalid JSON: ${(e as Error).message}`);
    }
    
    return errors;
  };

  const handleSaveModelConfig = () => {
    const errors = validateConfig(modelConfigText, 'model');
    setValidationErrors(errors);

    if (errors.length === 0) {
      setPendingSaveType('model');
      setShowSaveConfirmation(true);
    } else {
      toast.error('Configuration validation failed', {
        description: `Found ${errors.length} error${errors.length > 1 ? 's' : ''}. Please fix them before saving.`
      });
    }
  };

  const handleSaveTaskConfig = () => {
    const errors = validateConfig(taskConfigText, 'tasks');
    setValidationErrors(errors);

    if (errors.length === 0) {
      setPendingSaveType('tasks');
      setShowSaveConfirmation(true);
    } else {
      toast.error('Configuration validation failed', {
        description: `Found ${errors.length} error${errors.length > 1 ? 's' : ''}. Please fix them before saving.`
      });
    }
  };

  const confirmSave = () => {
    try {
      if (pendingSaveType === 'model') {
        const config = JSON.parse(modelConfigText);
        updateModelConfigMutation.mutate(config);
      } else if (pendingSaveType === 'tasks') {
        const config = JSON.parse(taskConfigText);
        updateTaskConfigMutation.mutate(config);
      }
      setShowSaveConfirmation(false);
      setPendingSaveType(null);
    } catch (e) {
      toast.error('Failed to parse JSON', {
        description: (e as Error).message
      });
      setShowSaveConfirmation(false);
      setPendingSaveType(null);
    }
  };

  const handleReset = () => {
    if (activeTab === 'model' && currentModelConfig) {
      setModelConfigText(JSON.stringify(currentModelConfig, null, 2));
    } else if (activeTab === 'tasks' && currentTaskConfig) {
      setTaskConfigText(JSON.stringify(currentTaskConfig, null, 2));
    }
    setHasUnsavedChanges(false);
    setValidationErrors([]);
  };

  const handleTextChange = (value: string) => {
    if (activeTab === 'model') {
      setModelConfigText(value);
    } else {
      setTaskConfigText(value);
    }
    setHasUnsavedChanges(true);
    setValidationErrors([]);
  };

  if (loadingModelConfig || loadingTaskConfig) {
    return (
      <div className="flex items-center justify-center h-64" role="status">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400" aria-hidden="true"></div>
        <span className="sr-only">Loading configuration...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-4 md:p-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">RLCF Configuration Manager</h1>
        <p className="text-slate-400">
          Manage model parameters and task definitions with live validation
        </p>
      </div>

      {/* Status Card */}
      <Card className={`border-2 ${hasUnsavedChanges ? 'border-yellow-600 bg-yellow-950/10' : 'border-green-600 bg-green-950/10'}`}>
        <CardContent className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <div className={`text-2xl ${hasUnsavedChanges ? 'text-yellow-400' : 'text-green-400'}`}>
              {hasUnsavedChanges ? '⚠️' : '✅'}
            </div>
            <div>
              <div className={`font-semibold ${hasUnsavedChanges ? 'text-yellow-400' : 'text-green-400'}`}>
                {hasUnsavedChanges ? 'Unsaved Changes' : 'Configuration Synchronized'}
              </div>
              <div className="text-slate-400 text-sm">
                {hasUnsavedChanges 
                  ? 'Remember to save your changes before switching tabs'
                  : 'All configurations are up to date'
                }
              </div>
            </div>
          </div>
          {hasUnsavedChanges && (
            <Button variant="outline" onClick={handleReset} size="sm">
              Reset Changes
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Tab Navigation */}
      <div className="flex space-x-1 border-b border-slate-700 overflow-x-auto" role="tablist" aria-label="Configuration tabs">
        {(['model', 'tasks', 'preview'] as const).map((tab) => (
          <button
            key={tab}
            role="tab"
            aria-selected={activeTab === tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 font-medium text-sm transition-colors capitalize whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
              activeTab === tab
                ? 'text-purple-400 border-b-2 border-purple-400'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab} Config
          </button>
        ))}
      </div>

      {/* Model Configuration Tab */}
      {activeTab === 'model' && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6" role="tabpanel" aria-label="Model configuration">
          <div className="xl:col-span-2">
            <Card className="border-slate-700">
              <CardHeader>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div>
                    <CardTitle>Model Configuration (YAML/JSON)</CardTitle>
                    <p className="text-slate-400">
                      Authority formula: A(t) = alpha*B + beta*T(t-1) + gamma*P(t)
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowSyntaxPreview(!showSyntaxPreview)}
                    className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  >
                    {showSyntaxPreview ? 'Edit Mode' : 'Preview'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {showSyntaxPreview ? (
                  <div className="border border-slate-600 rounded-lg overflow-hidden bg-slate-900">
                    <JsonView
                      value={JSON.parse(modelConfigText || '{}')}
                      style={darkTheme}
                      displayDataTypes={false}
                      enableClipboard={false}
                    />
                  </div>
                ) : (
                  <textarea
                    value={modelConfigText}
                    onChange={(e) => handleTextChange(e.target.value)}
                    className="w-full h-96 p-4 bg-slate-900 border border-slate-600 rounded-lg font-mono text-sm text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    spellCheck={false}
                    aria-label="Model configuration JSON editor"
                    aria-invalid={validationErrors.length > 0}
                    aria-describedby={validationErrors.length > 0 ? 'model-validation-errors' : undefined}
                  />
                )}

                {validationErrors.length > 0 && (
                  <div id="model-validation-errors" className="mt-4 p-3 bg-red-950/20 border border-red-700 rounded-lg" role="alert">
                    <h4 className="text-red-400 font-semibold mb-2">Validation Errors:</h4>
                    <ul className="text-red-300 text-sm space-y-1">
                      {validationErrors.map((error, idx) => (
                        <li key={idx}>* {error}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 mt-4">
                  <div className="text-sm text-slate-500">
                    Live validation
                  </div>
                  <Button
                    onClick={handleSaveModelConfig}
                    disabled={validationErrors.length > 0 || updateModelConfigMutation.isPending}
                    className="bg-purple-600 hover:bg-purple-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  >
                    {updateModelConfigMutation.isPending ? 'Saving...' : 'Save Model Config'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Model Config Helper */}
          <Card className="border-slate-700">
            <CardHeader>
              <CardTitle>Configuration Guide</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold text-slate-200 mb-2">Authority Weights</h4>
                <div className="text-sm text-slate-400 space-y-1">
                  <p>• α (baseline_credentials): 0.3</p>
                  <p>• β (track_record): 0.5</p>
                  <p>• γ (recent_performance): 0.2</p>
                  <p className="text-yellow-400">⚠️ Must sum to 1.0</p>
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold text-slate-200 mb-2">Thresholds</h4>
                <div className="text-sm text-slate-400 space-y-1">
                  <p>• uncertainty_threshold: 0.4</p>
                  <p>• min_evaluators: 3</p>
                  <p>• devils_advocate_probability: 0.1</p>
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold text-slate-200 mb-2">Scoring Functions</h4>
                <div className="text-sm text-slate-400 space-y-1">
                  <p>• Academic degrees mapping</p>
                  <p>• Experience formula coefficients</p>
                  <p>• Publication scoring parameters</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Task Configuration Tab */}
      {activeTab === 'tasks' && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6" role="tabpanel" aria-label="Task configuration">
          <div className="xl:col-span-2">
            <Card className="border-slate-700">
              <CardHeader>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div>
                    <CardTitle>Task Type Definitions</CardTitle>
                    <p className="text-slate-400">
                      Define input schemas and validation rules for each task type
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowSyntaxPreview(!showSyntaxPreview)}
                    className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  >
                    {showSyntaxPreview ? 'Edit Mode' : 'Preview'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {showSyntaxPreview ? (
                  <div className="border border-slate-600 rounded-lg overflow-hidden bg-slate-900">
                    <JsonView
                      value={JSON.parse(taskConfigText || '{}')}
                      style={darkTheme}
                      displayDataTypes={false}
                      enableClipboard={false}
                    />
                  </div>
                ) : (
                  <textarea
                    value={taskConfigText}
                    onChange={(e) => handleTextChange(e.target.value)}
                    className="w-full h-96 p-4 bg-slate-900 border border-slate-600 rounded-lg font-mono text-sm text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    spellCheck={false}
                    aria-label="Task configuration JSON editor"
                    aria-invalid={validationErrors.length > 0}
                    aria-describedby={validationErrors.length > 0 ? 'task-validation-errors' : undefined}
                  />
                )}

                {validationErrors.length > 0 && (
                  <div id="task-validation-errors" className="mt-4 p-3 bg-red-950/20 border border-red-700 rounded-lg" role="alert">
                    <h4 className="text-red-400 font-semibold mb-2">Validation Errors:</h4>
                    <ul className="text-red-300 text-sm space-y-1">
                      {validationErrors.map((error, idx) => (
                        <li key={idx}>* {error}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 mt-4">
                  <div className="text-sm text-slate-500">
                    {currentTaskConfig ? `${Object.keys(currentTaskConfig).length} task types configured` : ''}
                  </div>
                  <Button
                    onClick={handleSaveTaskConfig}
                    disabled={validationErrors.length > 0 || updateTaskConfigMutation.isPending}
                    className="bg-purple-600 hover:bg-purple-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  >
                    {updateTaskConfigMutation.isPending ? 'Saving...' : 'Save Task Config'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Task Config Helper */}
          <Card className="border-slate-700">
            <CardHeader>
              <CardTitle>Task Types</CardTitle>
            </CardHeader>
            <CardContent>
              {currentTaskConfig && (
                <div className="space-y-3">
                  {Object.keys(currentTaskConfig).map((taskType) => (
                    <div key={taskType} className="p-3 bg-slate-800/50 rounded-lg">
                      <Badge variant="outline" className="text-purple-400 border-purple-400 mb-2">
                        {taskType}
                      </Badge>
                      <div className="text-sm text-slate-400">
                        <p>• Input fields: {Object.keys(currentTaskConfig[taskType].input_data || {}).length}</p>
                        <p>• Ground truth: {(currentTaskConfig[taskType].ground_truth_keys || []).length} keys</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Preview Tab */}
      {activeTab === 'preview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6" role="tabpanel" aria-label="Configuration preview">
          <Card className="border-slate-700">
            <CardHeader>
              <CardTitle>Current Model Parameters</CardTitle>
            </CardHeader>
            <CardContent>
              {currentModelConfig ? (
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-slate-200 mb-2">Authority Formula Weights</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Baseline Credentials (α)</span>
                        <span className="font-mono text-blue-400">
                          {currentModelConfig.authority_weights?.baseline_credentials || 0.3}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Track Record (β)</span>
                        <span className="font-mono text-green-400">
                          {currentModelConfig.authority_weights?.track_record || 0.5}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Recent Performance (γ)</span>
                        <span className="font-mono text-yellow-400">
                          {currentModelConfig.authority_weights?.recent_performance || 0.2}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="border-t border-slate-600 pt-4">
                    <h4 className="font-semibold text-slate-200 mb-2">System Thresholds</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Uncertainty Threshold (τ)</span>
                        <span className="font-mono text-purple-400">
                          {currentModelConfig.thresholds?.uncertainty_threshold || 0.4}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Min Evaluators</span>
                        <span className="font-mono text-purple-400">
                          {currentModelConfig.thresholds?.min_evaluators || 3}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-slate-500">No model configuration loaded</p>
              )}
            </CardContent>
          </Card>

          <Card className="border-slate-700">
            <CardHeader>
              <CardTitle>Task Configuration Summary</CardTitle>
            </CardHeader>
            <CardContent>
              {currentTaskConfig ? (
                <div className="space-y-3">
                  {Object.entries(currentTaskConfig).map(([taskType, config]: [string, any]) => (
                    <div key={taskType} className="p-3 bg-slate-800/30 rounded-lg">
                      <div className="font-semibold text-slate-200 mb-1">{taskType}</div>
                      <div className="text-sm text-slate-400 space-y-1">
                        <p>Input fields: {Object.keys(config?.input_data || {}).join(', ')}</p>
                        <p>Ground truth: {(config?.ground_truth_keys || []).join(', ')}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-slate-500">No task configuration loaded</p>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Confirmation Dialog */}
      {showSaveConfirmation && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4" role="dialog" aria-modal="true" aria-labelledby="confirm-dialog-title">
          <Card className="max-w-md w-full border-yellow-600 bg-slate-900">
            <CardHeader>
              <CardTitle id="confirm-dialog-title" className="flex items-center gap-2 text-yellow-400">
                Confirm Configuration Change
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-slate-300">
                You are about to save changes to the <strong className="text-purple-400">
                  {pendingSaveType === 'model' ? 'Model Configuration' : 'Task Configuration'}
                </strong>.
              </p>
              <div className="p-3 bg-yellow-950/20 border border-yellow-700 rounded-lg" role="alert">
                <p className="text-yellow-300 text-sm">
                  <strong>Warning:</strong> These changes will affect how the RLCF framework calculates authority scores
                  {pendingSaveType === 'model' ? ' and validates feedback' : ' and processes task inputs'}.
                  This may impact ongoing evaluations.
                </p>
              </div>
              <div className="flex gap-3 justify-end">
                <Button
                  variant="outline"
                  className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  onClick={() => {
                    setShowSaveConfirmation(false);
                    setPendingSaveType(null);
                  }}
                >
                  Cancel
                </Button>
                <Button
                  onClick={confirmSave}
                  className="bg-purple-600 hover:bg-purple-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                >
                  Confirm & Save
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}