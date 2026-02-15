import { z } from 'zod';
import type { FieldErrors, UseFormRegister, UseFormSetValue } from 'react-hook-form';

// Schemas aligned with backend `task_config.yaml`
export const TASK_FORM_SCHEMAS = {
  QA: z.object({
    validated_answer: z.string().min(10, 'Provide a validated answer (min 10 chars)'),
    position: z.enum(['correct', 'incorrect']),
    reasoning: z.string().min(50, 'Reasoning must be at least 50 chars'),
    source_accuracy: z.enum(['accurate', 'partially_accurate', 'inaccurate']),
    completeness: z.enum(['complete', 'missing_minor', 'missing_major', 'incomplete']),
  }),
  STATUTORY_RULE_QA: z.object({
    validated_answer: z.string().min(10, 'Provide a validated answer (min 10 chars)'),
    position: z.enum(['correct', 'incorrect']),
    reasoning: z.string().min(50, 'Reasoning must be at least 50 chars'),
    legal_accuracy: z.enum(['accurate', 'partially_accurate', 'inaccurate']),
    citation_quality: z.enum(['excellent', 'good', 'fair', 'poor']),
    omitted_articles: z.string().optional(),
    citation_corrections: z.string().optional(),
  }),
  CLASSIFICATION: z.object({
    validated_labels: z.array(z.string()).min(1, 'Select at least one label'),
    reasoning: z.string().min(50, 'Reasoning must be at least 50 chars'),
    confidence_per_label: z.record(z.number().min(0).max(1)).optional(),
    missed_labels: z.string().optional(),
  }),
  SUMMARIZATION: z.object({
    revised_summary: z.string().min(30, 'Summary must be at least 30 chars'),
    rating: z.enum(['good', 'bad']),
    reasoning: z.string().min(50),
    key_points_coverage: z.enum(['excellent', 'good', 'fair', 'poor']),
    factual_accuracy: z.enum(['accurate', 'mostly_accurate', 'some_errors', 'many_errors']),
  }),
  PREDICTION: z.object({
    chosen_outcome: z.enum(['violation', 'no_violation']),
    reasoning: z.string().min(50),
    confidence: z.number().min(0).max(1),
    risk_factors: z.string().optional(),
  }),
  NLI: z.object({
    chosen_label: z.enum(['entail', 'contradict', 'neutral']),
    reasoning: z.string().min(50),
    confidence: z.number().min(0).max(1),
    logical_structure: z.string().optional(),
  }),
  NER: z.object({
    validated_tags: z.array(z.string()).min(1),
    reasoning: z.string().min(30),
    entity_corrections: z.string().optional(),
    missed_entities: z.string().optional(),
  }),
  DRAFTING: z.object({
    revised_target: z.string().min(30),
    rating: z.enum(['better', 'worse']),
    reasoning: z.string().min(50),
    style_improvements: z.string().optional(),
    legal_compliance: z.enum(['compliant', 'needs_review', 'non_compliant']),
  }),
  RISK_SPOTTING: z.object({
    validated_risk_labels: z.array(z.string()).min(1),
    validated_severity: z.number().min(0).max(10),
    reasoning: z.string().min(50),
    mitigation_suggestions: z.string().optional(),
    regulatory_references: z.string().optional(),
  }),
  DOCTRINE_APPLICATION: z.object({
    chosen_label: z.enum(['yes', 'no']),
    reasoning: z.string().min(50),
    doctrine_analysis: z.string().min(30),
    precedent_citations: z.string().optional(),
    alternative_interpretations: z.string().optional(),
  }),
} as const;

export function getSchemaForTaskType(taskType: string) {
  return TASK_FORM_SCHEMAS[taskType as keyof typeof TASK_FORM_SCHEMAS] ?? z.object({});
}

interface TaskFormFieldsProps {
  taskType: string;
  register: UseFormRegister<any>;
  setValue: UseFormSetValue<any>;
  errors: FieldErrors<any>;
}

export function TaskFormFields({ taskType, register, setValue, errors }: TaskFormFieldsProps) {
  if (taskType === 'QA') {
    return (
      <div className="space-y-6">
        <div className="space-y-1">
          <label htmlFor="qa-validated-answer" className="block text-sm font-medium text-slate-300">Validated Answer</label>
          <textarea
            id="qa-validated-answer"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={4}
            placeholder="Provide your validated answer..."
            aria-describedby={errors?.validated_answer ? 'qa-validated-answer-error' : undefined}
            {...register('validated_answer')}
          />
          {errors?.validated_answer && <p id="qa-validated-answer-error" className="text-red-400 text-xs" role="alert">{String(errors.validated_answer.message)}</p>}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label htmlFor="qa-position" className="block text-sm font-medium text-slate-300">Position</label>
            <select id="qa-position" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.position ? 'qa-position-error' : undefined} {...register('position')}>
              <option value="">Select position...</option>
              <option value="correct">Correct</option>
              <option value="incorrect">Incorrect</option>
            </select>
            {errors?.position && <p id="qa-position-error" className="text-red-400 text-xs" role="alert">{String(errors.position.message)}</p>}
          </div>

          <div className="space-y-1">
            <label htmlFor="qa-source-accuracy" className="block text-sm font-medium text-slate-300">Source Accuracy</label>
            <select id="qa-source-accuracy" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.source_accuracy ? 'qa-source-accuracy-error' : undefined} {...register('source_accuracy')}>
              <option value="">Select accuracy...</option>
              <option value="accurate">Accurate</option>
              <option value="partially_accurate">Partially Accurate</option>
              <option value="inaccurate">Inaccurate</option>
            </select>
            {errors?.source_accuracy && <p id="qa-source-accuracy-error" className="text-red-400 text-xs" role="alert">{String(errors.source_accuracy.message)}</p>}
          </div>
        </div>

        <div className="space-y-1">
          <label htmlFor="qa-completeness" className="block text-sm font-medium text-slate-300">Completeness</label>
          <select id="qa-completeness" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.completeness ? 'qa-completeness-error' : undefined} {...register('completeness')}>
            <option value="">Rate completeness...</option>
            <option value="complete">Complete</option>
            <option value="missing_minor">Missing Minor Details</option>
            <option value="missing_major">Missing Major Elements</option>
            <option value="incomplete">Incomplete</option>
          </select>
          {errors?.completeness && <p id="qa-completeness-error" className="text-red-400 text-xs" role="alert">{String(errors.completeness.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="qa-reasoning" className="block text-sm font-medium text-slate-300">Reasoning</label>
          <textarea
            id="qa-reasoning"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={6}
            placeholder="Explain your reasoning..."
            aria-describedby={errors?.reasoning ? 'qa-reasoning-error' : undefined}
            {...register('reasoning')}
          />
          {errors?.reasoning && <p id="qa-reasoning-error" className="text-red-400 text-xs" role="alert">{String(errors.reasoning.message)}</p>}
        </div>
      </div>
    );
  }

  if (taskType === 'STATUTORY_RULE_QA') {
    return (
      <div className="space-y-6">
        <div className="space-y-1">
          <label htmlFor="srqa-validated-answer" className="block text-sm font-medium text-slate-300">Validated Legal Answer</label>
          <textarea
            id="srqa-validated-answer"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={6}
            placeholder="Provide your expert legal analysis and answer..."
            aria-describedby={errors?.validated_answer ? 'srqa-validated-answer-error' : undefined}
            {...register('validated_answer')}
          />
          {errors?.validated_answer && <p id="srqa-validated-answer-error" className="text-red-400 text-xs" role="alert">{String(errors.validated_answer.message)}</p>}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label htmlFor="srqa-position" className="block text-sm font-medium text-slate-300">Overall Position</label>
            <select id="srqa-position" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.position ? 'srqa-position-error' : undefined} {...register('position')}>
              <option value="">Select position...</option>
              <option value="correct">Correct</option>
              <option value="incorrect">Incorrect</option>
            </select>
            {errors?.position && <p id="srqa-position-error" className="text-red-400 text-xs" role="alert">{String(errors.position.message)}</p>}
          </div>

          <div className="space-y-1">
            <label htmlFor="srqa-legal-accuracy" className="block text-sm font-medium text-slate-300">Legal Accuracy</label>
            <select id="srqa-legal-accuracy" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.legal_accuracy ? 'srqa-legal-accuracy-error' : undefined} {...register('legal_accuracy')}>
              <option value="">Select accuracy...</option>
              <option value="accurate">Accurate</option>
              <option value="partially_accurate">Partially Accurate</option>
              <option value="inaccurate">Inaccurate</option>
            </select>
            {errors?.legal_accuracy && <p id="srqa-legal-accuracy-error" className="text-red-400 text-xs" role="alert">{String(errors.legal_accuracy.message)}</p>}
          </div>
        </div>

        <div className="space-y-1">
          <label htmlFor="srqa-citation-quality" className="block text-sm font-medium text-slate-300">Citation Quality</label>
          <select id="srqa-citation-quality" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.citation_quality ? 'srqa-citation-quality-error' : undefined} {...register('citation_quality')}>
            <option value="">Rate citation quality...</option>
            <option value="excellent">Excellent</option>
            <option value="good">Good</option>
            <option value="fair">Fair</option>
            <option value="poor">Poor</option>
          </select>
          {errors?.citation_quality && <p id="srqa-citation-quality-error" className="text-red-400 text-xs" role="alert">{String(errors.citation_quality.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="srqa-reasoning" className="block text-sm font-medium text-slate-300">Legal Reasoning</label>
          <textarea
            id="srqa-reasoning"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={8}
            placeholder="Explain your legal reasoning..."
            aria-describedby={errors?.reasoning ? 'srqa-reasoning-error' : undefined}
            {...register('reasoning')}
          />
          {errors?.reasoning && <p id="srqa-reasoning-error" className="text-red-400 text-xs" role="alert">{String(errors.reasoning.message)}</p>}
        </div>

        <div className="space-y-4 pt-4 border-t border-slate-700">
            <h4 className="text-md font-semibold text-slate-200">Citation & Source Feedback</h4>
            <div className="space-y-1">
              <label htmlFor="srqa-omitted-articles" className="block text-sm font-medium text-slate-300">Omitted Articles / Sources</label>
              <textarea
                id="srqa-omitted-articles"
                className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                rows={3}
                placeholder="List any relevant articles, laws, or sources the AI missed. One per line."
                aria-describedby={errors?.omitted_articles ? 'srqa-omitted-articles-error' : undefined}
                {...register('omitted_articles')}
              />
              {errors?.omitted_articles && <p id="srqa-omitted-articles-error" className="text-red-400 text-xs" role="alert">{String(errors.omitted_articles.message)}</p>}
            </div>
            <div className="space-y-1">
              <label htmlFor="srqa-citation-corrections" className="block text-sm font-medium text-slate-300">Citation Corrections</label>
              <textarea
                id="srqa-citation-corrections"
                className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                rows={3}
                placeholder="Provide corrections for any inaccurate citations. E.g., 'The reference to Art. 5 should be Art. 5, comma 2.'"
                aria-describedby={errors?.citation_corrections ? 'srqa-citation-corrections-error' : undefined}
                {...register('citation_corrections')}
              />
              {errors?.citation_corrections && <p id="srqa-citation-corrections-error" className="text-red-400 text-xs" role="alert">{String(errors.citation_corrections.message)}</p>}
            </div>
        </div>
      </div>
    );
  }

  if (taskType === 'CLASSIFICATION') {
    return (
      <div className="space-y-6">
        <div className="space-y-1">
          <label htmlFor="cls-validated-labels" className="block text-sm font-medium text-slate-300">Validated Labels</label>
          <input
            id="cls-validated-labels"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            placeholder="Enter labels separated by commas (e.g., employment, confidentiality)"
            aria-describedby={errors?.validated_labels ? 'cls-validated-labels-error' : undefined}
            {...register('validated_labels', {
              setValueAs: (value: string) => value.split(',').map(s => s.trim()).filter(Boolean)
            })}
          />
          {errors?.validated_labels && <p id="cls-validated-labels-error" className="text-red-400 text-xs" role="alert">{String(errors.validated_labels.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="cls-reasoning" className="block text-sm font-medium text-slate-300">Reasoning</label>
          <textarea
            id="cls-reasoning"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={6}
            placeholder="Explain your classification reasoning..."
            aria-describedby={errors?.reasoning ? 'cls-reasoning-error' : undefined}
            {...register('reasoning')}
          />
          {errors?.reasoning && <p id="cls-reasoning-error" className="text-red-400 text-xs" role="alert">{String(errors.reasoning.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="cls-missed-labels" className="block text-sm font-medium text-slate-300">Missed Labels (Optional)</label>
          <textarea
            id="cls-missed-labels"
            className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={2}
            placeholder="List any labels the AI should have included but missed..."
            {...register('missed_labels')}
          />
        </div>
      </div>
    );
  }

  if (taskType === 'SUMMARIZATION') {
    return (
      <div className="space-y-6">
        <div className="space-y-1">
          <label htmlFor="sum-revised-summary" className="block text-sm font-medium text-slate-300">Revised Summary</label>
          <textarea
            id="sum-revised-summary"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={5}
            placeholder="Provide your improved summary..."
            aria-describedby={errors?.revised_summary ? 'sum-revised-summary-error' : undefined}
            {...register('revised_summary')}
          />
          {errors?.revised_summary && <p id="sum-revised-summary-error" className="text-red-400 text-xs" role="alert">{String(errors.revised_summary.message)}</p>}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-1">
            <label htmlFor="sum-rating" className="block text-sm font-medium text-slate-300">Overall Rating</label>
            <select id="sum-rating" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.rating ? 'sum-rating-error' : undefined} {...register('rating')}>
              <option value="">Select rating...</option>
              <option value="good">Good</option>
              <option value="bad">Bad</option>
            </select>
            {errors?.rating && <p id="sum-rating-error" className="text-red-400 text-xs" role="alert">{String(errors.rating.message)}</p>}
          </div>

          <div className="space-y-1">
            <label htmlFor="sum-key-points" className="block text-sm font-medium text-slate-300">Key Points Coverage</label>
            <select id="sum-key-points" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.key_points_coverage ? 'sum-key-points-error' : undefined} {...register('key_points_coverage')}>
              <option value="">Rate coverage...</option>
              <option value="excellent">Excellent</option>
              <option value="good">Good</option>
              <option value="fair">Fair</option>
              <option value="poor">Poor</option>
            </select>
            {errors?.key_points_coverage && <p id="sum-key-points-error" className="text-red-400 text-xs" role="alert">{String(errors.key_points_coverage.message)}</p>}
          </div>

          <div className="space-y-1">
            <label htmlFor="sum-factual-accuracy" className="block text-sm font-medium text-slate-300">Factual Accuracy</label>
            <select id="sum-factual-accuracy" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.factual_accuracy ? 'sum-factual-accuracy-error' : undefined} {...register('factual_accuracy')}>
              <option value="">Rate accuracy...</option>
              <option value="accurate">Accurate</option>
              <option value="mostly_accurate">Mostly Accurate</option>
              <option value="some_errors">Some Errors</option>
              <option value="many_errors">Many Errors</option>
            </select>
            {errors?.factual_accuracy && <p id="sum-factual-accuracy-error" className="text-red-400 text-xs" role="alert">{String(errors.factual_accuracy.message)}</p>}
          </div>
        </div>

        <div className="space-y-1">
          <label htmlFor="sum-reasoning" className="block text-sm font-medium text-slate-300">Reasoning</label>
          <textarea
            id="sum-reasoning"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={6}
            placeholder="Explain your evaluation of the summary..."
            aria-describedby={errors?.reasoning ? 'sum-reasoning-error' : undefined}
            {...register('reasoning')}
          />
          {errors?.reasoning && <p id="sum-reasoning-error" className="text-red-400 text-xs" role="alert">{String(errors.reasoning.message)}</p>}
        </div>
      </div>
    );
  }

  if (taskType === 'PREDICTION') {
    return (
      <div className="space-y-6">
        <div className="space-y-1">
          <label htmlFor="pred-outcome" className="block text-sm font-medium text-slate-300">Predicted Outcome</label>
          <select id="pred-outcome" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.chosen_outcome ? 'pred-outcome-error' : undefined} {...register('chosen_outcome')}>
            <option value="">Select outcome...</option>
            <option value="violation">Violation</option>
            <option value="no_violation">No Violation</option>
          </select>
          {errors?.chosen_outcome && <p id="pred-outcome-error" className="text-red-400 text-xs" role="alert">{String(errors.chosen_outcome.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="pred-confidence" className="block text-sm font-medium text-slate-300">Confidence Level</label>
          <input
            id="pred-confidence"
            type="range"
            min="0"
            max="1"
            step="0.1"
            className="w-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            aria-describedby={errors?.confidence ? 'pred-confidence-error' : undefined}
            {...register('confidence', { setValueAs: (value: string) => parseFloat(value) })}
          />
          <div className="text-xs text-slate-400 flex justify-between" aria-hidden="true">
            <span>0% (Not Confident)</span>
            <span>100% (Very Confident)</span>
          </div>
          {errors?.confidence && <p id="pred-confidence-error" className="text-red-400 text-xs" role="alert">{String(errors.confidence.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="pred-reasoning" className="block text-sm font-medium text-slate-300">Reasoning</label>
          <textarea
            id="pred-reasoning"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={6}
            placeholder="Explain your prediction reasoning..."
            aria-describedby={errors?.reasoning ? 'pred-reasoning-error' : undefined}
            {...register('reasoning')}
          />
          {errors?.reasoning && <p id="pred-reasoning-error" className="text-red-400 text-xs" role="alert">{String(errors.reasoning.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="pred-risk-factors" className="block text-sm font-medium text-slate-300">Risk Factors (Optional)</label>
          <textarea
            id="pred-risk-factors"
            className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={3}
            placeholder="Identify key risk factors..."
            {...register('risk_factors')}
          />
        </div>
      </div>
    );
  }

  if (taskType === 'NLI') {
    return (
      <div className="space-y-6">
        <div className="space-y-1">
          <label htmlFor="nli-chosen-label" className="block text-sm font-medium text-slate-300">Logical Relationship</label>
          <select id="nli-chosen-label" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.chosen_label ? 'nli-chosen-label-error' : undefined} {...register('chosen_label')}>
            <option value="">Select relationship...</option>
            <option value="entail">Entail (follows logically)</option>
            <option value="contradict">Contradict (conflict)</option>
            <option value="neutral">Neutral (no relationship)</option>
          </select>
          {errors?.chosen_label && <p id="nli-chosen-label-error" className="text-red-400 text-xs" role="alert">{String(errors.chosen_label.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="nli-confidence" className="block text-sm font-medium text-slate-300">Confidence Level</label>
          <input
            id="nli-confidence"
            type="range"
            min="0"
            max="1"
            step="0.1"
            className="w-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            aria-describedby={errors?.confidence ? 'nli-confidence-error' : undefined}
            {...register('confidence', { setValueAs: (value: string) => parseFloat(value) })}
          />
          <div className="text-xs text-slate-400 flex justify-between" aria-hidden="true">
            <span>0% (Not Confident)</span>
            <span>100% (Very Confident)</span>
          </div>
          {errors?.confidence && <p id="nli-confidence-error" className="text-red-400 text-xs" role="alert">{String(errors.confidence.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="nli-reasoning" className="block text-sm font-medium text-slate-300">Reasoning</label>
          <textarea
            id="nli-reasoning"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={6}
            placeholder="Explain the logical relationship..."
            aria-describedby={errors?.reasoning ? 'nli-reasoning-error' : undefined}
            {...register('reasoning')}
          />
          {errors?.reasoning && <p id="nli-reasoning-error" className="text-red-400 text-xs" role="alert">{String(errors.reasoning.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="nli-logical-structure" className="block text-sm font-medium text-slate-300">Logical Structure (Optional)</label>
          <textarea
            id="nli-logical-structure"
            className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={3}
            placeholder="Describe the logical structure or connections..."
            {...register('logical_structure')}
          />
        </div>
      </div>
    );
  }

  if (taskType === 'NER') {
    return (
      <div className="space-y-6">
        <div className="space-y-1">
          <label htmlFor="ner-validated-tags" className="block text-sm font-medium text-slate-300">Validated Entity Tags</label>
          <input
            id="ner-validated-tags"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            placeholder="Enter tags separated by commas (e.g., PERSON, ORG, DATE)"
            aria-describedby={errors?.validated_tags ? 'ner-validated-tags-error' : undefined}
            {...register('validated_tags', {
              setValueAs: (value: string) => value.split(',').map(s => s.trim()).filter(Boolean)
            })}
          />
          {errors?.validated_tags && <p id="ner-validated-tags-error" className="text-red-400 text-xs" role="alert">{String(errors.validated_tags.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="ner-reasoning" className="block text-sm font-medium text-slate-300">Reasoning</label>
          <textarea
            id="ner-reasoning"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={4}
            placeholder="Explain your entity recognition decisions..."
            aria-describedby={errors?.reasoning ? 'ner-reasoning-error' : undefined}
            {...register('reasoning')}
          />
          {errors?.reasoning && <p id="ner-reasoning-error" className="text-red-400 text-xs" role="alert">{String(errors.reasoning.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="ner-entity-corrections" className="block text-sm font-medium text-slate-300">Entity Corrections (Optional)</label>
          <textarea
            id="ner-entity-corrections"
            className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={3}
            placeholder="Describe any corrections needed..."
            {...register('entity_corrections')}
          />
        </div>

        <div className="space-y-1">
          <label htmlFor="ner-missed-entities" className="block text-sm font-medium text-slate-300">Missed Entities (Optional)</label>
          <textarea
            id="ner-missed-entities"
            className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={3}
            placeholder="List entities the AI missed..."
            {...register('missed_entities')}
          />
        </div>
      </div>
    );
  }

  if (taskType === 'DRAFTING') {
    return (
      <div className="space-y-6">
        <div className="space-y-1">
          <label htmlFor="draft-revised-target" className="block text-sm font-medium text-slate-300">Revised Target Document</label>
          <textarea
            id="draft-revised-target"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={8}
            placeholder="Provide your improved document draft..."
            aria-describedby={errors?.revised_target ? 'draft-revised-target-error' : undefined}
            {...register('revised_target')}
          />
          {errors?.revised_target && <p id="draft-revised-target-error" className="text-red-400 text-xs" role="alert">{String(errors.revised_target.message)}</p>}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label htmlFor="draft-rating" className="block text-sm font-medium text-slate-300">Overall Rating</label>
            <select id="draft-rating" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.rating ? 'draft-rating-error' : undefined} {...register('rating')}>
              <option value="">Select rating...</option>
              <option value="better">Better</option>
              <option value="worse">Worse</option>
            </select>
            {errors?.rating && <p id="draft-rating-error" className="text-red-400 text-xs" role="alert">{String(errors.rating.message)}</p>}
          </div>

          <div className="space-y-1">
            <label htmlFor="draft-legal-compliance" className="block text-sm font-medium text-slate-300">Legal Compliance</label>
            <select id="draft-legal-compliance" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.legal_compliance ? 'draft-legal-compliance-error' : undefined} {...register('legal_compliance')}>
              <option value="">Rate compliance...</option>
              <option value="compliant">Compliant</option>
              <option value="needs_review">Needs Review</option>
              <option value="non_compliant">Non-Compliant</option>
            </select>
            {errors?.legal_compliance && <p id="draft-legal-compliance-error" className="text-red-400 text-xs" role="alert">{String(errors.legal_compliance.message)}</p>}
          </div>
        </div>

        <div className="space-y-1">
          <label htmlFor="draft-reasoning" className="block text-sm font-medium text-slate-300">Reasoning</label>
          <textarea
            id="draft-reasoning"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={6}
            placeholder="Explain your evaluation and revisions..."
            aria-describedby={errors?.reasoning ? 'draft-reasoning-error' : undefined}
            {...register('reasoning')}
          />
          {errors?.reasoning && <p id="draft-reasoning-error" className="text-red-400 text-xs" role="alert">{String(errors.reasoning.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="draft-style-improvements" className="block text-sm font-medium text-slate-300">Style Improvements (Optional)</label>
          <textarea
            id="draft-style-improvements"
            className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={3}
            placeholder="Suggest style and language improvements..."
            {...register('style_improvements')}
          />
        </div>
      </div>
    );
  }

  if (taskType === 'RISK_SPOTTING') {
    return (
      <div className="space-y-6">
        <div className="space-y-1">
          <label htmlFor="risk-labels" className="block text-sm font-medium text-slate-300">Risk Labels</label>
          <input
            id="risk-labels"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            placeholder="Enter risk types separated by commas (e.g., privacy, compliance, regulatory)"
            aria-describedby={errors?.validated_risk_labels ? 'risk-labels-error' : undefined}
            {...register('validated_risk_labels', {
              setValueAs: (value: string) => value.split(',').map(s => s.trim()).filter(Boolean)
            })}
          />
          {errors?.validated_risk_labels && <p id="risk-labels-error" className="text-red-400 text-xs" role="alert">{String(errors.validated_risk_labels.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="risk-severity" className="block text-sm font-medium text-slate-300">Risk Severity (0-10)</label>
          <input
            id="risk-severity"
            type="range"
            min="0"
            max="10"
            step="1"
            className="w-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            aria-describedby={errors?.validated_severity ? 'risk-severity-error' : undefined}
            {...register('validated_severity', { setValueAs: (value: string) => parseInt(value, 10) })}
          />
          <div className="text-xs text-slate-400 flex justify-between" aria-hidden="true">
            <span>0 (Low Risk)</span>
            <span>10 (Critical Risk)</span>
          </div>
          {errors?.validated_severity && <p id="risk-severity-error" className="text-red-400 text-xs" role="alert">{String(errors.validated_severity.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="risk-reasoning" className="block text-sm font-medium text-slate-300">Risk Analysis</label>
          <textarea
            id="risk-reasoning"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={6}
            placeholder="Explain the identified risks..."
            aria-describedby={errors?.reasoning ? 'risk-reasoning-error' : undefined}
            {...register('reasoning')}
          />
          {errors?.reasoning && <p id="risk-reasoning-error" className="text-red-400 text-xs" role="alert">{String(errors.reasoning.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="risk-mitigation" className="block text-sm font-medium text-slate-300">Mitigation Suggestions (Optional)</label>
          <textarea
            id="risk-mitigation"
            className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={4}
            placeholder="Suggest how to mitigate these risks..."
            {...register('mitigation_suggestions')}
          />
        </div>

        <div className="space-y-1">
          <label htmlFor="risk-regulatory-refs" className="block text-sm font-medium text-slate-300">Regulatory References (Optional)</label>
          <textarea
            id="risk-regulatory-refs"
            className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={3}
            placeholder="Reference relevant regulations, laws, or guidelines..."
            {...register('regulatory_references')}
          />
        </div>
      </div>
    );
  }

  if (taskType === 'DOCTRINE_APPLICATION') {
    return (
      <div className="space-y-6">
        <div className="space-y-1">
          <label htmlFor="doc-chosen-label" className="block text-sm font-medium text-slate-300">Doctrine Application</label>
          <select id="doc-chosen-label" className="w-full p-2 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500" aria-describedby={errors?.chosen_label ? 'doc-chosen-label-error' : undefined} {...register('chosen_label')}>
            <option value="">Select application...</option>
            <option value="yes">Yes - Doctrine Applies</option>
            <option value="no">No - Doctrine Does Not Apply</option>
          </select>
          {errors?.chosen_label && <p id="doc-chosen-label-error" className="text-red-400 text-xs" role="alert">{String(errors.chosen_label.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="doc-analysis" className="block text-sm font-medium text-slate-300">Doctrine Analysis</label>
          <textarea
            id="doc-analysis"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={5}
            placeholder="Analyze how the doctrine applies to these facts..."
            aria-describedby={errors?.doctrine_analysis ? 'doc-analysis-error' : undefined}
            {...register('doctrine_analysis')}
          />
          {errors?.doctrine_analysis && <p id="doc-analysis-error" className="text-red-400 text-xs" role="alert">{String(errors.doctrine_analysis.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="doc-reasoning" className="block text-sm font-medium text-slate-300">Legal Reasoning</label>
          <textarea
            id="doc-reasoning"
            className="w-full p-3 bg-slate-900 border border-purple-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={6}
            placeholder="Explain your legal reasoning..."
            aria-describedby={errors?.reasoning ? 'doc-reasoning-error' : undefined}
            {...register('reasoning')}
          />
          {errors?.reasoning && <p id="doc-reasoning-error" className="text-red-400 text-xs" role="alert">{String(errors.reasoning.message)}</p>}
        </div>

        <div className="space-y-1">
          <label htmlFor="doc-precedent-citations" className="block text-sm font-medium text-slate-300">Precedent Citations (Optional)</label>
          <textarea
            id="doc-precedent-citations"
            className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={4}
            placeholder="Cite relevant case law and precedents..."
            {...register('precedent_citations')}
          />
        </div>

        <div className="space-y-1">
          <label htmlFor="doc-alt-interpretations" className="block text-sm font-medium text-slate-300">Alternative Interpretations (Optional)</label>
          <textarea
            id="doc-alt-interpretations"
            className="w-full p-3 bg-slate-900 border border-slate-700 rounded text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            rows={4}
            placeholder="Discuss alternative legal interpretations..."
            {...register('alternative_interpretations')}
          />
        </div>
      </div>
    );
  }

  // Fallback for unknown task types
  return (
    <div className="text-center py-8">
      <p className="text-slate-400">⚠️ No specific form available for task type: <span className="font-mono text-purple-400">{taskType}</span></p>
      <p className="text-xs text-slate-500 mt-2">Please contact support to add this task type.</p>
    </div>
  );
}

export type TaskFormSchema = typeof TASK_FORM_SCHEMAS[keyof typeof TASK_FORM_SCHEMAS];
export type TaskFormValues<T extends TaskFormSchema> = z.infer<T>;