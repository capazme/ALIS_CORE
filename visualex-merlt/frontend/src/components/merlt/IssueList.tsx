/**
 * IssueList
 * =========
 *
 * Componente per visualizzare e votare sulle segnalazioni del Knowledge Graph.
 * Fa parte del ciclo RLCF: la community vota per decidere se le issue sono valide.
 *
 * Se una issue raggiunge la soglia di upvote (2.0), l'entita' torna in validazione.
 */

import { useState, useEffect, useCallback } from 'react';
import {
  AlertTriangle,
  ThumbsUp,
  ThumbsDown,
  Loader2,
  RefreshCw,
  Filter,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Clock,
  MessageSquare,
  ArrowRight,
  FileText,
  Link2,
  Box,
  Info,
  ExternalLink,
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { getOpenIssues, voteOnIssue } from '../../services/merltService';
import { formatUrnToReadable } from '../../utils/normattivaParser';
import { abbreviateCase } from '../../utils/graphLabels';
import type {
  EntityIssue,
  IssueStatus,
  IssueSeverity,
  IssueType,
} from '../../types/merlt';
import { ISSUE_TYPE_LABELS, ISSUE_SEVERITY_LABELS } from '../../types/merlt';

// =============================================================================
// HELPER: Format entity label to readable form
// =============================================================================

/**
 * Formatta un ID/URN in formato leggibile usando le utils esistenti.
 * Gestisce URN normattiva, riferimenti giurisprudenziali, e ID generici.
 */
function formatEntityLabel(label: string | undefined | null): string {
  if (!label) return 'Sconosciuto';

  // Se è un URN normattiva o URL
  if (label.includes('normattiva.it') || label.includes('urn:nir:')) {
    return formatUrnToReadable(label);
  }

  // Se contiene pattern di sentenza (numero/anno)
  if (/\d+\/\d{4}/.test(label) || label.toLowerCase().includes('cassazione')) {
    return abbreviateCase(label);
  }

  // Se è un ID con underscore, convertilo in formato leggibile
  if (label.includes('_') && !label.includes('/')) {
    // massima_cassazione_civile_7288_2023 -> Massima Cassazione Civile 7288 2023
    const parts = label.split('_');
    // Cerca pattern numero/anno
    const nums = parts.filter(p => /^\d+$/.test(p));
    if (nums.length >= 2) {
      // Probabilmente è un riferimento giurisprudenziale
      const tipo = parts.filter(p => !/^\d+$/.test(p)).map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' ');
      return `${tipo} ${nums[0]}/${nums[1]}`;
    }
    // Altrimenti capitalizza
    return parts.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' ');
  }

  // Se è troppo lungo, tronca
  if (label.length > 50) {
    return label.substring(0, 47) + '...';
  }

  return label;
}

// =============================================================================
// TYPES
// =============================================================================

interface IssueListProps {
  userId: string;
  className?: string;
}

// =============================================================================
// STATUS BADGE COMPONENT
// =============================================================================

function StatusBadge({ status }: { status: IssueStatus }) {
  const config: Record<IssueStatus, { icon: React.ReactNode; label: string; className: string }> = {
    open: {
      icon: <Clock size={12} />,
      label: 'Aperta',
      className: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300',
    },
    threshold_reached: {
      icon: <CheckCircle2 size={12} />,
      label: 'Soglia raggiunta',
      className: 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300',
    },
    dismissed: {
      icon: <XCircle size={12} />,
      label: 'Respinta',
      className: 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400',
    },
    resolved: {
      icon: <CheckCircle2 size={12} />,
      label: 'Risolta',
      className: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300',
    },
  };

  const { icon, label, className } = config[status];

  return (
    <span className={cn('inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium', className)}>
      {icon}
      {label}
    </span>
  );
}

// =============================================================================
// SEVERITY BADGE COMPONENT
// =============================================================================

function SeverityBadge({ severity }: { severity: IssueSeverity }) {
  const config: Record<IssueSeverity, { className: string }> = {
    low: { className: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' },
    medium: { className: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300' },
    high: { className: 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300' },
  };

  return (
    <span className={cn('px-2 py-0.5 rounded-full text-xs font-medium', config[severity].className)}>
      {ISSUE_SEVERITY_LABELS[severity].label}
    </span>
  );
}

// =============================================================================
// ISSUE CARD COMPONENT
// =============================================================================

interface IssueCardProps {
  issue: EntityIssue;
  userId: string;
  onVote: (issueId: string, vote: 'upvote' | 'downvote') => Promise<void>;
  isVoting: boolean;
}

function IssueCard({ issue, userId, onVote, isVoting }: IssueCardProps) {
  const [showCommentInput, setShowCommentInput] = useState(false);
  const [comment, setComment] = useState('');
  const [showFullDetails, setShowFullDetails] = useState(false);

  const typeInfo = ISSUE_TYPE_LABELS[issue.issue_type];
  const isError = typeInfo.category === 'error';
  const details = issue.entity_details;

  const handleVote = async (vote: 'upvote' | 'downvote') => {
    await onVote(issue.issue_id, vote);
    setShowCommentInput(false);
    setComment('');
  };

  // Calculate progress towards threshold (2.0)
  const threshold = 2.0;
  const progress = Math.min((issue.upvote_score / threshold) * 100, 100);

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      {/* Header */}
      <div className={cn(
        'px-4 py-3 border-b',
        isError
          ? 'bg-red-50 dark:bg-red-900/10 border-red-100 dark:border-red-900/30'
          : 'bg-blue-50 dark:bg-blue-900/10 border-blue-100 dark:border-blue-900/30'
      )}>
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className={cn(
              'w-8 h-8 rounded-lg flex items-center justify-center',
              isError ? 'bg-red-100 dark:bg-red-900/30' : 'bg-blue-100 dark:bg-blue-900/30'
            )}>
              <AlertTriangle
                size={16}
                className={isError ? 'text-red-600 dark:text-red-400' : 'text-blue-600 dark:text-blue-400'}
              />
            </div>
            <div>
              <p className="font-medium text-slate-900 dark:text-white text-sm">
                {typeInfo.label}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {details?.is_relation ? 'Relazione' : (details?.node_type || issue.entity_type || 'Entita')}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <SeverityBadge severity={issue.severity} />
            <StatusBadge status={issue.status} />
          </div>
        </div>
      </div>

      {/* Entity Details Section */}
      <div className="px-4 py-3 bg-slate-50 dark:bg-slate-900/50 border-b border-slate-100 dark:border-slate-700/50">
        {details?.is_relation ? (
          // Visualizzazione per RELAZIONI
          <div className="space-y-2">
            <div className="flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400">
              <Link2 size={12} />
              <span>Relazione segnalata</span>
            </div>
            <div className="flex items-center gap-2 flex-wrap overflow-hidden">
              {/* Source Node */}
              <div className="flex items-center gap-1.5 px-2.5 py-1.5 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-600 min-w-0">
                <Box size={14} className="text-purple-500" />
                <div>
                  <p className="text-xs text-slate-500 dark:text-slate-400">{details.source_type || 'Nodo'}</p>
                  <p className="text-sm font-medium text-slate-900 dark:text-white">
                    {formatEntityLabel(details.source_label)}
                  </p>
                </div>
              </div>

              {/* Arrow with Relation Type */}
              <div className="flex items-center gap-1 px-2 py-1 bg-amber-100 dark:bg-amber-900/30 rounded text-amber-700 dark:text-amber-300">
                <ArrowRight size={14} />
                <span className="text-xs font-medium">{details.relation_type || 'Relazione'}</span>
              </div>

              {/* Target Node */}
              <div className="flex items-center gap-1.5 px-2.5 py-1.5 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-600">
                <Box size={14} className="text-blue-500" />
                <div>
                  <p className="text-xs text-slate-500 dark:text-slate-400">{details.target_type || 'Nodo'}</p>
                  <p className="text-sm font-medium text-slate-900 dark:text-white">
                    {formatEntityLabel(details.target_label)}
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : details ? (
          // Visualizzazione per NODI
          <div className="space-y-2">
            <div className="flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400">
              <FileText size={12} />
              <span>Nodo segnalato</span>
            </div>
            <div className="flex items-start gap-3">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className="px-2 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-xs font-medium rounded">
                    {details.node_type}
                  </span>
                  {details.ambito && (
                    <span className="px-2 py-0.5 bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 text-xs rounded">
                      {details.ambito}
                    </span>
                  )}
                </div>
                <p className="text-base font-semibold text-slate-900 dark:text-white">
                  {formatEntityLabel(details.label)}
                </p>
                {details.urn && (
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                    {formatUrnToReadable(details.urn)}
                  </p>
                )}
                {/* Key properties */}
                {details.properties && Object.keys(details.properties).length > 0 && (
                  <div className="mt-2 space-y-1">
                    {Object.entries(details.properties).slice(0, 2).map(([key, value]) => (
                      <p key={key} className="text-xs text-slate-600 dark:text-slate-400">
                        <span className="font-medium">{key}:</span> {String(value).slice(0, 100)}{String(value).length > 100 ? '...' : ''}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          // Fallback se non ci sono dettagli
          <div className="text-sm text-slate-500 dark:text-slate-400">
            <span className="font-mono text-xs">{issue.entity_id}</span>
          </div>
        )}

        {/* Toggle per dettagli completi */}
        {details && (
          <button
            onClick={() => setShowFullDetails(!showFullDetails)}
            className="mt-3 flex items-center gap-1 text-xs text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 transition-colors"
          >
            {showFullDetails ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <Info size={12} />
            <span>{showFullDetails ? 'Nascondi dettagli completi' : 'Mostra dettagli completi'}</span>
          </button>
        )}

        {/* Sezione dettagli completi espandibile */}
        {showFullDetails && details && (
          <div className="mt-3 p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-600 space-y-3">
            <h4 className="text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wide flex items-center gap-1">
              <Info size={12} />
              Metadati Completi
            </h4>

            {details.is_relation ? (
              // Dettagli completi per RELAZIONI
              <div className="space-y-4">
                {/* Source Node Details */}
                <div className="p-2 bg-purple-50 dark:bg-purple-900/20 rounded border border-purple-200 dark:border-purple-800/30">
                  <p className="text-xs font-semibold text-purple-700 dark:text-purple-300 mb-2">
                    Nodo Sorgente
                  </p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-slate-500 dark:text-slate-400">Tipo:</span>
                      <span className="ml-1 text-slate-900 dark:text-white font-medium">{details.source_type || 'N/D'}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 dark:text-slate-400">Label:</span>
                      <span className="ml-1 text-slate-900 dark:text-white font-medium">{formatEntityLabel(details.source_label)}</span>
                    </div>
                  </div>
                  {details.source_label && (details.source_label.includes('normattiva') || details.source_label.includes('urn:nir')) && (
                    <div className="mt-2">
                      <span className="text-xs text-slate-500 dark:text-slate-400">URN completo:</span>
                      <p className="text-xs font-mono text-slate-600 dark:text-slate-400 break-all mt-0.5">
                        {details.source_label}
                      </p>
                      <a
                        href={details.source_label.startsWith('http') ? details.source_label : `https://www.normattiva.it/uri-res/N2Ls?${details.source_label}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 mt-1 text-xs text-primary-600 dark:text-primary-400 hover:underline"
                      >
                        <ExternalLink size={10} />
                        Apri su Normattiva
                      </a>
                    </div>
                  )}
                </div>

                {/* Relation Type */}
                <div className="p-2 bg-amber-50 dark:bg-amber-900/20 rounded border border-amber-200 dark:border-amber-800/30">
                  <p className="text-xs font-semibold text-amber-700 dark:text-amber-300 mb-1">
                    Tipo Relazione
                  </p>
                  <p className="text-sm font-medium text-amber-900 dark:text-amber-100">{details.relation_type || 'N/D'}</p>
                </div>

                {/* Target Node Details */}
                <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800/30">
                  <p className="text-xs font-semibold text-blue-700 dark:text-blue-300 mb-2">
                    Nodo Target
                  </p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-slate-500 dark:text-slate-400">Tipo:</span>
                      <span className="ml-1 text-slate-900 dark:text-white font-medium">{details.target_type || 'N/D'}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 dark:text-slate-400">Label:</span>
                      <span className="ml-1 text-slate-900 dark:text-white font-medium">{formatEntityLabel(details.target_label)}</span>
                    </div>
                  </div>
                  {details.target_label && (details.target_label.includes('normattiva') || details.target_label.includes('urn:nir')) && (
                    <div className="mt-2">
                      <span className="text-xs text-slate-500 dark:text-slate-400">URN completo:</span>
                      <p className="text-xs font-mono text-slate-600 dark:text-slate-400 break-all mt-0.5">
                        {details.target_label}
                      </p>
                      <a
                        href={details.target_label.startsWith('http') ? details.target_label : `https://www.normattiva.it/uri-res/N2Ls?${details.target_label}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 mt-1 text-xs text-primary-600 dark:text-primary-400 hover:underline"
                      >
                        <ExternalLink size={10} />
                        Apri su Normattiva
                      </a>
                    </div>
                  )}
                </div>

                {/* Entity ID raw */}
                <div className="pt-2 border-t border-slate-200 dark:border-slate-700">
                  <span className="text-xs text-slate-500 dark:text-slate-400">Entity ID (raw):</span>
                  <p className="text-xs font-mono text-slate-600 dark:text-slate-400 break-all mt-0.5">
                    {issue.entity_id}
                  </p>
                </div>
              </div>
            ) : (
              // Dettagli completi per NODI
              <div className="space-y-3">
                {/* Basic info grid */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <span className="text-slate-500 dark:text-slate-400">Tipo nodo:</span>
                    <p className="text-slate-900 dark:text-white font-medium">{details.node_type || 'N/D'}</p>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400">Ambito:</span>
                    <p className="text-slate-900 dark:text-white font-medium">{details.ambito || 'N/D'}</p>
                  </div>
                  <div className="col-span-2">
                    <span className="text-slate-500 dark:text-slate-400">Label:</span>
                    <p className="text-slate-900 dark:text-white font-medium">{details.label || 'N/D'}</p>
                  </div>
                </div>

                {/* URN con link */}
                {details.urn && (
                  <div className="p-2 bg-slate-100 dark:bg-slate-700/50 rounded">
                    <span className="text-xs text-slate-500 dark:text-slate-400">URN:</span>
                    <p className="text-xs font-mono text-slate-700 dark:text-slate-300 break-all mt-0.5">
                      {details.urn}
                    </p>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                      Formattato: <span className="font-medium">{formatUrnToReadable(details.urn)}</span>
                    </p>
                    {(details.urn.includes('normattiva') || details.urn.includes('urn:nir')) && (
                      <a
                        href={details.urn.startsWith('http') ? details.urn : `https://www.normattiva.it/uri-res/N2Ls?${details.urn}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 mt-1 text-xs text-primary-600 dark:text-primary-400 hover:underline"
                      >
                        <ExternalLink size={10} />
                        Apri su Normattiva
                      </a>
                    )}
                  </div>
                )}

                {/* Tutte le properties */}
                {details.properties && Object.keys(details.properties).length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-2">
                      Proprietà ({Object.keys(details.properties).length})
                    </p>
                    <div className="space-y-2">
                      {Object.entries(details.properties).map(([key, value]) => (
                        <div key={key} className="p-2 bg-slate-100 dark:bg-slate-700/50 rounded">
                          <span className="text-xs font-medium text-slate-600 dark:text-slate-400">{key}:</span>
                          <p className="text-xs text-slate-800 dark:text-slate-200 mt-0.5 whitespace-pre-wrap">
                            {String(value)}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Entity ID raw */}
                <div className="pt-2 border-t border-slate-200 dark:border-slate-700">
                  <span className="text-xs text-slate-500 dark:text-slate-400">Entity ID (raw):</span>
                  <p className="text-xs font-mono text-slate-600 dark:text-slate-400 break-all mt-0.5">
                    {issue.entity_id}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Problem Description */}
      <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-700/50">
        <p className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">
          Problema segnalato
        </p>
        <p className="text-sm text-slate-700 dark:text-slate-300">
          {issue.description}
        </p>
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-2">
          Segnalato da utente con authority {issue.reporter_authority.toFixed(2)} •{' '}
          {new Date(issue.created_at).toLocaleDateString('it-IT', {
            day: 'numeric',
            month: 'short',
            year: 'numeric',
          })}
        </p>
      </div>

      {/* Content */}
      <div className="p-4">

        {/* Progress bar towards threshold */}
        {issue.status === 'open' && (
          <div className="mb-4">
            <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
              <span>Progresso verso soglia</span>
              <span>{issue.upvote_score.toFixed(2)} / {threshold.toFixed(1)}</span>
            </div>
            <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-amber-500 to-amber-600 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Stats */}
        <div className="flex items-center gap-4 text-xs text-slate-500 dark:text-slate-400 mb-4">
          <span className="flex items-center gap-1">
            <ThumbsUp size={12} className="text-green-500" />
            {issue.upvote_score.toFixed(2)}
          </span>
          <span className="flex items-center gap-1">
            <ThumbsDown size={12} className="text-red-500" />
            {issue.downvote_score.toFixed(2)}
          </span>
          <span>{issue.votes_count} voti totali</span>
        </div>

        {/* Comment input */}
        {showCommentInput && (
          <div className="mb-4">
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Aggiungi un commento (opzionale)..."
              rows={2}
              className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-sm resize-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none"
            />
          </div>
        )}

        {/* Vote buttons */}
        {issue.status === 'open' && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleVote('upvote')}
              disabled={isVoting}
              className={cn(
                'flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg font-medium text-sm transition-all',
                'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300',
                'hover:bg-green-200 dark:hover:bg-green-900/50',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
              )}
            >
              {isVoting ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <ThumbsUp size={16} />
              )}
              Confermo
            </button>
            <button
              onClick={() => handleVote('downvote')}
              disabled={isVoting}
              className={cn(
                'flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg font-medium text-sm transition-all',
                'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300',
                'hover:bg-red-200 dark:hover:bg-red-900/50',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
              )}
            >
              {isVoting ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <ThumbsDown size={16} />
              )}
              Non valida
            </button>
            <button
              onClick={() => setShowCommentInput(!showCommentInput)}
              className="p-2.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-500 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
              title="Aggiungi commento"
              aria-label="Aggiungi commento"
              aria-expanded={showCommentInput}
            >
              <MessageSquare size={16} aria-hidden="true" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function IssueList({ userId, className }: IssueListProps) {
  // State
  const [issues, setIssues] = useState([] as EntityIssue[]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null as string | null);
  const [votingIssueId, setVotingIssueId] = useState(null as string | null);

  // Filters
  const [statusFilter, setStatusFilter] = useState('open' as IssueStatus | 'all');
  const [severityFilter, setSeverityFilter] = useState('all' as IssueSeverity | 'all');
  const [showFilterDropdown, setShowFilterDropdown] = useState(false);

  // Pagination
  const [hasMore, setHasMore] = useState(false);
  const [total, setTotal] = useState(0);

  // Fetch issues
  const fetchIssues = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await getOpenIssues({
        status: statusFilter === 'all' ? undefined : statusFilter,
        severity: severityFilter === 'all' ? undefined : severityFilter,
        limit: 20,
      });

      setIssues(response.issues);
      setHasMore(response.has_more);
      setTotal(response.total);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Errore nel caricamento';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [statusFilter, severityFilter]);

  // Initial fetch
  useEffect(() => {
    fetchIssues();
  }, [fetchIssues]);

  // Handle vote
  const handleVote = async (issueId: string, vote: 'upvote' | 'downvote') => {
    setVotingIssueId(issueId);

    try {
      const response = await voteOnIssue({
        issue_id: issueId,
        vote,
        user_id: userId,
      });

      // Update issue in list
      setIssues((prev: EntityIssue[]) => prev.map((issue: EntityIssue) => {
        if (issue.issue_id === issueId) {
          return {
            ...issue,
            status: response.new_status,
            upvote_score: response.upvote_score,
            downvote_score: response.downvote_score,
            votes_count: response.votes_count,
          };
        }
        return issue;
      }));

      // Show success message if entity was reopened
      if (response.entity_reopened) {
        // Could show a toast here
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Errore nel voto';
      setError(message);
    } finally {
      setVotingIssueId(null);
    }
  };

  return (
    <div className={cn('space-y-4', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white flex items-center gap-2">
            <AlertTriangle className="text-amber-600" size={20} />
            Segnalazioni
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {total} segnalazioni totali
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Filter dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowFilterDropdown(!showFilterDropdown)}
              className="flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
              aria-label="Apri filtri"
              aria-expanded={showFilterDropdown}
            >
              <Filter size={16} aria-hidden="true" />
              Filtri
              <ChevronDown size={14} aria-hidden="true" />
            </button>

            {showFilterDropdown && (
              <>
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setShowFilterDropdown(false)}
                />
                <div className="absolute right-0 top-full mt-1 w-48 bg-white dark:bg-slate-800 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 py-2 z-20">
                  {/* Status filter */}
                  <div className="px-3 py-1">
                    <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                      Stato
                    </p>
                    {(['all', 'open', 'threshold_reached', 'dismissed'] as const).map((status) => (
                      <button
                        key={status}
                        onClick={() => {
                          setStatusFilter(status);
                          setShowFilterDropdown(false);
                        }}
                        className={cn(
                          'w-full px-2 py-1.5 text-left text-sm rounded hover:bg-slate-100 dark:hover:bg-slate-700',
                          statusFilter === status
                            ? 'text-primary-600 dark:text-primary-400 font-medium'
                            : 'text-slate-700 dark:text-slate-300'
                        )}
                      >
                        {status === 'all' ? 'Tutti' :
                         status === 'open' ? 'Aperte' :
                         status === 'threshold_reached' ? 'Soglia raggiunta' : 'Respinte'}
                      </button>
                    ))}
                  </div>

                  <div className="border-t border-slate-200 dark:border-slate-700 my-1" />

                  {/* Severity filter */}
                  <div className="px-3 py-1">
                    <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                      Gravita
                    </p>
                    {(['all', 'high', 'medium', 'low'] as const).map((severity) => (
                      <button
                        key={severity}
                        onClick={() => {
                          setSeverityFilter(severity);
                          setShowFilterDropdown(false);
                        }}
                        className={cn(
                          'w-full px-2 py-1.5 text-left text-sm rounded hover:bg-slate-100 dark:hover:bg-slate-700',
                          severityFilter === severity
                            ? 'text-primary-600 dark:text-primary-400 font-medium'
                            : 'text-slate-700 dark:text-slate-300'
                        )}
                      >
                        {severity === 'all' ? 'Tutte' : ISSUE_SEVERITY_LABELS[severity].label}
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Refresh */}
          <button
            onClick={fetchIssues}
            disabled={loading}
            className="p-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            aria-label="Aggiorna segnalazioni"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-600 dark:text-red-400" role="alert">
          <AlertCircle size={16} className="flex-shrink-0 mt-0.5" aria-hidden="true" />
          {error}
        </div>
      )}

      {/* Loading */}
      {loading ? (
        <div className="space-y-4" role="status" aria-label="Caricamento segnalazioni">
          {[...Array(3)].map((_, i) => (
            <div
              key={i}
              className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden animate-pulse"
            >
              <div className="h-16 bg-slate-100 dark:bg-slate-700" />
              <div className="p-4 space-y-3">
                <div className="h-4 bg-slate-100 dark:bg-slate-700 rounded w-3/4" />
                <div className="h-4 bg-slate-100 dark:bg-slate-700 rounded w-1/2" />
                <div className="h-10 bg-slate-100 dark:bg-slate-700 rounded" />
              </div>
            </div>
          ))}
        </div>
      ) : issues.length === 0 ? (
        <div className="text-center py-12" role="status">
          <AlertTriangle size={48} className="mx-auto text-slate-300 dark:text-slate-600 mb-4" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
            Nessuna segnalazione
          </h3>
          <p className="text-slate-600 dark:text-slate-400">
            {statusFilter !== 'all' || severityFilter !== 'all'
              ? 'Prova a modificare i filtri'
              : 'Non ci sono segnalazioni aperte al momento'}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {issues.map((issue: EntityIssue) => (
            <IssueCard
              key={issue.issue_id}
              issue={issue}
              userId={userId}
              onVote={handleVote}
              isVoting={votingIssueId === issue.issue_id}
            />
          ))}

          {hasMore && (
            <div className="text-center">
              <button
                onClick={() => {/* TODO: Load more */}}
                className="px-4 py-2 text-sm text-primary-600 dark:text-primary-400 hover:underline"
              >
                Carica altre segnalazioni
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
