/**
 * IssueSidebar
 * ============
 *
 * Pannello laterale collassabile per visualizzare e votare sulle segnalazioni
 * del Knowledge Graph. Integrato direttamente nella pagina del grafo.
 *
 * Le card sono progettate per dare all'utente tutto il contesto necessario
 * per poter valutare se la segnalazione e' valida.
 */

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AlertTriangle,
  ThumbsUp,
  ThumbsDown,
  Loader2,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Clock,
  User,
  Target,
  FileText,
  CheckCircle2,
  XCircle,
  MessageSquare,
  ExternalLink,
  Sparkles,
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { getOpenIssues, voteOnIssue } from '../../services/merltService';
import type { EntityIssue, IssueStatus } from '../../types/merlt';
import { ISSUE_TYPE_LABELS, ISSUE_SEVERITY_LABELS } from '../../types/merlt';

// =============================================================================
// TYPES
// =============================================================================

interface IssueSidebarProps {
  userId: string;
  isOpen: boolean;
  onToggle: () => void;
  onNavigateToEntity?: (entityId: string) => void;
  className?: string;
}

// =============================================================================
// DETAILED ISSUE CARD
// =============================================================================

interface DetailedIssueCardProps {
  issue: EntityIssue;
  userId: string;
  onVote: (issueId: string, vote: 'upvote' | 'downvote', comment?: string) => Promise<void>;
  onNavigate?: (entityId: string) => void;
  isVoting: boolean;
}

function DetailedIssueCard({
  issue,
  userId,
  onVote,
  onNavigate,
  isVoting,
}: DetailedIssueCardProps) {
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState('');

  const typeInfo = ISSUE_TYPE_LABELS[issue.issue_type];
  const severityInfo = ISSUE_SEVERITY_LABELS[issue.severity];
  const isError = typeInfo.category === 'error';

  // Progress towards threshold (2.0)
  const threshold = 2.0;
  const progress = Math.min((issue.upvote_score / threshold) * 100, 100);
  const netScore = issue.upvote_score - issue.downvote_score;

  // Parse entity_id per estrarre info utili
  const parseEntityId = (id: string) => {
    // Format tipici: "massima_cassazione_civile_1234_2020", "concetto:buona_fede", "dottrina_brocardi_art1337"
    const parts = id.split(/[_:]/);
    const type = parts[0];
    const rest = parts.slice(1).join(' ').replace(/_/g, ' ');
    return { type, readable: rest || id };
  };

  const entityInfo = parseEntityId(issue.entity_id);

  // Format date
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) return `${diffMins}m fa`;
    if (diffHours < 24) return `${diffHours}h fa`;
    if (diffDays < 7) return `${diffDays}g fa`;
    return date.toLocaleDateString('it-IT', { day: 'numeric', month: 'short' });
  };

  const handleVote = async (vote: 'upvote' | 'downvote') => {
    await onVote(issue.issue_id, vote, comment || undefined);
    setComment('');
    setShowComment(false);
  };

  // Status config
  const statusConfig: Record<IssueStatus, { icon: React.ReactNode; label: string; className: string }> = {
    open: {
      icon: <Clock size={12} />,
      label: 'In votazione',
      className: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    },
    threshold_reached: {
      icon: <CheckCircle2 size={12} />,
      label: 'Confermata',
      className: 'bg-green-500/20 text-green-400 border-green-500/30',
    },
    dismissed: {
      icon: <XCircle size={12} />,
      label: 'Respinta',
      className: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
    },
    resolved: {
      icon: <CheckCircle2 size={12} />,
      label: 'Risolta',
      className: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    },
  };

  const status = statusConfig[issue.status];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-800/60 backdrop-blur rounded-xl border border-slate-700/50 overflow-hidden"
    >
      {/* Header con tipo problema e severity */}
      <div className={cn(
        'px-3 py-2 border-b flex items-center justify-between',
        isError
          ? 'bg-red-500/10 border-red-500/20'
          : 'bg-blue-500/10 border-blue-500/20'
      )}>
        <div className="flex items-center gap-2">
          <AlertTriangle
            size={14}
            className={isError ? 'text-red-400' : 'text-blue-400'}
          />
          <span className={cn(
            'text-xs font-medium',
            isError ? 'text-red-300' : 'text-blue-300'
          )}>
            {typeInfo.label}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn(
            'px-1.5 py-0.5 rounded text-[10px] font-medium',
            issue.severity === 'high' ? 'bg-red-500/20 text-red-300' :
            issue.severity === 'medium' ? 'bg-amber-500/20 text-amber-300' :
            'bg-blue-500/20 text-blue-300'
          )}>
            {severityInfo.label}
          </span>
          <span className={cn(
            'px-1.5 py-0.5 rounded text-[10px] font-medium flex items-center gap-1 border',
            status.className
          )}>
            {status.icon}
            {status.label}
          </span>
        </div>
      </div>

      {/* Contenuto principale */}
      <div className="p-3 space-y-3">
        {/* Entita' interessata */}
        <div className="flex items-start gap-2">
          <div className="w-8 h-8 rounded-lg bg-slate-700/50 flex items-center justify-center flex-shrink-0">
            <Target size={14} className="text-slate-400" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-wide text-slate-500">
                {issue.entity_type || entityInfo.type}
              </span>
              {onNavigate && (
                <button
                  onClick={() => onNavigate(issue.entity_id)}
                  className="text-primary-400 hover:text-primary-300 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
                  title="Vai all'entita' nel grafo"
                  aria-label="Vai all'entita' nel grafo"
                >
                  <ExternalLink size={10} aria-hidden="true" />
                </button>
              )}
            </div>
            <p className="text-sm text-white font-medium truncate" title={issue.entity_id}>
              {entityInfo.readable}
            </p>
          </div>
        </div>

        {/* Descrizione del problema */}
        <div className="bg-slate-900/50 rounded-lg p-2.5 border border-slate-700/50">
          <div className="flex items-center gap-1.5 mb-1.5">
            <FileText size={12} className="text-slate-500" />
            <span className="text-[10px] uppercase tracking-wide text-slate-500">
              Problema segnalato
            </span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            {issue.description}
          </p>
        </div>

        {/* Chi ha segnalato e quando */}
        <div className="flex items-center justify-between text-[10px] text-slate-500">
          <div className="flex items-center gap-1.5">
            <User size={10} />
            <span>
              Segnalato da utente con authority {issue.reporter_authority.toFixed(2)}
            </span>
          </div>
          <span>{formatDate(issue.created_at)}</span>
        </div>

        {/* Barra progresso SOLO se issue aperta */}
        {issue.status === 'open' && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-slate-500">Progresso verso conferma</span>
              <span className="text-slate-400 font-mono">
                {issue.upvote_score.toFixed(2)} / {threshold}
              </span>
            </div>
            <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                className={cn(
                  'h-full rounded-full',
                  progress >= 80 ? 'bg-green-500' :
                  progress >= 50 ? 'bg-amber-500' :
                  'bg-slate-500'
                )}
              />
            </div>

            {/* Stats voti */}
            <div className="flex items-center justify-center gap-4 text-xs">
              <span className="flex items-center gap-1 text-green-400">
                <ThumbsUp size={12} />
                {issue.upvote_score.toFixed(2)}
              </span>
              <span className="flex items-center gap-1 text-red-400">
                <ThumbsDown size={12} />
                {issue.downvote_score.toFixed(2)}
              </span>
              <span className="text-slate-500">
                {issue.votes_count} voti
              </span>
            </div>
          </div>
        )}

        {/* Area commento espandibile */}
        <AnimatePresence>
          {showComment && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
            >
              <textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                placeholder="Spiega il tuo voto (opzionale)..."
                rows={2}
                className="w-full px-2.5 py-2 rounded-lg border border-slate-700 bg-slate-900/50 text-xs text-white placeholder-slate-500 resize-none focus:ring-1 focus:ring-primary-500/50 focus:border-primary-500 outline-none"
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Pulsanti voto - solo se issue aperta */}
        {issue.status === 'open' && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleVote('upvote')}
              disabled={isVoting}
              className={cn(
                'flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-all',
                'bg-green-500/20 text-green-400 border border-green-500/30',
                'hover:bg-green-500/30 hover:border-green-500/50',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
              )}
            >
              {isVoting ? (
                <Loader2 size={12} className="animate-spin" />
              ) : (
                <ThumbsUp size={12} />
              )}
              Confermo
            </button>
            <button
              onClick={() => handleVote('downvote')}
              disabled={isVoting}
              className={cn(
                'flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-all',
                'bg-red-500/20 text-red-400 border border-red-500/30',
                'hover:bg-red-500/30 hover:border-red-500/50',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500'
              )}
            >
              {isVoting ? (
                <Loader2 size={12} className="animate-spin" />
              ) : (
                <ThumbsDown size={12} />
              )}
              Non valida
            </button>
            <button
              onClick={() => setShowComment(!showComment)}
              className={cn(
                'p-2 rounded-lg border transition-colors',
                showComment
                  ? 'bg-primary-500/20 border-primary-500/30 text-primary-400'
                  : 'bg-slate-700/50 border-slate-600 text-slate-400 hover:text-white'
              )}
              title="Aggiungi commento"
              aria-label="Aggiungi commento"
            >
              <MessageSquare size={12} aria-hidden="true" />
            </button>
          </div>
        )}
      </div>
    </motion.div>
  );
}

// =============================================================================
// MAIN SIDEBAR COMPONENT
// =============================================================================

export function IssueSidebar({
  userId,
  isOpen,
  onToggle,
  onNavigateToEntity,
  className,
}: IssueSidebarProps) {
  const [issues, setIssues] = useState([] as EntityIssue[]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null as string | null);
  const [votingIssueId, setVotingIssueId] = useState(null as string | null);

  // Fetch issues
  const fetchIssues = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await getOpenIssues({
        status: 'open',
        limit: 10,
      });
      setIssues(response.issues);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Errore nel caricamento';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      fetchIssues();
    }
  }, [isOpen, fetchIssues]);

  // Handle vote
  const handleVote = async (issueId: string, vote: 'upvote' | 'downvote', comment?: string) => {
    setVotingIssueId(issueId);

    try {
      const response = await voteOnIssue({
        issue_id: issueId,
        vote,
        comment,
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

    } catch (_err) {
      // Vote error handled silently - could show toast
    } finally {
      setVotingIssueId(null);
    }
  };

  const openIssuesCount = issues.filter((i: EntityIssue) => i.status === 'open').length;

  return (
    <>
      {/* Toggle button - sempre visibile */}
      <button
        onClick={onToggle}
        className={cn(
          'absolute top-4 right-4 z-30 flex items-center gap-2 px-3 py-2 rounded-lg transition-all',
          'bg-slate-800/90 backdrop-blur border border-slate-700 hover:border-slate-600',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
          isOpen && 'bg-amber-500/20 border-amber-500/30'
        )}
        aria-label={isOpen ? 'Chiudi segnalazioni' : 'Apri segnalazioni'}
        aria-expanded={isOpen}
      >
        <AlertTriangle size={16} className={isOpen ? 'text-amber-400' : 'text-slate-400'} aria-hidden="true" />
        <span className="text-xs font-medium text-white">Segnalazioni</span>
        {openIssuesCount > 0 && (
          <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-amber-500 text-white">
            {openIssuesCount}
          </span>
        )}
        {isOpen ? (
          <ChevronRight size={14} className="text-slate-400" />
        ) : (
          <ChevronLeft size={14} className="text-slate-400" />
        )}
      </button>

      {/* Sidebar panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ x: '100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '100%', opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className={cn(
              'absolute top-0 right-0 bottom-0 w-80 z-20',
              'bg-slate-900/95 backdrop-blur-xl border-l border-slate-700',
              'flex flex-col',
              className
            )}
          >
            {/* Header */}
            <div className="p-4 border-b border-slate-700/50">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center">
                    <AlertTriangle size={16} className="text-amber-400" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white">
                      Segnalazioni Aperte
                    </h3>
                    <p className="text-[10px] text-slate-500">
                      Vota per validare o respingere
                    </p>
                  </div>
                </div>
                <button
                  onClick={fetchIssues}
                  disabled={loading}
                  className="p-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-400 hover:text-white transition-colors disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  aria-label="Aggiorna segnalazioni"
                >
                  <RefreshCw size={14} className={loading ? 'animate-spin' : ''} aria-hidden="true" />
                </button>
              </div>

              {/* Info box */}
              <div className="flex items-start gap-2 p-2 bg-slate-800/50 rounded-lg border border-slate-700/50">
                <Sparkles size={12} className="text-amber-400 flex-shrink-0 mt-0.5" />
                <p className="text-[10px] text-slate-400 leading-relaxed">
                  Quando una segnalazione raggiunge la soglia di consenso, l'entita' torna in validazione per essere corretta.
                </p>
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-3 space-y-3">
              {loading ? (
                <div className="flex flex-col items-center justify-center py-8" role="status" aria-label="Caricamento segnalazioni">
                  <Loader2 size={24} className="animate-spin text-primary-500 mb-2" aria-hidden="true" />
                  <p className="text-xs text-slate-500">Caricamento...</p>
                </div>
              ) : error ? (
                <div className="text-center py-8" role="alert">
                  <AlertTriangle size={24} className="mx-auto text-red-400 mb-2" aria-hidden="true" />
                  <p className="text-xs text-red-400">{error}</p>
                </div>
              ) : issues.length === 0 ? (
                <div className="text-center py-8" role="status">
                  <CheckCircle2 size={32} className="mx-auto text-green-400 mb-3" />
                  <p className="text-sm font-medium text-white mb-1">
                    Nessuna segnalazione
                  </p>
                  <p className="text-xs text-slate-500">
                    Il Knowledge Graph e' in ottimo stato!
                  </p>
                </div>
              ) : (
                issues.map((issue: EntityIssue) => (
                  <DetailedIssueCard
                    key={issue.issue_id}
                    issue={issue}
                    userId={userId}
                    onVote={handleVote}
                    onNavigate={onNavigateToEntity}
                    isVoting={votingIssueId === issue.issue_id}
                  />
                ))
              )}
            </div>

            {/* Footer */}
            <div className="p-3 border-t border-slate-700/50">
              <p className="text-[10px] text-slate-500 text-center">
                I voti sono pesati per la tua authority ({userId === 'anonymous' ? '0.50' : 'calcolata'})
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

export default IssueSidebar;
