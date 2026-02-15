/**
 * RLCFExplainer
 * =============
 *
 * Sezione educativa che spiega il sistema RLCF (Reinforcement Learning
 * from Crowd Feedback) in modo accessibile.
 *
 * Obiettivi:
 * - Spiegare il "perché" dietro il sistema
 * - Mostrare come aumentare la propria authority
 * - Trasparenza sulla formula e meccanismi
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Users,
  Brain,
  Target,
  Sparkles,
  CheckCircle2,
  ArrowRight,
  Scale,
  TrendingUp,
  GraduationCap,
  Zap,
} from 'lucide-react';
import { cn } from '../../../lib/utils';

// =============================================================================
// FAQ ITEM
// =============================================================================

interface FAQItemProps {
  question: string;
  answer: React.ReactNode;
  icon: typeof HelpCircle;
  isOpen: boolean;
  onToggle: () => void;
}

function FAQItem({ question, answer, icon: Icon, isOpen, onToggle }: FAQItemProps) {
  return (
    <div className="border-b border-slate-100 dark:border-slate-800 last:border-0">
      <button
        onClick={onToggle}
        aria-expanded={isOpen}
        className="w-full py-3 flex items-center gap-3 text-left hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors rounded-lg px-2 -mx-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
      >
        <Icon size={16} className="text-blue-500 flex-shrink-0" aria-hidden="true" />
        <span className="flex-1 text-sm font-medium text-slate-700 dark:text-slate-300">
          {question}
        </span>
        {isOpen ? (
          <ChevronUp size={16} className="text-slate-400" aria-hidden="true" />
        ) : (
          <ChevronDown size={16} className="text-slate-400" aria-hidden="true" />
        )}
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="pb-3 pl-7 pr-2 text-sm text-slate-600 dark:text-slate-400">
              {answer}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface RLCFExplainerProps {
  className?: string;
  defaultExpanded?: boolean;
}

export function RLCFExplainer({ className, defaultExpanded = false }: RLCFExplainerProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [openFAQ, setOpenFAQ] = useState(null as number | null);

  const faqs = [
    {
      question: "Cos'è il sistema RLCF?",
      icon: Brain,
      answer: (
        <div className="space-y-2">
          <p>
            <strong>RLCF</strong> (Reinforcement Learning from Crowd Feedback) è un sistema che combina
            l'intelligenza artificiale con la validazione della community giuridica.
          </p>
          <p>
            L'AI estrae automaticamente concetti e relazioni dai testi di legge, ma sono gli
            esperti come te a validare se queste estrazioni sono corrette.
          </p>
        </div>
      ),
    },
    {
      question: 'Come viene calcolata la mia authority?',
      icon: Scale,
      answer: (
        <div className="space-y-3">
          <p>La tua authority è calcolata con la formula:</p>
          <div className="p-2 bg-slate-100 dark:bg-slate-800 rounded font-mono text-xs">
            A_u = 0.3·B_u + 0.5·T_u + 0.2·P_u
          </div>
          <ul className="space-y-1.5 text-xs">
            <li className="flex items-start gap-2">
              <GraduationCap size={12} className="text-blue-500 mt-0.5 flex-shrink-0" />
              <span><strong>B_u (30%)</strong>: Qualifiche professionali verificate</span>
            </li>
            <li className="flex items-start gap-2">
              <TrendingUp size={12} className="text-emerald-500 mt-0.5 flex-shrink-0" />
              <span><strong>T_u (50%)</strong>: Il tuo storico di validazioni corrette</span>
            </li>
            <li className="flex items-start gap-2">
              <Zap size={12} className="text-amber-500 mt-0.5 flex-shrink-0" />
              <span><strong>P_u (20%)</strong>: Performance recente nei domini attivi</span>
            </li>
          </ul>
        </div>
      ),
    },
    {
      question: 'Come posso aumentare la mia authority?',
      icon: TrendingUp,
      answer: (
        <div className="space-y-2">
          <p>Ecco come far crescere la tua reputazione nel sistema:</p>
          <ul className="space-y-1.5">
            <li className="flex items-start gap-2">
              <CheckCircle2 size={12} className="text-emerald-500 mt-0.5 flex-shrink-0" />
              <span><strong>Valida accuratamente</strong>: Voti corretti aumentano il track record</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 size={12} className="text-emerald-500 mt-0.5 flex-shrink-0" />
              <span><strong>Proponi entità di qualità</strong>: Proposte approvate dalla community</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 size={12} className="text-emerald-500 mt-0.5 flex-shrink-0" />
              <span><strong>Specializzati</strong>: Contributi costanti nello stesso dominio</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 size={12} className="text-emerald-500 mt-0.5 flex-shrink-0" />
              <span><strong>Sii costante</strong>: I contributi recenti pesano di più (λ=0.95)</span>
            </li>
          </ul>
        </div>
      ),
    },
    {
      question: 'Quanto pesa il mio voto?',
      icon: Target,
      answer: (
        <div className="space-y-2">
          <p>
            Il peso del tuo voto è esattamente uguale alla tua authority.
            Se hai authority 0.65, il tuo voto vale 0.65 punti.
          </p>
          <p>
            Per approvare un'entità o relazione servono <strong>2.0 punti totali</strong>.
            Questo significa che servono circa 3 utenti con authority media, oppure
            2 esperti ad alta authority.
          </p>
        </div>
      ),
    },
    {
      question: 'Perché questo sistema?',
      icon: Sparkles,
      answer: (
        <div className="space-y-2">
          <p>
            Il diritto è complesso e richiede competenza per essere interpretato correttamente.
            Non tutti i voti dovrebbero pesare uguale: chi ha dimostrato competenza
            dovrebbe avere più influenza.
          </p>
          <p>
            RLCF crea un <strong>circolo virtuoso</strong>: più contribuisci correttamente,
            più la tua voce conta, più il Knowledge Graph diventa accurato.
          </p>
        </div>
      ),
    },
  ];

  return (
    <div
      className={cn(
        'rounded-xl border border-slate-200 dark:border-slate-700',
        'bg-white dark:bg-slate-900 overflow-hidden',
        className
      )}
    >
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        className="w-full p-4 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-inset"
      >
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-500 flex items-center justify-center flex-shrink-0">
          <HelpCircle size={20} className="text-white" aria-hidden="true" />
        </div>
        <div className="flex-1 text-left">
          <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
            Come funziona il sistema RLCF?
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Scopri come la tua authority influenza il Knowledge Graph
          </p>
        </div>
        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown size={20} className="text-slate-400" aria-hidden="true" />
        </motion.div>
      </button>

      {/* Expandable content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 border-t border-slate-100 dark:border-slate-800">
              {/* Quick intro */}
              <div className="py-4 flex flex-wrap items-center gap-4 sm:gap-6 justify-center border-b border-slate-100 dark:border-slate-800">
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <Users size={14} className="text-blue-500" aria-hidden="true" />
                  <span>Community-driven</span>
                </div>
                <ArrowRight size={12} className="text-slate-300 hidden sm:block" aria-hidden="true" />
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <Brain size={14} className="text-purple-500" aria-hidden="true" />
                  <span>AI-assisted</span>
                </div>
                <ArrowRight size={12} className="text-slate-300 hidden sm:block" aria-hidden="true" />
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <Target size={14} className="text-emerald-500" aria-hidden="true" />
                  <span>Quality-focused</span>
                </div>
              </div>

              {/* FAQ section */}
              <div className="pt-4">
                {faqs.map((faq, index) => (
                  <FAQItem
                    key={index}
                    question={faq.question}
                    answer={faq.answer}
                    icon={faq.icon}
                    isOpen={openFAQ === index}
                    onToggle={() => setOpenFAQ(openFAQ === index ? null : index)}
                  />
                ))}
              </div>

              {/* CTA */}
              <div className="mt-4 p-3 rounded-lg bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20">
                <p className="text-xs text-center text-slate-600 dark:text-slate-400">
                  Inizia a validare per costruire la tua reputazione e contribuire
                  al Knowledge Graph giuridico italiano.
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default RLCFExplainer;
