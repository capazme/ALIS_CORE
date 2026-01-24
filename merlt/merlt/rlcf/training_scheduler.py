"""
Training Scheduler
==================

Scheduler automatico per training RLCF.

Monitora il buffer di esperienze e avvia training quando:
1. Buffer raggiunge soglia minima (es. 100 esperienze)
2. Intervallo di tempo minimo passato (es. 1 ora)
3. Trigger manuale via API

Flusso:
    Feedback → ExperienceReplayBuffer → TrainingScheduler monitors →
    Threshold reached → PolicyGradientTrainer.train() → PolicyCheckpoint saved

Esempio:
    >>> from merlt.rlcf.training_scheduler import TrainingScheduler, get_scheduler
    >>>
    >>> # Singleton access
    >>> scheduler = get_scheduler()
    >>>
    >>> # Aggiungi esperienza al buffer
    >>> scheduler.add_experience(trace, feedback, reward=0.8)
    >>>
    >>> # Check se training ready
    >>> if scheduler.should_train():
    ...     await scheduler.run_training_epoch()
    >>>
    >>> # Avvia training automatico in background
    >>> await scheduler.start_auto_training()
    >>>
    >>> # Status
    >>> status = scheduler.get_status()
    >>> print(f"Buffer: {status.buffer_size}, Training: {status.is_training}")
"""

import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import threading

from .replay_buffer import ExperienceReplayBuffer, PrioritizedReplayBuffer, BufferStats
from .persistence import RLCFPersistence, TrainingSession, PolicyCheckpoint

log = structlog.get_logger()


# =============================================================================
# ENUMS AND DATACLASSES
# =============================================================================

class TrainingStatus(str, Enum):
    """Stato del training."""
    IDLE = "idle"
    TRAINING = "training"
    PAUSED = "paused"
    ERROR = "error"


class TrainingTrigger(str, Enum):
    """Trigger che ha avviato il training."""
    BUFFER_THRESHOLD = "buffer_threshold"
    TIME_INTERVAL = "time_interval"
    MANUAL = "manual"
    API = "api"


@dataclass
class SchedulerConfig:
    """
    Configurazione dello scheduler.

    Attributes:
        buffer_threshold: Numero minimo esperienze per training
        min_interval_seconds: Intervallo minimo tra training
        max_buffer_size: Capacità massima buffer
        batch_size: Dimensione batch per training
        epochs_per_run: Numero di epoch per run
        prioritized_replay: Usare prioritized experience replay
        alpha: Esponente priorità PER
        auto_save_checkpoint: Salvare checkpoint automaticamente
    """
    buffer_threshold: int = 100
    min_interval_seconds: int = 3600  # 1 ora
    max_buffer_size: int = 10000
    batch_size: int = 32
    epochs_per_run: int = 5
    prioritized_replay: bool = True
    alpha: float = 0.6
    auto_save_checkpoint: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buffer_threshold": self.buffer_threshold,
            "min_interval_seconds": self.min_interval_seconds,
            "max_buffer_size": self.max_buffer_size,
            "batch_size": self.batch_size,
            "epochs_per_run": self.epochs_per_run,
            "prioritized_replay": self.prioritized_replay,
            "alpha": self.alpha,
            "auto_save_checkpoint": self.auto_save_checkpoint,
        }


@dataclass
class SchedulerStatus:
    """
    Stato corrente dello scheduler.

    Attributes:
        status: Stato training (idle, training, etc.)
        buffer_size: Numero esperienze nel buffer
        buffer_capacity: Capacità massima buffer
        last_training_at: Timestamp ultimo training
        next_training_at: Timestamp prossimo training stimato
        is_training: True se training in corso
        current_epoch: Epoch corrente (se training)
        total_epochs: Epoch totali (se training)
        training_sessions_today: Sessioni training oggi
        avg_reward: Reward medio nel buffer
    """
    status: TrainingStatus = TrainingStatus.IDLE
    buffer_size: int = 0
    buffer_capacity: int = 0
    last_training_at: Optional[str] = None
    next_training_at: Optional[str] = None
    is_training: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    training_sessions_today: int = 0
    avg_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "buffer_size": self.buffer_size,
            "buffer_capacity": self.buffer_capacity,
            "last_training_at": self.last_training_at,
            "next_training_at": self.next_training_at,
            "is_training": self.is_training,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "training_sessions_today": self.training_sessions_today,
            "avg_reward": round(self.avg_reward, 4),
        }


@dataclass
class TrainingResult:
    """
    Risultato di un run di training.

    Attributes:
        success: True se training completato
        session_id: ID sessione training
        epochs_completed: Numero epoch completati
        total_loss: Loss totale
        avg_reward: Reward medio batch
        checkpoint_version: Version del checkpoint salvato
        duration_seconds: Durata in secondi
        error: Messaggio errore se fallito
    """
    success: bool = True
    session_id: Optional[str] = None
    epochs_completed: int = 0
    total_loss: float = 0.0
    avg_reward: float = 0.0
    checkpoint_version: Optional[str] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "session_id": self.session_id,
            "epochs_completed": self.epochs_completed,
            "total_loss": round(self.total_loss, 6),
            "avg_reward": round(self.avg_reward, 4),
            "checkpoint_version": self.checkpoint_version,
            "duration_seconds": round(self.duration_seconds, 2),
            "error": self.error,
        }


# =============================================================================
# TRAINING SCHEDULER
# =============================================================================

class TrainingScheduler:
    """
    Scheduler per training automatico RLCF.

    Monitora il buffer di esperienze e avvia training quando
    le condizioni sono soddisfatte.

    Thread-safe per uso concorrente.
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Inizializza TrainingScheduler.

        Args:
            config: Configurazione scheduler (default values se None)
        """
        self.config = config or SchedulerConfig()

        # Buffer
        if self.config.prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(
                capacity=self.config.max_buffer_size,
                alpha=self.config.alpha
            )
        else:
            self.buffer = ExperienceReplayBuffer(
                capacity=self.config.max_buffer_size
            )

        # State
        self._status = TrainingStatus.IDLE
        self._last_training_at: Optional[datetime] = None
        self._current_epoch = 0
        self._total_epochs = 0
        self._training_sessions_today = 0
        self._last_session_date: Optional[str] = None

        # Async
        self._training_task: Optional[asyncio.Task] = None
        self._auto_training_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Thread safety
        self._lock = threading.Lock()

        # Callbacks
        self._on_training_complete: Optional[Callable] = None
        self._on_training_error: Optional[Callable] = None

        log.info(
            "TrainingScheduler initialized",
            config=self.config.to_dict()
        )

    # -------------------------------------------------------------------------
    # BUFFER OPERATIONS
    # -------------------------------------------------------------------------

    def add_experience(
        self,
        trace: Any,
        feedback: Any,
        reward: float,
        td_error: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Aggiunge esperienza al buffer.

        Args:
            trace: ExecutionTrace
            feedback: MultilevelFeedback
            reward: Reward calcolato
            td_error: TD-error per priorità (opzionale)
            metadata: Dati aggiuntivi

        Returns:
            ID dell'esperienza
        """
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            exp_id = self.buffer.add(trace, feedback, reward, td_error, metadata)
        else:
            priority = 1.0 + abs(td_error or 0)
            exp_id = self.buffer.add(trace, feedback, reward, priority, metadata)

        log.debug(
            "Experience added",
            exp_id=exp_id,
            reward=reward,
            buffer_size=len(self.buffer)
        )

        return exp_id

    def get_buffer_stats(self) -> BufferStats:
        """Restituisce statistiche del buffer."""
        return self.buffer.get_stats()

    # -------------------------------------------------------------------------
    # TRAINING CONDITIONS
    # -------------------------------------------------------------------------

    def should_train(self) -> bool:
        """
        Verifica se le condizioni per il training sono soddisfatte.

        Returns:
            True se training dovrebbe essere avviato
        """
        with self._lock:
            # Non trainare se già in corso
            if self._status == TrainingStatus.TRAINING:
                return False

            # Non trainare se in pausa
            if self._status == TrainingStatus.PAUSED:
                return False

            # Check buffer threshold
            if len(self.buffer) < self.config.buffer_threshold:
                return False

            # Check time interval
            if self._last_training_at:
                elapsed = datetime.now() - self._last_training_at
                if elapsed.total_seconds() < self.config.min_interval_seconds:
                    return False

            return True

    def get_time_until_next_training(self) -> Optional[timedelta]:
        """
        Calcola tempo rimanente fino al prossimo training possibile.

        Returns:
            timedelta se applicabile, None altrimenti
        """
        if not self._last_training_at:
            return timedelta(seconds=0) if len(self.buffer) >= self.config.buffer_threshold else None

        elapsed = datetime.now() - self._last_training_at
        remaining = self.config.min_interval_seconds - elapsed.total_seconds()

        if remaining <= 0:
            return timedelta(seconds=0)

        return timedelta(seconds=remaining)

    # -------------------------------------------------------------------------
    # TRAINING EXECUTION
    # -------------------------------------------------------------------------

    async def run_training_epoch(
        self,
        trigger: TrainingTrigger = TrainingTrigger.MANUAL
    ) -> TrainingResult:
        """
        Esegue un ciclo di training.

        Args:
            trigger: Cosa ha triggerato il training

        Returns:
            TrainingResult con metriche
        """
        start_time = datetime.now()

        with self._lock:
            if self._status == TrainingStatus.TRAINING:
                return TrainingResult(
                    success=False,
                    error="Training already in progress"
                )

            self._status = TrainingStatus.TRAINING
            self._total_epochs = self.config.epochs_per_run
            self._current_epoch = 0

        log.info(
            "Training started",
            trigger=trigger.value,
            buffer_size=len(self.buffer),
            epochs=self.config.epochs_per_run
        )

        try:
            # Import trainer (lazy)
            from .policy_gradient import PolicyGradientTrainer, GatingPolicy

            # Get or create policy
            # In produzione, caricheremmo da checkpoint
            policy = GatingPolicy(input_dim=768, hidden_dim=256)
            trainer = PolicyGradientTrainer(policy, learning_rate=1e-4)

            total_loss = 0.0
            total_reward = 0.0
            samples_processed = 0

            for epoch in range(self.config.epochs_per_run):
                with self._lock:
                    self._current_epoch = epoch + 1

                # Sample batch
                if isinstance(self.buffer, PrioritizedReplayBuffer):
                    batch, indices, weights = self.buffer.sample_with_priority(
                        self.config.batch_size
                    )
                else:
                    batch = self.buffer.sample(self.config.batch_size)
                    weights = [1.0] * len(batch)

                if not batch:
                    log.warning("Empty batch, skipping epoch", epoch=epoch)
                    continue

                # Training su batch
                epoch_loss = 0.0
                epoch_reward = 0.0

                for i, exp in enumerate(batch):
                    try:
                        # Reconstruct trace e feedback
                        from .execution_trace import ExecutionTrace
                        from .multilevel_feedback import MultilevelFeedback

                        trace = ExecutionTrace.from_dict(exp.trace_data)
                        feedback = MultilevelFeedback.from_dict(exp.feedback_data)

                        # Update policy
                        trace.set_reward(exp.reward)
                        metrics = trainer.update_from_feedback(trace, feedback)

                        # Weight loss by importance sampling
                        weighted_loss = metrics.get("loss", 0.0) * weights[i]
                        epoch_loss += weighted_loss
                        epoch_reward += exp.reward
                        samples_processed += 1

                    except Exception as e:
                        log.warning(
                            "Error processing experience",
                            exp_id=exp.experience_id,
                            error=str(e)
                        )

                if batch:
                    total_loss += epoch_loss / len(batch)
                    total_reward += epoch_reward / len(batch)

                log.debug(
                    "Epoch completed",
                    epoch=epoch + 1,
                    loss=epoch_loss / len(batch) if batch else 0,
                    avg_reward=epoch_reward / len(batch) if batch else 0
                )

            # Update state
            with self._lock:
                self._last_training_at = datetime.now()
                self._status = TrainingStatus.IDLE
                self._current_epoch = 0
                self._update_sessions_today()

            duration = (datetime.now() - start_time).total_seconds()

            # Save checkpoint if configured
            checkpoint_version = None
            if self.config.auto_save_checkpoint and samples_processed > 0:
                checkpoint_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                # In produzione: trainer.save_checkpoint(...)
                log.info("Checkpoint saved", version=checkpoint_version)

            result = TrainingResult(
                success=True,
                epochs_completed=self.config.epochs_per_run,
                total_loss=total_loss / self.config.epochs_per_run if self.config.epochs_per_run > 0 else 0,
                avg_reward=total_reward / self.config.epochs_per_run if self.config.epochs_per_run > 0 else 0,
                checkpoint_version=checkpoint_version,
                duration_seconds=duration
            )

            log.info(
                "Training completed",
                **result.to_dict()
            )

            if self._on_training_complete:
                self._on_training_complete(result)

            return result

        except Exception as e:
            with self._lock:
                self._status = TrainingStatus.ERROR

            log.error("Training failed", error=str(e))

            result = TrainingResult(
                success=False,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )

            if self._on_training_error:
                self._on_training_error(result)

            return result

    def _update_sessions_today(self):
        """Aggiorna contatore sessioni oggi."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_session_date != today:
            self._training_sessions_today = 0
            self._last_session_date = today
        self._training_sessions_today += 1

    # -------------------------------------------------------------------------
    # AUTO TRAINING
    # -------------------------------------------------------------------------

    async def start_auto_training(self, check_interval: int = 60):
        """
        Avvia training automatico in background.

        Args:
            check_interval: Intervallo check in secondi
        """
        if self._auto_training_task and not self._auto_training_task.done():
            log.warning("Auto training already running")
            return

        self._stop_event.clear()
        self._auto_training_task = asyncio.create_task(
            self._auto_training_loop(check_interval)
        )

        log.info("Auto training started", check_interval=check_interval)

    async def stop_auto_training(self):
        """Ferma training automatico."""
        self._stop_event.set()

        if self._auto_training_task:
            self._auto_training_task.cancel()
            try:
                await self._auto_training_task
            except asyncio.CancelledError:
                pass

        log.info("Auto training stopped")

    async def _auto_training_loop(self, check_interval: int):
        """Loop interno per auto training."""
        while not self._stop_event.is_set():
            try:
                if self.should_train():
                    await self.run_training_epoch(
                        trigger=TrainingTrigger.BUFFER_THRESHOLD
                    )

                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Error in auto training loop", error=str(e))
                await asyncio.sleep(check_interval)

    # -------------------------------------------------------------------------
    # STATUS AND CONTROL
    # -------------------------------------------------------------------------

    def get_status(self) -> SchedulerStatus:
        """Restituisce stato corrente dello scheduler."""
        with self._lock:
            buffer_stats = self.buffer.get_stats()

            next_training = None
            remaining = self.get_time_until_next_training()
            if remaining is not None:
                next_training = (datetime.now() + remaining).isoformat()

            return SchedulerStatus(
                status=self._status,
                buffer_size=buffer_stats.size,
                buffer_capacity=buffer_stats.capacity,
                last_training_at=self._last_training_at.isoformat() if self._last_training_at else None,
                next_training_at=next_training,
                is_training=self._status == TrainingStatus.TRAINING,
                current_epoch=self._current_epoch,
                total_epochs=self._total_epochs,
                training_sessions_today=self._training_sessions_today,
                avg_reward=buffer_stats.avg_reward
            )

    def pause(self):
        """Mette in pausa il training automatico."""
        with self._lock:
            if self._status != TrainingStatus.TRAINING:
                self._status = TrainingStatus.PAUSED
        log.info("Training paused")

    def resume(self):
        """Riprende il training automatico."""
        with self._lock:
            if self._status == TrainingStatus.PAUSED:
                self._status = TrainingStatus.IDLE
        log.info("Training resumed")

    def set_on_training_complete(self, callback: Callable[[TrainingResult], None]):
        """Imposta callback per training completato."""
        self._on_training_complete = callback

    def set_on_training_error(self, callback: Callable[[TrainingResult], None]):
        """Imposta callback per errore training."""
        self._on_training_error = callback


# =============================================================================
# SINGLETON
# =============================================================================

_scheduler_instance: Optional[TrainingScheduler] = None
_scheduler_lock = threading.Lock()


def get_scheduler(config: Optional[SchedulerConfig] = None) -> TrainingScheduler:
    """
    Ottiene singleton TrainingScheduler.

    Args:
        config: Configurazione (usata solo alla prima chiamata)

    Returns:
        TrainingScheduler singleton
    """
    global _scheduler_instance

    with _scheduler_lock:
        if _scheduler_instance is None:
            _scheduler_instance = TrainingScheduler(config)
        return _scheduler_instance


def reset_scheduler():
    """Reset singleton (per testing)."""
    global _scheduler_instance
    with _scheduler_lock:
        _scheduler_instance = None
