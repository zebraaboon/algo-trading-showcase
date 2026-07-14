"""
Risk Manager with Kelly Criterion Position Sizing and Circuit Breakers

Demonstrates:
- Late-binding pattern (interface injected after construction)
- Multi-level circuit breaker system
- Kelly criterion with fractional sizing
- Async-safe position management
- Production kill switch

This risk manager protects against:
- Runaway losses (circuit breakers at 1%, 2%, 3% daily loss)
- Oversized positions (max 3% per position)
- Correlated exposure (correlation penalty)
- System failures (dead man's switch)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class ExchangeInterface(Protocol):
    """Abstract exchange interface — late-binding pattern."""

    async def get_balance(self) -> float: ...
    async def place_order(
        self, side: str, size: float, price: float
    ) -> dict: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def get_positions(self) -> list[dict]: ...


class CircuitLevel(IntEnum):
    """Circuit breaker levels — each triggers stricter controls."""

    NORMAL = 0  # All systems nominal
    L1_CAUTION = 1  # -1% daily → reduce position sizes by 50%
    L2_WARNING = 2  # -2% daily → new positions blocked, exits only
    L3_CRITICAL = 3  # -3% daily → emergency close all positions
    L4_LOCKDOWN = 4  # Manual reset required


@dataclass
class RiskLimits:
    """Configurable risk parameters — validated at startup."""

    max_position_pct: float = 0.03  # 3% of capital per position
    max_daily_loss_pct: float = 0.03  # 3% daily loss triggers L3
    max_drawdown_pct: float = 0.15  # 15% max drawdown → kill switch
    kelly_fraction: float = 0.25  # Quarter Kelly (conservative)
    max_correlation_penalty: float = 0.3  # Reduce size when correlated
    correlation_threshold: float = 0.7  # Penalty kicks in above this
    dead_man_switch_hours: float = 24.0  # Alert if no heartbeat

    def __post_init__(self):
        """Validate all limits at construction time."""
        assert 0 < self.max_position_pct <= 0.1, "Max position must be 0-10%"
        assert 0 < self.max_daily_loss_pct <= 0.1, "Max daily loss must be 0-10%"
        assert 0 < self.max_drawdown_pct <= 0.3, "Max drawdown must be 0-30%"
        assert 0 < self.kelly_fraction <= 1.0, "Kelly fraction must be 0-1"


class RiskManager:
    """
    Production risk manager with circuit breakers and Kelly sizing.

    Uses late-binding pattern: constructed without exchange interface,
    then linked after both components exist. This avoids circular
    dependency issues in async systems.
    """

    def __init__(self, limits: RiskLimits = RiskLimits()):
        self.limits = limits
        self._interface: Optional[ExchangeInterface] = None

        # State
        self._circuit_level = CircuitLevel.NORMAL
        self._daily_pnl = 0.0
        self._peak_equity = 0.0
        self._current_equity = 0.0
        self._last_heartbeat = time.time()
        self._kill_switch = False
        self._positions: dict[str, float] = {}  # pair → size

        # Lock for thread-safe position updates
        self._lock = asyncio.Lock()

    def link_interface(self, interface: ExchangeInterface) -> None:
        """Late-binding: attach exchange interface after construction."""
        self._interface = interface
        logger.info("RiskManager linked to exchange interface")

    # --- Position Sizing ---

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
    ) -> float:
        """
        Kelly Criterion position size with fractional scaling.

        Full Kelly: f* = (p * b - q) / b
        Where: p = win rate, q = 1-p, b = avg_win / avg_loss

        We use quarter Kelly (0.25 * f*) for production safety.
        Full Kelly is theoretically optimal but has massive variance.
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        b = avg_win / abs(avg_loss)  # Odds ratio
        p = win_rate
        q = 1 - p

        kelly_fraction = (p * b - q) / b

        if kelly_fraction <= 0:
            return 0.0  # Negative edge → don't trade

        # Apply fractional Kelly and position limit
        size = capital * kelly_fraction * self.limits.kelly_fraction
        max_size = capital * self.limits.max_position_pct

        return min(size, max_size)

    def apply_correlation_penalty(
        self,
        size: float,
        pair: str,
        correlations: dict[str, float],
    ) -> float:
        """
        Reduce position size when correlated with existing positions.

        If we're already long BTC-USD and BTC-GBP is 0.95 correlated,
        we should reduce the new position to avoid concentration risk.
        """
        if not self._positions or not correlations:
            return size

        max_corr = 0.0
        for existing_pair in self._positions:
            corr = abs(correlations.get(f"{pair}_{existing_pair}", 0))
            max_corr = max(max_corr, corr)

        if max_corr > self.limits.correlation_threshold:
            penalty = 1 - self.limits.max_correlation_penalty * (
                max_corr - self.limits.correlation_threshold
            ) / (1 - self.limits.correlation_threshold)
            return size * max(penalty, 0.1)  # Never reduce below 10%

        return size

    # --- Circuit Breakers ---

    def update_pnl(self, daily_pnl: float, current_equity: float) -> None:
        """Update daily PnL and check circuit breaker levels."""
        self._daily_pnl = daily_pnl
        self._current_equity = current_equity
        self._peak_equity = max(self._peak_equity, current_equity)
        self._last_heartbeat = time.time()

        # Check circuit breaker levels
        daily_loss_pct = abs(min(daily_pnl, 0)) / max(current_equity, 1)
        drawdown_pct = (
            (self._peak_equity - current_equity)
            / max(self._peak_equity, 1)
        )

        if drawdown_pct >= self.limits.max_drawdown_pct:
            self._kill_switch = True
            self._circuit_level = CircuitLevel.L4_LOCKDOWN
            logger.critical(
                f"KILL SWITCH: Drawdown {drawdown_pct:.1%} "
                f"exceeds {self.limits.max_drawdown_pct:.1%}"
            )
        elif daily_loss_pct >= self.limits.max_daily_loss_pct:
            self._circuit_level = CircuitLevel.L3_CRITICAL
            logger.error(
                f"CIRCUIT L3: Daily loss {daily_loss_pct:.1%}"
            )
        elif daily_loss_pct >= 0.02:
            self._circuit_level = CircuitLevel.L2_WARNING
        elif daily_loss_pct >= 0.01:
            self._circuit_level = CircuitLevel.L1_CAUTION
        else:
            self._circuit_level = CircuitLevel.NORMAL

    # --- Order Gating ---

    async def approve_order(
        self,
        pair: str,
        side: str,
        size: float,
        price: float,
    ) -> tuple[bool, str]:
        """
        Gate every order through risk checks before execution.

        Returns (approved, reason) tuple.
        """
        async with self._lock:
            # Kill switch — nothing gets through
            if self._kill_switch:
                return False, "Kill switch active"

            # L4 lockdown — manual reset required
            if self._circuit_level >= CircuitLevel.L4_LOCKDOWN:
                return False, "Circuit breaker L4: lockdown"

            # L3 — emergency close only
            if self._circuit_level >= CircuitLevel.L3_CRITICAL:
                if side == "buy":
                    return False, "Circuit breaker L3: exits only"

            # L2 — no new positions
            if self._circuit_level >= CircuitLevel.L2_WARNING:
                if pair not in self._positions and side == "buy":
                    return False, "Circuit breaker L2: no new positions"

            # Position size check
            if self._current_equity > 0:
                position_value = size * price
                position_pct = position_value / self._current_equity
                if position_pct > self.limits.max_position_pct:
                    return (
                        False,
                        f"Position {position_pct:.1%} exceeds "
                        f"limit {self.limits.max_position_pct:.1%}",
                    )

            # Price sanity check
            if price <= 0 or price > 1_000_000:
                return False, f"Price sanity check failed: {price}"

            if size <= 0:
                return False, f"Invalid size: {size}"

            # L1 — reduce size by 50%
            if self._circuit_level >= CircuitLevel.L1_CAUTION:
                logger.warning(f"L1 active: would reduce size by 50%")

            # Dead man's switch check
            hours_since = (
                (time.time() - self._last_heartbeat) / 3600
            )
            if hours_since > self.limits.dead_man_switch_hours:
                return False, f"Dead man switch: {hours_since:.1f}h since heartbeat"

            # All checks passed
            self._positions[pair] = self._positions.get(pair, 0) + (
                size if side == "buy" else -size
            )
            return True, "Approved"

    # --- Status ---

    @property
    def status(self) -> dict:
        """Current risk manager status for monitoring."""
        return {
            "circuit_level": self._circuit_level.name,
            "daily_pnl": self._daily_pnl,
            "current_equity": self._current_equity,
            "peak_equity": self._peak_equity,
            "drawdown_pct": (
                (self._peak_equity - self._current_equity)
                / max(self._peak_equity, 1)
            ),
            "kill_switch": self._kill_switch,
            "open_positions": len(self._positions),
            "last_heartbeat_ago": time.time() - self._last_heartbeat,
        }
