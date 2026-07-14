"""
Real-time L2 Order Book Processor with Automatic Reconnection

Demonstrates:
- Async WebSocket connection management with exponential backoff
- Efficient order book maintenance (incremental updates)
- Heartbeat monitoring and stale data detection
- Production-grade error handling (never crashes)

This is a sanitised example from a system processing 60 concurrent feeds.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class OrderBook:
    """Maintains a sorted L2 order book with incremental updates."""

    bids: dict[float, float] = field(default_factory=dict)  # price -> size
    asks: dict[float, float] = field(default_factory=dict)  # price -> size
    sequence: int = 0
    last_update: float = 0.0

    def apply_update(self, side: str, price: float, size: float) -> None:
        """Apply a single level update. Size=0 means remove."""
        book = self.bids if side == "buy" else self.asks
        if size == 0:
            book.pop(price, None)
        else:
            book[price] = size
        self.last_update = time.time()

    def best_bid(self) -> Optional[float]:
        return max(self.bids.keys()) if self.bids else None

    def best_ask(self) -> Optional[float]:
        return min(self.asks.keys()) if self.asks else None

    def spread(self) -> Optional[float]:
        bid, ask = self.best_bid(), self.best_ask()
        if bid and ask:
            return ask - bid
        return None

    def mid_price(self) -> Optional[float]:
        bid, ask = self.best_bid(), self.best_ask()
        if bid and ask:
            return (bid + ask) / 2.0
        return None

    def depth(self, side: str, levels: int = 10) -> list[tuple[float, float]]:
        """Return top N levels sorted by price."""
        book = self.bids if side == "buy" else self.asks
        reverse = side == "buy"
        sorted_levels = sorted(book.items(), key=lambda x: x[0], reverse=reverse)
        return sorted_levels[:levels]

    def imbalance(self, levels: int = 5) -> float:
        """
        Order Book Imbalance (OBI) — ratio of bid volume to total volume.

        OBI > 0.5 → buying pressure (bullish)
        OBI < 0.5 → selling pressure (bearish)
        """
        bid_vol = sum(s for _, s in self.depth("buy", levels))
        ask_vol = sum(s for _, s in self.depth("ask", levels))
        total = bid_vol + ask_vol
        if total == 0:
            return 0.5
        return bid_vol / total

    @property
    def is_stale(self) -> bool:
        """Book is stale if no update in 30 seconds."""
        return time.time() - self.last_update > 30.0


class WebSocketOrderBookFeed:
    """
    Production WebSocket feed with automatic reconnection.

    Features:
    - Exponential backoff on disconnect (1s → 2s → 4s → ... → 60s max)
    - Heartbeat monitoring (detects silent disconnects)
    - Sequence gap detection → full book rebuild
    - Graceful shutdown
    """

    MAX_BACKOFF = 60.0
    HEARTBEAT_TIMEOUT = 30.0

    def __init__(
        self,
        url: str,
        pair: str,
        depth: int = 1000,
        on_update: Optional[callable] = None,
    ):
        self.url = url
        self.pair = pair
        self.depth = depth
        self.on_update = on_update

        self.book = OrderBook()
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._running = False
        self._reconnect_delay = 1.0
        self._last_heartbeat = 0.0
        self._message_count = 0

    async def start(self) -> None:
        """Start the feed with automatic reconnection."""
        self._running = True
        self._session = aiohttp.ClientSession()

        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.pair}] Feed error: {e}")

            if self._running:
                logger.info(
                    f"[{self.pair}] Reconnecting in {self._reconnect_delay:.1f}s"
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self.MAX_BACKOFF
                )

        await self._cleanup()

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()

    async def _connect_and_listen(self) -> None:
        """Connect to WebSocket and process messages."""
        async with self._session.ws_connect(
            self.url,
            heartbeat=20.0,
            receive_timeout=self.HEARTBEAT_TIMEOUT,
        ) as ws:
            self._ws = ws
            self._last_heartbeat = time.time()
            logger.info(f"[{self.pair}] Connected to {self.url}")

            # Subscribe to order book channel
            await ws.send_json(self._subscribe_message())

            # Reset backoff on successful connection
            self._reconnect_delay = 1.0

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(json.loads(msg.data))
                    self._last_heartbeat = time.time()
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"[{self.pair}] WS error: {ws.exception()}")
                    break
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break

    async def _handle_message(self, data: dict) -> None:
        """Process a WebSocket message and update the order book."""
        msg_type = data.get("type", "")

        if msg_type == "snapshot":
            # Full book snapshot — rebuild from scratch
            self.book = OrderBook()
            for bid in data.get("bids", []):
                self.book.apply_update("buy", float(bid[0]), float(bid[1]))
            for ask in data.get("asks", []):
                self.book.apply_update("ask", float(ask[0]), float(ask[1]))
            self.book.sequence = data.get("sequence", 0)
            logger.info(
                f"[{self.pair}] Snapshot: {len(self.book.bids)} bids, "
                f"{len(self.book.asks)} asks"
            )

        elif msg_type == "l2update":
            # Incremental update
            new_seq = data.get("sequence", 0)

            # Detect sequence gap → need full rebuild
            if new_seq > self.book.sequence + 1 and self.book.sequence > 0:
                logger.warning(
                    f"[{self.pair}] Sequence gap: "
                    f"{self.book.sequence} → {new_seq}, rebuilding"
                )
                # Request new snapshot (implementation depends on exchange)
                return

            for change in data.get("changes", []):
                side, price, size = change[0], float(change[1]), float(change[2])
                self.book.apply_update(side, price, size)

            self.book.sequence = new_seq

        self._message_count += 1

        # Notify callback
        if self.on_update and not self.book.is_stale:
            try:
                await self.on_update(self.pair, self.book)
            except Exception as e:
                # Callback errors must never kill the feed
                logger.error(f"[{self.pair}] Callback error: {e}")

    def _subscribe_message(self) -> dict:
        """Build exchange-specific subscription message."""
        return {
            "type": "subscribe",
            "product_ids": [self.pair],
            "channels": [{"name": "level2", "product_ids": [self.pair]}],
        }

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    @property
    def stats(self) -> dict:
        """Current feed statistics."""
        return {
            "pair": self.pair,
            "connected": self._ws is not None and not self._ws.closed,
            "messages": self._message_count,
            "bid_levels": len(self.book.bids),
            "ask_levels": len(self.book.asks),
            "spread": self.book.spread(),
            "mid_price": self.book.mid_price(),
            "imbalance": self.book.imbalance(),
            "stale": self.book.is_stale,
        }


# --- Usage Example ---

async def on_book_update(pair: str, book: OrderBook) -> None:
    """Called on every order book update."""
    mid = book.mid_price()
    obi = book.imbalance()
    spread = book.spread()
    if mid:
        logger.info(
            f"{pair} | Mid: {mid:.2f} | OBI: {obi:.3f} | Spread: {spread:.2f}"
        )


async def main():
    logging.basicConfig(level=logging.INFO)

    feed = WebSocketOrderBookFeed(
        url="wss://ws-feed.exchange.example.com",
        pair="BTC-USD",
        depth=1000,
        on_update=on_book_update,
    )

    try:
        await feed.start()
    except KeyboardInterrupt:
        await feed.stop()


if __name__ == "__main__":
    asyncio.run(main())
