"""
Abstract Exchange Adapter with Rate Limiting and Failover

Demonstrates:
- Abstract base class pattern for multi-exchange support
- Token bucket rate limiter (async-safe)
- Automatic retry with exponential backoff
- Request/response logging for debugging
- Clean separation of concerns

This pattern supports 6 exchanges through a single interface.
"""

import asyncio
import hmac
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Standardised order result across all exchanges."""

    order_id: str
    pair: str
    side: str
    size: float
    price: float
    filled_size: float
    average_price: float
    status: str  # "filled", "partial", "pending", "cancelled", "error"
    fee: float
    timestamp: float
    raw_response: dict


class TokenBucketRateLimiter:
    """
    Async token bucket rate limiter.

    Smoothly distributes requests over time rather than bursting.
    Each exchange has different rate limits — this is configurable.
    """

    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: Tokens per second (e.g., 10 = 10 requests/sec)
            burst: Maximum tokens that can accumulate
        """
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self.burst, self._tokens + elapsed * self.rate
                )
                self._last_refill = now

                if self._tokens >= 1:
                    self._tokens -= 1
                    return

                # Wait for next token
                wait_time = (1 - self._tokens) / self.rate
                await asyncio.sleep(wait_time)


class ExchangeBase(ABC):
    """
    Abstract base class for exchange adapters.

    Provides:
    - Authenticated HTTP client with rate limiting
    - Automatic retry with exponential backoff
    - Request/response logging
    - Error normalisation

    Subclasses implement exchange-specific:
    - Authentication (sign_request)
    - Order placement (place_order)
    - Balance retrieval (get_balance)
    - WebSocket feeds (subscribe_orderbook)
    """

    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        api_secret: str,
        rate_limit: float = 10.0,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._api_secret = api_secret
        self._rate_limiter = TokenBucketRateLimiter(rate=rate_limit)
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0

    async def start(self) -> None:
        """Initialise HTTP session."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "TradingBot/1.0"},
        )

    async def stop(self) -> None:
        """Clean shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        authenticated: bool = False,
    ) -> dict:
        """
        Make an HTTP request with rate limiting, retry, and logging.

        Automatically handles:
        - Rate limiting (waits for token)
        - Authentication (signs request if needed)
        - Retries with exponential backoff on 429/5xx
        - Timeout handling
        - Response parsing
        """
        url = f"{self.base_url}{path}"
        headers = {}

        if authenticated:
            headers = self.sign_request(method, path, body or {})

        for attempt in range(self.MAX_RETRIES):
            await self._rate_limiter.acquire()
            self._request_count += 1

            try:
                async with self._session.request(
                    method,
                    url,
                    params=params,
                    json=body,
                    headers=headers,
                ) as resp:
                    data = await resp.json()

                    if resp.status == 200:
                        return data

                    if resp.status == 429:
                        # Rate limited — wait and retry
                        retry_after = float(
                            resp.headers.get("Retry-After", 5)
                        )
                        logger.warning(
                            f"[{self.name}] Rate limited, "
                            f"waiting {retry_after}s"
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status >= 500:
                        # Server error — retry with backoff
                        delay = self.BASE_RETRY_DELAY * (2**attempt)
                        logger.warning(
                            f"[{self.name}] Server error {resp.status}, "
                            f"retry in {delay}s"
                        )
                        await asyncio.sleep(delay)
                        continue

                    # Client error — don't retry
                    logger.error(
                        f"[{self.name}] {method} {path} → "
                        f"{resp.status}: {data}"
                    )
                    return {"error": data, "status": resp.status}

            except asyncio.TimeoutError:
                delay = self.BASE_RETRY_DELAY * (2**attempt)
                logger.warning(
                    f"[{self.name}] Timeout on {method} {path}, "
                    f"retry in {delay}s"
                )
                await asyncio.sleep(delay)
            except aiohttp.ClientError as e:
                logger.error(f"[{self.name}] Connection error: {e}")
                await asyncio.sleep(self.BASE_RETRY_DELAY)

        logger.error(
            f"[{self.name}] {method} {path} failed after "
            f"{self.MAX_RETRIES} retries"
        )
        return {"error": "max_retries_exceeded"}

    # --- Abstract methods (exchange-specific) ---

    @abstractmethod
    def sign_request(
        self, method: str, path: str, body: dict
    ) -> dict[str, str]:
        """Sign a request — returns headers dict."""
        ...

    @abstractmethod
    async def get_balance(self) -> dict[str, float]:
        """Get account balances — returns {currency: amount}."""
        ...

    @abstractmethod
    async def place_order(
        self,
        pair: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        order_type: str = "limit",
        post_only: bool = True,
    ) -> OrderResult:
        """Place an order on the exchange."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        ...

    @abstractmethod
    async def get_orderbook(
        self, pair: str, depth: int = 100
    ) -> dict:
        """Fetch current order book snapshot."""
        ...

    @property
    def stats(self) -> dict:
        """Connection statistics."""
        return {
            "exchange": self.name,
            "requests": self._request_count,
            "connected": self._session is not None
            and not self._session.closed,
        }


# --- Example Implementation ---


class CoinbaseAdapter(ExchangeBase):
    """
    Coinbase Advanced Trade API adapter.

    Implements HMAC-SHA256 authentication and maps
    Coinbase-specific responses to standardised format.
    """

    def __init__(self, api_key: str, api_secret: str):
        super().__init__(
            name="coinbase",
            base_url="https://api.coinbase.com",
            api_key=api_key,
            api_secret=api_secret,
            rate_limit=10.0,  # 10 requests/sec
        )

    def sign_request(
        self, method: str, path: str, body: dict
    ) -> dict[str, str]:
        """Coinbase HMAC-SHA256 signature."""
        timestamp = str(int(time.time()))
        body_str = json.dumps(body) if body else ""
        message = f"{timestamp}{method.upper()}{path}{body_str}"

        signature = hmac.new(
            self._api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        return {
            "CB-ACCESS-KEY": self._api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    async def get_balance(self) -> dict[str, float]:
        resp = await self._request(
            "GET", "/api/v3/brokerage/accounts", authenticated=True
        )
        balances = {}
        for account in resp.get("accounts", []):
            currency = account["currency"]
            available = float(account.get("available_balance", {}).get("value", 0))
            if available > 0:
                balances[currency] = available
        return balances

    async def place_order(
        self,
        pair: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        order_type: str = "limit",
        post_only: bool = True,
    ) -> OrderResult:
        body = {
            "product_id": pair,
            "side": side.upper(),
            "order_configuration": {},
        }

        if order_type == "limit" and price:
            body["order_configuration"]["limit_limit_gtc"] = {
                "base_size": str(size),
                "limit_price": str(price),
                "post_only": post_only,
            }
        else:
            body["order_configuration"]["market_market_ioc"] = {
                "base_size": str(size),
            }

        resp = await self._request(
            "POST", "/api/v3/brokerage/orders", body=body, authenticated=True
        )

        if "error" in resp:
            return OrderResult(
                order_id="",
                pair=pair,
                side=side,
                size=size,
                price=price or 0,
                filled_size=0,
                average_price=0,
                status="error",
                fee=0,
                timestamp=time.time(),
                raw_response=resp,
            )

        order = resp.get("success_response", {})
        return OrderResult(
            order_id=order.get("order_id", ""),
            pair=pair,
            side=side,
            size=size,
            price=price or 0,
            filled_size=0,
            average_price=0,
            status="pending",
            fee=0,
            timestamp=time.time(),
            raw_response=resp,
        )

    async def cancel_order(self, order_id: str) -> bool:
        resp = await self._request(
            "POST",
            "/api/v3/brokerage/orders/batch_cancel",
            body={"order_ids": [order_id]},
            authenticated=True,
        )
        results = resp.get("results", [])
        return bool(results and results[0].get("success"))

    async def get_orderbook(
        self, pair: str, depth: int = 100
    ) -> dict:
        return await self._request(
            "GET",
            f"/api/v3/brokerage/market/product_book",
            params={"product_id": pair, "limit": depth},
        )
