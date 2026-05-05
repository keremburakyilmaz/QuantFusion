"""LangGraph agent + Nemotron regime commentary.

Two responsibilities:
  1. regime_commentary(regime, holdings, risk) - one LLM call producing
     a 2-3 sentence narrative. Cached in Redis by (regime, sharpe-bucket,
     ticker-set) to keep NIM cost bounded on the public analyzer.
  2. query(query, portfolio_id) - LangGraph graph
     (intent_router → tool_call → response_formatter) returning a
     natural-language answer drawn from one of five tools.
"""


import asyncio
import json
import logging
import re
import uuid
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from redis.asyncio import Redis

from app.schemas.agent import AgentQueryResponse
from app.schemas.common import HoldingInput
from app.schemas.regime import RegimeSnapshotResponse
from app.schemas.risk import RiskMetrics


logger = logging.getLogger(__name__)


COMMENTARY_TTL = 3600
COMMENTARY_FALLBACK = ""

LEGAL_INTENTS = {"risk", "optimize", "regime", "backtest", "holdings", "earnings"}
DEFAULT_INTENT = "holdings"


class AgentState(TypedDict, total=False):
    query: str
    portfolio_id: str
    intent: str | None
    tool_args: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    response: str | None


ToolFn = Callable[..., Awaitable[dict[str, Any]]]


class AgentService:
    def __init__(
        self,
        llm_factory: Callable[[], Any | None],
        redis: Redis | None = None,
        tools: dict[str, ToolFn] | None = None,
    ) -> None:
        self._llm_factory = llm_factory
        self.redis = redis
        self.tools: dict[str, ToolFn] = tools or {}
        self._graph = self._build_graph()

    @property
    def llm(self):
        return self._llm_factory()

    @property
    def enabled(self) -> bool:
        return self.llm is not None

    # ---------- regime commentary ----------

    async def regime_commentary(
        self,
        regime: RegimeSnapshotResponse,
        holdings: list[HoldingInput],
        risk: RiskMetrics,
        earnings: dict | None = None,
    ) -> str:
        llm = self.llm
        if llm is None:
            return COMMENTARY_FALLBACK

        cache_key = self._commentary_key(regime, holdings, risk, earnings)
        if self.redis is not None and cache_key:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return cached
            except Exception:
                logger.exception("redis read failed for commentary")

        prompt = self._build_commentary_prompt(regime, holdings, risk, earnings)
        try:
            response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=50.0)
            text = _strip_preamble((response.content or "").strip())
        except Exception:
            logger.exception("regime_commentary LLM call failed")
            return COMMENTARY_FALLBACK

        if not text:
            return COMMENTARY_FALLBACK

        if self.redis is not None and cache_key:
            try:
                await self.redis.set(cache_key, text, ex=COMMENTARY_TTL)
            except Exception:
                logger.exception("redis write failed for commentary")

        return text

    def _build_commentary_prompt(
        self,
        regime: RegimeSnapshotResponse,
        holdings: list[HoldingInput],
        risk: RiskMetrics,
        earnings: dict | None = None,
    ) -> str:
        tickers_str = ", ".join(
            f"{h.ticker} ({h.weight * 100:.0f}%)" for h in holdings
        )
        sharpe = _fmt_num(risk.sharpe)
        var_hist = _fmt_pct(risk.var_historical)
        max_dd = _fmt_pct(risk.max_drawdown)
        beta = _fmt_num(risk.beta)
        confidence = _fmt_pct(regime.confidence) if regime.confidence else "?"

        earnings_block = ""
        if earnings:
            lines = []
            for ticker, sig in earnings.items():
                eps_a = sig.get("eps_actual") if isinstance(sig, dict) else getattr(sig, "eps_actual", None)
                eps_e = sig.get("eps_estimate") if isinstance(sig, dict) else getattr(sig, "eps_estimate", None)
                beat = sig.get("eps_beat") if isinstance(sig, dict) else getattr(sig, "eps_beat", None)
                guidance = sig.get("guidance") if isinstance(sig, dict) else getattr(sig, "guidance", None)
                sentiment = sig.get("sentiment", "neutral") if isinstance(sig, dict) else getattr(sig, "sentiment", "neutral")
                parts = [f"{ticker}:"]
                if eps_a is not None:
                    parts.append(f"EPS ${eps_a:.2f} actual")
                if eps_e is not None:
                    parts.append(f"vs ${eps_e:.2f} est")
                if beat is not None:
                    parts.append("(beat)" if beat else "(missed)")
                if guidance:
                    parts.append(f"guidance {guidance}")
                parts.append(f"sentiment {sentiment}")
                lines.append(" ".join(parts))
            earnings_block = "\nRecent earnings:\n" + "\n".join(f"- {l}" for l in lines) + "\n"

        return (
            f"Market regime: {regime.regime} (confidence {confidence})\n"
            f"Portfolio: {tickers_str}\n"
            f"Sharpe={sharpe}, VaR_95={var_hist}, "
            f"Max Drawdown={max_dd}, Beta={beta}\n"
            f"{earnings_block}\n"
            "Write 2-3 sentences explaining how this portfolio sits in the "
            "current regime and the main risk to watch. Be specific to these "
            "tickers, not generic. Use only the values above; do not fabricate "
            "numbers.\n\n"
            "Output ONLY the commentary itself. Do NOT include any preamble, "
            "introduction, headers, bullet points, or phrases like 'Here is' "
            "or 'Here are'. Start directly with the analysis."
        )

    def _commentary_key(
        self,
        regime: RegimeSnapshotResponse,
        holdings: list[HoldingInput],
        risk: RiskMetrics,
        earnings: dict | None = None,
    ) -> str | None:
        if regime is None or risk is None:
            return None
        sharpe = risk.sharpe if risk.sharpe is not None else 0.0
        bucket = round(sharpe * 4) / 4
        tickers = ",".join(sorted(h.ticker for h in holdings))
        # Include a hash of earnings filing dates so stale cache is invalidated
        # when new earnings are fetched
        if earnings:
            filing_sig = ":".join(
                f"{t}={getattr(s, 'filing_date', None) or (s.get('filing_date') if isinstance(s, dict) else '')}"
                for t, s in sorted(earnings.items())
            )
            import hashlib
            eh = hashlib.md5(filing_sig.encode()).hexdigest()[:8]
        else:
            eh = "none"
        return f"commentary:{regime.regime}:{bucket}:{tickers}:{eh}"

    # ---------- LangGraph agent ----------

    async def query(
        self, query: str, portfolio_id: uuid.UUID
    ) -> AgentQueryResponse:
        if not self.enabled:
            raise RuntimeError("Agent service unavailable; LLM not configured")
        initial: AgentState = {
            "query": query,
            "portfolio_id": str(portfolio_id),
        }
        final: AgentState = await self._graph.ainvoke(initial)
        return AgentQueryResponse(
            response=final.get("response") or "",
            intent=final.get("intent") or DEFAULT_INTENT,
            data=final.get("tool_result"),
        )

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("intent_router", self._intent_router_node)
        graph.add_node("tool_call", self._tool_call_node)
        graph.add_node("response_formatter", self._formatter_node)
        graph.add_edge(START, "intent_router")
        graph.add_edge("intent_router", "tool_call")
        graph.add_edge("tool_call", "response_formatter")
        graph.add_edge("response_formatter", END)
        return graph.compile()

    async def _intent_router_node(self, state: AgentState) -> AgentState:
        llm = self.llm
        prompt = (
            "Classify the user's question into exactly one intent and "
            "extract any structured arguments. Respond ONLY with a JSON "
            "object with keys 'intent' and 'args'.\n\n"
            "Legal intents: 'risk', 'optimize', 'regime', 'backtest', "
            "'holdings', 'earnings'.\n"
            "  - risk: questions about Sharpe, VaR, drawdown, volatility, beta\n"
            "  - optimize: requests to rebalance, find optimal weights, max Sharpe, min vol\n"
            "  - regime: questions about market regime, bull/bear, current conditions\n"
            "  - backtest: historical performance, equity curve, monthly returns\n"
            "  - holdings: what's in my portfolio, list tickers, show weights\n"
            "  - earnings: questions about earnings results, EPS beats/misses, guidance, "
            "revenue, recent quarterly results\n\n"
            "Args examples: optimize -> {\"method\": \"mvo\", \"target\": \"max_sharpe\"}; "
            "backtest -> {\"lookback_years\": 1}.\n\n"
            f"User question: {state['query']}\n\n"
            "JSON:"
        )
        try:
            response = await llm.ainvoke(prompt)
            text = (response.content or "").strip()
            parsed = _parse_json_loose(text)
        except Exception:
            logger.exception("intent_router LLM call failed")
            parsed = {}

        intent = parsed.get("intent") if isinstance(parsed, dict) else None
        if intent not in LEGAL_INTENTS:
            intent = DEFAULT_INTENT
        args = parsed.get("args") if isinstance(parsed, dict) else None
        if not isinstance(args, dict):
            args = {}
        return {"intent": intent, "tool_args": args}

    async def _tool_call_node(self, state: AgentState) -> AgentState:
        intent = state.get("intent") or DEFAULT_INTENT
        args = state.get("tool_args") or {}
        tool = self.tools.get(intent)
        if tool is None:
            return {"tool_result": {"error": f"no tool registered for intent: {intent}"}}
        try:
            portfolio_id = uuid.UUID(state["portfolio_id"])
            result = await tool(portfolio_id, **args)
        except TypeError:
            try:
                result = await tool(uuid.UUID(state["portfolio_id"]))
            except Exception as exc:
                logger.exception("tool %s failed", intent)
                result = {"error": str(exc)}
        except Exception as exc:
            logger.exception("tool %s failed", intent)
            result = {"error": str(exc)}
        return {"tool_result": result}

    async def _formatter_node(self, state: AgentState) -> AgentState:
        llm = self.llm
        intent = state.get("intent") or DEFAULT_INTENT
        result = state.get("tool_result") or {}
        prompt = (
            "You are a quantitative-analytics assistant. Use the data below "
            "to answer the user's question in 2-4 sentences. Be specific. "
            "Cite numbers from the data; do not invent any number not "
            "present below.\n\n"
            f"User question: {state['query']}\n"
            f"Intent: {intent}\n"
            f"Data: {json.dumps(result, default=str)[:3000]}\n\n"
            "Answer:"
        )
        try:
            response = await llm.ainvoke(prompt)
            text = (response.content or "").strip()
        except Exception:
            logger.exception("formatter LLM call failed")
            text = ""
        if not text:
            text = "I couldn't form a response from the available data."
        return {"response": text}


# ---------- helpers ----------


def _parse_json_loose(text: str) -> dict[str, Any]:
    """Try to parse JSON, falling back to the first {...} block in the text."""
    text = text.strip()
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except (ValueError, json.JSONDecodeError):
        return {}


_PREAMBLE_RE = re.compile(
    r"^\s*(here\s+(is|are|'s)|sure[,!.]?|certainly[,!.]?|below\s+is)\b[^\n:]*[:\n]+",
    re.IGNORECASE,
)


def _strip_preamble(text: str) -> str:
    """Drop a leading 'Here is...:' / 'Sure!...' style preamble line."""
    if not text:
        return text
    cleaned = _PREAMBLE_RE.sub("", text, count=1).strip()
    # Also drop a leading blank-line-separated intro paragraph if it's clearly
    # meta ("Here are 2-3 sentences..." with no numbers).
    parts = cleaned.split("\n\n", 1)
    if len(parts) == 2:
        head, rest = parts
        head_low = head.lower()
        if (
            len(head) < 200
            and any(p in head_low for p in ("here are", "here is", "below is", "as requested"))
        ):
            cleaned = rest.strip()
    return cleaned or text


def _fmt_num(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "?"


def _fmt_pct(v: float | None) -> str:
    return f"{v:.2%}" if v is not None else "?"
