"""SQLite persistence behind `RunManager` (Phase 7 PRD §6).

`RunStore` is the only module allowed to import `sqlite3`. It owns the schema,
migrations, event projections, recovery, and retention, and speaks domain
objects (`RunRecord`, `RunEvent`, `RoundSnapshot`) from `app.runs` so the
in-memory owner can pass through without translating. The dependency direction
is `app.db -> app.runs/config/schemas`; `app.runs` mentions `RunStore` only
under TYPE_CHECKING so no cycle exists at runtime.

Durability model: one short-lived connection per operation, WAL, one
transaction per public write. The event log is the spine — `append_event`
projects event rows into runs/steps/messages tables in the same transaction,
so durable state never drifts from the stream the SSE clients saw.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app import logging_config as log
from app.config import Settings
from app.llm.base import ToolCall
from app.runs import (
    ErrorInfo,
    RoundSnapshot,
    RunEvent,
    RunEventType,
    RunRecord,
    RunStatus,
    StepKind,
    StepRecord,
    StepStatus,
)
from app.schemas import (
    AccountabilityReport,
    AgentMessage,
    AgentRole,
    Claim,
    ClaimStatus,
    ClaimVerdict,
    Evidence,
    ManagerDecision,
    MessageType,
    ToolResult,
    VerificationReport,
    Verdict,
)

logger = logging.getLogger("ai_nexus.db")

__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "PersistenceError",
    "RunStore",
    "build_run_store",
]

SCHEMA_VERSION = 2
_BUSY_TIMEOUT_S = 5.0

# `has_claims`/`has_verdicts` distinguish `None` from `[]` on AgentMessage —
# the wire payload differs (null vs []) and replay must match it exactly.
_V1 = """
CREATE TABLE runs (
    id               TEXT PRIMARY KEY,
    task             TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('running','completed','failed')),
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    error_json       TEXT,
    final_message_id TEXT
);

CREATE TABLE messages (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    from_agent    TEXT NOT NULL,
    to_agent      TEXT,
    type          TEXT NOT NULL,
    content       TEXT NOT NULL DEFAULT '',
    decision_json TEXT,
    created_at    TEXT NOT NULL,
    has_claims    INTEGER NOT NULL DEFAULT 0,
    has_verdicts  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_messages_run ON messages(run_id);

CREATE TABLE steps (
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    agent       TEXT NOT NULL,
    round       INTEGER,
    status      TEXT NOT NULL,
    skipped     INTEGER NOT NULL DEFAULT 0,
    duration_ms REAL,
    error_json  TEXT,
    message_id  TEXT,
    started_at  TEXT NOT NULL,
    PRIMARY KEY (run_id, idx)
);

CREATE TABLE claims (
    pk         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    claim_id   TEXT NOT NULL,
    statement  TEXT NOT NULL,
    status     TEXT NOT NULL,
    confidence REAL
);
CREATE INDEX idx_claims_message ON claims(message_id);

CREATE TABLE claim_evidence (
    claim_pk INTEGER NOT NULL REFERENCES claims(pk) ON DELETE CASCADE,
    ordinal  INTEGER NOT NULL,
    source   TEXT NOT NULL,
    quote    TEXT,
    PRIMARY KEY (claim_pk, ordinal)
);

CREATE TABLE verdicts (
    pk         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    claim_id   TEXT NOT NULL,
    verdict    TEXT NOT NULL,
    objection  TEXT NOT NULL
);
CREATE INDEX idx_verdicts_message ON verdicts(message_id);

CREATE TABLE verdict_evidence (
    verdict_pk INTEGER NOT NULL REFERENCES verdicts(pk) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    source     TEXT NOT NULL,
    quote      TEXT,
    PRIMARY KEY (verdict_pk, ordinal)
);

CREATE TABLE tool_calls (
    pk             INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id     TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    ordinal        INTEGER NOT NULL,
    name           TEXT NOT NULL,
    call_id        TEXT,
    arguments_json TEXT NOT NULL,
    result_content TEXT,
    error          TEXT,
    duration_ms    REAL,
    executed       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_tool_calls_message ON tool_calls(message_id);

CREATE TABLE rounds (
    run_id           TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    round_number     INTEGER NOT NULL,
    supported_count  INTEGER NOT NULL,
    unresolved_count INTEGER NOT NULL,
    decision_json    TEXT,
    snapshot_json    TEXT NOT NULL,
    PRIMARY KEY (run_id, round_number)
);

CREATE TABLE events (
    run_id       TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    type         TEXT NOT NULL,
    ts           TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""

# Phase 11: retry visibility + audit reports ride the message row.
_V2 = """
ALTER TABLE messages ADD COLUMN retries INTEGER NOT NULL DEFAULT 0;
ALTER TABLE messages ADD COLUMN verification_json TEXT;
ALTER TABLE messages ADD COLUMN accountability_json TEXT;
"""

MIGRATIONS: tuple[str, ...] = (_V1, _V2)


class PersistenceError(Exception):
    """A durable-state operation failed; surfaced to operators and the API."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


# --------------------------------------------------------------------- helpers


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _dt(value: str) -> datetime:
    """Strict parse for required timestamps (malformed data must not degrade)."""
    return datetime.fromisoformat(value)


def _dt_opt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _error_json(error: ErrorInfo | None) -> str | None:
    return error.model_dump_json() if error is not None else None


def _error_from_json(raw: str | None) -> ErrorInfo | None:
    return ErrorInfo.model_validate_json(raw) if raw else None


def _round_to_json(snapshot: RoundSnapshot) -> str:
    payload = {
        "claims": [claim.model_dump(mode="json") for claim in snapshot.claims],
        "origins": {
            claim_id: role.value for claim_id, role in snapshot.origins.items()
        },
        "verdicts": [verdict.model_dump(mode="json") for verdict in snapshot.verdicts],
        "worker_content": {
            role.value: text for role, text in snapshot.worker_content.items()
        },
        "skeptic_content": snapshot.skeptic_content,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _round_from_json(
    round_number: int,
    supported: int,
    unresolved: int,
    decision_json: str | None,
    raw: str,
) -> RoundSnapshot:
    data = json.loads(raw)
    return RoundSnapshot(
        round_number=round_number,
        claims=[Claim.model_validate(item) for item in data["claims"]],
        origins={cid: AgentRole(role) for cid, role in data["origins"].items()},
        verdicts=[Verdict.model_validate(item) for item in data["verdicts"]],
        worker_content={
            AgentRole(role): text
            for role, text in data["worker_content"].items()
        },
        skeptic_content=data["skeptic_content"],
        supported_count=supported,
        unresolved_count=unresolved,
        decision=(
            ManagerDecision.model_validate_json(decision_json)
            if decision_json
            else None
        ),
    )


def _insert_message(conn: sqlite3.Connection, run_id: str, message: AgentMessage) -> None:
    """Idempotent message-tree insert (messages are immutable once written)."""
    if conn.execute("SELECT 1 FROM messages WHERE id = ?", (message.id,)).fetchone():
        return
    conn.execute(
        "INSERT INTO messages(id, run_id, from_agent, to_agent, type, content,"
        " decision_json, created_at, has_claims, has_verdicts, retries,"
        " verification_json, accountability_json)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            message.id,
            run_id,
            message.from_agent.value,
            message.to_agent.value if message.to_agent else None,
            message.type.value,
            message.content,
            message.decision.model_dump_json() if message.decision else None,
            _iso(message.created_at),
            int(message.claims is not None),
            int(message.verdicts is not None),
            message.retries,
            (
                message.verification.model_dump_json()
                if message.verification else None
            ),
            (
                message.accountability.model_dump_json()
                if message.accountability else None
            ),
        ),
    )
    for ordinal, claim in enumerate(message.claims or []):
        cursor = conn.execute(
            "INSERT INTO claims(message_id, ordinal, claim_id, statement, status,"
            " confidence) VALUES (?,?,?,?,?,?)",
            (
                message.id, ordinal, claim.id, claim.statement,
                claim.status.value, claim.confidence,
            ),
        )
        for ev_ordinal, evidence in enumerate(claim.evidence):
            conn.execute(
                "INSERT INTO claim_evidence(claim_pk, ordinal, source, quote)"
                " VALUES (?,?,?,?)",
                (cursor.lastrowid, ev_ordinal, evidence.source, evidence.quote),
            )
    for ordinal, verdict in enumerate(message.verdicts or []):
        cursor = conn.execute(
            "INSERT INTO verdicts(message_id, ordinal, claim_id, verdict, objection)"
            " VALUES (?,?,?,?,?)",
            (
                message.id, ordinal, verdict.claim_id, verdict.verdict.value,
                verdict.objection,
            ),
        )
        for ev_ordinal, evidence in enumerate(verdict.evidence):
            conn.execute(
                "INSERT INTO verdict_evidence(verdict_pk, ordinal, source, quote)"
                " VALUES (?,?,?,?)",
                (cursor.lastrowid, ev_ordinal, evidence.source, evidence.quote),
            )
    for ordinal, call in enumerate(message.tool_calls or []):
        result = (
            message.tool_results[ordinal]
            if message.tool_results is not None
            and ordinal < len(message.tool_results)
            else None
        )
        conn.execute(
            "INSERT INTO tool_calls(message_id, ordinal, name, call_id,"
            " arguments_json, result_content, error, duration_ms, executed)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                message.id, ordinal, call.name, call.id,
                json.dumps(call.arguments, ensure_ascii=False, sort_keys=True),
                result.content if result else None,
                result.error if result else None,
                result.duration_ms if result else None,
                int(result is not None),
            ),
        )


def _load_message(conn: sqlite3.Connection, message_id: str) -> AgentMessage:
    row = conn.execute(
        "SELECT * FROM messages WHERE id = ?", (message_id,)
    ).fetchone()
    if row is None:
        raise PersistenceError(f"message {message_id} not found")

    claims: list[Claim] | None = None
    if row["has_claims"]:
        claims = []
        for claim_row in conn.execute(
            "SELECT pk, claim_id, statement, status, confidence FROM claims"
            " WHERE message_id = ? ORDER BY ordinal",
            (message_id,),
        ):
            evidence = [
                Evidence(source=item["source"], quote=item["quote"])
                for item in conn.execute(
                    "SELECT source, quote FROM claim_evidence WHERE claim_pk = ?"
                    " ORDER BY ordinal",
                    (claim_row["pk"],),
                )
            ]
            claims.append(
                Claim(
                    id=claim_row["claim_id"],
                    statement=claim_row["statement"],
                    status=ClaimStatus(claim_row["status"]),
                    confidence=claim_row["confidence"],
                    evidence=evidence,
                )
            )

    verdicts: list[Verdict] | None = None
    if row["has_verdicts"]:
        verdicts = []
        for verdict_row in conn.execute(
            "SELECT pk, claim_id, verdict, objection FROM verdicts"
            " WHERE message_id = ? ORDER BY ordinal",
            (message_id,),
        ):
            evidence = [
                Evidence(source=item["source"], quote=item["quote"])
                for item in conn.execute(
                    "SELECT source, quote FROM verdict_evidence WHERE verdict_pk = ?"
                    " ORDER BY ordinal",
                    (verdict_row["pk"],),
                )
            ]
            verdicts.append(
                Verdict(
                    claim_id=verdict_row["claim_id"],
                    verdict=ClaimVerdict(verdict_row["verdict"]),
                    objection=verdict_row["objection"],
                    evidence=evidence,
                )
            )

    call_rows = list(
        conn.execute(
            "SELECT name, call_id, arguments_json, result_content, error,"
            " duration_ms, executed FROM tool_calls WHERE message_id = ?"
            " ORDER BY ordinal",
            (message_id,),
        )
    )
    tool_calls = [
        ToolCall(
            name=item["name"],
            arguments=json.loads(item["arguments_json"]),
            id=item["call_id"],
        )
        for item in call_rows
    ]
    tool_results = [
        ToolResult(
            name=item["name"],
            content=item["result_content"],
            error=item["error"],
            duration_ms=item["duration_ms"],
        )
        for item in call_rows
        if item["executed"]
    ]

    return AgentMessage(
        id=row["id"],
        from_agent=AgentRole(row["from_agent"]),
        to_agent=AgentRole(row["to_agent"]) if row["to_agent"] else None,
        type=MessageType(row["type"]),
        content=row["content"],
        claims=claims,
        verdicts=verdicts,
        decision=(
            ManagerDecision.model_validate_json(row["decision_json"])
            if row["decision_json"]
            else None
        ),
        verification=(
            VerificationReport.model_validate_json(row["verification_json"])
            if row["verification_json"]
            else None
        ),
        accountability=(
            AccountabilityReport.model_validate_json(row["accountability_json"])
            if row["accountability_json"]
            else None
        ),
        confidence=None,   # reserved, always None on the write path
        tool_calls=tool_calls or None,
        tool_results=tool_results or None,
        retries=row["retries"] or 0,
        round=None,        # reserved, always None on the write path
        created_at=_dt(row["created_at"]),
    )


# -------------------------------------------------------------------- RunStore


class RunStore:
    """Durable backend for `RunManager`. One connection per operation."""

    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(self, path: str | Path, *, retention_runs: int = 0) -> None:
        self._path = Path(path).expanduser()
        self._retention_runs = retention_runs
        try:
            if str(self._path.parent):
                self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._read() as conn:
                # persistent property; must run outside any transaction
                conn.execute("PRAGMA journal_mode=WAL").fetchone()
            self._migrate()
        except PersistenceError:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise PersistenceError(
                f"cannot open database {self._path}: {type(exc).__name__}: {exc}",
                hint="Check AI_NEXUS_DB_PATH.",
            ) from exc
        self.prune(self._retention_runs)

    # ------------------------------------------------------------- connections

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._path), timeout=_BUSY_TIMEOUT_S, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = self._open()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @contextmanager
    def _guard(self, operation: str) -> Iterator[None]:
        """Convert infrastructure/data errors to typed PersistenceError."""
        try:
            yield
        except PersistenceError:
            raise
        except (
            sqlite3.Error, OSError, ValidationError, ValueError, KeyError, TypeError,
            AttributeError,
        ) as exc:
            log.persistence_error(
                logger, operation=operation, error=f"{type(exc).__name__}: {exc}"
            )
            raise PersistenceError(
                f"{operation} failed: {type(exc).__name__}: {exc}",
                hint="The database may be corrupt or from an incompatible build.",
            ) from exc

    def _migrate(self) -> None:
        with self._read() as conn:
            try:
                current = conn.execute("PRAGMA user_version").fetchone()[0]
            except sqlite3.DatabaseError as exc:
                raise PersistenceError(
                    f"{self._path} is not a readable SQLite database: {exc}",
                    hint="Check AI_NEXUS_DB_PATH or delete the file to re-create it.",
                ) from exc
            if current > SCHEMA_VERSION:
                raise PersistenceError(
                    f"database schema is newer than this build "
                    f"(user_version={current}, code={SCHEMA_VERSION})",
                    hint="Upgrade the code or point AI_NEXUS_DB_PATH elsewhere.",
                )
            for version in range(current, SCHEMA_VERSION):
                script = (
                    "BEGIN;\n"
                    + MIGRATIONS[version]
                    + f"\nPRAGMA user_version = {version + 1};\nCOMMIT;"
                )
                try:
                    conn.executescript(script)
                except sqlite3.Error as exc:
                    with contextlib.suppress(sqlite3.Error):
                        conn.execute("ROLLBACK")
                    raise PersistenceError(
                        f"migration to schema v{version + 1} failed: {exc}",
                        hint="The database file was left unchanged.",
                    ) from exc

    # ------------------------------------------------------------------ writes

    def save_run(self, run: RunRecord) -> None:
        with self._guard("save_run"), self._write() as conn:
            conn.execute(
                "INSERT INTO runs(id, task, status, created_at, started_at,"
                " finished_at, error_json, final_message_id) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run.id, run.task, run.status.value, _iso(run.created_at),
                    _iso(run.started_at) if run.started_at else None,
                    _iso(run.finished_at) if run.finished_at else None,
                    _error_json(run.error),
                    run.final_message.id if run.final_message else None,
                ),
            )
        self.prune(self._retention_runs)

    def append_event(self, event: RunEvent) -> None:
        with self._guard("append_event"), self._write() as conn:
            conn.execute(
                "INSERT INTO events(run_id, seq, type, ts, payload_json)"
                " VALUES (?,?,?,?,?)",
                (
                    event.run_id, event.seq, event.type.value, _iso(event.ts),
                    event.model_dump_json(),
                ),
            )
            self._project(conn, event)

    @staticmethod
    def _project(conn: sqlite3.Connection, event: RunEvent) -> None:
        """One transaction: the event row above plus this event's state change."""
        if event.type is RunEventType.RUN_STARTED:
            conn.execute(
                "UPDATE runs SET started_at = ?, status = 'running' WHERE id = ?",
                (_iso(event.ts), event.run_id),
            )
            return
        if event.type is RunEventType.STEP_STARTED:
            RunStore._upsert_step(conn, event, status="running")
            return
        if event.type is RunEventType.STEP_COMPLETED:
            status = "skipped" if event.skipped else "completed"
            if event.message is not None:
                _insert_message(conn, event.run_id, event.message)
            RunStore._upsert_step(conn, event, status=status)
            return
        if event.type is RunEventType.RUN_COMPLETED:
            if event.message is not None:
                _insert_message(conn, event.run_id, event.message)
            conn.execute(
                "UPDATE runs SET status = 'completed', finished_at = ?,"
                " final_message_id = COALESCE(?, final_message_id) WHERE id = ?",
                (
                    _iso(event.ts),
                    event.message.id if event.message else None,
                    event.run_id,
                ),
            )
            return
        if event.type is RunEventType.RUN_FAILED:
            conn.execute(
                "UPDATE runs SET status = 'failed', finished_at = ?,"
                " error_json = ? WHERE id = ?",
                (_iso(event.ts), _error_json(event.error), event.run_id),
            )
            conn.execute(
                "UPDATE steps SET status = 'failed', error_json = ?"
                " WHERE run_id = ? AND status = 'running'",
                (_error_json(event.error), event.run_id),
            )

    @staticmethod
    def _upsert_step(conn: sqlite3.Connection, event: RunEvent, *, status: str) -> None:
        if event.step is None or event.kind is None or event.agent is None:
            raise PersistenceError(
                f"step event missing step/kind/agent fields: {event.model_dump()}"
            )
        conn.execute(
            "INSERT INTO steps(run_id, idx, kind, agent, round, status, skipped,"
            " duration_ms, error_json, message_id, started_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(run_id, idx) DO UPDATE SET"
            " kind = excluded.kind, agent = excluded.agent, round = excluded.round,"
            " status = excluded.status, skipped = excluded.skipped,"
            " duration_ms = excluded.duration_ms, error_json = excluded.error_json,"
            " message_id = excluded.message_id",
            (
                event.run_id, event.step, event.kind.value, event.agent.value,
                event.round, status, int(bool(event.skipped)), event.duration_ms,
                _error_json(event.error),
                event.message.id if event.message else None,
                _iso(event.ts),
            ),
        )

    def record_round(self, run_id: str, snapshot: RoundSnapshot) -> None:
        with self._guard(f"record_round({run_id})"), self._write() as conn:
            conn.execute(
                "INSERT INTO rounds(run_id, round_number, supported_count,"
                " unresolved_count, decision_json, snapshot_json)"
                " VALUES (?,?,?,?,?,?)",
                (
                    run_id, snapshot.round_number, snapshot.supported_count,
                    snapshot.unresolved_count,
                    snapshot.decision.model_dump_json()
                    if snapshot.decision
                    else None,
                    _round_to_json(snapshot),
                ),
            )

    def record_round_decision(
        self, run_id: str, round_number: int, decision: ManagerDecision
    ) -> None:
        with self._guard(f"record_round_decision({run_id})"), self._write() as conn:
            cursor = conn.execute(
                "UPDATE rounds SET decision_json = ? WHERE run_id = ?"
                " AND round_number = ?",
                (decision.model_dump_json(), run_id, round_number),
            )
            if cursor.rowcount == 0:
                raise PersistenceError(
                    f"round {round_number} of run {run_id} not found for decision"
                )

    def recover_interrupted(self) -> list[str]:
        recovered: list[str] = []
        with self._guard("recover_interrupted"):
            with self._read() as conn:
                rows = conn.execute(
                    "SELECT id FROM runs WHERE status = 'running' ORDER BY created_at"
                ).fetchall()
            for row in rows:
                run_id = row["id"]
                with self._write() as conn:
                    step = conn.execute(
                        "SELECT idx, kind, agent, round FROM steps"
                        " WHERE run_id = ? AND status = 'running'"
                        " ORDER BY idx DESC LIMIT 1",
                        (run_id,),
                    ).fetchone()
                    max_seq = conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) FROM events WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()[0]
                    now = datetime.now(UTC)
                    error = ErrorInfo(
                        type="ServerRestart",
                        message="run was interrupted by a server restart",
                        hint="Runs cannot be resumed; create a new run.",
                    )
                    event = RunEvent(
                        seq=max_seq + 1,
                        type=RunEventType.RUN_FAILED,
                        run_id=run_id,
                        ts=now,
                        step=step["idx"] if step else None,
                        kind=StepKind(step["kind"]) if step else None,
                        agent=AgentRole(step["agent"]) if step else None,
                        round=step["round"] if step else None,
                        error=error,
                    )
                    conn.execute(
                        "UPDATE runs SET status = 'failed', finished_at = ?,"
                        " error_json = ? WHERE id = ?",
                        (_iso(now), error.model_dump_json(), run_id),
                    )
                    conn.execute(
                        "UPDATE steps SET status = 'failed', error_json = ?"
                        " WHERE run_id = ? AND status = 'running'",
                        (error.model_dump_json(), run_id),
                    )
                    conn.execute(
                        "INSERT INTO events(run_id, seq, type, ts, payload_json)"
                        " VALUES (?,?,?,?,?)",
                        (
                            run_id, event.seq, event.type.value, _iso(event.ts),
                            event.model_dump_json(),
                        ),
                    )
                recovered.append(run_id)
        return recovered

    def prune(self, keep: int) -> int:
        if keep <= 0:
            return 0
        with self._guard("prune"), self._write() as conn:
            rows = conn.execute(
                "SELECT id FROM runs WHERE status != 'running'"
                " ORDER BY created_at DESC, id DESC LIMIT -1 OFFSET ?",
                (keep,),
            ).fetchall()
            conn.executemany(
                "DELETE FROM runs WHERE id = ?", [(row["id"],) for row in rows]
            )
        deleted = len(rows)
        if deleted:
            logger.info("persistence_pruned deleted=%s kept=%s", deleted, keep)
        return deleted

    # ------------------------------------------------------------------- reads

    def load_run(self, run_id: str) -> RunRecord | None:
        with self._guard(f"load_run({run_id})"), self._read() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None

            steps = []
            for step_row in conn.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY idx", (run_id,)
            ):
                steps.append(
                    StepRecord(
                        index=step_row["idx"],
                        kind=StepKind(step_row["kind"]),
                        agent=AgentRole(step_row["agent"]),
                        status=StepStatus(step_row["status"]),
                        started_at=_dt(step_row["started_at"]),
                        duration_ms=step_row["duration_ms"],
                        message=(
                            _load_message(conn, step_row["message_id"])
                            if step_row["message_id"]
                            else None
                        ),
                        skipped=bool(step_row["skipped"]),
                        error=_error_from_json(step_row["error_json"]),
                        round=step_row["round"],
                    )
                )

            rounds = [
                _round_from_json(
                    round_row["round_number"],
                    round_row["supported_count"],
                    round_row["unresolved_count"],
                    round_row["decision_json"],
                    round_row["snapshot_json"],
                )
                for round_row in conn.execute(
                    "SELECT * FROM rounds WHERE run_id = ? ORDER BY round_number",
                    (run_id,),
                )
            ]

            events = [
                RunEvent.model_validate_json(event_row["payload_json"])
                for event_row in conn.execute(
                    "SELECT payload_json FROM events WHERE run_id = ? ORDER BY seq",
                    (run_id,),
                )
            ]

            final_message = (
                _load_message(conn, row["final_message_id"])
                if row["final_message_id"]
                else None
            )

            return RunRecord(
                id=row["id"],
                task=row["task"],
                status=RunStatus(row["status"]),
                created_at=_dt(row["created_at"]),
                started_at=_dt_opt(row["started_at"]),
                finished_at=_dt_opt(row["finished_at"]),
                steps=steps,
                final_message=final_message,
                error=_error_from_json(row["error_json"]),
                events=events,
                rounds=rounds,
            )

    def list_summaries(self, limit: int) -> list[RunRecord]:
        """Stub records (timestamps only) for merged history views."""
        with self._guard("list_summaries"), self._read() as conn:
            rows = conn.execute(
                "SELECT id, task, status, created_at, started_at, finished_at"
                " FROM runs ORDER BY created_at DESC, id DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [
            RunRecord(
                id=row["id"],
                task=row["task"],
                status=RunStatus(row["status"]),
                created_at=_dt(row["created_at"]),
                started_at=_dt_opt(row["started_at"]),
                finished_at=_dt_opt(row["finished_at"]),
            )
            for row in rows
        ]

    def load_events(self, run_id: str, since_seq: int = 0) -> list[RunEvent]:
        with self._guard(f"load_events({run_id})"), self._read() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM events WHERE run_id = ? AND seq > ?"
                " ORDER BY seq",
                (run_id, since_seq),
            ).fetchall()
        return [RunEvent.model_validate_json(row["payload_json"]) for row in rows]


def build_run_store(settings: Settings) -> RunStore | None:
    """Factory: `AI_NEXUS_DB_PATH=""` disables persistence entirely."""
    if not settings.db_path:
        return None
    return RunStore(
        settings.db_path, retention_runs=settings.db_retention_runs
    )
