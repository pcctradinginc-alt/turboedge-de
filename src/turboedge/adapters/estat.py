"""Japanese e-Stat (Portal Site of Official Statistics of Japan) adapter.

**This source is an addition beyond the written specification**, requested
directly by the user -- it is not one of the Wave 1 sources named in the
adapter build contract, and is not wired into `configs/sources.yaml` or any
curated series list by this change.

Implements `turboedge.external.adapter.ExternalSeriesAdapter` against
e-Stat's public REST API 3.0, `getStatsData`:
``https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData``, documented at
https://www.e-stat.go.jp/api/en/api-dev/how_to_use and (parameter reference,
Japanese-language page, English parameter names) https://www.e-stat.go.jp/api/api-info/e-stat-manual3-0.

DOCUMENTATION-BASED, NOT LIVE-VERIFIED: no `ESTAT_APP_ID` is available in
this environment (the user holds a key and will add it as a GitHub secret
separately). Every response shape parsed here -- the
`GET_STATS_DATA.RESULT.{STATUS,ERROR_MSG}` envelope, the STATUS convention
("0"-"2" success, ">=100" error, per the fetched parameter reference),
`STATISTICAL_DATA.CLASS_INF.CLASS_OBJ[].CLASS[]` (`@id`/`@code`/`@name`),
and `STATISTICAL_DATA.DATA_INF.VALUE[]` (`@time`, other `@<dim>` attributes,
`$` for the scalar value) -- comes from that fetched documentation, not from
an observed live response. See the adapter report for exactly what was
fetched and what is still a best-effort inference (time-code parsing, see
below).

NO CURATED statsDataId: the task brief is explicit that no `statsDataId`
should be hard-coded unless it has actually been seen in official
documentation or a catalog, and no such id was available to verify here.
This adapter is therefore fully generic -- `spec.native_identifier` is
whatever `statsDataId` a caller supplies -- and no default/curated list of
ids is defined anywhere in this module.

ERROR ENVELOPE: e-Stat answers HTTP 200 even for a request it considers an
error (an unknown `statsDataId`, a malformed parameter, ...); the failure is
signalled only by `RESULT.STATUS` inside the JSON body. `fetch()` does not
inspect the body at all (it is a raw-bytes retrieval, like every adapter on
this contract), so a non-zero STATUS is *not* an HTTP failure and does not
raise there -- it is `parse()`'s job to check `STATUS` and refuse to treat
the body as data, surfacing it as a `ParseResult` warning (which blocks
readiness, per the adapter contract) rather than fabricating observations
from an error payload.

TIME-CODE PARSING (best-effort, documented limitation): e-Stat's `@time`
dimension code is table-specific -- annual, quarterly, monthly and
fiscal-year tables each encode it differently, and this repository has never
seen a real table's `CLASS_INF` to pin an exact convention down. Rather than
guess a table-specific code offset, `_resolve_time_code` only recognises
well-known, widely-documented e-Stat time-label shapes taken from
`CLASS_INF`'s human-readable `@name` for that code (`"2020年"`,
`"2020年04月"`, `"2020年04月05日"`) plus a few unambiguous plain-digit code
shapes (`YYYY`, `YYYYMM`, `YYYYMMDD`). A `@time` code this cannot resolve
produces a parser warning and the row is skipped -- never a fabricated date
(CLAUDE.md rule 29).

MULTI-DIMENSIONAL TABLES: a `statsDataId` can return more than one series at
once (e.g. sliced by area and category as well as time). Since no specific
table has been verified, this adapter cannot know which dimension, if any,
already disambiguates a single intended series. To stay both generic and
collision-free, every `@<dim>` attribute on a `VALUE` row other than
`@time`, `@tab` and `@unit` is appended, sorted, to `spec.series_id` (e.g.
``"estat.my_series#area=13000,cat01=A123"``) -- callers who configure a
`SeriesSpec` against a `statsDataId` that is already sliced to one series
will simply never see a suffix (no other `@<dim>` keys will vary).

AVAILABILITY: e-Stat gives no intraday publication timestamp, only the
`@time` period. `available_at` is therefore always produced by
`external.adapter.resolve_available_at()` using the parsed period and the
series' declared `CONSERVATIVE_DATE` lag -- never invented here.

VINTAGES: e-Stat's `getStatsData` exposes no revision/vintage concept (no
ALFRED-style realtime window). `vintage_time` and `revision_index` are
always `None`; only FRED (`adapters/fred.py`) is vintage-aware in Wave 1.

CREDENTIAL: `ESTAT_APP_ID` is read from the environment on every fetch and
never allowed to silently fall back to an unauthenticated request --
`_require_app_id` raises `EstatCredentialError` (a typed `AdapterError`)
naming the environment variable when it is unset. The id is baked into the
*actual* request's query parameters but never into anything persisted:
`FetchedPayload.url`/`.request_fingerprint` are built from a credential-free
parameter dict, never from the real, credentialed request URL.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, date, datetime, time
from typing import Any
from urllib.parse import urlencode

from turboedge.adapters.base import AdapterError, HttpClient
from turboedge.external.adapter import FetchedPayload, ParseResult, resolve_available_at
from turboedge.external.schemas import SeriesSpec
from turboedge.storage.schemas import ExternalObservation

_SOURCE_ID = "estat"
_PARSER_VERSION = "1"
_SOURCE_VERSION = "estat_getstatsdata_json_v3"
_DEFAULT_BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
_APP_ID_ENV_VAR = "ESTAT_APP_ID"

#: STATUS codes 0-2 are documented as "normal completion"; 100+ are errors
#: (fetched from the e-Stat API 3.0 parameter reference).
_SUCCESS_STATUS_MAX = 2

#: Dimension attribute keys never treated as a series-disambiguating
#: category: `@time` becomes `observation_time`, `@tab`/`@unit` are
#: tabulation/unit metadata, not category slices.
_NON_CATEGORY_DIM_KEYS = frozenset({"@time", "@tab", "@unit"})

_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)

__all__ = [
    "EstatAdapter",
    "EstatCredentialError",
    "parse_native_identifier",
    "parse_stats_data",
]


class EstatCredentialError(AdapterError):
    """`ESTAT_APP_ID` is not set. Never substituted with an unauthenticated
    request or a fabricated id -- every e-Stat request requires a real one."""


def _require_app_id() -> str:
    app_id = os.environ.get(_APP_ID_ENV_VAR)
    if not app_id:
        raise EstatCredentialError(
            f"{_APP_ID_ENV_VAR} is not set. e-Stat requires an application ID for every "
            "request (https://www.e-stat.go.jp/api/); refusing to make an unauthenticated "
            "request or substitute a fabricated id."
        )
    return app_id


def _public_url(base_url: str, params: dict[str, str]) -> str:
    """Build the archivable request URL from parameters that never include
    the credential -- callers must pass a dict with `appId` already
    excluded."""
    return f"{base_url}?{urlencode(params)}"


def _as_list(value: Any) -> list[Any]:
    """e-Stat's JSON mirrors an XML origin: a repeating element with exactly
    one member is a bare object, not a one-item list. Both shapes are
    normalised to a list here; anything else (missing/None/wrong type)
    normalises to an empty list rather than raising, leaving the caller to
    decide whether an empty result is itself a warning."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _class_time_lookup(class_inf: Any) -> dict[str, dict[str, str]]:
    """`{time-code -> its CLASS_INF attributes}` for the `"time"` dimension,
    or `{}` if `CLASS_INF` is absent/malformed -- callers must treat an empty
    lookup as "no metadata available", not as an error, since `metaGetFlg`
    could in principle be turned off by a caller and this adapter must still
    degrade to code-shape parsing rather than crash."""
    if not isinstance(class_inf, dict):
        return {}
    lookup: dict[str, dict[str, str]] = {}
    for class_obj in _as_list(class_inf.get("CLASS_OBJ")):
        if not isinstance(class_obj, dict) or class_obj.get("@id") != "time":
            continue
        for entry in _as_list(class_obj.get("CLASS")):
            if isinstance(entry, dict) and "@code" in entry:
                lookup[entry["@code"]] = entry
    return lookup


#: "2020年04月05日" / "2020年04月" / "2020年" -- e-Stat's standard
#: human-readable time-class label shapes.
_JAPANESE_YMD_RE = re.compile(r"^(\d{4})年(?:(\d{1,2})月(?:(\d{1,2})日)?)?$")


def _parse_time_label(label: str) -> date | None:
    """Best-effort parse of an e-Stat time label/code into a calendar date.
    Returns `None` (never a guess) for anything not matching a known shape --
    see module docstring "TIME-CODE PARSING"."""
    s = label.strip()

    m = _JAPANESE_YMD_RE.match(s)
    if m:
        year = int(m.group(1))
        month = int(m.group(2)) if m.group(2) else 1
        day = int(m.group(3)) if m.group(3) else 1
        try:
            return date(year, month, day)
        except ValueError:
            return None

    if re.fullmatch(r"\d{4}", s):
        return date(int(s), 1, 1)

    if re.fullmatch(r"\d{6}", s):
        year, month = int(s[:4]), int(s[4:6])
        if 1 <= month <= 12:
            try:
                return date(year, month, 1)
            except ValueError:
                return None
        return None

    if re.fullmatch(r"\d{8}", s):
        year, month, day = int(s[:4]), int(s[4:6]), int(s[6:8])
        try:
            return date(year, month, day)
        except ValueError:
            return None

    return None


def _resolve_time_code(time_code: str, time_lookup: dict[str, dict[str, str]]) -> date | None:
    """Map a `@time` dimension code to a calendar date: prefer CLASS_INF's
    human-readable `@name` for that code, fall back to parsing the raw code
    itself if it happens to already be a plain-digit shape. `None` (never a
    guess) if neither resolves."""
    entry = time_lookup.get(time_code)
    if entry is not None:
        name = entry.get("@name")
        if name:
            parsed = _parse_time_label(name)
            if parsed is not None:
                return parsed
    return _parse_time_label(time_code)


def parse_stats_data(
    content: bytes,
    spec: SeriesSpec,
    *,
    retrieved_at: datetime,
) -> ParseResult:
    """Parse one `getStatsData` JSON payload into `ExternalObservation`s.

    Pure and network-free. A non-zero `RESULT.STATUS` is treated as an error
    and the body is never parsed as data (module docstring "ERROR ENVELOPE").
    Any other structural surprise -- not valid JSON, a missing
    `GET_STATS_DATA`/`STATISTICAL_DATA`/`DATA_INF.VALUE`, a `VALUE` row
    missing `@time` or `$`, an unresolvable `@time` code, a non-numeric `$`
    -- is reported as a warning and the affected row (or the whole payload)
    is skipped, never guessed at. An empty `$` string is the one expected
    "no data published" shape and is skipped silently.
    """
    try:
        raw: Any = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: payload is not valid UTF-8 JSON: {exc}",),
            missing_series=(spec.series_id,),
        )

    if not isinstance(raw, dict) or "GET_STATS_DATA" not in raw:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: response missing top-level 'GET_STATS_DATA' key",),
            missing_series=(spec.series_id,),
        )
    root = raw["GET_STATS_DATA"]
    if not isinstance(root, dict):
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: 'GET_STATS_DATA' is not a JSON object",),
            missing_series=(spec.series_id,),
        )

    result = root.get("RESULT")
    if not isinstance(result, dict) or "STATUS" not in result:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: response missing 'GET_STATS_DATA.RESULT.STATUS'",),
            missing_series=(spec.series_id,),
        )
    try:
        status = int(result["STATUS"])
    except (TypeError, ValueError):
        status_raw = result["STATUS"]
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: RESULT.STATUS is not an integer: {status_raw!r}",),
            missing_series=(spec.series_id,),
        )
    if status > _SUCCESS_STATUS_MAX:
        error_msg = result.get("ERROR_MSG", "")
        return ParseResult(
            observations=[],
            warnings=(
                f"{spec.qualified_id}: e-Stat STATUS={status} ({error_msg!r}); "
                "not parsing the response body as data",
            ),
            missing_series=(spec.series_id,),
        )

    stats_data = root.get("STATISTICAL_DATA")
    if not isinstance(stats_data, dict):
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: response missing 'STATISTICAL_DATA'",),
            missing_series=(spec.series_id,),
        )
    data_inf = stats_data.get("DATA_INF")
    if not isinstance(data_inf, dict) or "VALUE" not in data_inf:
        return ParseResult(
            observations=[],
            warnings=(f"{spec.qualified_id}: response missing 'STATISTICAL_DATA.DATA_INF.VALUE'",),
            missing_series=(spec.series_id,),
        )

    time_lookup = _class_time_lookup(stats_data.get("CLASS_INF"))
    warnings: list[str] = []
    observations: list[ExternalObservation] = []

    for row in _as_list(data_inf["VALUE"]):
        if not isinstance(row, dict):
            warnings.append(f"{spec.qualified_id}: VALUE entry is not a JSON object: {row!r}")
            continue
        if "@time" not in row:
            warnings.append(f"{spec.qualified_id}: VALUE entry missing '@time': {row!r}")
            continue
        if "$" not in row:
            warnings.append(f"{spec.qualified_id}: VALUE entry missing '$' (value): {row!r}")
            continue

        time_code = row["@time"]
        observation_day = _resolve_time_code(time_code, time_lookup)
        if observation_day is None:
            warnings.append(
                f"{spec.qualified_id}: could not resolve a calendar date for "
                f"@time={time_code!r} -- unrecognised time-code/label shape"
            )
            continue

        stripped = str(row["$"]).strip()
        if stripped == "":
            # An explicitly empty value cell: e-Stat's documented shape for
            # "no figure published for this cell". Normal, not a warning.
            continue
        try:
            value = float(stripped)
        except _PARSE_ERROR_TYPES as exc:
            warnings.append(
                f"{spec.qualified_id}: unparseable VALUE '$'={row['$']!r} at "
                f"@time={time_code!r}: {exc}. e-Stat's non-numeric missing-data "
                "markers (if any) are not documented in the fetched API reference, "
                "so this is flagged rather than silently treated as missing."
            )
            continue

        other_dims = {
            key[1:]: val
            for key, val in row.items()
            if key.startswith("@") and key not in _NON_CATEGORY_DIM_KEYS
        }
        series_id = spec.series_id
        if other_dims:
            suffix = ",".join(f"{k}={v}" for k, v in sorted(other_dims.items()))
            series_id = f"{spec.series_id}#{suffix}"

        available_at, precision = resolve_available_at(spec, observation_day)
        observations.append(
            ExternalObservation(
                series_id=series_id,
                value=value,
                unit=spec.unit,
                frequency=spec.frequency,
                source_version=_SOURCE_VERSION,
                observation_time=datetime.combine(observation_day, time(0, 0), tzinfo=UTC),
                available_at=available_at,
                retrieved_at=retrieved_at,
                source=_SOURCE_ID,
                parser_version=_PARSER_VERSION,
                quality_score=1.0,
                is_stale=False,
                # e-Stat's getStatsData response carries no per-observation
                # release timestamp (module docstring) -- never fabricated.
                source_release_time=None,
                vintage_time=None,  # e-Stat exposes no revision/vintage concept
                availability_precision=str(precision),
                revision_index=None,
            )
        )

    return ParseResult(observations=observations, warnings=tuple(warnings), missing_series=())


def parse_native_identifier(raw: str) -> tuple[str, dict[str, str]]:
    """`"<statsDataId>"` or `"<statsDataId>|<cdKey=value,cdKey=value>"`.

    The filter segment exists because an e-Stat table is not a series. The
    2020-base CPI table (`0003427113`) carries every item, every region and
    every month at once, and this adapter turns each distinct dimension
    combination into its own `series_id`. Fetching it unfiltered would write
    an uncontrolled number of series from a single request -- so the catalog
    narrows it, and an entry that does not narrow it says so deliberately.

    Only `cd*` parameters are accepted. Anything else is either a paging
    control this adapter owns, or the credential, and neither belongs in a
    configured identifier.
    """
    head, _, tail = raw.partition("|")
    stats_data_id = head.strip()
    if not stats_data_id:
        raise AdapterError(f"estat: native_identifier {raw!r} has no statsDataId")
    filters: dict[str, str] = {}
    for chunk in tail.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, sep, value = chunk.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            raise AdapterError(f"estat: filter {chunk!r} in {raw!r} is not 'cdKey=value'")
        if not key.startswith("cd"):
            raise AdapterError(
                f"estat: only 'cd*' filters may be configured, not {key!r}; "
                "paging is this adapter's own concern and the credential is never "
                "part of an identifier"
            )
        filters[key] = value
    return stats_data_id, filters


class EstatAdapter:
    """Fetches one e-Stat `statsDataId` table's full data.

    Fully generic: `spec.native_identifier` is the `statsDataId` sent to the
    endpoint; `spec.series_id` is TurboEdge's own id (suffixed with any
    non-time dimension codes that vary within the table -- see module
    docstring "MULTI-DIMENSIONAL TABLES").
    """

    def __init__(self, http: HttpClient, *, base_url: str = _DEFAULT_BASE_URL) -> None:
        self._http = http
        self._base_url = base_url

    @property
    def source_id(self) -> str:
        return _SOURCE_ID

    @property
    def parser_version(self) -> str:
        return _PARSER_VERSION

    def fetch(self, spec: SeriesSpec, *, since: date | None = None) -> FetchedPayload:
        """Retrieve one e-Stat table, narrowed by whatever filters the
        `native_identifier` declares.

        `since` is accepted for protocol compliance but not translated into
        a `cdTimeFrom` filter: e-Stat's `@time` codes are table-specific
        (module docstring "TIME-CODE PARSING") and this adapter has never
        seen a real table's `CLASS_INF` to learn its code convention --
        guessing a `cdTimeFrom` value from a calendar date could silently
        narrow the request to the wrong period instead of just being
        inefficient. Always fetching the whole table can only return *more*
        than `since` onward would need, never less, so the protocol's "never
        return less than since onwards" contract still holds.
        """
        app_id = _require_app_id()
        stats_data_id, filters = parse_native_identifier(spec.native_identifier)
        public_params: dict[str, str] = {"statsDataId": stats_data_id, **filters}
        request_params = {**public_params, "appId": app_id}

        retrieved_at = datetime.now(UTC)
        # See `adapters/ecb_data.py` / `adapters/fred.py`: `_request` is the
        # one place HttpClient applies retry, per-host rate limiting and the
        # honest User-Agent, and hands back the real `httpx.Response` so
        # `http_status`/`content_type` are recorded rather than invented.
        # `.request.url` is NOT used for anything persisted -- it carries the
        # real, credentialed query string; `url`/`request_fingerprint` are
        # built only from `public_params`.
        response = self._http._request("GET", self._base_url, params=request_params)

        public_url = _public_url(self._base_url, public_params)
        return FetchedPayload(
            source=_SOURCE_ID,
            dataset=spec.series_id,
            url=public_url,
            content=response.content,
            http_status=response.status_code,
            content_type=response.headers.get("content-type", ""),
            retrieved_at=retrieved_at,
            request_fingerprint=f"GET {public_url}",
        )

    def parse(self, payload: FetchedPayload, spec: SeriesSpec) -> ParseResult:
        return parse_stats_data(payload.content, spec, retrieved_at=payload.retrieved_at)
