"""OpenAI-compatible judge for comparing teacher and student answers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
from typing import Any

import httpx
from dotenv import load_dotenv

from src.models.cache import SQLiteCache


LOGGER = logging.getLogger(__name__)

SCORE_FIELDS = (
    "a_correctness",
    "a_relevance",
    "a_helpfulness",
    "a_overall",
    "b_correctness",
    "b_relevance",
    "b_helpfulness",
    "b_overall",
)

SYSTEM = """
Evaluate two candidate answers to the user prompt.

The reference answer is guidance and does not require identical wording.

Return exactly one complete valid JSON object with:
- a_correctness: integer 1..5
- a_relevance: integer 1..5
- a_helpfulness: integer 1..5
- a_overall: integer 1..5
- b_correctness: integer 1..5
- b_relevance: integer 1..5
- b_helpfulness: integer 1..5
- b_overall: integer 1..5
- preferred: "A", "B", or "tie"
- rationale: a short explanation

Ensure that the closing brace is present.
Do not answer the user prompt.
Do not return Markdown.
Return JSON only.
""".strip()


class AnswerJudge:
    def __init__(self, cfg: dict[str, Any]):
        load_dotenv(
            cfg.get("env_file", ".env"),
            override=False,
        )

        self.cfg = cfg

        self.base_url = os.getenv(
            cfg["base_url_env"],
            "",
        ).rstrip("/")

        self.model = os.getenv(
            cfg["model_env"],
            "",
        )

        self.token = os.getenv(
            cfg["token_env"],
            "",
        )

        missing = []

        if not self.base_url:
            missing.append(cfg["base_url_env"])

        if not self.model:
            missing.append(cfg["model_env"])

        if not self.token:
            missing.append(cfg["token_env"])

        if missing:
            raise RuntimeError(
                "Missing judge environment variables: "
                + ", ".join(missing)
            )

        self.url = (
            self.base_url
            + cfg.get(
                "endpoint",
                "/v1/chat/completions",
            )
        )

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

        self.timeout = float(
            cfg.get("timeout_seconds", 180)
        )

        self.max_retries = int(
            cfg.get("max_retries", 5)
        )

        self.parse_retries = int(
            cfg.get("parse_retries", 2)
        )

        self.cache = SQLiteCache(
            cfg["cache_path"]
        )

    @staticmethod
    def _parse(raw_content: str) -> dict[str, Any]:
        if not isinstance(raw_content, str):
            raise TypeError(
                "Judge content must be a string, "
                f"got {type(raw_content).__name__}"
            )

        text = raw_content.strip()

        if not text:
            raise ValueError(
                "Judge returned empty content"
            )

        text = (
            text
            .removeprefix("```json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        start = text.find("{")
        end = text.rfind("}")

        if start < 0 or end <= start:
            raise ValueError(
                "Judge returned no complete JSON object: "
                f"{text[:500]!r}"
            )

        value = json.loads(text[start:end + 1])

        if not isinstance(value, dict):
            raise TypeError(
                "Judge JSON result must be an object, "
                f"got {type(value).__name__}"
            )

        normalized: dict[str, Any] = {}

        for field in SCORE_FIELDS:
            if field not in value:
                raise ValueError(
                    f"Judge response is missing field {field!r}"
                )

            if isinstance(value[field], bool):
                raise ValueError(
                    f"Invalid boolean value for {field}: "
                    f"{value[field]!r}"
                )

            try:
                score = int(value[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid score for {field}: "
                    f"{value[field]!r}"
                ) from exc

            if not 1 <= score <= 5:
                raise ValueError(
                    f"Score outside 1..5 for {field}: "
                    f"{score}"
                )

            normalized[field] = score

        preferred = str(
            value.get("preferred", "")
        ).strip().upper()

        if preferred == "TIE":
            preferred = "tie"

        if preferred not in {"A", "B", "tie"}:
            raise ValueError(
                "Invalid preferred value: "
                f"{value.get('preferred')!r}"
            )

        normalized["preferred"] = preferred
        normalized["rationale"] = str(
            value.get("rationale", "")
        ).strip()[:2000]

        return normalized

    async def score(
        self,
        prompt: str,
        reply: str,
        tgo: str,
        sgo: str,
    ) -> dict[str, Any]:
        prompt = str(prompt or "")
        reply = str(reply or "")
        tgo = str(tgo or "")
        sgo = str(sgo or "")

        swapped = (
            int(
                hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                16,
            )
            % 2
            == 1
        )

        if swapped:
            candidate_a = sgo
            candidate_b = tgo
            first = "sgo"
            second = "tgo"
            judge_order = "SGO,TGO"
        else:
            candidate_a = tgo
            candidate_b = sgo
            first = "tgo"
            second = "sgo"
            judge_order = "TGO,SGO"

        user_content = (
            f"PROMPT:\n{prompt}\n\n"
            f"REFERENCE:\n{reply}\n\n"
            f"CANDIDATE A:\n{candidate_a}\n\n"
            f"CANDIDATE B:\n{candidate_b}"
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "temperature": float(
                self.cfg.get("temperature", 0)
            ),
            "max_tokens": int(
                self.cfg.get("max_tokens", 2048)
            ),
        }

        if self.cfg.get(
            "disable_thinking",
            True,
        ):
            payload["chat_template_kwargs"] = {
                "enable_thinking": False,
            }

        if self.cfg.get(
            "send_top_level_enable_thinking",
            False,
        ):
            payload["enable_thinking"] = False

        if self.cfg.get(
            "response_format_json",
            True,
        ):
            payload["response_format"] = {
                "type": "json_object",
            }

        key = self.cache.key(payload)
        data = self.cache.get(key)

        if data is None:
            total_attempts = max(
                1,
                self.max_retries + 1,
                self.parse_retries + 1,
            )

            last_error: Exception | None = None

            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:
                for attempt in range(total_attempts):
                    try:
                        response = await client.post(
                            self.url,
                            headers=self.headers,
                            json=payload,
                        )

                        response.raise_for_status()
                        body = response.json()

                        choices = body.get("choices") or []

                        if not choices:
                            raise RuntimeError(
                                "Judge returned no choices: "
                                f"{json.dumps(body)[:1000]}"
                            )

                        choice = choices[0]
                        message = (
                            choice.get("message")
                            or {}
                        )

                        content = message.get("content")
                        content_source = "content"

                        if not (
                            isinstance(content, str)
                            and content.strip()
                        ):
                            content = message.get(
                                "reasoning_content"
                            )
                            content_source = (
                                "reasoning_content"
                            )

                        if not (
                            isinstance(content, str)
                            and content.strip()
                        ):
                            raise ValueError(
                                "Judge returned empty content; "
                                f"finish_reason="
                                f"{choice.get('finish_reason')!r}; "
                                f"message_keys="
                                f"{list(message.keys())}; "
                                f"usage={body.get('usage')!r}"
                            )

                        LOGGER.info(
                            "Judge response: "
                            "attempt=%d/%d "
                            "finish_reason=%r "
                            "source=%s "
                            "content_len=%d "
                            "reasoning_len=%d "
                            "usage=%r",
                            attempt + 1,
                            total_attempts,
                            choice.get("finish_reason"),
                            content_source,
                            len(
                                message.get("content")
                                or ""
                            ),
                            len(
                                message.get(
                                    "reasoning_content"
                                )
                                or ""
                            ),
                            body.get("usage"),
                        )

                        data = self._parse(content)
                        self.cache.put(key, data)
                        break

                    except Exception as exc:
                        last_error = exc

                        LOGGER.warning(
                            "Judge attempt failed "
                            "(%d/%d): %s",
                            attempt + 1,
                            total_attempts,
                            exc,
                        )

                        if attempt + 1 >= total_attempts:
                            raise RuntimeError(
                                "Judge failed after "
                                f"{total_attempts} attempts: "
                                f"{last_error}"
                            ) from last_error

                        await asyncio.sleep(
                            min(
                                30.0,
                                2**attempt
                                + random.random(),
                            )
                        )

        if not isinstance(data, dict):
            raise TypeError(
                "Invalid cached judge result: "
                f"{type(data).__name__}"
            )

        out: dict[str, Any] = {
            "judge_order": judge_order,
            "judge_rationale": str(
                data.get("rationale", "")
            ),
        }

        for metric in (
            "correctness",
            "relevance",
            "helpfulness",
            "overall",
        ):
            out[f"{first}_judge_{metric}"] = data[
                f"a_{metric}"
            ]

            out[f"{second}_judge_{metric}"] = data[
                f"b_{metric}"
            ]

        preferred = str(
            data["preferred"]
        ).upper()

        if preferred == "A":
            out["judge_preferred"] = (
                first.upper()
            )
        elif preferred == "B":
            out["judge_preferred"] = (
                second.upper()
            )
        else:
            out["judge_preferred"] = "tie"

        return out