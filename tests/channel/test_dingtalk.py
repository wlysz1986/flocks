"""
Tests for the DingTalk active-outbound send library.

Layout:
  - send library  → flocks.channel.builtin.dingtalk.{config,client,send}
  - inbound owner → .flocks/plugins/channels/dingtalk/dingtalk.py (Node.js)

Only the OAPI app-robot ("stream/app push") path is supported; custom
group-robot incoming webhooks are intentionally out of scope.

The builtin package does NOT register a ChannelPlugin (to avoid id
collisions with the local plugin), so registry-side tests are absent.
The local Node.js plugin is owned separately and not exercised here.
"""

from __future__ import annotations

import importlib.util
import json
from unittest.mock import AsyncMock, patch

import pytest

from flocks.channel.builtin.dingtalk.client import (
    DingTalkApiError,
    ensure_api_success,
)
from flocks.channel.builtin.dingtalk.config import (
    list_account_configs,
    resolve_account_credentials,
    resolve_target_kind,
    strip_target_prefix,
)
from flocks.channel.builtin.dingtalk.send import (
    build_app_payload,
    send_message_app,
)


# ------------------------------------------------------------------
# config helpers
# ------------------------------------------------------------------

class TestConfigHelpers:
    def test_strip_target_prefix(self):
        assert strip_target_prefix("user:abc") == "abc"
        assert strip_target_prefix("chat:cidXYZ") == "cidXYZ"
        assert strip_target_prefix("plain") == "plain"
        assert strip_target_prefix("") == ""

    def test_resolve_target_kind(self):
        assert resolve_target_kind("user:zhangsan") == "user"
        assert resolve_target_kind("chat:cid1") == "group"
        assert resolve_target_kind("cidABC123") == "group"
        assert resolve_target_kind("zhangsan") == "user"
        assert resolve_target_kind("") == "user"

    def test_resolve_account_credentials_default(self):
        cfg = {"appKey": "K", "appSecret": "S", "robotCode": "R"}
        assert resolve_account_credentials(cfg, None) == ("K", "S", "R")
        assert resolve_account_credentials(cfg, "default") == ("K", "S", "R")

    def test_resolve_account_credentials_override(self):
        cfg = {
            "appKey": "K", "appSecret": "S", "robotCode": "R",
            "accounts": {
                "alice": {"appKey": "K2", "appSecret": "S2"},
            },
        }
        assert resolve_account_credentials(cfg, "alice") == ("K2", "S2", "R")

    def test_resolve_account_credentials_accepts_client_id_alias(self):
        # DingTalk Stream config uses clientId/clientSecret; the send library
        # must transparently treat them as appKey/appSecret.
        cfg = {"clientId": "K", "clientSecret": "S", "robotCode": "R"}
        assert resolve_account_credentials(cfg, None) == ("K", "S", "R")

    def test_resolve_account_credentials_app_key_wins_over_alias(self):
        cfg = {
            "appKey": "PRIMARY", "clientId": "FALLBACK",
            "appSecret": "PS", "clientSecret": "FS",
            "robotCode": "R",
        }
        assert resolve_account_credentials(cfg, None) == ("PRIMARY", "PS", "R")

    def test_list_account_configs_top_level_app(self):
        cfg = {"appKey": "k", "appSecret": "s", "robotCode": "r"}
        accounts = list_account_configs(cfg, require_credentials=True)
        assert len(accounts) == 1
        assert accounts[0]["_account_id"] == "default"

    def test_list_account_configs_accepts_client_id_alias(self):
        cfg = {"clientId": "k", "clientSecret": "s"}
        accounts = list_account_configs(cfg, require_credentials=True)
        assert len(accounts) == 1

    def test_list_account_configs_skips_disabled(self):
        cfg = {
            "robotCode": "r",
            "accounts": {
                "alice": {"appKey": "k", "appSecret": "s", "enabled": False},
                "bob":   {"appKey": "k2", "appSecret": "s2"},
            },
        }
        accounts = list_account_configs(cfg, require_credentials=True)
        ids = {a["_account_id"] for a in accounts}
        assert ids == {"bob"}

    def test_list_account_configs_filters_missing_credentials(self):
        cfg = {
            "accounts": {
                "alice": {"appKey": "k"},  # missing appSecret
            },
        }
        accounts = list_account_configs(cfg, require_credentials=True)
        assert accounts == []


# ------------------------------------------------------------------
# Payload builder
# ------------------------------------------------------------------

class TestAppPayloadBuilder:
    def test_plain_text(self):
        msg_key, msg_param = build_app_payload("hello", "plain")
        assert msg_key == "sampleText"
        assert json.loads(msg_param) == {"content": "hello"}

    def test_markdown_default(self):
        msg_key, msg_param = build_app_payload("# 标题\n正文", "auto")
        assert msg_key == "sampleMarkdown"
        param = json.loads(msg_param)
        assert param["title"] == "标题"
        assert "正文" in param["text"]

    def test_markdown_uses_fallback_title_when_blank(self):
        msg_key, msg_param = build_app_payload("\n\n   ", "card")
        param = json.loads(msg_param)
        assert msg_key == "sampleMarkdown"
        assert param["title"] == "通知"


# ------------------------------------------------------------------
# send_message_app — routing between user and group
# ------------------------------------------------------------------

class TestSendApp:
    async def test_user_target_uses_oto_endpoint(self):
        captured: dict = {}

        async def _fake_request(method, path, *, config, account_id, json_body=None, params=None):
            captured["path"] = path
            captured["body"] = json_body
            return {"processQueryKey": "pqk-1"}

        cfg = {"appKey": "k", "appSecret": "s", "robotCode": "r"}
        with patch(
            "flocks.channel.builtin.dingtalk.send.api_request_for_account",
            new=AsyncMock(side_effect=_fake_request),
        ):
            result = await send_message_app(
                config=cfg, to="user:zhangsan", text="hello",
            )

        assert captured["path"] == "/v1.0/robot/oToMessages/batchSend"
        assert captured["body"]["userIds"] == ["zhangsan"]
        assert captured["body"]["msgKey"] == "sampleMarkdown"
        assert captured["body"]["robotCode"] == "r"
        assert result["message_id"] == "pqk-1"
        assert result["chat_id"] == "zhangsan"

    async def test_chat_target_uses_group_endpoint(self):
        captured: dict = {}

        async def _fake_request(method, path, *, config, account_id, json_body=None, params=None):
            captured["path"] = path
            captured["body"] = json_body
            return {"processQueryKey": "pqk-2"}

        cfg = {
            "appKey": "k", "appSecret": "s", "robotCode": "r",
            "renderMode": "plain",
        }
        with patch(
            "flocks.channel.builtin.dingtalk.send.api_request_for_account",
            new=AsyncMock(side_effect=_fake_request),
        ):
            await send_message_app(
                config=cfg, to="chat:cid_GROUP_1", text="hi all",
            )

        assert captured["path"] == "/v1.0/robot/groupMessages/send"
        assert captured["body"]["openConversationId"] == "cid_GROUP_1"
        assert captured["body"]["msgKey"] == "sampleText"

    async def test_app_send_works_with_client_id_alias(self):
        # Reuses the DingTalk Stream credential fields end-to-end.
        captured: dict = {}

        async def _fake_request(method, path, *, config, account_id, json_body=None, params=None):
            captured["path"] = path
            captured["body"] = json_body
            return {"processQueryKey": "pqk-3"}

        cfg = {"clientId": "ck", "clientSecret": "cs", "robotCode": "r"}
        with patch(
            "flocks.channel.builtin.dingtalk.send.api_request_for_account",
            new=AsyncMock(side_effect=_fake_request),
        ):
            await send_message_app(config=cfg, to="user:u1", text="hello")

        assert captured["body"]["userIds"] == ["u1"]
        assert captured["body"]["robotCode"] == "r"

    async def test_missing_robot_code_raises(self):
        cfg = {"appKey": "k", "appSecret": "s"}
        with pytest.raises(ValueError, match="robotCode not configured"):
            await send_message_app(config=cfg, to="user:abc", text="hi")

    async def test_empty_target_raises(self):
        cfg = {"appKey": "k", "appSecret": "s", "robotCode": "r"}
        with pytest.raises(ValueError, match="empty target"):
            await send_message_app(config=cfg, to="", text="hi")

    async def test_long_text_chunks_into_multiple_calls(self):
        calls: list[dict] = []

        async def _fake_request(method, path, *, config, account_id, json_body=None, params=None):
            calls.append({"path": path, "body": json_body})
            return {"processQueryKey": f"pqk-{len(calls)}"}

        cfg = {
            "appKey": "k", "appSecret": "s", "robotCode": "r",
            "textChunkLimit": 10,
        }
        with patch(
            "flocks.channel.builtin.dingtalk.send.api_request_for_account",
            new=AsyncMock(side_effect=_fake_request),
        ):
            await send_message_app(
                config=cfg, to="user:u1",
                text="abcde\nfghij\nklmno\npqrst",
            )

        assert len(calls) >= 2


# ------------------------------------------------------------------
# Client error parsing
# ------------------------------------------------------------------

class TestEnsureApiSuccess:
    def test_legacy_errcode_zero_passes(self):
        data = ensure_api_success({"errcode": 0, "errmsg": "ok"}, context="ctx")
        assert data["errcode"] == 0

    def test_legacy_errcode_non_zero_raises(self):
        with pytest.raises(DingTalkApiError) as exc:
            ensure_api_success(
                {"errcode": 310000, "errmsg": "keywords not in content"},
                context="oapi",
            )
        assert exc.value.code == "310000"

    def test_v1_code_field_raises(self):
        with pytest.raises(DingTalkApiError) as exc:
            ensure_api_success(
                {"code": "InvalidParameter", "message": "bad robotCode"},
                context="oapi",
                http_status=400,
            )
        assert exc.value.code == "InvalidParameter"
        assert exc.value.http_status == 400

    def test_throttling_marked_retryable(self):
        with pytest.raises(DingTalkApiError) as exc:
            ensure_api_success(
                {"code": "Throttling.Api", "message": "rate limit exceeded"},
                context="oapi",
                http_status=429,
            )
        assert exc.value.retryable is True

    def test_success_payload_passes_through(self):
        # v1.0 success payloads typically lack a ``code`` field altogether.
        data = ensure_api_success(
            {"processQueryKey": "abc"},
            context="oapi",
            http_status=200,
        )
        assert data["processQueryKey"] == "abc"


# ------------------------------------------------------------------
# Builtin package no longer registers a ChannelPlugin
# ------------------------------------------------------------------

class TestBuiltinHasNoChannelClass:
    def test_no_dingtalk_in_builtin_registry(self):
        from flocks.channel.registry import ChannelRegistry
        reg = ChannelRegistry()
        reg._register_builtin_channels()
        # The builtin package intentionally exposes only a send library;
        # the dingtalk id is owned by the project-local plugin.
        assert reg.get("dingtalk") is None

    def test_builtin_package_has_no_channel_module(self):
        spec = importlib.util.find_spec(
            "flocks.channel.builtin.dingtalk.channel"
        )
        assert spec is None
