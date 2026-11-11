"""The direct HTTP backends against a real local server: the exact request, the reply, usage, errors and keys."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from memshort import backends, doctor, installation, om


class Server:
    """A local HTTP server that records each request and answers from a queue of (status, body) pairs."""

    def __init__(self):
        self.requests: list[dict] = []
        self.replies: list[tuple[int, dict]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                server.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": json.loads(body) if body else None,
                    }
                )
                status, reply = server.replies.pop(0)
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _answer

            def log_message(self, format, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for variable in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MEMSHORT_OPENAI_API_KEY",
        "MEMSHORT_ANTHROPIC_API_KEY",
        "LAB_KEY",
        om.ENV_MODEL,
    ):
        monkeypatch.delenv(variable, raising=False)


def openai_reply(text="<observations>\n* 🟡 ok\n</observations>", **usage):
    return {
        "choices": [
            {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": usage
        or {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 100},
        },
    }


def anthropic_reply(text="<observations>\n* 🟡 ok\n</observations>"):
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 20,
            "output_tokens": 30,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 0,
        },
    }


def test_openai_sends_only_the_prompt_and_reads_text_and_usage(server, monkeypatch):
    monkeypatch.setenv("LAB_KEY", "sk-test-value")
    endpoint = backends.endpoint_of(
        "lab",
        {
            "api": "openai",
            "base_url": server.url + "/v1",
            "api_key_env": "LAB_KEY",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )
    server.replies.append((200, openai_reply()))
    text, usage = backends.complete(
        backends.route("lab/qwen3", {"lab": endpoint}), "SYSTEM", "INPUT", 10
    )
    assert text.startswith("<observations>")
    (req,) = server.requests
    assert (req["method"], req["path"]) == ("POST", "/v1/chat/completions")
    assert req["headers"]["authorization"] == "Bearer sk-test-value"
    # No tools, no history, no harness prompt: exactly memshort's system prompt and one user message.
    assert req["body"] == {
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "INPUT"},
        ],
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert usage["input"] == 20 and usage["cacheRead"] == 100
    assert usage["output"] == 30 and usage["totalTokens"] == 150
    assert usage["cost"] == {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "total": 0,
    }


def test_anthropic_sends_a_cached_system_prompt_and_reads_usage(server, monkeypatch):
    monkeypatch.setenv("MEMSHORT_ANTHROPIC_API_KEY", "sk-ant-memshort")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-harness")
    endpoint = backends.Endpoint(
        "anthropic",
        server.url,
        backends.BUILTINS["anthropic"].api_key_env,
        key_required=True,
    )
    server.replies.append((200, anthropic_reply()))
    text, usage = backends.complete(
        backends.route("x/claude-haiku-4-5", {"x": endpoint}), "SYSTEM", "INPUT", 10
    )
    (req,) = server.requests
    assert req["path"] == "/v1/messages"
    # memshort's own variable wins, so compaction need not use the key the harness bills.
    assert req["headers"]["x-api-key"] == "sk-ant-memshort"
    assert req["headers"]["anthropic-version"] == backends.ANTHROPIC_VERSION
    assert "authorization" not in req["headers"]
    assert req["body"] == {
        "model": "claude-haiku-4-5",
        "max_tokens": backends.DEFAULT_MAX_TOKENS,
        "system": [
            {"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": "INPUT"}],
    }
    assert text.startswith("<observations>")
    assert (usage["input"], usage["output"], usage["cacheRead"]) == (20, 30, 1000)
    assert usage["totalTokens"] == 1050


def test_a_hosted_endpoint_without_its_key_fails_before_sending(server):
    endpoint = backends.Endpoint("openai", server.url, ("LAB_KEY",), key_required=True)
    with pytest.raises(backends.BackendError, match="no API key: set LAB_KEY"):
        backends.complete(backends.route("x/m", {"x": endpoint}), "s", "p", 10)
    assert server.requests == []


def test_a_local_server_needs_no_key(server):
    endpoint = backends.Endpoint("openai", server.url)
    server.replies.append((200, openai_reply("hi", prompt_tokens=5)))
    assert (
        backends.complete(backends.route("x/m", {"x": endpoint}), "s", "p", 10)[0]
        == "hi"
    )
    assert "authorization" not in server.requests[0]["headers"]


def test_an_http_error_shows_the_provider_message_but_never_the_key(
    server, monkeypatch
):
    monkeypatch.setenv("LAB_KEY", "sk-very-secret")
    endpoint = backends.Endpoint("openai", server.url, ("LAB_KEY",), key_required=True)
    server.replies.append(
        (401, {"error": {"message": "Incorrect API key provided: sk-very-secret"}})
    )
    with pytest.raises(backends.BackendError) as err:
        backends.complete(backends.route("x/m", {"x": endpoint}), "s", "p", 10)
    # The provider's message says what is wrong, so it is shown, but never with the key in it.
    assert "HTTP 401" in str(err.value) and "Incorrect API key provided: [key]" in str(
        err.value
    )
    assert "sk-very-secret" not in str(err.value)


def test_a_truncated_reply_is_an_error(server):
    endpoint = backends.Endpoint("openai", server.url)
    reply = openai_reply()
    reply["choices"][0]["finish_reason"] = "length"
    server.replies.append((200, reply))
    with pytest.raises(backends.BackendError, match="stopped: length"):
        backends.complete(backends.route("x/m", {"x": endpoint}), "s", "p", 10)


def test_a_redirect_is_not_followed(server):
    endpoint = backends.Endpoint("openai", server.url)
    server.replies.append((302, {}))
    with pytest.raises(backends.BackendError, match="HTTP 302"):
        backends.complete(backends.route("x/m", {"x": endpoint}), "s", "p", 10)
    assert len(server.requests) == 1


def test_om_runs_a_whole_observation_on_a_direct_endpoint(
    server, monkeypatch, tmp_path
):
    path = installation.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "features": ["compact"],
                "harnesses": ["pi"],
                "model": "lab/qwen3",
                "endpoints": {"lab": {"api": "openai", "base_url": server.url}},
            }
        )
    )
    server.replies.append(
        (
            200,
            openai_reply(
                "<observations>\n* 🔴 user wants X\n</observations>\n<current-task>\ndo X\n</current-task>"
            ),
        )
    )
    out = om.run({"log": "", "task": "", "transcript": "[User]: I want X"})
    assert "user wants X" in out["log"] and out["task"] == "do X"
    assert out["usage"]["totalTokens"] == 150
    assert (
        "<conversation>\n[User]: I want X\n</conversation>"
        in (server.requests[0]["body"]["messages"][1]["content"])
    )


def test_om_does_not_retry_a_missing_key(monkeypatch, tmp_path):
    path = installation.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "features": ["compact"],
                "harnesses": ["pi"],
                "model": "openai/gpt-x",
            }
        )
    )
    with pytest.raises(
        om.ModelError,
        match="openai/gpt-x: no API key: set MEMSHORT_OPENAI_API_KEY or OPENAI_API_KEY",
    ):
        om.complete("observer", "hello")


def test_doctor_checks_key_and_model_list_and_probes_on_request(server, monkeypatch):
    monkeypatch.setenv("LAB_KEY", "sk-hidden")
    path = installation.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "features": ["compact"],
                "harnesses": ["pi"],
                "model": "lab/qwen3",
                "reflector_model": "lab/absent",
                "endpoints": {
                    "lab": {
                        "api": "openai",
                        "base_url": server.url,
                        "api_key_env": "LAB_KEY",
                    }
                },
            }
        )
    )
    server.replies += [
        (200, {"data": [{"id": "qwen3"}, {"id": "gemma"}]}),
        (200, openai_reply("READY")),
        (200, {"data": [{"id": "qwen3"}, {"id": "gemma"}]}),
    ]
    report = doctor.check(probe=True)
    out = report.render()
    assert "sk-hidden" not in out
    assert "API key (lab/qwen3): set in LAB_KEY" in out
    assert f"endpoint (lab/qwen3): {server.url} lists qwen3" in out
    assert "probe (lab/qwen3): replied" in out
    assert "does not list absent; it lists gemma, qwen3" in out
    assert not report.ok
    # The failed model is not probed.
    assert [r["path"] for r in server.requests] == [
        "/models",
        "/chat/completions",
        "/models",
    ]


@pytest.mark.parametrize(
    ("model", "reason"),
    [
        ("pi/deepseek", "pi/<provider>/<model>"),
        ("openai/-rf", "one word"),
        ("openai/a b", "one word"),
        ("deepseek", "<endpoint>/<model-id>"),
    ],
)
def test_bad_model_names(model, reason):
    with pytest.raises(backends.BackendError, match=reason):
        backends.route(model)


def test_bad_endpoints():
    with pytest.raises(backends.BackendError, match="extra_body cannot set model"):
        backends.endpoint_of(
            "x", {"api": "openai", "base_url": "http://h", "extra_body": {"model": "y"}}
        )
    with pytest.raises(backends.BackendError, match="no credentials"):
        backends.endpoint_of("x", {"api": "openai", "base_url": "http://u:p@h"})
    with pytest.raises(backends.BackendError, match="api must be one of"):
        backends.endpoint_of("x", {"api": "gemini", "base_url": "http://h"})
