"""Unit tests for the E2B cloud sandbox environment backend."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock E2B SDK objects
# ---------------------------------------------------------------------------

def _make_run_result(stdout="", stderr="", exit_code=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)


def _make_command_handle(result=None):
    """Create a mock CommandHandle returned by commands.run(background=True)."""
    if result is None:
        result = _make_run_result()
    handle = MagicMock()
    handle.wait.return_value = result
    handle.kill.return_value = True
    return handle


def _make_sandbox(sandbox_id="sb-e2b-123"):
    """Create a mock sandbox.

    commands.run dispatches based on ``background`` kwarg:
    - background=True  → returns a CommandHandle (for _run_bash)
    - otherwise        → returns a CommandResult directly (for init, sync)
    """
    sb = MagicMock()
    sb.sandbox_id = sandbox_id
    # Default: direct result for non-background calls (init, sync)
    default_result = _make_run_result()

    def _run_dispatch(*args, **kwargs):
        if kwargs.get("background"):
            return _make_command_handle(default_result)
        return default_result

    sb.commands.run.side_effect = _run_dispatch
    sb.files = MagicMock()
    return sb


def _patch_e2b_imports(monkeypatch):
    """Patch the e2b SDK so E2BEnvironment can be imported without it."""
    import enum
    import types as _types

    e2b_mod = _types.ModuleType("e2b")
    e2b_mod.Sandbox = MagicMock

    sandbox_api_mod = _types.ModuleType("e2b.sandbox.sandbox_api")
    sandbox_api_mod.SandboxQuery = MagicMock

    # Mock SandboxState enum used for listing running/paused sandboxes
    class _SandboxState(str, enum.Enum):
        RUNNING = "running"
        PAUSED = "paused"

    state_mod = _types.ModuleType("e2b.api.client.models.sandbox_state")
    state_mod.SandboxState = _SandboxState

    monkeypatch.setitem(__import__("sys").modules, "e2b", e2b_mod)
    monkeypatch.setitem(__import__("sys").modules, "e2b.sandbox.sandbox_api", sandbox_api_mod)
    monkeypatch.setitem(__import__("sys").modules, "e2b.api", _types.ModuleType("e2b.api"))
    monkeypatch.setitem(__import__("sys").modules, "e2b.api.client", _types.ModuleType("e2b.api.client"))
    monkeypatch.setitem(__import__("sys").modules, "e2b.api.client.models", _types.ModuleType("e2b.api.client.models"))
    monkeypatch.setitem(__import__("sys").modules, "e2b.api.client.models.sandbox_state", state_mod)
    return e2b_mod, sandbox_api_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def e2b_sdk(monkeypatch):
    """Provide mock e2b SDK modules and return them for assertions."""
    e2b_mod, sandbox_api_mod = _patch_e2b_imports(monkeypatch)
    return e2b_mod


@pytest.fixture()
def make_env(e2b_sdk, monkeypatch):
    """Factory that creates an E2BEnvironment with a mocked SDK."""
    # Prevent is_interrupted from interfering
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    # Prevent skills/credential sync from consuming mock exec calls
    monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", lambda: [])
    monkeypatch.setattr("tools.credential_files.get_skills_directory_mount", lambda **kw: None)
    monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kw: [])

    def _factory(
        sandbox=None,
        list_items=None,
        connect_side_effect=None,
        home_dir="/home/user",
        persistent=True,
        **kwargs,
    ):
        custom_sandbox = sandbox is not None
        sandbox = sandbox or _make_sandbox()
        if not custom_sandbox:
            # Override side_effect so the home_dir value is returned for
            # non-background calls (side_effect takes precedence over
            # return_value).
            home_result = _make_run_result(stdout=home_dir)

            def _home_dispatch(*args, **kwargs):
                if kwargs.get("background"):
                    return _make_command_handle(home_result)
                return home_result

            sandbox.commands.run.side_effect = _home_dispatch

        # Mock Sandbox class
        mock_sandbox_cls = MagicMock()
        mock_sandbox_cls.create.return_value = sandbox

        # Mock list() for metadata-based lookup
        mock_page = MagicMock()
        if list_items is not None:
            mock_page.next_items.return_value = list_items
        else:
            mock_page.next_items.return_value = []
        mock_sandbox_cls.list.return_value = mock_page

        if connect_side_effect is not None:
            mock_sandbox_cls.connect.side_effect = connect_side_effect
        else:
            mock_sandbox_cls.connect.side_effect = Exception("not found")

        e2b_sdk.Sandbox = mock_sandbox_cls

        from tools.environments.e2b import E2BEnvironment

        env = E2BEnvironment(
            image="test-template",
            persistent_filesystem=persistent,
            **kwargs,
        )
        env._mock_sandbox_cls = mock_sandbox_cls
        return env

    return _factory


# ---------------------------------------------------------------------------
# Constructor / cwd resolution
# ---------------------------------------------------------------------------

class TestCwdResolution:
    def test_default_cwd_resolves_home(self, make_env):
        env = make_env(home_dir="/home/testuser")
        assert env.cwd == "/home/testuser"

    def test_tilde_cwd_resolves_home(self, make_env):
        env = make_env(cwd="~", home_dir="/home/testuser")
        assert env.cwd == "/home/testuser"

    def test_explicit_cwd_not_overridden(self, make_env):
        env = make_env(cwd="/workspace", home_dir="/root")
        assert env.cwd == "/workspace"

    def test_home_detection_failure_keeps_default_cwd(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = RuntimeError("exec failed")
        env = make_env(sandbox=sb)
        assert env.cwd == "/home/user"  # keeps constructor default

    def test_empty_home_keeps_default_cwd(self, make_env):
        env = make_env(home_dir="")
        assert env.cwd == "/home/user"  # keeps constructor default


# ---------------------------------------------------------------------------
# Sandbox persistence / reconnect via metadata
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persistent_reconnects_via_metadata_list(self, make_env):
        existing = _make_sandbox(sandbox_id="sb-existing")
        existing.commands.run.return_value = _make_run_result(stdout="/home/user")
        # list() returns an item, connect() returns the sandbox
        list_item = SimpleNamespace(sandbox_id="sb-existing", metadata={"hermes_provider": "hermes", "hermes_template": "test-template"})
        env = make_env(
            list_items=[list_item],
            connect_side_effect=lambda sid: existing,
            persistent=True,
            task_id="mytask",
        )
        env._mock_sandbox_cls.list.assert_called_once()
        env._mock_sandbox_cls.connect.assert_called_once_with("sb-existing")
        env._mock_sandbox_cls.create.assert_not_called()

    def test_persistent_creates_new_when_no_match(self, make_env):
        env = make_env(
            list_items=[],
            persistent=True,
            task_id="mytask",
        )
        env._mock_sandbox_cls.create.assert_called_once()

    def test_persistent_creates_new_when_connect_fails(self, make_env):
        list_item = SimpleNamespace(sandbox_id="sb-dead", metadata={"hermes_provider": "hermes", "hermes_template": "test-template"})
        env = make_env(
            list_items=[list_item],
            connect_side_effect=Exception("sandbox dead"),
            persistent=True,
            task_id="mytask",
        )
        env._mock_sandbox_cls.create.assert_called_once()

    def test_non_persistent_skips_list(self, make_env):
        env = make_env(persistent=False)
        env._mock_sandbox_cls.list.assert_not_called()
        env._mock_sandbox_cls.connect.assert_not_called()
        env._mock_sandbox_cls.create.assert_called_once()

    def test_metadata_labels_passed_to_create(self, make_env):
        env = make_env(persistent=True, task_id="mytask")
        call_kwargs = env._mock_sandbox_cls.create.call_args[1]
        assert call_kwargs["metadata"] == {"hermes_provider": "hermes", "hermes_template": "test-template"}


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_persistent_cleanup_pauses_sandbox(self, make_env):
        env = make_env(persistent=True)
        sb = env._sandbox
        env.cleanup()
        sb.pause.assert_called_once()

    def test_non_persistent_cleanup_kills_sandbox(self, make_env):
        env = make_env(persistent=False)
        sb = env._sandbox
        env.cleanup()
        sb.kill.assert_called_once()

    def test_cleanup_idempotent(self, make_env):
        env = make_env(persistent=True)
        env.cleanup()
        env.cleanup()  # should not raise

    def test_cleanup_swallows_errors(self, make_env):
        env = make_env(persistent=True)
        env._sandbox.pause.side_effect = RuntimeError("pause failed")
        env.cleanup()  # should not raise
        assert env._sandbox is None


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

def _sequenced_run(results):
    """Build a side_effect for commands.run that dispatches background calls.

    Non-background calls return CommandResult directly (used by init).
    Background calls return a CommandHandle whose wait() yields the result.
    """
    it = iter(results)

    def _side_effect(*args, **kwargs):
        r = next(it)
        if isinstance(r, Exception):
            raise r
        if kwargs.get("background"):
            return _make_command_handle(r)
        return r

    return _side_effect


class TestExecute:
    def test_basic_command(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = _sequenced_run([
            _make_run_result(stdout="/home/user"),         # $HOME
            _make_run_result(stdout="", exit_code=0),      # init_session
            _make_run_result(stdout="hello", exit_code=0), # actual cmd
        ])
        env = make_env(sandbox=sb)

        result = env.execute("echo hello")
        assert "hello" in result["output"]
        assert result["returncode"] == 0

    def test_nonzero_exit_code(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = _sequenced_run([
            _make_run_result(stdout="/home/user"),
            _make_run_result(stdout="", exit_code=0),
            _make_run_result(stdout="not found", exit_code=127),
        ])
        env = make_env(sandbox=sb)

        result = env.execute("bad_cmd")
        assert result["returncode"] == 127

    def test_stderr_included_in_output(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = _sequenced_run([
            _make_run_result(stdout="/home/user"),
            _make_run_result(stdout="", exit_code=0),
            _make_run_result(stdout="out", stderr="err", exit_code=0),
        ])
        env = make_env(sandbox=sb)

        result = env.execute("echo hello")
        assert "out" in result["output"]
        assert "err" in result["output"]

    def test_stdin_data_wraps_heredoc(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = _sequenced_run([
            _make_run_result(stdout="/home/user"),
            _make_run_result(stdout="", exit_code=0),
            _make_run_result(stdout="ok", exit_code=0),
        ])
        env = make_env(sandbox=sb)

        env.execute("python3", stdin_data="print('hi')")
        call_args = sb.commands.run.call_args_list[-1]
        cmd = call_args[0][0]
        assert "HERMES_STDIN_" in cmd
        assert "print" in cmd
        assert "hi" in cmd

    def test_custom_cwd_in_command_wrapper(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = _sequenced_run([
            _make_run_result(stdout="/home/user"),
            _make_run_result(stdout="", exit_code=0),
            _make_run_result(stdout="/tmp", exit_code=0),
        ])
        env = make_env(sandbox=sb)

        env.execute("pwd", cwd="/tmp")
        call_args = sb.commands.run.call_args_list[-1]
        cmd = call_args[0][0]
        assert "cd /tmp" in cmd


# ---------------------------------------------------------------------------
# Interrupt handling
# ---------------------------------------------------------------------------

class TestInterrupt:
    def test_interrupt_kills_handle_and_returns_130(self, make_env, monkeypatch):
        sb = _make_sandbox()
        event = threading.Event()
        calls = {"n": 0}
        long_handle = _make_command_handle()

        # Make wait() block until event is set, simulating a long-running cmd
        def _blocking_wait():
            event.wait(timeout=5)
            return _make_run_result(stdout="done", exit_code=0)

        long_handle.wait.side_effect = _blocking_wait

        def run_side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _make_run_result(stdout="/home/user")  # $HOME detection
            if calls["n"] == 2:
                return _make_run_result(stdout="", exit_code=0)  # init_session
            # The actual command: background=True → return a CommandHandle
            return long_handle

        sb.commands.run.side_effect = run_side_effect
        env = make_env(sandbox=sb)

        monkeypatch.setattr(
            "tools.environments.base.is_interrupted", lambda: True
        )
        try:
            result = env.execute("sleep 10")
            assert result["returncode"] == 130
            # Cancel should kill the command handle, NOT the sandbox
            long_handle.kill.assert_called()
        finally:
            event.set()


# ---------------------------------------------------------------------------
# SDK error surfaces directly
# ---------------------------------------------------------------------------

class TestSDKError:
    def test_sdk_error_surfaces_as_rc1(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = _sequenced_run([
            _make_run_result(stdout="/home/user"),       # $HOME
            _make_run_result(stdout="", exit_code=0),    # init_session
            RuntimeError("sdk error"),                    # actual command fails
        ])
        env = make_env(sandbox=sb)

        result = env.execute("echo x")
        assert result["returncode"] == 1
