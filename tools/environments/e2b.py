"""E2B cloud execution environment.

Uses the E2B Python SDK to run commands in cloud sandboxes.
Supports persistent sandboxes: when enabled, sandboxes are looked up via
metadata labels and reconnected, preserving the sandbox across sessions.
"""

import logging
import math
import threading
from pathlib import Path

from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_rm_command,
)

logger = logging.getLogger(__name__)


class E2BEnvironment(BaseEnvironment):
    """E2B cloud sandbox execution backend.

    Spawn-per-call via _ThreadedProcessHandle wrapping blocking SDK calls.
    cancel_fn wired to sandbox.kill() for interrupt support.
    Shell timeout wrapper preserved (SDK timeout may be unreliable).
    """

    _stdin_mode = "heredoc"

    def __init__(
        self,
        image: str = "hermes",
        cwd: str = "/home/user",
        timeout: int = 60,
        cpu: int = 1,
        memory: int = 5120,
        disk: int = 10240,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        from e2b import Sandbox
        from e2b.sandbox.sandbox_api import SandboxQuery

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._Sandbox = Sandbox
        self._SandboxQuery = SandboxQuery
        self._sandbox = None
        self._lock = threading.Lock()

        # Resolve template: if cpu/memory are explicitly set, pick a
        # size-tagged template (e2b/hermes:<cpu>-<ram>).  Valid values
        # are 1 and even numbers up to 8.  Memory is converted from MiB
        # to GiB.  Falls back to e2b/hermes:lts when using defaults.
        _VALID_SIZES = {1, 2, 4, 6, 8}
        memory_gib = max(1, math.ceil(memory / 1024))
        custom_resources = cpu != 1 or memory_gib != 5

        _valid_str = ", ".join(str(s) for s in sorted(_VALID_SIZES))
        custom_image = image != "hermes"

        if custom_image:
            # User provided a custom template — use it as-is.
            logger.info("E2B: using custom template %s", image)
        elif custom_resources:
            if cpu not in _VALID_SIZES:
                raise ValueError(
                    f"E2B cpu must be one of [{_valid_str}] — got {cpu}. "
                    f"Available templates: e2b/hermes:<cpu>-<memory_gib>")
            if memory_gib not in _VALID_SIZES:
                raise ValueError(
                    f"E2B memory (GiB) must be one of [{_valid_str}] — got {memory_gib} "
                    f"(from {memory} MiB). Available templates: e2b/hermes:<cpu>-<memory_gib>")
            image = f"e2b/hermes:{cpu}-{memory_gib}"
            logger.info("E2B: using size-tagged template %s", image)

        # Use a stable label so sandboxes persist across CLI invocations.
        # The ephemeral task_id changes every run; the template is constant.
        labels = {"hermes_provider": "hermes", "hermes_template": image}

        # Try to reconnect to an existing sandbox via metadata lookup
        if self._persistent:
            try:
                page = Sandbox.list(
                    query=SandboxQuery(metadata=labels), limit=1,
                )
                items = page.next_items()
                if items:
                    self._sandbox = Sandbox.connect(items[0].sandbox_id)
                    self._sandbox.set_timeout(600)
                    logger.info("E2B: reconnected to sandbox %s (template=%s)",
                                self._sandbox.sandbox_id, image)
            except Exception as e:
                logger.debug("E2B: could not reconnect (template=%s): %s",
                             image, e)
                self._sandbox = None

        if self._sandbox is None:
            self._sandbox = Sandbox.create(
                template=image,
                timeout=600,
                metadata=labels,
            )
            logger.info("E2B: created sandbox %s (template=%s)",
                        self._sandbox.sandbox_id, image)

        # Detect remote home dir
        self._remote_home = "/home/user"
        try:
            result = self._sandbox.commands.run("echo $HOME")
            home = (result.stdout or "").strip()
            if home:
                self._remote_home = home
                if requested_cwd in ("~", "/home/user"):
                    self.cwd = home
        except Exception:
            pass
        logger.info("E2B: resolved home to %s, cwd to %s", self._remote_home, self.cwd)

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._e2b_upload,
            delete_fn=self._e2b_delete,
            bulk_upload_fn=self._e2b_bulk_upload,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    def _e2b_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via E2B SDK."""
        parent = str(Path(remote_path).parent)
        self._sandbox.commands.run(f"mkdir -p {parent}")
        content = Path(host_path).read_bytes()
        self._sandbox.files.write(remote_path, content)

    def _e2b_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files as a single tar archive.

        Packs all files into an in-memory tar.gz, uploads it once via
        ``sandbox.files.write()``, and extracts on the sandbox.
        """
        if not files:
            return

        import io
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for host_path, remote_path in files:
                tar.add(host_path, arcname=remote_path)
        buf.seek(0)

        tar_remote = "/tmp/_hermes_sync.tar.gz"
        self._sandbox.files.write(tar_remote, buf.read())
        self._sandbox.commands.run(
            f"tar xzf {tar_remote} -C / && rm -f {tar_remote}"
        )

    def _e2b_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files via SDK exec."""
        self._sandbox.commands.run(quoted_rm_command(remote_paths))

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    def _ensure_sandbox_ready(self) -> None:
        """Verify sandbox is still alive by extending its timeout."""
        try:
            self._sandbox.set_timeout(600)
        except Exception as e:
            logger.warning("E2B: sandbox may be dead, attempting reconnect: %s", e)
            labels = {"hermes_task_id": self._task_id}
            try:
                page = self._Sandbox.list(
                    query=self._SandboxQuery(metadata=labels), limit=1,
                )
                items = page.next_items()
                if items:
                    self._sandbox = self._Sandbox.connect(items[0].sandbox_id)
                    self._sandbox.set_timeout(600)
                    logger.info("E2B: reconnected to sandbox %s",
                                self._sandbox.sandbox_id)
                    return
            except Exception:
                pass
            raise RuntimeError("E2B: sandbox is no longer available") from e

    def _before_execute(self) -> None:
        """Ensure sandbox is ready, then sync files via FileSyncManager."""
        with self._lock:
            self._ensure_sandbox_ready()
        self._sync_manager.sync()

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None):
        """Return a _ThreadedProcessHandle wrapping a blocking E2B SDK call.

        Uses ``background=True`` to get a CommandHandle, so cancel only
        kills the running process — not the entire sandbox.
        """
        sandbox = self._sandbox
        handle_ref: list = []

        def cancel():
            if handle_ref:
                try:
                    handle_ref[0].kill()
                except Exception:
                    pass

        def exec_fn() -> tuple[str, int]:
            handle = sandbox.commands.run(cmd_string, background=True, timeout=timeout)
            handle_ref.append(handle)
            result = handle.wait()
            output = (result.stdout or "") + (result.stderr or "")
            return (output, result.exit_code)

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            if self._sandbox is None:
                return
            try:
                if self._persistent:
                    self._sandbox.set_timeout(600)
                    logger.info("E2B: kept sandbox %s alive (persistent)",
                                self._sandbox.sandbox_id)
                else:
                    self._sandbox.kill()
                    logger.info("E2B: killed sandbox %s",
                                self._sandbox.sandbox_id)
            except Exception as e:
                logger.warning("E2B: cleanup failed: %s", e)
            self._sandbox = None
