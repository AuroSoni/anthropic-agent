from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import aiofiles

from .sandbox_types import ExecResult, FileEntry, Sandbox, SandboxConfig


@dataclass
class LocalSandboxConfig(SandboxConfig):
    """Serializable configuration for LocalSandbox."""

    sandbox_type: str = "local"
    sandbox_id: str = ""
    base_dir: str = ""
    default_timeout: float = 30.0


class LocalSandbox(Sandbox):
    """Path-restricted sandbox on the local filesystem.

    Each agent session gets its own directory tree::

        {base_dir}/{sandbox_id}/
        ├── workspace/          # default cwd for exec
        ├── .imported/          # (created by caller, not by sandbox)
        ├── .exports/           # (created by caller)
        └── ...

    All relative paths are resolved against the sandbox root
    ({base_dir}/{sandbox_id}/). Path traversal (../../etc/passwd)
    is blocked by _resolve().

    This is Level 1 isolation — organizational boundary only.
    Commands run as host subprocesses and can still access the wider
    system. Use DockerSandbox for real process isolation.
    """

    def __init__(
        self,
        sandbox_id: str,
        base_dir: str | Path,
        default_timeout: float = 30.0,
    ) -> None:
        if not sandbox_id:
            raise ValueError("sandbox_id must not be empty")
        if "/" in sandbox_id or "\\" in sandbox_id:
            raise ValueError("sandbox_id must not contain path separators")

        self.sandbox_id: str = sandbox_id
        self.root: Path = Path(base_dir).resolve() / sandbox_id
        self.workspace: Path = self.root / "workspace"
        self.default_timeout: float = default_timeout
        self._cwd: Path = self.workspace

    # ─── Configuration ─────────────────────────────────────────────────

    @property
    def config(self) -> LocalSandboxConfig:
        return LocalSandboxConfig(
            sandbox_id=self.sandbox_id,
            base_dir=str(self.root.parent),
            default_timeout=self.default_timeout,
        )

    @classmethod
    def from_config(cls, config: LocalSandboxConfig) -> LocalSandbox:
        """Create a LocalSandbox from a serialized config."""
        return cls(
            sandbox_id=config.sandbox_id,
            base_dir=config.base_dir,
            default_timeout=config.default_timeout,
        )

    # ─── Path containment ────────────────────────────────────────────

    def _resolve(self, path: str) -> Path:
        """Resolve a relative path to an absolute path within the sandbox root.

        Raises ValueError if the resolved path escapes the sandbox boundary.
        """
        resolved = (self.root / path).resolve()
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError:
            raise ValueError(
                f"Path traversal blocked: '{path}' resolves outside sandbox"
            ) from None
        return resolved

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def setup(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for zone in (".imported", "workspace", ".exports"):
            (self.root / zone).mkdir(exist_ok=True)
        self._cwd = self.workspace

    async def teardown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    # ─── Filesystem ──────────────────────────────────────────────────

    async def read_file(self, path: str) -> str:
        resolved = self._resolve(path)
        async with aiofiles.open(resolved, "r", encoding="utf-8", errors="replace") as f:
            return await f.read()

    async def write_file(self, path: str, content: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(resolved, "w", encoding="utf-8") as f:
            await f.write(content)

    async def read_file_bytes(self, path: str) -> bytes:
        resolved = self._resolve(path)
        async with aiofiles.open(resolved, "rb") as f:
            return await f.read()

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(resolved, "wb") as f:
            await f.write(content)

    async def list_dir(self, path: str = ".") -> list[FileEntry]:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Directory not found: '{path}'")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: '{path}'")

        entries: list[FileEntry] = []
        for item in sorted(resolved.iterdir(), key=lambda p: p.name):
            try:
                size = item.stat().st_size if item.is_file() else 0
            except OSError:
                size = 0
            entries.append(FileEntry(
                name=item.name,
                is_dir=item.is_dir(),
                size_bytes=size,
            ))
        return entries

    async def file_exists(self, path: str) -> bool:
        try:
            resolved = self._resolve(path)
        except ValueError:
            return False
        return resolved.exists()

    async def delete(self, path: str) -> bool:
        resolved = self._resolve(path)
        if not resolved.exists():
            return False
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        return True

    # ─── File Coordination ────────────────────────────────────────────

    async def import_file(self, file_id: str, filename: str, content: bytes) -> str:
        sandbox_path = f".imported/{file_id}/{filename}"
        resolved = self._resolve(sandbox_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(resolved, "wb") as f:
            await f.write(content)
        return sandbox_path

    async def get_exported_files(self) -> list[tuple[str, bytes]]:
        exports_root = self.root / ".exports"
        if not exports_root.exists():
            return []
        results: list[tuple[str, bytes]] = []
        await self._collect_exports(exports_root, exports_root, results)
        return results

    async def _collect_exports(
        self, current: Path, root: Path, results: list[tuple[str, bytes]]
    ) -> None:
        """Recursively collect files from the exports directory."""
        for item in sorted(current.iterdir(), key=lambda p: p.name):
            if item.is_dir():
                await self._collect_exports(item, root, results)
            elif item.is_file():
                relative = str(item.relative_to(root))
                async with aiofiles.open(item, "rb") as f:
                    content = await f.read()
                results.append((relative, content))

    # ─── Execution ───────────────────────────────────────────────────

    # Unique marker to separate command output from the pwd probe.
    _PWD_MARKER = "__SANDBOX_PWD_8f3a1b__"

    async def exec(
        self,
        command: str,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        effective_timeout = timeout if timeout is not None else self.default_timeout
        if cwd is not None:
            work_dir = self._resolve(cwd)
            self._cwd = work_dir
        else:
            work_dir = self._cwd
        effective_env = {**os.environ, **(env or {})}

        # Append a pwd probe so we can track directory changes from cd commands.
        probed_command = f"{command}\n__exit_code=$?\necho {self._PWD_MARKER}\npwd\nexit $__exit_code"

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_shell(
                probed_command,
                cwd=str(work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=effective_env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()  # drain pipes to prevent resource leak
                return ExecResult(
                    exit_code=-1,
                    timed_out=True,
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            # Extract the pwd probe output and update _cwd.
            stdout_text = self._extract_cwd(stdout_text)

            return ExecResult(
                exit_code=proc.returncode if proc.returncode is not None else -1,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return ExecResult(
                exit_code=-1,
                stderr=str(exc),
                duration_ms=(time.monotonic() - start) * 1000,
            )

    def _extract_cwd(self, stdout: str) -> str:
        """Parse the pwd probe from stdout, update self._cwd, and return clean output."""
        marker_pos = stdout.rfind(self._PWD_MARKER)
        if marker_pos == -1:
            return stdout

        clean_output = stdout[:marker_pos]
        # Strip trailing newline that precedes the marker
        if clean_output.endswith("\n"):
            clean_output = clean_output[:-1]

        after_marker = stdout[marker_pos + len(self._PWD_MARKER):]
        pwd_line = after_marker.strip()
        if pwd_line:
            new_cwd = Path(pwd_line).resolve()
            try:
                new_cwd.relative_to(self.root.resolve())
                self._cwd = new_cwd
            except ValueError:
                pass  # cd'd outside sandbox — keep previous _cwd

        return clean_output

    async def exec_stream(
        self,
        command: str,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        effective_timeout = timeout if timeout is not None else self.default_timeout
        if cwd is not None:
            work_dir = self._resolve(cwd)
            self._cwd = work_dir
        else:
            work_dir = self._cwd
        effective_env = {**os.environ, **(env or {})}

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(work_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # merge stderr into stdout
            env=effective_env,
        )
        try:
            assert proc.stdout is not None
            deadline = time.monotonic() + effective_timeout
            async for line in proc.stdout:
                yield line.decode("utf-8", errors="replace")
                if time.monotonic() > deadline:
                    break
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
