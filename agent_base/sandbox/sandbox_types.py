from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ExecResult:
    """Captured output of a command executed inside a sandbox.

    Callers should inspect exit_code and timed_out before using stdout/stderr.
    """

    exit_code: int = 0
    """Process return code. -1 when timed_out is True (killed before natural exit)."""

    stdout: str = ""
    """Decoded standard output (UTF-8, errors='replace')."""

    stderr: str = ""
    """Decoded standard error (UTF-8, errors='replace')."""

    timed_out: bool = False
    """True if the process was killed due to timeout expiry."""

    duration_ms: float = 0.0
    """Wall-clock milliseconds from command start to return."""


@dataclass
class FileEntry:
    """A single entry returned by Sandbox.list_dir()."""

    name: str
    """Bare filename or directory name (no path prefix)."""

    is_dir: bool = False
    """True if this entry is a directory."""

    size_bytes: int = 0
    """File size in bytes. 0 for directories."""


@dataclass
class SandboxConfig:
    """Base sandbox configuration for serialization and reconstruction.

    Every Sandbox implementation defines a corresponding SandboxConfig subclass
    that captures all constructor parameters. This enables:

      1. Serialization — ``config.to_dict()`` produces a JSON-safe dict.
      2. Reconstruction — ``SandboxConfig.from_dict(d)`` restores the config,
         and the sandbox's ``from_config()`` classmethod recreates the instance.

    The ``sandbox_type`` field acts as a dispatch key so callers can determine
    which Sandbox subclass to instantiate (analogous to ``provider`` on
    ``AgentConfig`` for LLMConfig dispatch).

    Subclasses add implementation-specific fields (paths, container IDs, etc.).
    """

    sandbox_type: str = ""
    """Dispatch key identifying the Sandbox implementation (e.g. "local")."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SandboxConfig:
        """Reconstruct a SandboxConfig from a dict.

        Filters keys to only those that are valid dataclass fields on ``cls``,
        so unknown keys from older/newer schemas are silently ignored.
        """
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class Sandbox(ABC):
    """Abstract execution environment for agent tools.

    A Sandbox provides two capabilities:
      1. Filesystem — read, write, list, delete files in an isolated namespace.
      2. Execution — run shell commands and capture output within the sandbox.

    All file paths accepted by public methods are RELATIVE to the sandbox root.
    The sandbox resolves them to absolute paths internally. No caller should
    ever see or construct absolute paths.

    Lifecycle: use as an async context manager.

        async with sandbox:
            content = await sandbox.read_file("src/main.py")

    Implementations:
      - LocalSandbox  (sandbox/local.py)   — path-restricted host directory
      - DockerSandbox (sandbox/docker.py)  — container isolation (future)
      - E2BSandbox    (sandbox/e2b.py)     — remote VM isolation (future)
    """

    # ─── Configuration ─────────────────────────────────────────────────

    @property
    @abstractmethod
    def config(self) -> SandboxConfig:
        """Return a serializable config that can recreate this sandbox."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: SandboxConfig) -> Sandbox:
        """Recreate a sandbox instance from a serialized config."""
        ...

    # ─── Lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    async def setup(self) -> None:
        """Initialize the sandbox. Must be called before any other method.

        Idempotent: calling setup() on an already-set-up sandbox is safe.
        """
        ...

    @abstractmethod
    async def teardown(self) -> None:
        """Destroy the sandbox and clean up all resources.

        After teardown(), no other method should be called.
        """
        ...

    async def __aenter__(self) -> Sandbox:
        await self.setup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.teardown()

    # ─── Filesystem ──────────────────────────────────────────────────

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read a text file and return its contents as a string.

        Args:
            path: Relative path within the sandbox (e.g. "src/main.py").

        Returns:
            File contents decoded as UTF-8 (errors='replace').

        Raises:
            ValueError: If path resolves outside the sandbox.
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If path is a directory.
        """
        ...

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write a string to a file, creating parent directories as needed.

        Overwrites the file if it already exists.

        Args:
            path: Relative path within the sandbox.
            content: String content to write (UTF-8 encoded).

        Raises:
            ValueError: If path resolves outside the sandbox.
        """
        ...

    @abstractmethod
    async def read_file_bytes(self, path: str) -> bytes:
        """Read a binary file and return its raw bytes.

        Args:
            path: Relative path within the sandbox.

        Raises:
            ValueError: If path resolves outside the sandbox.
            FileNotFoundError: If the file does not exist.
        """
        ...

    @abstractmethod
    async def write_file_bytes(self, path: str, content: bytes) -> None:
        """Write raw bytes to a file, creating parent directories as needed.

        Args:
            path: Relative path within the sandbox.
            content: Raw bytes to write.

        Raises:
            ValueError: If path resolves outside the sandbox.
        """
        ...

    @abstractmethod
    async def list_dir(self, path: str = ".") -> list[FileEntry]:
        """List the contents of a directory.

        Args:
            path: Relative path to the directory. Defaults to sandbox root.

        Returns:
            Sorted list of FileEntry objects (sorted by name, ascending).

        Raises:
            ValueError: If path resolves outside the sandbox.
            NotADirectoryError: If path is a file.
            FileNotFoundError: If path does not exist.
        """
        ...

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Check whether a file or directory exists at the given path.

        Returns False (does not raise) if path escapes the sandbox.
        """
        ...

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """Delete a file or directory (recursively for directories).

        Returns:
            True if deleted, False if it did not exist.

        Raises:
            ValueError: If path resolves outside the sandbox.
        """
        ...

    # ─── File Coordination ────────────────────────────────────────────

    @abstractmethod
    async def import_file(self, file_id: str, filename: str, content: bytes) -> str:
        """Accept a file for tool use, placing it where tools can access it.

        If a file with this file_id and filename already exists, it is
        overwritten with the new content.

        The sandbox decides the internal layout. Callers must not assume
        a specific path structure.

        Args:
            file_id: Unique identifier for the file (typically a media_id).
            filename: Human-readable filename (e.g. "photo.png").
            content: Raw file bytes.

        Returns:
            Sandbox-relative path where the file is accessible.
        """
        ...

    @abstractmethod
    async def get_exported_files(self) -> list[tuple[str, bytes]]:
        """Collect all files that tools have produced in the exports area.

        Scans the exports zone recursively and returns every file found.

        Returns:
            List of (relative_filename, content) tuples. relative_filename
            is the path relative to the exports area root (e.g. "report.csv"
            or "subdir/data.json"). Empty list if no exports exist.
        """
        ...

    # ─── Execution ───────────────────────────────────────────────────

    @abstractmethod
    async def exec(
        self,
        command: str,
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a shell command and capture its output.

        The command string is passed to the shell (sh -c), preserving pipes,
        redirections, and globbing.

        Args:
            command: Shell command string (e.g. "python main.py | head -20").
            timeout: Maximum seconds before killing the process.
            cwd: Working directory as a relative sandbox path.
                 Defaults to "workspace/" within the sandbox root.
            env: Additional environment variables. Merged over the current
                 process environment in LocalSandbox.

        Returns:
            ExecResult with captured output. timed_out=True and exit_code=-1
            if timeout was exceeded.

        Raises:
            ValueError: If cwd resolves outside the sandbox.
        """
        ...

    @abstractmethod
    async def exec_stream(
        self,
        command: str,
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        """Execute a command and yield output lines as they arrive.

        stderr is merged into stdout in the stream.

        Args:
            command: Shell command string.
            timeout: Maximum seconds before killing the process.
            cwd: Relative sandbox path for the working directory.
            env: Additional environment variables.

        Yields:
            Lines of output (stdout + stderr merged) as produced.

        Raises:
            ValueError: If cwd resolves outside the sandbox.
        """
        ...
        # Make this an abstract async generator — yield is needed for type
        # checkers to recognize the return as AsyncIterator.
        yield ""  # pragma: no cover
