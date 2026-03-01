from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


# ─── Constants ────────────────────────────────────────────────────────

TOKEN_COUNTING_SIZE_THRESHOLD: int = 1 * 1024 * 1024  # 1MB
"""Files larger than this are skipped for token counting."""

READ_CHUNK_SIZE: int = 64 * 1024  # 64KB
"""Chunk size for streaming file reads/writes."""

MAX_READ_LINES: int = 10_000
"""Maximum number of lines read_file can return in a single call."""

TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
    ".xml", ".html", ".htm", ".css", ".csv", ".toml", ".cfg", ".ini", ".sh",
    ".bash", ".c", ".h", ".cpp", ".hpp", ".java", ".rs", ".go", ".rb",
    ".sql", ".log", ".env", ".rst", ".tex", ".svg", ".graphql", ".proto",
    ".swift", ".kt", ".scala", ".lua",
})
"""File extensions considered text-based for token counting and read_file."""


# ─── Exceptions ───────────────────────────────────────────────────────


class SandboxPathEscapeError(ValueError):
    """Raised when a path resolves outside the sandbox boundary."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Path traversal blocked: '{path}' resolves outside sandbox")


class SandboxNotATextFileError(Exception):
    """Raised when read_file is called on a non-text file."""

    def __init__(self, path: str, extension: str):
        self.path = path
        self.extension = extension
        super().__init__(
            f"Cannot read non-text file with read_file: '{path}' "
            f"(extension '{extension}' not in TEXT_EXTENSIONS). "
            f"Use read_file_bytes for binary files."
        )


# ─── Data types ───────────────────────────────────────────────────────


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
    """A single entry returned by Sandbox.list_dir() and Sandbox.file_exists()."""

    name: str
    """Bare filename or directory name (no path prefix)."""

    is_dir: bool = False
    """True if this entry is a directory."""

    size_bytes: int = 0
    """File size in bytes. 0 for directories."""

    extension: str = ""
    """File extension including the dot (e.g. ".py"). Empty for directories."""

    tokens: int | None = None
    """Estimated LLM token count (size_bytes // 3). None for directories,
    binary files, or files exceeding TOKEN_COUNTING_SIZE_THRESHOLD."""


@dataclass
class ExportedFileMetadata:
    """Metadata for a file in the sandbox exports area."""

    filename: str
    """Bare filename (e.g. "report.csv")."""

    extension: str
    """File extension including the dot (e.g. ".csv")."""

    size_bytes: int
    """File size in bytes."""

    blake3_hash: str
    """Hex digest of the file contents (BLAKE3)."""

    path: str
    """Path relative to the exports root (e.g. "subdir/report.csv")."""


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


# ─── Sandbox ABC ──────────────────────────────────────────────────────


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
    async def read_file(
        self,
        path: str,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        """Read a text file and return its contents as a string.

        Only works with text-based files (extension in TEXT_EXTENSIONS).

        Args:
            path: Relative path within the sandbox (e.g. "src/main.py").
            offset: Number of lines to skip from the start.
            limit: Maximum number of lines to return. Capped at MAX_READ_LINES.

        Returns:
            File contents decoded as UTF-8 (errors='replace').

        Raises:
            SandboxPathEscapeError: If path resolves outside the sandbox.
            SandboxNotATextFileError: If the file is not a text file.
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
            SandboxPathEscapeError: If path resolves outside the sandbox.
        """
        ...

    @abstractmethod
    async def read_file_bytes(self, path: str) -> AsyncIterator[bytes]:
        """Read a file and yield its contents as a stream of byte chunks.

        Args:
            path: Relative path within the sandbox.

        Yields:
            Byte chunks of up to READ_CHUNK_SIZE bytes.

        Raises:
            SandboxPathEscapeError: If path resolves outside the sandbox.
            FileNotFoundError: If the file does not exist.
        """
        ...
        yield b""  # pragma: no cover

    @abstractmethod
    async def write_file_bytes(
        self, path: str, data: AsyncIterator[bytes]
    ) -> None:
        """Write a stream of byte chunks to a file.

        Creates parent directories as needed. Overwrites the file if it exists.

        Args:
            path: Relative path within the sandbox.
            data: Async iterator yielding byte chunks.

        Raises:
            SandboxPathEscapeError: If path resolves outside the sandbox.
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
            SandboxPathEscapeError: If path resolves outside the sandbox.
            NotADirectoryError: If path is a file.
            FileNotFoundError: If path does not exist.
        """
        ...

    @abstractmethod
    async def file_exists(self, path: str) -> tuple[bool, FileEntry | None]:
        """Check whether a file or directory exists at the given path.

        Returns (False, None) without raising if path escapes the sandbox.

        Returns:
            Tuple of (exists, file_entry). file_entry is None when not found.
        """
        ...

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """Delete a file or directory (recursively for directories).

        Returns:
            True if deleted, False if it did not exist.

        Raises:
            SandboxPathEscapeError: If path resolves outside the sandbox.
        """
        ...

    # ─── File Coordination ────────────────────────────────────────────

    @abstractmethod
    async def import_file(
        self, filename: str, data: AsyncIterator[bytes]
    ) -> str:
        """Accept a file stream for tool use, placing it where tools can access it.

        The sandbox decides the internal layout. Callers must not assume
        a specific path structure.

        Args:
            filename: Human-readable filename (e.g. "photo.png").
            data: Async iterator yielding byte chunks of the file content.

        Returns:
            Sandbox-relative path where the file is accessible.
        """
        ...

    @abstractmethod
    async def list_exported_files(self) -> list[str]:
        """List all file paths in the exports area.

        Scans the exports zone recursively and returns every file path found.

        Returns:
            List of paths relative to the exports root (e.g. ["report.csv",
            "subdir/data.json"]). Empty list if no exports exist.
        """
        ...

    @abstractmethod
    async def get_exported_file(self, path: str) -> AsyncIterator[bytes]:
        """Read a single exported file as a byte stream.

        Args:
            path: Path relative to the exports root (as returned by
                  list_exported_files).

        Yields:
            Byte chunks of up to READ_CHUNK_SIZE bytes.

        Raises:
            FileNotFoundError: If the file does not exist in the exports area.
            SandboxPathEscapeError: If path escapes the exports area.
        """
        ...
        yield b""  # pragma: no cover

    @abstractmethod
    async def get_exported_file_metadata(self) -> list[ExportedFileMetadata]:
        """Return metadata for all files in the exports area.

        Scans the exports zone recursively and returns metadata (filename,
        extension, size, blake3 hash, relative path) for every file found.

        Returns:
            List of ExportedFileMetadata objects. Empty list if no exports exist.
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
            SandboxPathEscapeError: If cwd resolves outside the sandbox.
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
            SandboxPathEscapeError: If cwd resolves outside the sandbox.
        """
        ...
        # Make this an abstract async generator — yield is needed for type
        # checkers to recognize the return as AsyncIterator.
        yield ""  # pragma: no cover
