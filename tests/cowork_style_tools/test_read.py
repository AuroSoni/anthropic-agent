"""Tests for the cowork-style Read tool."""
import os
import pytest
from pathlib import Path

from anthropic_agent.cowork_style_tools.read import create_read_tool
from anthropic_agent.tools.base import ToolResult, ImageBlock, DocumentBlock


@pytest.fixture
def tool_fn():
    return create_read_tool()


@pytest.fixture
def tmp_workspace(tmp_path):
    return tmp_path


def create_file(workspace, rel_path, content):
    p = workspace / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


class TestReadTextFiles:
    def test_basic_read(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "test.py", "line1\nline2\nline3\n")
        result = tool_fn(file_path=str(f))
        assert "1\tline1" in result
        assert "2\tline2" in result
        assert "3\tline3" in result

    def test_cat_n_format(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "test.txt", "alpha\nbeta\n")
        result = tool_fn(file_path=str(f))
        lines = result.split("\n")
        assert lines[0].endswith("alpha")
        assert lines[1].endswith("beta")
        # Line numbers should be right-aligned with tab separator
        assert "\t" in lines[0]

    def test_offset_and_limit(self, tool_fn, tmp_workspace):
        content = "\n".join(f"line{i}" for i in range(1, 21)) + "\n"
        f = create_file(tmp_workspace, "test.txt", content)
        result = tool_fn(file_path=str(f), offset=5, limit=3)
        assert "line5" in result
        assert "line6" in result
        assert "line7" in result
        assert "line4" not in result
        assert "line8" not in result

    def test_default_limit_2000(self, tool_fn, tmp_workspace):
        content = "\n".join(f"line{i}" for i in range(1, 3001)) + "\n"
        f = create_file(tmp_workspace, "big.txt", content)
        result = tool_fn(file_path=str(f))
        # Should contain line1 through line2000, but not line2001
        assert "line1\n" in result or "1\tline1" in result
        assert "line2000" in result
        assert "line2001" not in result

    def test_line_truncation(self, tool_fn, tmp_workspace):
        long_line = "x" * 3000
        f = create_file(tmp_workspace, "long.txt", long_line + "\n")
        result = tool_fn(file_path=str(f))
        assert "[truncated]" in result
        assert len(result.split("\t", 1)[1].split("\n")[0]) < 3000

    def test_empty_file(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "empty.txt", "")
        result = tool_fn(file_path=str(f))
        assert "Warning" in result
        assert "empty" in result.lower()

    def test_offset_beyond_file(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "short.txt", "one\ntwo\n")
        result = tool_fn(file_path=str(f), offset=100)
        assert "Error" in result
        assert "exceeds" in result

    def test_file_without_trailing_newline(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "no_newline.txt", "abc\ndef")
        result = tool_fn(file_path=str(f))
        assert "abc" in result
        assert "def" in result


class TestReadErrors:
    def test_nonexistent_file(self, tool_fn):
        result = tool_fn(file_path="/nonexistent/path/file.txt")
        assert "Error" in result
        assert "does not exist" in result

    def test_directory_path(self, tool_fn, tmp_workspace):
        result = tool_fn(file_path=str(tmp_workspace))
        assert "Error" in result
        assert "directory" in result.lower()

    def test_relative_path(self, tool_fn):
        result = tool_fn(file_path="relative/path.txt")
        assert "Error" in result
        assert "absolute" in result.lower()


class TestReadImages:
    def test_png_returns_tool_result(self, tool_fn, tmp_workspace):
        img_path = tmp_workspace / "test.png"
        # Write minimal PNG header bytes
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        result = tool_fn(file_path=str(img_path))
        assert isinstance(result, ToolResult)
        # Check that it contains an ImageBlock
        assert isinstance(result.content, list)
        has_image = any(isinstance(item, ImageBlock) for item in result.content)
        assert has_image

    def test_jpg_returns_tool_result(self, tool_fn, tmp_workspace):
        img_path = tmp_workspace / "test.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        result = tool_fn(file_path=str(img_path))
        assert isinstance(result, ToolResult)

    def test_image_media_type(self, tool_fn, tmp_workspace):
        img_path = tmp_workspace / "test.webp"
        img_path.write_bytes(b"RIFF" + b"\x00" * 100)
        result = tool_fn(file_path=str(img_path))
        assert isinstance(result, ToolResult)
        image_blocks = [i for i in result.content if isinstance(i, ImageBlock)]
        assert image_blocks[0].media_type == "image/webp"


class TestReadPDFs:
    def test_pdf_returns_tool_result(self, tool_fn, tmp_workspace):
        pdf_path = tmp_workspace / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4" + b"\x00" * 100)
        result = tool_fn(file_path=str(pdf_path))
        assert isinstance(result, ToolResult)
        assert isinstance(result.content, list)
        has_doc = any(isinstance(item, DocumentBlock) for item in result.content)
        assert has_doc

    def test_pdf_media_type(self, tool_fn, tmp_workspace):
        pdf_path = tmp_workspace / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.7" + b"\x00" * 100)
        result = tool_fn(file_path=str(pdf_path))
        doc_blocks = [i for i in result.content if isinstance(i, DocumentBlock)]
        assert doc_blocks[0].media_type == "application/pdf"

    def test_pdf_too_large(self, tool_fn, tmp_workspace):
        pdf_path = tmp_workspace / "huge.pdf"
        # Create a file slightly over 32MB
        pdf_path.write_bytes(b"%PDF" + b"\x00" * (33 * 1024 * 1024))
        result = tool_fn(file_path=str(pdf_path))
        assert isinstance(result, str)
        assert "Error" in result
        assert "32 MB" in result


class TestReadSchema:
    def test_has_tool_schema(self, tool_fn):
        assert hasattr(tool_fn, "__tool_schema__")
        schema = tool_fn.__tool_schema__
        assert schema["name"] == "read_file"

    def test_schema_parameters(self, tool_fn):
        props = tool_fn.__tool_schema__["input_schema"]["properties"]
        assert "file_path" in props
        assert "offset" in props
        assert "limit" in props
