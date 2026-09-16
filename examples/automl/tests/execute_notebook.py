"""Execute repository notebooks and persist standard Jupyter cell outputs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from jupyter_client import KernelManager

logger = logging.getLogger(__name__)


def _output_from_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one IOPub message to a notebook output object."""
    message_type = message["header"]["msg_type"]
    content = message["content"]
    if message_type == "stream":
        return {"name": content["name"], "output_type": "stream", "text": content["text"]}
    if message_type in {"display_data", "execute_result"}:
        output = {
            "data": content["data"],
            "metadata": content.get("metadata", {}),
            "output_type": message_type,
        }
        if message_type == "execute_result":
            output["execution_count"] = content["execution_count"]
        return output
    if message_type == "error":
        return {
            "ename": content["ename"],
            "evalue": content["evalue"],
            "output_type": "error",
            "traceback": content["traceback"],
        }
    return None


def execute_notebook(path: Path, *, working_directory: Path) -> None:
    """Execute code cells in one notebook with the current Python environment."""
    notebook = json.loads(path.read_text(encoding="utf-8"))

    def save() -> None:
        """Persist progress so a later failing cell keeps earlier outputs."""
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    manager = KernelManager()
    with tempfile.TemporaryDirectory(
        prefix="fmlib_ipython_",
        ignore_cleanup_errors=True,
    ) as ipython_directory:
        environment = os.environ.copy()
        environment["IPYTHONDIR"] = ipython_directory
        manager.start_kernel(cwd=str(working_directory), env=environment)
        client = manager.client()
        client.start_channels()
        client.wait_for_ready(timeout=60)
        try:
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                cell["outputs"] = []
                message_id = client.execute("".join(cell["source"]), stop_on_error=True)
                while True:
                    message = client.get_iopub_msg(timeout=3600)
                    if message.get("parent_header", {}).get("msg_id") != message_id:
                        continue
                    if (
                        message["header"]["msg_type"] == "status"
                        and message["content"]["execution_state"] == "idle"
                    ):
                        break
                    output = _output_from_message(message)
                    if output is not None:
                        cell["outputs"].append(output)
                        if output["output_type"] == "error":
                            save()
                            msg = f"Notebook cell failed: {output['ename']}: {output['evalue']}"
                            raise RuntimeError(msg)
                    if message["header"]["msg_type"] == "execute_input":
                        cell["execution_count"] = message["content"]["execution_count"]
                save()
        finally:
            client.stop_channels()
            manager.shutdown_kernel(now=True)
    save()


def main() -> None:
    """Execute every notebook path passed on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--working-directory", type=Path, default=Path.cwd())
    args = parser.parse_args()
    for notebook in args.notebooks:
        logger.info("Executing %s", notebook)
        execute_notebook(notebook.resolve(), working_directory=args.working_directory.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
