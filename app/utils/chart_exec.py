"""Shared utility: run LLM-generated matplotlib code in an isolated subprocess.

Each chart runs in its own Python process so:
- Errors in one chart don't kill the others.
- matplotlib state is always clean (no bleed between figures).
- A hard 30-second timeout prevents runaway scripts.

The caller is responsible for passing only LLM-authored code. The subprocess
has no network access restrictions beyond what the OS enforces, which is
acceptable for a local research tool where the LLM is the trusted source.
"""

import logging
import os
import subprocess
import sys
import tempfile
import uuid

logger = logging.getLogger(__name__)
CHARTS_DIR = "static/charts"


def run_chart_code(code: str, extra_setup: str = "") -> str:
    """Execute matplotlib chart code and save the result as a PNG.

    Args:
        code:        The chart body — no imports, no savefig/show calls.
                     The figure must be created with plt.subplots() or plt.figure().
        extra_setup: Optional lines injected before `code` (e.g. loading a DataFrame).

    Returns:
        Public URL of the PNG on success (e.g. "/static/charts/chart_abc123.png"),
        or an empty string if the script fails or times out.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)

    filename = f"chart_{uuid.uuid4().hex[:10]}.png"
    filepath = os.path.abspath(os.path.join(CHARTS_DIR, filename))

    # Build the wrapper script.
    indented_code = "\n".join("    " + line for line in code.splitlines())
    script = f"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
try:
    import seaborn as _sns
except ImportError:
    _sns = None

{extra_setup}

# Apply seaborn theme after extra_setup (which sets BG/PANEL/TEXT/PALETTE)
if _sns is not None:
    try:
        _style = "whitegrid" if BG in ("white", "#ffffff", "#FFFFFF") else "darkgrid"
        _sns.set_theme(style=_style, font_scale=1.05,
                       rc={{"axes.spines.right": False, "axes.spines.top": False,
                            "font.family": "DejaVu Sans"}})
    except Exception:
        pass

try:
{indented_code}
    plt.tight_layout(pad=1.5)
    plt.savefig({filepath!r}, dpi=140, bbox_inches="tight")
    plt.close("all")
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        tmp = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(
                "chart_exec_failed file=%r stderr=%s",
                tmp,
                result.stderr[:600],
            )
            return ""
        return f"/static/charts/{filename}"
    except subprocess.TimeoutExpired:
        logger.error("chart_exec_timeout code_preview=%r", code[:120])
        return ""
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
