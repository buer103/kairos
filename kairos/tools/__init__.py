"""Kairos tools package.

Auto-discovered tool modules. Importing a module registers its tools
via the @register_tool decorator.
"""

from kairos.tools import browser_tools  # noqa: F401
from kairos.tools import mcp_tools  # noqa: F401
from kairos.tools import vision_tools  # noqa: F401
