#!/usr/bin/env python3
"""
Start MCP Server for Solar System Tools
"""

import asyncio
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from solar_mcp_server import main

if __name__ == "__main__":
    print("Starting Solar System MCP Server...")
    asyncio.run(main())
