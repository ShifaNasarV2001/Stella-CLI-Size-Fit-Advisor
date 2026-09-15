#!/usr/bin/env python3
"""Entrypoint. All behaviour lives in the stella package."""

import sys

from stella.cli import main

if __name__ == "__main__":
    sys.exit(main())
