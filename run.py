#!/usr/bin/env python3

import sys
import curses
from smartcash.cli import main

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("\nKeluar dari SmartCash...")
        sys.exit(0)
