#!/usr/bin/env python3
"""Compatibility entrypoint for validated author-level research experiments.

The former script mixed post-level uncertainty with user-level claims and wrote
clinical effect-size constants as results. Its historical source is archived
under research/archive for audit, and it must not generate publication metrics.
"""
from src.research_benchmark import main

if __name__ == '__main__':
    main()
