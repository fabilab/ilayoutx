#!/bin/sh
.venv/bin/coverage run -m pytest && .venv/bin/coverage xml && .venv/bin/genbadge coverage -i coverage.xml
