#!/bin/sh
COV=.venv/bin/coverage
GENBADGE=.venv/bin/genbadge

$COV erase
$COV run -m pytest && $COV xml --omit="*/experimental/*" && $COV html && $GENBADGE coverage -i coverage.xml
xdg-open htmlcov/index.html &
