#!/bin/sh
RTD_TOKEN=$(<.rtd_token)
curl -X POST https://app.readthedocs.org/api/v2/webhook/ilayoutx/317850/ -H "Content-Type: application/json" -d "{\"token\":\"${RTD_TOKEN}\"}"
