#!/bin/bash

echo "Running Unit tests"

pytest --random-order --cov=chaindrift --cov-config=.coveragerc tests/
