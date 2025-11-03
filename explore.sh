#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
./venv/bin/python ./exploration "$@"