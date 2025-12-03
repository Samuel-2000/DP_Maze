#!/bin/bash

# Train all architectures
python run.py train --network-type lstm --epochs 5000
python run.py train --network-type transformer --epochs 5000 --auxiliary-tasks
python run.py train --network-type multimemory --epochs 5000 --auxiliary-tasks