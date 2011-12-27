#!/bin/bash
java -cp build/classes/ neuralnet.experiments.BreastCancerExperiment -f data/wdbc.data $@

