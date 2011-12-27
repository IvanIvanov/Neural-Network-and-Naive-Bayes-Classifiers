#!/bin/bash
java -cp build/classes naivebayes.experiments.BreastCancerExperiment -f 'data/wdbc.data' $@

