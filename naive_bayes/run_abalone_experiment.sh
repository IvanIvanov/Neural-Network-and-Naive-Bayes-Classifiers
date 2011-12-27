#!/bin/bash
java -cp build/classes naivebayes.experiments.AbaloneExperiment -f 'data/abalone.data' $@

