#!/bin/bash
java -cp build/classes naivebayes.experiments.WineExperiment -f 'data/wine.data' $@

