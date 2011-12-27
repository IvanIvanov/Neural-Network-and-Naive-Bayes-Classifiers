#!/bin/bash
rm -rf build
mkdir -p build/classes
javac -d build/classes/ src/neuralnet/*.java src/neuralnet/experiments/*.java

