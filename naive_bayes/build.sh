#!/bin/bash
rm -rf build
mkdir -p build/classes
javac -d build/classes src/naivebayes/*.java src/naivebayes/experiments/*.java

