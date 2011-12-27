Author: Ivan Vladimirov Ivanov (ivan.vladimirov.ivanov@gmail.com)

This project implements a basic Multilayer Feedforward Neural Network
and Naive Bayes classifier framework. The implementations are compared
against each other on 3 generic data sets which can be found at:

* http://archive.ics.uci.edu/ml/datasets/Abalone
* http://archive.ics.uci.edu/ml/datasets/Wine
* http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Performance.pdf provides a summary of the performance of various topologies
of the Neural Network and the Naive Bayes classifier on the example data sets.

The learning curves of the Neural Network and Naive Bayes classifier can be
found in:

* feedforward_neural_network/figures/
* naive_bayes/figures/

The feedforward_neural_network/ directory contains the Neural Network
implementation, 3 experiments and the following scripts:

* ./build.sh                       - builds the Java source code.
* ./run_abalone_experiment.sh      - runs the first Abalone experiment.
* ./run_breastcancer_experiment.sh - runs the breast cancer experiment.
* ./run_wine_experiment.sh         - runs the wine experiment.
* ./ANN_learning_curve.py          - generates a learning curve diagram
                                     (requires the matplotlib module).

The naive_bayes/ directory contains the Naive Bayes
implementation, 3 experiments and the following scripts:

* ./build.sh                       - builds the Java source code.
* ./run_abalone_experiment.sh      - runs the first Abalone experiment.
* ./run_breastcancer_experiment.sh - runs the breast cancer experiment.
* ./run_wine_experiment.sh         - runs the wine experiment.
* ./NB_learning_curve.py           - generates a learning curve diagram
                                     (requires the matplotlib module).

