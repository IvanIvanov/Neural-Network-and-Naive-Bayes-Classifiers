#!/usr/bin/python
import os
import sys

from matplotlib import pyplot

def main():
  name = 'Abalone'
  start = 1
  end =  3133
  step = 100
  plot = True
  for i in range(1, len(sys.argv[1:]), 2):
    if sys.argv[i] == '-name':
      name = sys.argv[i + 1]
    elif sys.argv[i] == '-start':
      start = int(sys.argv[i + 1])
    elif sys.argv[i] == '-end':
      end = int(sys.argv[i + 1])
    elif sys.argv[i] == '-step':
      step = int(sys.argv[i + 1])
    elif sys.argv[i] == '-plot':
      plot = True if sys.argv[i + 1] == 'True' else False

  examples = start
  x = []
  y = []
  if not plot:
    print 'examples,percent'
  while examples <= end:
    result = float(os.popen('./run_%s_experiment.sh -e %d' %
        (name.lower(), examples), 'r').readline())
    if not plot:
      print '%d,%.4f' % (examples, result)
    x.append(examples)
    y.append(result)
    examples += step

  if plot:
    pyplot.plot(x, y)
    pyplot.xlabel('Training Set Size')
    pyplot.ylabel('Proportion Correct')
    pyplot.title('Neural Network Learning Curve - ' + name)
    pyplot.show()


if __name__ == '__main__':
  main()

