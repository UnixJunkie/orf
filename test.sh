#!/bin/bash

dune build src/test.exe

time _build/default/src/test.exe -n 1 -np 1 -feats 0.5 -samps 1.0 -min 1 \
     -tr data/train.csv -te data/test.csv \
     -rtr data/train_regr.csv -rte data/test_regr.csv
