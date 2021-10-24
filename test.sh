#!/bin/bash

set -e

for i in {32..512..7}
do
  echo "Testing for i = $i"
  ./target/*/*.out $i
  echo ""
done