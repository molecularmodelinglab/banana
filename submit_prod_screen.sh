#!/usr/bin/bash
for i in {0..799};
do
sbatch prod_screen.sh $@ $i
done
