squeue -u $USER | grep infer | awk '{print $1}' | tail -n+2 | xargs scancel
rm -f *.err *.out
rm out_*.txt
