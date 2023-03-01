#!/bin/bash
#SBATCH -J infer
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=infer-%x.%j.out
#SBATCH --error=infer-%x.%j.err

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/work/users/m/i/mixarcid/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/work/users/m/i/mixarcid/mixarcid/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/work/users/m/i/mixarcid/mixarcid/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/work/users/m/i/mixarcid/mixarcid/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate banana

cd /nas/longleaf/home/mixarcid/banana

python infer_nsp3.py $@
