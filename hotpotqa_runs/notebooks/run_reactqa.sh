set -x
for n in $(seq $1 $2); do
    python ReactQA.py --n $n | tee reactqa_$n.log
done
