for n in $(seq $1 $2); do
    python CotQA_context.py --n $n | tee cotqa_context_$n.log
done
