for filename in jobs/tdmpc/*.pbs; do
    echo $filename;
    qsub $filename;
done