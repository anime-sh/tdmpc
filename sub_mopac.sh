for filename in jobs/mopac/*.pbs; do
    echo $filename;
    qsub $filename;
done