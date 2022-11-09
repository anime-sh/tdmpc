for filename in jobs/*.pbs; do
    echo $filename;
    qsub $filename;
done