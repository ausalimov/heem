cd JGibbLDA
for filename in ../lda_inputs/*; do
    base_name=`basename "$filename"`
    output_file=../JGibbLDA_models/$base_name
    mkdir -p $output_file
    cp "$filename" $output_file/"$base_name".tt
    
    echo $output_file
    echo $output_file/$base_name
    echo $base_name
    jgibbs_cmd="java -cp bin:lib/args4j-2.0.6.jar jgibblda.LDA -est -dir $output_file -dfile ${base_name}.tt"
    # echo $jgibbs_cmd
    $jgibbs_cmd
    
done
cd .. 
