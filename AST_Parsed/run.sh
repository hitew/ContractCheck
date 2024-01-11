#!/bin/bash




directory="./data/vun_data/"
command="java -classpath ./AST_Parsed/antlr4.jar:./AST_Parsed/target Tokenize"


for file in "$directory"0x*; do
   
    filename=$(basename "$file")
    extension="${filename##*.}"

    
    contract_name="${filename%.*}"

    
    $command "$file" "./AST_Parsed/output/${contract_name}"
done
