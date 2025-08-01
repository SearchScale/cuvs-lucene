How to run
==========

● To run the NativeHNSWSerialization program, use this command:

  rm -rf myindex

  export LD_LIBRARY_PATH=/home/ishan/code/cuvs/cpp/build:$LD_LIBRARY_PATH && java -cp "target/test-classes:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout 
  -q)" com.nvidia.cuvs.lucene.NativeHNSWSerialization

  Make sure to:
  1. First compile the project: mvn test-compile
  2. Clean the index directory if needed: rm -rf myindex/
  3. Then run the command above

  The program will create a Lucene index with Native HNSW (CAGRA) vectors and display the vector field information.


