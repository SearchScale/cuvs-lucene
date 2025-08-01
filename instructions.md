 # 1. Clean any existing index
  rm -rf myindex_lucene101

  # 2. Compile the program
  javac -cp "target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" NativeHNSWLucene101.java

  # 3. Run the program to generate index
  java -cp ".:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" NativeHNSWLucene101

  Output: Creates myindex_lucene101/ directory with native Lucene HNSW files

  Program 2: Verify Index with Lucene101 Codec

  File: TestLucene101Index.java

  Steps:
  # 1. Compile the test program
  javac -cp "target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" TestLucene101Index.java

  # 2. Run the verification program
  java -cp ".:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" TestLucene101Index

  What it verifies:
  - Index opens with Lucene101Codec (without CuVSCodec)
  - Vector fields are accessible (knn1, knn2)
  - Vector search queries return reasonable results

  Complete Workflow

  # Full regeneration and verification
  rm -rf myindex_lucene101
  java -cp ".:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" NativeHNSWLucene101
  java -cp ".:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" TestLucene101Index
