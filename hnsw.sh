rm -rf myindex *.class
export LD_LIBRARY_PATH=/home/ishan/code/cuvs/cpp/build

mvn -DskipTests=true clean compile package

# Create myindex
javac -cp "target/test-classes:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" NativeHNSWSerialization.java && java -cp  ".:target/test-classes:target/classes:$(mvn dependency:build-classpath   -Dmdep.outputFile=/dev/stdout -q)" NativeHNSWSerialization

# Search on myindex without using GPU
javac -cp "target/test-classes:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" TestIndexWithLucene101.java && java -cp  ".:target/test-classes:target/classes:$(mvn dependency:build-classpath   -Dmdep.outputFile=/dev/stdout -q)" TestIndexWithLucene101
