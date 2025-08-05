rm -rf helloindex *.class
mvn -DskipTests=true clean compile package

# Create myindex
javac -cp "target/test-classes:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" NativeHNSWSerialization.java && java -cp  ".:target/test-classes:target/classes:$(mvn dependency:build-classpath   -Dmdep.outputFile=/dev/stdout -q)" NativeHNSWSerialization

# Search on myindex without using GPU
#javac -cp "target/test-classes:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" TestIndexWithLucene101.java && java -cp  ".:target/test-classes:target/classes:$(mvn dependency:build-classpath   -Dmdep.outputFile=/dev/stdout -q)" TestIndexWithLucene101

# Ensure CuVSCodec SPI registration file exists
mkdir -p src/main/resources/META-INF/services
echo "com.nvidia.cuvs.lucene.CuVSCodec" > src/main/resources/META-INF/services/org.apache.lucene.codecs.Codec

# Clean and compile the project
mvn -DskipTests=true clean compile package

# Compile and run TestIndexWithLucene101
javac -cp "target/test-classes:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" TestIndexWithLucene101.java && java -cp  ".:target/test-classes:target/classes:$(mvn dependency:build-classpath   -Dmdep.outputFile=/dev/stdout -q)" TestIndexWithLucene101
