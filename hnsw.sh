rm -rf helloindex *.class
mvn -DskipTests=true clean compile package

# Create myindex
javac -cp "target/test-classes:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" NativeHNSWSerialization.java && java -cp  ".:target/test-classes:target/classes:$(mvn dependency:build-classpath   -Dmdep.outputFile=/dev/stdout -q)" NativeHNSWSerialization

# Compile and run TestIndexWithLucene101
javac -cp "target/test-classes:target/classes:$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)" TestIndexWithLucene101.java && java -cp  ".:target/test-classes:target/classes:$(mvn dependency:build-classpath   -Dmdep.outputFile=/dev/stdout -q)" TestIndexWithLucene101
