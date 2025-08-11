#!/bin/bash

# CuVS-Lucene CPUSearchCodec Benchmark using vectorsearch-benchmarks
echo "🚀 CuVS-Lucene CPUSearchCodec Benchmark (using vectorsearch-benchmarks)"
echo "========================================================================"

# Parse arguments
DATASET_TYPE=${1:-"sift"}
JOB_CONFIG_FILE=""

echo "📋 Benchmark Configuration:"
echo "  - Dataset: $DATASET_TYPE"

# Set up environment
echo "🔧 Setting up environment..."

# Set Java 22
export JAVA_HOME=/opt/jdk-22.0.2/
export PATH=$JAVA_HOME/bin:$PATH

echo "☕ Using Java: $(java -version 2>&1 | head -n 1)"

# Activate conda environment
echo "🐍 Activating conda environment cuvs-25.08..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cuvs-25.08

# Set library paths for CuVS native libraries
export LD_LIBRARY_PATH=/home/puneet/miniforge3/envs/cuvs-25.08/lib:$LD_LIBRARY_PATH
export JAVA_LIBRARY_PATH="/home/puneet/miniforge3/envs/cuvs-25.08/lib"

# Verify environment
echo "🔍 Environment check:"
echo "  - JAVA_HOME: $JAVA_HOME"
echo "  - Java version: $(java -version 2>&1 | head -n 1)"
echo "  - Conda env: $CONDA_DEFAULT_ENV"
echo "  - LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check if CuVS native library exists
if [ -f "/home/puneet/miniforge3/envs/cuvs-25.08/lib/libcuvs_c.so" ]; then
    echo "✅ CuVS native library found: libcuvs_c.so"
else
    echo "⚠️  CuVS native library not found - will fall back to CPU"
fi

# CuVS-Java dependency is now installed locally in Maven repository
echo "🔍 Checking CuVS-Java dependency..."
CUVS_JAVA_JAR=$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q | tr ':' '\n' | grep "cuvs-java.*jar" | head -1)

if [ -z "$CUVS_JAVA_JAR" ]; then
    echo "❌ CuVS-Java JAR not found in Maven dependencies"
    echo "💡 Run: mvn install:install-file -Dfile=/home/puneet/code/cuvs-repo/cuvs/java/cuvs-java/target/cuvs-java-25.10.0-lorenzo-9362a-SNAPSHOT-jar-with-dependencies.jar -DgroupId=com.nvidia.cuvs -DartifactId=cuvs-java -Dversion=25.10.0-lorenzo-9362a-SNAPSHOT -Dpackaging=jar"
    exit 1
fi
echo "✅ CuVS-Java JAR found: $CUVS_JAVA_JAR"

# Build cuvs-lucene if needed
echo "🔍 Checking cuvs-lucene build..."
CUVS_LUCENE_JAR="/home/puneet/code/1cuvs-lucene/cuvs-lucene/target/cuvs-lucene-0.0.1-SNAPSHOT.jar"
if [ ! -f "$CUVS_LUCENE_JAR" ]; then
    echo "🔨 Building cuvs-lucene..."
    mvn clean compile package -DskipTests
    if [ $? -ne 0 ]; then
        echo "❌ Failed to build cuvs-lucene"
        exit 1
    fi
fi
echo "✅ CuVS-Lucene JAR found: $CUVS_LUCENE_JAR"

# Determine job configuration file
case "$DATASET_TYPE" in
    "sift"|"sift-small")
        JOB_CONFIG_FILE="jobs/job_cpu_search_sift.json"
        echo "📋 Using SIFT dataset configuration"
        ;;
    "wikipedia"|"wiki")
        JOB_CONFIG_FILE="jobs/job_cpu_search_wikipedia.json"
        echo "📋 Using Wikipedia dataset configuration (1M vectors)"
        ;;
    "wikipedia-small"|"wiki-small")
        JOB_CONFIG_FILE="jobs/job_cpu_search_wikipedia_small.json"
        echo "📋 Using Wikipedia dataset configuration (10K vectors - test mode)"
        ;;
    "wikipedia-2M"|"wiki-2M")
        JOB_CONFIG_FILE="jobs/job_cpu_search_wikipedia_2M.json"
        echo "📋 Using Wikipedia dataset configuration (2M vectors)"
        ;;
    "wikipedia-2M-optimized"|"wiki-2M-opt")
        JOB_CONFIG_FILE="jobs/job_cpu_search_wikipedia_2M_optimized.json"
        echo "📋 Using Wikipedia dataset configuration (2M vectors - optimized for memory)"
        ;;
    *)
        echo "❌ Unknown dataset type: $DATASET_TYPE"
        echo "Available options: sift, wikipedia, wikipedia-small, wikipedia-2M, wikipedia-2M-optimized"
        exit 1
        ;;
esac

if [ ! -f "$JOB_CONFIG_FILE" ]; then
    echo "❌ Job configuration file not found: $JOB_CONFIG_FILE"
    exit 1
fi

echo "✅ Job configuration: $JOB_CONFIG_FILE"

# Set Java memory settings based on dataset size
if [[ "$DATASET_TYPE" == "wikipedia-2M" ]]; then
    # For 2M vectors with 2048 dimensions, need more heap memory
    export JAVA_OPTS="-Xmx20g -Xms4g"
    echo "🔧 Setting Java heap size to 20GB for large dataset"
elif [[ "$DATASET_TYPE" == "wikipedia-2M-optimized" ]]; then
    # For optimized 2M vectors, use more conservative memory
    export JAVA_OPTS="-Xmx18g -Xms2g"
    echo "🔧 Setting Java heap size to 18GB for optimized large dataset"
elif [[ "$DATASET_TYPE" == "wikipedia" ]]; then
    # For 1M vectors, moderate memory
    export JAVA_OPTS="-Xmx12g -Xms2g"
    echo "🔧 Setting Java heap size to 12GB for medium dataset"
else
    # For smaller datasets like SIFT, default memory
    export JAVA_OPTS="-Xmx8g -Xms1g"
    echo "🔧 Setting Java heap size to 8GB for small dataset"
fi

# Build the comprehensive benchmark class
echo "🔨 Building comprehensive benchmark..."
MAVEN_DEPS=$(mvn dependency:build-classpath -Dmdep.outputFile=/dev/stdout -q)
mkdir -p target/benchmark-classes

# First ensure main classes are compiled
mvn compile -q
if [ $? -ne 0 ]; then
    echo "❌ Failed to compile main cuvs-lucene classes"
    exit 1
fi

# Compile all benchmark classes together with proper classpath
echo "  - Compiling benchmark classes..."
javac -cp "$MAVEN_DEPS:$CUVS_JAVA_JAR:target/classes" \
      -d target/benchmark-classes \
      src/main/java/com/nvidia/cuvs/lucene/benchmark/*.java

if [ $? -ne 0 ]; then
    echo "❌ Failed to compile comprehensive benchmark"
    echo "  - Trying alternative compilation method..."
    
    # Alternative: compile each file individually to see which one fails
    for java_file in src/main/java/com/nvidia/cuvs/lucene/benchmark/*.java; do
        echo "  - Compiling: $(basename $java_file)"
        javac -cp "$MAVEN_DEPS:$CUVS_JAVA_JAR:target/classes" \
              -d target/benchmark-classes \
              "$java_file"
        if [ $? -ne 0 ]; then
            echo "❌ Failed to compile: $(basename $java_file)"
            exit 1
        fi
    done
fi

echo "✅ Comprehensive benchmark compiled successfully"

# Create results directory
mkdir -p results

# Run the benchmark
echo "🚀 Running CuVS-Lucene CPUSearchCodec comprehensive benchmark..."
echo "  - Job Config: $JOB_CONFIG_FILE"

java $JAVA_OPTS \
     --enable-native-access=ALL-UNNAMED \
     -Djava.library.path="/home/puneet/miniforge3/envs/cuvs-25.08/lib" \
     -cp ".:target/classes:target/benchmark-classes:$MAVEN_DEPS:$CUVS_JAVA_JAR" \
     com.nvidia.cuvs.lucene.benchmark.ComprehensiveBenchmark \
     "$JOB_CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Benchmark completed successfully!"
    echo "📁 Results saved in: results/"
    if [ -d "results" ]; then
        ls -la results/
    fi
else
    echo "❌ Benchmark failed!"
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "   1. Ensure CuVS native library is available"
    echo "   2. Check dataset file paths in the job configuration"
    echo "   3. Verify cuvs-java JAR is built correctly"
    echo "   4. Try the simple benchmark first: ./run_benchmark.sh"
fi

echo ""
echo "📖 Available benchmark options:"
echo "  ./run_vectorsearch_benchmark.sh sift                    # SIFT dataset with CPUSearchCodec"
echo "  ./run_vectorsearch_benchmark.sh wikipedia               # Wikipedia dataset with CPUSearchCodec (1M vectors)"
echo "  ./run_vectorsearch_benchmark.sh wikipedia-small         # Wikipedia dataset with CPUSearchCodec (10K vectors - test)"
echo "  ./run_vectorsearch_benchmark.sh wikipedia-2M            # Wikipedia dataset with CPUSearchCodec (2M vectors)"
echo "  ./run_vectorsearch_benchmark.sh wikipedia-2M-optimized  # Wikipedia dataset with CPUSearchCodec (2M vectors - memory optimized)"