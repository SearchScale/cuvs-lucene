/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.cuvs.lucene.benchmark;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.nvidia.cuvs.lucene.CuVSCPUSearchCodec;
import com.nvidia.cuvs.lucene.CuVSKnnFloatVectorQuery;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.zip.GZIPInputStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.IndexTreeList;

/**
 * Comprehensive CuVS-Lucene CPUSearchCodec Benchmark
 *
 * This benchmark supports the vectorsearch-benchmarks JSON configuration format
 * and provides comprehensive metrics for the CuVSCPUSearchCodec workflow:
 * 1. GPU indexing with CuVS (CAGRA graph construction)
 * 2. Graph serialization to Lucene format
 * 3. CPU search using Lucene HNSW on the serialized graph
 */
public class ComprehensiveBenchmark {

  private static final ObjectMapper mapper = new ObjectMapper();

  public static void main(String[] args) throws Exception {
    if (args.length != 1) {
      System.err.println("Usage: java ComprehensiveBenchmark <job-config.json>");
      System.err.println("Example job configs are in the jobs/ directory");
      return;
    }

    System.out.println("🚀 CuVS-Lucene CPUSearchCodec Comprehensive Benchmark");
    System.out.println("=====================================================");
    System.out.println("Workflow: GPU Indexing (CuVS CAGRA) → CPU Search (Lucene HNSW)");
    System.out.println();

    // Load configuration
    BenchmarkConfig config = mapper.readValue(new File(args[0]), BenchmarkConfig.class);
    config.debugPrintArguments();
    System.out.println();

    // Validate configuration
    if (!validateConfig(config)) {
      System.err.println("❌ Configuration validation failed");
      return;
    }

    // Load dataset
    System.out.println("📂 Loading dataset...");
    BenchmarkDataset dataset = loadDataset(config);

    if (dataset == null) {
      System.err.println("❌ Failed to load dataset");
      return;
    }

    System.out.println("✅ Dataset loaded successfully:");
    System.out.println("  - Base vectors: " + dataset.vectors.size() + " × " + dataset.dimensions);
    System.out.println(
        "  - Query vectors: "
            + dataset.queries.size()
            + " × "
            + (dataset.queries.isEmpty() ? 0 : dataset.queries.get(0).length));
    if (dataset.hasGroundTruth()) {
      System.out.println(
          "  - Ground truth: "
              + dataset.groundTruth.size()
              + " × "
              + (dataset.groundTruth.isEmpty() ? 0 : dataset.groundTruth.get(0).length));
    }
    System.out.println();

    // Run benchmark
    ObjectNode results = runBenchmark(config, dataset);

    // Print results
    printResults(config, results);

    // Save results if requested
    if (config.saveResultsOnDisk) {
      saveResults(config, results);
    }
  }

  private static boolean validateConfig(BenchmarkConfig config) {
    System.out.println("🔍 Validating configuration...");

    // Check required fields
    if (config.benchmarkID == null || config.benchmarkID.trim().isEmpty()) {
      System.err.println("❌ benchmarkID is required");
      return false;
    }

    if (config.datasetFile == null || config.datasetFile.trim().isEmpty()) {
      System.err.println("❌ datasetFile is required");
      return false;
    }

    if (config.queryFile == null || config.queryFile.trim().isEmpty()) {
      System.err.println("❌ queryFile is required");
      return false;
    }

    if (config.numDocs <= 0) {
      System.err.println("❌ numDocs must be positive, got: " + config.numDocs);
      return false;
    }

    if (config.numQueriesToRun <= 0) {
      System.err.println("❌ numQueriesToRun must be positive, got: " + config.numQueriesToRun);
      return false;
    }

    if (config.topK <= 0) {
      System.err.println("❌ topK must be positive, got: " + config.topK);
      return false;
    }

    // Validate file extensions
    String datasetExt = getFileExtension(config.datasetFile);
    String queryExt = getFileExtension(config.queryFile);

    if (!isSupportedDatasetFormat(datasetExt)) {
      System.err.println("❌ Unsupported dataset format: " + datasetExt);
      System.err.println("   Supported formats: .fvecs, .csv, .csv.gz");
      return false;
    }

    if (!isSupportedQueryFormat(queryExt)) {
      System.err.println("❌ Unsupported query format: " + queryExt);
      System.err.println("   Supported formats: .fvecs, .csv, .csv.gz");
      return false;
    }

    // Validate column indices
    if (config.indexOfVector < 0) {
      System.err.println("❌ indexOfVector must be non-negative, got: " + config.indexOfVector);
      return false;
    }

    if (config.queryIndexOfVector != null && config.queryIndexOfVector < 0) {
      System.err.println(
          "❌ queryIndexOfVector must be non-negative, got: " + config.queryIndexOfVector);
      return false;
    }

    System.out.println("✅ Configuration validation passed");
    return true;
  }

  private static String getFileExtension(String filename) {
    int lastDot = filename.lastIndexOf('.');
    if (lastDot == -1) return "";

    // Handle .csv.gz case
    if (filename.endsWith(".csv.gz")) return ".csv.gz";

    return filename.substring(lastDot);
  }

  private static boolean isSupportedDatasetFormat(String ext) {
    return ext.equals(".fvecs") || ext.equals(".csv") || ext.equals(".csv.gz");
  }

  private static boolean isSupportedQueryFormat(String ext) {
    return ext.equals(".fvecs") || ext.equals(".csv") || ext.equals(".csv.gz");
  }

  private static BenchmarkDataset loadDataset(BenchmarkConfig config) throws IOException {
    List<float[]> vectors = null;
    List<float[]> queries = null;
    List<int[]> groundTruth = null;

    // Validate file existence first
    if (!new File(config.datasetFile).exists()) {
      System.err.println("❌ Dataset file not found: " + config.datasetFile);
      return null;
    }

    if (!new File(config.queryFile).exists()) {
      System.err.println("❌ Query file not found: " + config.queryFile);
      return null;
    }

    // Load base vectors with MapDB caching
    System.out.println("📖 Loading base vectors from: " + config.datasetFile);

    // Check for existing MapDB cache in dataset directory first (prioritize full caches)
    String datasetDir = new File(config.datasetFile).getParent();
    String datasetName = new File(config.datasetFile).getName();
    String[] possibleMapdbFiles = {
      datasetDir + "/" + datasetName + ".mapdb", // Full cache in dataset directory (preferred)
      "cache_" + datasetName + ".mapdb", // Current directory cache
      datasetDir + "/cache_" + datasetName + ".mapdb" // Dataset directory with cache prefix
    };

    String selectedMapdbFile = null;
    for (String mapdbFile : possibleMapdbFiles) {
      if (new File(mapdbFile).exists()) {
        selectedMapdbFile = mapdbFile;
        break;
      }
    }

    if (selectedMapdbFile != null) {
      System.out.println("  - Found MapDB cache: " + selectedMapdbFile);
      try {
        // Load from existing cache
        vectors = loadVectorsFromMapDB(selectedMapdbFile, config.numDocs);
        if (vectors != null && !vectors.isEmpty()) {
          System.out.println("✅ Loaded " + vectors.size() + " vectors from MapDB cache");
        } else {
          System.err.println("  ⚠️  MapDB cache is empty or corrupted, will parse file");
          selectedMapdbFile = null; // Fall back to file parsing
        }
      } catch (Exception e) {
        System.err.println(
            "  ⚠️  Failed to load from MapDB cache, falling back to file parsing: "
                + e.getMessage());
        selectedMapdbFile = null; // Fall back to file parsing
      }
    }

    if (selectedMapdbFile == null) {
      System.out.println("  - No existing cache found, will parse file and create cache...");
      try {
        // Parse the file and create cache
        vectors =
            loadCSVVectors(
                config.datasetFile, config.indexOfVector, config.numDocs, config.hasColNames);
        if (vectors != null && !vectors.isEmpty()) {
          // Create cache for future use
          String cacheFile = "cache_" + datasetName + ".mapdb";
          System.out.println("  - Creating MapDB cache: " + cacheFile);
          try {
            createMapDBCache(vectors, cacheFile);
            System.out.println("✅ MapDB cache created successfully");
          } catch (Exception e) {
            System.err.println("  ⚠️  Failed to create MapDB cache: " + e.getMessage());
          }
        }
      } catch (Exception e) {
        System.err.println("❌ Failed to parse source file: " + e.getMessage());
        return null;
      }
    }

    // Load query vectors
    System.out.println("📖 Loading query vectors from: " + config.queryFile);
    if (config.queryFile.endsWith(".fvecs")) {
      queries = FvecsReader.readFvecs(config.queryFile, config.numQueriesToRun);
    } else if (config.queryFile.endsWith(".csv") || config.queryFile.endsWith(".csv.gz")) {
      // Check for existing MapDB cache first
      String queryMapdbFile = "cache_" + new File(config.queryFile).getName() + ".mapdb";
      String[] possibleQueryMapdbFiles = {
        queryMapdbFile, // Current directory
        new File(config.queryFile).getParent()
            + "/"
            + new File(config.queryFile).getName()
            + ".mapdb", // Same directory as query file
        new File(config.queryFile).getParent()
            + "/cache_"
            + new File(config.queryFile).getName()
            + ".mapdb" // Same directory with cache prefix
      };

      String selectedQueryMapdbFile = null;
      for (String mapdbFile : possibleQueryMapdbFiles) {
        if (new File(mapdbFile).exists()) {
          selectedQueryMapdbFile = mapdbFile;
          break;
        }
      }

      if (selectedQueryMapdbFile != null) {
        System.out.println("  - Found query MapDB cache: " + selectedQueryMapdbFile);
        try {
          queries = loadVectorsFromMapDB(selectedQueryMapdbFile, config.numQueriesToRun);
          System.out.println("✅ Loaded " + queries.size() + " query vectors from MapDB cache");
        } catch (Exception e) {
          System.err.println(
              "  ⚠️  Failed to load from MapDB cache, falling back to file parsing: "
                  + e.getMessage());
          selectedQueryMapdbFile = null; // Fall back to file parsing
        }
      }

      if (selectedQueryMapdbFile == null) {
        System.out.println("  - No existing query cache found, will parse file...");
        try {
          // Try to detect query file format by examining first few lines
          boolean isLineBasedFormat = detectQueryFileFormat(config.queryFile, config.hasColNames);

          if (isLineBasedFormat) {
            // Use line-based parsing (like vectorsearch-benchmarks)
            queries =
                loadQueryVectorsFromLineStrings(
                    config.queryFile, config.numQueriesToRun, config.hasColNames);
          } else {
            // Use column-based parsing (original approach)
            int queryColumnIndex = config.queryIndexOfVector;
            queries =
                loadCSVVectors(
                    config.queryFile, queryColumnIndex, config.numQueriesToRun, config.hasColNames);
          }
        } catch (Exception e) {
          System.err.println("❌ Failed to load query vectors from CSV: " + e.getMessage());
          e.printStackTrace();
          return null;
        }
      }
    } else {
      System.err.println("❌ Unsupported query file format: " + config.queryFile);
      System.err.println("   Supported formats: .fvecs, .csv, .csv.gz");
      return null;
    }

    // Validate query vector dimensions match base vector dimensions
    if (!queries.isEmpty() && !vectors.isEmpty()) {
      int queryDimensions = queries.get(0).length;
      int baseDimensions = vectors.get(0).length;
      if (queryDimensions != baseDimensions) {
        System.err.println("❌ Vector dimension mismatch:");
        System.err.println("   - Base vectors: " + baseDimensions + " dimensions");
        System.err.println("   - Query vectors: " + queryDimensions + " dimensions");
        System.err.println("   - This will cause search failures");
        System.err.println("   - Check your queryIndexOfVector configuration");
        System.err.println("   - Base file: " + config.datasetFile);
        System.err.println("   - Query file: " + config.queryFile);
        return null;
      }
    }

    // Load ground truth if available
    if (config.groundTruthFile != null) {
      if (!new File(config.groundTruthFile).exists()) {
        System.err.println("⚠️  Ground truth file not found: " + config.groundTruthFile);
        System.err.println("   Continuing without ground truth evaluation");
      } else {
        System.out.println("📖 Loading ground truth from: " + config.groundTruthFile);
        try {
          if (config.groundTruthFile.endsWith(".ivecs")) {
            groundTruth = IvecsReader.readIvecs(config.groundTruthFile, config.numQueriesToRun);
          } else if (config.groundTruthFile.endsWith(".csv")) {
            groundTruth = loadCSVGroundTruth(config.groundTruthFile, config.numQueriesToRun);
          } else {
            System.err.println(
                "❌ Unsupported ground truth format. Supported formats: .ivecs, .csv");
          }
        } catch (Exception e) {
          System.err.println("⚠️  Failed to load ground truth: " + e.getMessage());
          groundTruth = null;
        }
      }
    }

    int dimensions = vectors.isEmpty() ? 0 : vectors.get(0).length;
    return new BenchmarkDataset(config.benchmarkID, vectors, queries, groundTruth, dimensions);
  }

  private static ObjectNode runBenchmark(BenchmarkConfig config, BenchmarkDataset dataset)
      throws IOException {
    ObjectNode results = mapper.createObjectNode();
    results.put("benchmarkID", config.benchmarkID);
    results.put("dataset", dataset.name);
    results.put("numVectors", dataset.vectors.size());
    results.put("dimensions", dataset.dimensions);
    results.put("numQueries", dataset.queries.size());
    results.put("topK", config.topK);
    results.put("algorithm", config.algoToRun);

    // Create index directory
    Path indexDir = createIndexDirectory(config);

    try (Directory directory = FSDirectory.open(indexDir)) {

      // Create CuVSCPUSearchCodec
      System.out.println("🧪 Creating CuVSCPUSearchCodec...");
      CuVSCPUSearchCodec codec;
      try {
        codec = new CuVSCPUSearchCodec();
        System.out.println("✅ CuVSCPUSearchCodec created successfully");
      } catch (Exception e) {
        System.out.println("❌ CuVSCPUSearchCodec creation failed: " + e.getMessage());
        e.printStackTrace();
        results.put("status", "FAILED");
        results.put("error", "Codec creation failed: " + e.getMessage());
        return results;
      }

      // GPU indexing phase
      System.out.println("🔧 Starting GPU indexing phase (CAGRA graph construction)...");
      ObjectNode indexingResults = runIndexing(config, dataset, directory, codec);
      results.setAll(indexingResults);

      if ("FAILED".equals(indexingResults.get("status").asText())) {
        return results;
      }

      // CPU search phase
      System.out.println("🔍 Starting CPU search phase (Lucene HNSW on serialized graph)...");
      ObjectNode searchResults = runSearch(config, dataset, directory);
      results.setAll(searchResults);

    } catch (Exception e) {
      System.out.println("❌ Benchmark failed: " + e.getMessage());
      e.printStackTrace();
      results.put("status", "FAILED");
      results.put("error", "Benchmark failed: " + e.getMessage());
    } finally {
      // Clean up if requested
      if (config.cleanIndexDirectory) {
        cleanupIndexDirectory(indexDir);
      }
    }

    return results;
  }

  private static Path createIndexDirectory(BenchmarkConfig config) throws IOException {
    Path indexDir;
    if (config.createIndexInMemory) {
      // For in-memory, we still need a temp directory for the FSDirectory
      indexDir = Files.createTempDirectory("cuvs_benchmark_memory");
    } else {
      indexDir = Path.of(config.cuvsIndexDirPath != null ? config.cuvsIndexDirPath : "cuvsIndex");
      if (Files.exists(indexDir) && config.cleanIndexDirectory) {
        Files.walk(indexDir)
            .sorted((a, b) -> b.compareTo(a))
            .forEach(
                path -> {
                  try {
                    Files.delete(path);
                  } catch (IOException e) {
                    System.err.println("Failed to delete: " + path);
                  }
                });
      }
      Files.createDirectories(indexDir);
    }

    System.out.println("📁 Using index directory: " + indexDir);
    return indexDir;
  }

  private static ObjectNode runIndexing(
      BenchmarkConfig config,
      BenchmarkDataset dataset,
      Directory directory,
      CuVSCPUSearchCodec codec)
      throws IOException {
    ObjectNode results = mapper.createObjectNode();

    long indexingStartTime = System.currentTimeMillis();
    long memoryBefore = getMemoryUsage();

    try {
      // Configure index writer
      IndexWriterConfig iwConfig = new IndexWriterConfig(new StandardAnalyzer());
      iwConfig.setCodec(codec);
      iwConfig.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
      iwConfig.setUseCompoundFile(false);
      iwConfig.setMaxBufferedDocs(config.flushFreq);

      // Create index writer and add documents
      try (IndexWriter writer = new IndexWriter(directory, iwConfig)) {
        System.out.println("📝 Adding " + dataset.vectors.size() + " documents to index...");

        for (int i = 0; i < dataset.vectors.size(); i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField(config.vectorColName, dataset.vectors.get(i), EUCLIDEAN));
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          writer.addDocument(doc);

          if ((i + 1) % 10000 == 0) {
            System.out.println("  - Added " + (i + 1) + " documents");
          }
        }

        // Force merge to ensure graph construction
        System.out.println("🔄 Force merging segments (triggers CAGRA graph construction)...");
        writer.forceMerge(1);
        writer.commit();
      }

      long indexingEndTime = System.currentTimeMillis();
      long indexingTime = indexingEndTime - indexingStartTime;
      long memoryAfter = getMemoryUsage();

      System.out.println("✅ GPU indexing completed successfully");
      System.out.println("  - Indexing time: " + indexingTime + " ms");
      System.out.println(
          "  - Memory usage: " + ((memoryAfter - memoryBefore) / 1024 / 1024) + " MB");

      results.put("indexingTimeMs", indexingTime);
      results.put("memoryUsageMB", (memoryAfter - memoryBefore) / 1024 / 1024);
      results.put("status", "INDEXING_SUCCESS");

    } catch (Exception e) {
      System.out.println("❌ GPU indexing failed: " + e.getMessage());
      e.printStackTrace();
      results.put("status", "FAILED");
      results.put("error", "Indexing failed: " + e.getMessage());
    }

    return results;
  }

  private static ObjectNode runSearch(
      BenchmarkConfig config, BenchmarkDataset dataset, Directory directory) throws IOException {
    ObjectNode results = mapper.createObjectNode();

    long searchStartTime = System.currentTimeMillis();

    try {
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        System.out.println("✅ Index opened successfully for CPU search");
        System.out.println("  - Number of documents: " + reader.numDocs());

        IndexSearcher searcher = new IndexSearcher(reader);

        // Run search queries
        System.out.println("🔎 Running " + dataset.queries.size() + " search queries...");
        List<int[]> searchResults = new ArrayList<>();
        int totalHits = 0;

        for (int i = 0; i < dataset.queries.size(); i++) {
          TopDocs topDocs;
          if ("CAGRA".equals(config.algoToRun)) {
            // Use CuVSKnnFloatVectorQuery for CAGRA mode with proper search parameters
            CuVSKnnFloatVectorQuery query =
                new CuVSKnnFloatVectorQuery(
                    config.vectorColName,
                    dataset.queries.get(i),
                    config.topK,
                    config.cagraITopK, // Use configured iTopK
                    config.cagraSearchWidth // Use configured searchWidth
                    );
            topDocs = searcher.search(query, config.topK);
          } else {
            // Use standard KnnFloatVectorQuery for Lucene HNSW search
            KnnFloatVectorQuery query =
                new KnnFloatVectorQuery(config.vectorColName, dataset.queries.get(i), config.topK);
            topDocs = searcher.search(query, config.topK);
          }
          totalHits += topDocs.scoreDocs.length;

          // Extract document IDs for recall calculation
          int[] resultIds = new int[topDocs.scoreDocs.length];
          for (int j = 0; j < topDocs.scoreDocs.length; j++) {
            Document doc = searcher.storedFields().document(topDocs.scoreDocs[j].doc);
            resultIds[j] = Integer.parseInt(doc.get("id"));
          }
          searchResults.add(resultIds);

          if ((i + 1) % 20 == 0) {
            System.out.println("  - Completed " + (i + 1) + " queries");
          }
        }

        long searchEndTime = System.currentTimeMillis();
        long searchTime = searchEndTime - searchStartTime;

        System.out.println("✅ CPU search completed successfully");
        System.out.println("  - Search time: " + searchTime + " ms");
        System.out.println("  - Total hits: " + totalHits);
        System.out.println("  - Average hits per query: " + (totalHits / dataset.queries.size()));

        results.put("searchTimeMs", searchTime);
        results.put("totalHits", totalHits);
        results.put("avgHitsPerQuery", totalHits / dataset.queries.size());
        results.put("throughputQPS", (dataset.queries.size() * 1000.0) / searchTime);

        // Calculate recall if ground truth is available
        if (dataset.hasGroundTruth()) {
          System.out.println("📊 Calculating recall@" + config.topK + "...");
          double recall = calculateRecall(searchResults, dataset.groundTruth, config.topK);
          System.out.println(
              "  - Recall@"
                  + config.topK
                  + ": "
                  + String.format("%.4f", recall)
                  + " ("
                  + String.format("%.2f", recall * 100)
                  + "%)");
          results.put("recall", recall);
        }

        results.put("status", "SUCCESS");
      }
    } catch (Exception e) {
      System.out.println("❌ CPU search failed: " + e.getMessage());
      e.printStackTrace();
      results.put("status", "FAILED");
      results.put("error", "Search failed: " + e.getMessage());
    }

    return results;
  }

  private static void printResults(BenchmarkConfig config, ObjectNode results) {
    System.out.println();
    System.out.println("📊 Benchmark Results:");
    System.out.println("  - Status: " + results.get("status").asText());
    if (results.has("indexingTimeMs")) {
      System.out.println(
          "  - GPU Indexing Time: " + results.get("indexingTimeMs").asLong() + " ms");
    }
    if (results.has("searchTimeMs")) {
      System.out.println("  - CPU Search Time: " + results.get("searchTimeMs").asLong() + " ms");
    }
    if (results.has("throughputQPS")) {
      System.out.println(
          "  - Throughput: "
              + String.format("%.2f", results.get("throughputQPS").asDouble())
              + " QPS");
    }
    if (results.has("memoryUsageMB")) {
      System.out.println("  - Memory Usage: " + results.get("memoryUsageMB").asLong() + " MB");
    }
    if (results.has("recall")) {
      System.out.println(
          "  - Recall@"
              + config.topK
              + ": "
              + String.format("%.4f", results.get("recall").asDouble())
              + " ("
              + String.format("%.2f", results.get("recall").asDouble() * 100)
              + "%)");
    }

    System.out.println();
    System.out.println("📄 Detailed Results JSON:");
    try {
      mapper.writerWithDefaultPrettyPrinter().writeValue(System.out, results);
    } catch (IOException e) {
      System.err.println("Failed to print JSON results: " + e.getMessage());
    }
    System.out.println();
  }

  private static void saveResults(BenchmarkConfig config, ObjectNode results) {
    try {
      // Create results directory
      Path resultsDir = Path.of("results");
      Files.createDirectories(resultsDir);

      // Generate timestamp
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());

      // Save JSON results
      String resultsFileName = config.benchmarkID + "__benchmark_results_" + timestamp + ".json";
      Path resultsFile = resultsDir.resolve(resultsFileName);
      mapper.writerWithDefaultPrettyPrinter().writeValue(resultsFile.toFile(), results);

      System.out.println("💾 Results saved to: " + resultsFile);

    } catch (IOException e) {
      System.err.println("Failed to save results: " + e.getMessage());
    }
  }

  private static void cleanupIndexDirectory(Path indexDir) {
    try {
      Files.walk(indexDir)
          .sorted((a, b) -> b.compareTo(a))
          .forEach(
              path -> {
                try {
                  Files.delete(path);
                } catch (IOException e) {
                  System.err.println("Failed to delete: " + path);
                }
              });
      System.out.println("🧹 Cleaned up index directory");
    } catch (IOException e) {
      System.err.println("Failed to cleanup index directory: " + e.getMessage());
    }
  }

  private static long getMemoryUsage() {
    Runtime runtime = Runtime.getRuntime();
    return runtime.totalMemory() - runtime.freeMemory();
  }

  private static double calculateRecall(
      List<int[]> searchResults, List<int[]> groundTruth, int topK) {
    if (searchResults.size() != groundTruth.size()) {
      return 0.0;
    }

    double totalRecall = 0.0;
    int validQueries = 0;

    for (int i = 0; i < searchResults.size(); i++) {
      int[] results = searchResults.get(i);
      int[] truth = groundTruth.get(i);

      if (results.length == 0 || truth.length == 0) {
        continue;
      }

      Set<Integer> truthSet = new HashSet<>();
      for (int id : truth) {
        truthSet.add(id);
      }

      int matches = 0;
      int checkCount = Math.min(results.length, topK);
      for (int j = 0; j < checkCount; j++) {
        if (truthSet.contains(results[j])) {
          matches++;
        }
      }

      totalRecall += (double) matches / Math.min(truth.length, topK);
      validQueries++;
    }

    return validQueries > 0 ? totalRecall / validQueries : 0.0;
  }

  // Simple dataset container
  static class BenchmarkDataset {
    final String name;
    final List<float[]> vectors;
    final List<float[]> queries;
    final List<int[]> groundTruth;
    final int dimensions;

    BenchmarkDataset(
        String name,
        List<float[]> vectors,
        List<float[]> queries,
        List<int[]> groundTruth,
        int dimensions) {
      this.name = name;
      this.vectors = vectors;
      this.queries = queries;
      this.groundTruth = groundTruth;
      this.dimensions = dimensions;
    }

    boolean hasGroundTruth() {
      return groundTruth != null && !groundTruth.isEmpty();
    }
  }

  /**
   * Load vectors from MapDB cache
   */
  private static List<float[]> loadVectorsFromMapDB(String mapdbFile, int maxVectors)
      throws Exception {
    DB db = DBMaker.fileDB(mapdbFile).readOnly().make();
    try {
      IndexTreeList<float[]> vectors =
          db.indexTreeList("vectors", org.mapdb.Serializer.FLOAT_ARRAY).createOrOpen();
      List<float[]> result = new ArrayList<>();
      int count = Math.min(maxVectors, vectors.size());
      for (int i = 0; i < count; i++) {
        result.add(vectors.get(i));
      }
      return result;
    } finally {
      db.close();
    }
  }

  /**
   * Create MapDB cache from source file
   */
  private static List<float[]> createMapDBCache(List<float[]> vectors, String mapdbFile)
      throws Exception {
    DB db = DBMaker.fileDB(mapdbFile).make();
    IndexTreeList<float[]> vectorsList =
        db.indexTreeList("vectors", org.mapdb.Serializer.FLOAT_ARRAY).createOrOpen();

    // Store in MapDB
    for (float[] vector : vectors) {
      vectorsList.add(vector);
    }

    db.commit();
    return vectors;
  }

  /**
   * Load vectors from CSV file (supports .csv and .csv.gz)
   * Based on vectorsearch-benchmarks Util.parseCSVFile
   * Supports both formats:
   * 1. String-encoded vectors: "[1.0, 2.0, 3.0]" in a single column
   * 2. Column-based vectors: 1.0, 2.0, 3.0 spread across multiple columns
   */
  private static List<float[]> loadCSVVectors(
      String datasetFile, int vectorColumnIndex, int maxVectors, boolean hasColNames)
      throws IOException, CsvValidationException {
    System.out.println("🔄 Parsing CSV file (this may take a while for large datasets)...");
    System.out.println("  - Using column index: " + vectorColumnIndex);

    List<float[]> vectors = new ArrayList<>();
    InputStreamReader isr;

    if (datasetFile.endsWith(".gz")) {
      var fis = new FileInputStream(datasetFile);
      var bis = new BufferedInputStream(fis, 65536);
      isr = new InputStreamReader(new GZIPInputStream(bis));
    } else {
      var fis = new FileInputStream(datasetFile);
      var bis = new BufferedInputStream(fis, 65536);
      isr = new InputStreamReader(bis);
    }

    try (CSVReader csvReader = new CSVReader(isr)) {
      String[] csvLine;
      int countOfDocuments = 0;
      int malformedLines = 0;
      boolean formatDetected = false;
      boolean isStringEncoded = false;

      while ((csvLine = csvReader.readNext()) != null) {
        if (hasColNames && countOfDocuments == 0) {
          // Skip header line
          countOfDocuments++;
          continue;
        }

        try {
          float[] vector;

          // Auto-detect format on first valid line
          if (!formatDetected) {
            isStringEncoded = detectVectorFormat(csvLine, vectorColumnIndex);
            formatDetected = true;
            System.out.println(
                "  - Detected format: "
                    + (isStringEncoded ? "String-encoded vectors" : "Column-based vectors"));
          }

          if (isStringEncoded) {
            vector = parseFloatArrayFromString(csvLine[vectorColumnIndex]);
          } else {
            vector = parseFloatArrayFromColumns(csvLine, vectorColumnIndex);
          }

          if (countOfDocuments < 3) { // Show first few vector dimensions for debugging
            System.out.println("  - Vector " + countOfDocuments + " dimension: " + vector.length);
          }
          vectors.add(vector);
          countOfDocuments++;

          if (countOfDocuments % 10000 == 0) {
            System.out.println("  📊 Loaded " + countOfDocuments + " vectors...");
          }

          if (countOfDocuments >= maxVectors) {
            break;
          }
        } catch (Exception e) {
          // Count and report malformed lines
          malformedLines++;
          if (malformedLines <= 5) { // Show first 5 errors for debugging
            System.err.println(
                "  ⚠️  Malformed line "
                    + (countOfDocuments + malformedLines)
                    + ": "
                    + e.getMessage());
            if (csvLine.length > vectorColumnIndex) {
              System.err.println(
                  "    - Column "
                      + vectorColumnIndex
                      + " content: '"
                      + csvLine[vectorColumnIndex]
                      + "'");
            } else {
              System.err.println(
                  "    - Line has only "
                      + csvLine.length
                      + " columns, need at least "
                      + (vectorColumnIndex + 1));
            }
          } else if (malformedLines == 6) {
            System.err.println("  ⚠️  Additional malformed lines will be skipped silently...");
          }

          // Print progress indicator
          if (malformedLines % 100 == 0) {
            System.out.print("⚠️");
          } else {
            System.out.print("#");
          }
        }
      }

      System.out.println("\n✅ Loaded " + vectors.size() + " vectors from CSV");
      if (malformedLines > 0) {
        System.out.println("  ⚠️  Skipped " + malformedLines + " malformed lines");
      }
      if (!vectors.isEmpty()) {
        System.out.println("  - Vector dimensions: " + vectors.get(0).length);
      }
    }

    return vectors;
  }

  /**
   * Detect if the CSV contains string-encoded vectors or column-based vectors
   */
  private static boolean detectVectorFormat(String[] csvLine, int vectorColumnIndex) {
    if (vectorColumnIndex >= csvLine.length) {
      throw new IllegalArgumentException(
          "Vector column index "
              + vectorColumnIndex
              + " exceeds CSV line length "
              + csvLine.length);
    }

    String cell = csvLine[vectorColumnIndex];

    // Check if it's a string-encoded vector like "[1.0, 2.0, 3.0]"
    if (cell.startsWith("[") && cell.endsWith("]") && cell.contains(",")) {
      return true;
    }

    // Check if it's a single float value (column-based format)
    try {
      Float.parseFloat(cell);
      return false; // Column-based format
    } catch (NumberFormatException e) {
      // If it's not a single float, assume it's string-encoded
      return true;
    }
  }

  /**
   * Parse vector from multiple consecutive columns starting at vectorColumnIndex
   */
  private static float[] parseFloatArrayFromColumns(String[] csvLine, int vectorColumnIndex) {
    // Count how many consecutive numeric columns we have
    int vectorLength = 0;
    for (int i = vectorColumnIndex; i < csvLine.length; i++) {
      try {
        Float.parseFloat(csvLine[i]);
        vectorLength++;
      } catch (NumberFormatException e) {
        break; // Stop at first non-numeric column
      }
    }

    if (vectorLength == 0) {
      throw new IllegalArgumentException(
          "No valid numeric columns found starting at index " + vectorColumnIndex);
    }

    float[] vector = new float[vectorLength];
    for (int i = 0; i < vectorLength; i++) {
      vector[i] = Float.parseFloat(csvLine[vectorColumnIndex + i]);
    }

    return vector;
  }

  /**
   * Load ground truth from CSV file
   * Each line contains comma-separated integer IDs
   */
  private static List<int[]> loadCSVGroundTruth(String groundTruthFile, int maxQueries)
      throws IOException {
    System.out.println("🔄 Loading CSV ground truth...");

    List<int[]> groundTruth = new ArrayList<>();

    try (BufferedReader reader = new BufferedReader(new FileReader(groundTruthFile))) {
      String line;
      int count = 0;

      while ((line = reader.readLine()) != null && count < maxQueries) {
        try {
          int[] truthIds = parseIntArrayFromString(line);
          groundTruth.add(truthIds);
          count++;
        } catch (Exception e) {
          // Skip malformed lines
          System.out.print("#");
        }
      }
    }

    System.out.println("✅ Loaded ground truth for " + groundTruth.size() + " queries");
    return groundTruth;
  }

  /**
   * Load query vectors from CSV file where each line is a complete vector string
   * This matches the vectorsearch-benchmarks approach for query files
   */
  private static List<float[]> loadQueryVectorsFromLineStrings(
      String queryFile, int maxQueries, boolean hasColNames) throws IOException {
    System.out.println("🔄 Parsing query file as line-based vectors...");

    List<float[]> queries = new ArrayList<>();
    BufferedReader reader = null;

    try {
      if (queryFile.endsWith(".gz")) {
        reader =
            new BufferedReader(
                new InputStreamReader(new GZIPInputStream(new FileInputStream(queryFile))));
      } else {
        reader = new BufferedReader(new FileReader(queryFile));
      }

      String line;
      int lineCount = 0;
      int skippedLines = 0;

      while ((line = reader.readLine()) != null && queries.size() < maxQueries) {
        lineCount++;

        // Skip header if present
        if (hasColNames && lineCount == 1) {
          continue;
        }

        try {
          // Parse the entire line as a vector string
          float[] vector = parseFloatArrayFromString(line);
          queries.add(vector);

          if (queries.size() <= 3) { // Show first few for debugging
            System.out.println("  - Query " + queries.size() + " dimension: " + vector.length);
          }
        } catch (Exception e) {
          skippedLines++;
          if (skippedLines <= 3) {
            System.err.println(
                "  ⚠️  Skipping malformed query line " + lineCount + ": " + e.getMessage());
          }
        }
      }

      System.out.println("  - Loaded " + queries.size() + " query vectors");
      if (skippedLines > 0) {
        System.out.println("  - Skipped " + skippedLines + " malformed lines");
      }

    } finally {
      if (reader != null) {
        reader.close();
      }
    }

    return queries;
  }

  /**
   * Detect if the query file is line-based (e.g., each line is a complete vector)
   * or column-based (e.g., vectors are in specific columns)
   */
  private static boolean detectQueryFileFormat(String queryFile, boolean hasColNames) {
    try {
      if (queryFile.endsWith(".fvecs")) {
        return true; // Line-based format
      } else if (queryFile.endsWith(".csv") || queryFile.endsWith(".csv.gz")) {
        // For now, assume CSV files are line-based if they contain many comma-separated values
        // This is a heuristic based on the Wikipedia query file format
        BufferedReader reader = null;
        try {
          if (queryFile.endsWith(".gz")) {
            reader =
                new BufferedReader(
                    new InputStreamReader(new GZIPInputStream(new FileInputStream(queryFile))));
          } else {
            reader = new BufferedReader(new FileReader(queryFile));
          }

          String line;
          int lineCount = 0;
          while ((line = reader.readLine()) != null && lineCount < 3) {
            lineCount++;
            if (hasColNames && lineCount == 1) continue; // Skip header

            // Count commas to estimate if this is a vector line
            long commaCount = line.chars().filter(ch -> ch == ',').count();
            if (commaCount > 100) { // If more than 100 commas, likely a vector line
              return true;
            }
          }
        } finally {
          if (reader != null) reader.close();
        }
        return false; // Default to column-based
      }
    } catch (Exception e) {
      // If detection fails, default to column-based
      System.out.println(
          "  - Format detection failed, defaulting to column-based: " + e.getMessage());
    }
    return false; // Default to column-based
  }

  /**
   * Parse a string of comma-separated float values into a float array
   * This matches the vectorsearch-benchmarks parseFloatArrayFromStringArray method
   * Handles both formats: "1.0, 2.0, 3.0" and "[1.0, 2.0, 3.0]"
   */
  private static float[] parseFloatArrayFromString(String str) {
    if (str == null || str.trim().isEmpty()) {
      throw new IllegalArgumentException("Empty string");
    }

    // Remove brackets if present (like vectorsearch-benchmarks does)
    String cleanStr = str.trim();
    if (cleanStr.startsWith("[") && cleanStr.endsWith("]")) {
      cleanStr = cleanStr.substring(1, cleanStr.length() - 1);
    }

    // Split by comma and parse each value
    String[] parts = cleanStr.split(",");
    float[] result = new float[parts.length];

    for (int i = 0; i < parts.length; i++) {
      try {
        result[i] = Float.parseFloat(parts[i].trim());
      } catch (NumberFormatException e) {
        throw new IllegalArgumentException(
            "Invalid float value at position " + i + ": '" + parts[i].trim() + "'");
      }
    }

    return result;
  }

  /**
   * Parse integer array from string representation like "1, 2, 3"
   */
  private static int[] parseIntArrayFromString(String str) {
    String[] parts = str.split(", ");
    int[] result = new int[parts.length];
    for (int i = 0; i < parts.length; i++) {
      result[i] = Integer.parseInt(parts[i].trim());
    }
    return result;
  }
}
