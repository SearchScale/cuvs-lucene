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
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * CuVS-Lucene CPUSearchCodec Benchmark
 *
 * This benchmark demonstrates the CuVSCPUSearchCodec workflow:
 * 1. GPU indexing with CuVS (CAGRA graph construction)
 * 2. Graph serialization to Lucene format
 * 3. CPU search using Lucene HNSW on the serialized graph
 */
public class CuVSLuceneBenchmark {

  private static final ObjectMapper mapper = new ObjectMapper();

  public static void main(String[] args) throws Exception {
    System.out.println("🚀 CuVS-Lucene CPUSearchCodec Benchmark");
    System.out.println("========================================");
    System.out.println("Workflow: GPU Indexing (CuVS) → CPU Search (Lucene)");
    System.out.println();

    // Parse command line arguments
    String datasetType = args.length > 0 ? args[0] : "sift-small";
    int maxVectors = args.length > 1 ? Integer.parseInt(args[1]) : 10000;
    int topK = args.length > 2 ? Integer.parseInt(args[2]) : 10;

    // Load dataset
    BenchmarkDataset dataset = loadDataset(datasetType, maxVectors);

    System.out.println("📋 Configuration:");
    System.out.println("  - Dataset: " + dataset.name);
    System.out.println("  - Vectors: " + dataset.vectors.size());
    System.out.println("  - Dimensions: " + dataset.dimensions);
    System.out.println("  - Queries: " + dataset.queries.size());
    System.out.println("  - Top-K: " + topK);
    System.out.println(
        "  - Ground Truth: " + (dataset.hasGroundTruth() ? "✅ Available" : "❌ Not available"));
    System.out.println();

    // Run benchmark
    ObjectNode results = runBenchmark(dataset, topK);

    // Print results
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
              + topK
              + ": "
              + String.format("%.4f", results.get("recall").asDouble())
              + " ("
              + String.format("%.2f", results.get("recall").asDouble() * 100)
              + "%)");
    }

    System.out.println();
    System.out.println("📄 Detailed Results JSON:");
    mapper.writerWithDefaultPrettyPrinter().writeValue(System.out, results);
    System.out.println();
  }

  private static BenchmarkDataset loadDataset(String datasetType, int maxVectors)
      throws IOException {
    switch (datasetType.toLowerCase()) {
      case "sift-small":
      case "sift":
        return loadSiftDataset(maxVectors);

      case "wikipedia":
      case "wiki":
        return loadWikipediaDataset(maxVectors);

      case "synthetic-small":
        return generateSyntheticDataset(1000, 50, 128);

      case "synthetic":
        return generateSyntheticDataset(10000, 100, 128);

      case "synthetic-large":
        return generateSyntheticDataset(100000, 1000, 128);

      default:
        System.out.println("⚠️  Unknown dataset type: " + datasetType + ", using synthetic small");
        return generateSyntheticDataset(1000, 50, 128);
    }
  }

  private static BenchmarkDataset loadSiftDataset(int maxVectors) throws IOException {
    System.out.println("📂 Loading SIFT dataset from /data/faiss/siftsmall/");

    // Load base vectors
    List<float[]> vectors =
        FvecsReader.readFvecs("/data/faiss/siftsmall/siftsmall_base.fvecs", maxVectors);

    // Load query vectors
    List<float[]> queries =
        FvecsReader.readFvecs("/data/faiss/siftsmall/siftsmall_query.fvecs", 100);

    // Load ground truth
    List<int[]> groundTruth = null;
    try {
      groundTruth = IvecsReader.readIvecs("/data/faiss/siftsmall/siftsmall_groundtruth.ivecs", 100);
    } catch (IOException e) {
      System.out.println("⚠️  Ground truth not available: " + e.getMessage());
    }

    int dimensions = vectors.isEmpty() ? 0 : vectors.get(0).length;

    System.out.println("✅ SIFT dataset loaded:");
    System.out.println("  - Base vectors: " + vectors.size() + " × " + dimensions);
    System.out.println(
        "  - Query vectors: "
            + queries.size()
            + " × "
            + (queries.isEmpty() ? 0 : queries.get(0).length));
    if (groundTruth != null) {
      System.out.println(
          "  - Ground truth: "
              + groundTruth.size()
              + " × "
              + (groundTruth.isEmpty() ? 0 : groundTruth.get(0).length));
    }

    return new BenchmarkDataset("sift", vectors, queries, groundTruth, dimensions);
  }

  private static BenchmarkDataset loadWikipediaDataset(int maxVectors) throws IOException {
    System.out.println("📂 Loading Wikipedia dataset from ~/datasetwiki/");
    System.out.println(
        "⚠️  Wikipedia dataset loading requires CSV parsing - this is a simplified implementation");
    System.out.println(
        "⚠️  For full functionality, use the benchmark classes from"
            + " /home/puneet/simple-benchmark/");

    // For now, return a synthetic dataset with Wikipedia-like dimensions
    // In a full implementation, this would parse the CSV files
    return generateSyntheticDataset(maxVectors, 100, 2048);
  }

  private static BenchmarkDataset generateSyntheticDataset(
      int numVectors, int numQueries, int dimensions) {
    System.out.println("📈 Generating synthetic dataset...");
    System.out.println("  - Vectors: " + numVectors);
    System.out.println("  - Queries: " + numQueries);
    System.out.println("  - Dimensions: " + dimensions);

    Random random = new Random(42); // Fixed seed for reproducibility

    List<float[]> vectors = new ArrayList<>();
    for (int i = 0; i < numVectors; i++) {
      float[] vector = new float[dimensions];
      for (int j = 0; j < dimensions; j++) {
        vector[j] = (float) random.nextGaussian();
      }
      vectors.add(vector);
    }

    List<float[]> queries = new ArrayList<>();
    for (int i = 0; i < numQueries; i++) {
      float[] query = new float[dimensions];
      for (int j = 0; j < dimensions; j++) {
        query[j] = (float) random.nextGaussian();
      }
      queries.add(query);
    }

    return new BenchmarkDataset("synthetic", vectors, queries, null, dimensions);
  }

  private static ObjectNode runBenchmark(BenchmarkDataset dataset, int topK) throws IOException {
    ObjectNode results = mapper.createObjectNode();
    results.put("dataset", dataset.name);
    results.put("numVectors", dataset.vectors.size());
    results.put("dimensions", dataset.dimensions);
    results.put("numQueries", dataset.queries.size());
    results.put("topK", topK);

    // Create temporary directory for index
    Path tempDir = Files.createTempDirectory("cuvs_lucene_benchmark");
    System.out.println("📁 Using temporary directory: " + tempDir);

    try (Directory directory = FSDirectory.open(tempDir)) {

      // Test CuVSCPUSearchCodec
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
      long indexingStartTime = System.currentTimeMillis();
      long memoryBefore = getMemoryUsage();

      try {
        // Configure index writer with CuVSCPUSearchCodec
        IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
        config.setCodec(codec);
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        config.setUseCompoundFile(false); // Disable CFS to see individual index files

        // Create index writer and add documents
        try (IndexWriter writer = new IndexWriter(directory, config)) {
          System.out.println("📝 Adding " + dataset.vectors.size() + " documents to index...");

          for (int i = 0; i < dataset.vectors.size(); i++) {
            Document doc = new Document();
            doc.add(new KnnFloatVectorField("vector", dataset.vectors.get(i), EUCLIDEAN));
            doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
            writer.addDocument(doc);

            if ((i + 1) % 10000 == 0) {
              System.out.println("  - Added " + (i + 1) + " documents");
            }
          }

          // Force merge to ensure all data is written and graph is constructed
          System.out.println("🔄 Force merging segments (triggers graph construction)...");
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

      } catch (Exception e) {
        System.out.println("❌ GPU indexing failed: " + e.getMessage());
        e.printStackTrace();
        results.put("status", "FAILED");
        results.put("error", "Indexing failed: " + e.getMessage());
        return results;
      }

      // CPU search phase
      System.out.println("🔍 Starting CPU search phase (Lucene HNSW on serialized graph)...");
      long searchStartTime = System.currentTimeMillis();

      try {
        // Open index reader
        try (DirectoryReader reader = DirectoryReader.open(directory)) {
          System.out.println("✅ Index opened successfully for CPU search");
          System.out.println("  - Number of documents: " + reader.numDocs());

          // Create searcher
          IndexSearcher searcher = new IndexSearcher(reader);

          // Run search queries
          System.out.println("🔎 Running " + dataset.queries.size() + " search queries...");
          int totalHits = 0;
          List<int[]> searchResults = new ArrayList<>();

          for (int i = 0; i < dataset.queries.size(); i++) {
            KnnFloatVectorQuery query =
                new KnnFloatVectorQuery("vector", dataset.queries.get(i), topK);
            TopDocs topDocs = searcher.search(query, topK);
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
            System.out.println("📊 Calculating recall@" + topK + "...");
            double recall = calculateRecall(searchResults, dataset.groundTruth, topK);
            System.out.println(
                "  - Recall@"
                    + topK
                    + ": "
                    + String.format("%.4f", recall)
                    + " ("
                    + String.format("%.2f", recall * 100)
                    + "%)");
            results.put("recall", recall);
          }

          results.put("status", "SUCCESS");

        } catch (Exception e) {
          System.out.println("❌ CPU search failed: " + e.getMessage());
          e.printStackTrace();
          results.put("status", "FAILED");
          results.put("error", "Search failed: " + e.getMessage());
          return results;
        }
      } catch (Exception e) {
        System.out.println("❌ CPU search failed: " + e.getMessage());
        e.printStackTrace();
        results.put("status", "FAILED");
        results.put("error", "Search failed: " + e.getMessage());
        return results;
      }

    } catch (Exception e) {
      System.out.println("❌ Directory creation failed: " + e.getMessage());
      e.printStackTrace();
      results.put("status", "FAILED");
      results.put("error", "Directory creation failed: " + e.getMessage());
    } finally {
      // Clean up temporary directory
      try {
        Files.walk(tempDir)
            .sorted((a, b) -> b.compareTo(a)) // Delete files first, then directories
            .forEach(
                path -> {
                  try {
                    Files.delete(path);
                  } catch (IOException e) {
                    System.err.println("Failed to delete: " + path);
                  }
                });
        System.out.println("🧹 Cleaned up temporary directory");
      } catch (IOException e) {
        System.err.println("Failed to clean up temporary directory: " + e.getMessage());
      }
    }

    return results;
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
}
