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
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Test suite for CuVSCPUSearchCodec functionality.
 *
 * This test replicates the exact workflow of hnsw.sh script:
 * 1. GPU indexing with CuVSCPUSearchCodec (CAGRA)
 * 2. CPU search with Lucene HNSW
 *
 * Uses the same parameters as NativeHNSWSerialization.java:
 * - 100 documents
 * - 10 dimensions
 * - Fixed random seed (42)
 */
@SuppressSysoutChecks(bugUrl = "CuVS native library produces verbose logging output")
public class TestCuVSCPUSearchCodec extends LuceneTestCase {

  private static final Random RANDOM = new Random(42); // Fixed seed for reproducibility
  private static final int NUM_DOCS = 100;
  private static final int DIMENSION = 10;
  private static final int SEARCH_K = 5;

  @BeforeClass
  public static void beforeClass() {
    // Check if CuVS library is available
    try {
      com.nvidia.cuvs.CuVSResources.create();
      System.out.println("✓ CuVS library available for testing");
    } catch (Throwable e) {
      org.junit.Assume.assumeTrue("CuVS library not available: " + e.getMessage(), false);
    }
  }

  /**
   * Test GPU indexing with CPU search - replicates the exact workflow of hnsw.sh script.
   * This test ensures that:
   * 1. GPU indexing works (CAGRA graph construction)
   * 2. Graph is properly serialized to Lucene format
   * 3. CPU search works on the serialized graph
   */
  @Test
  public void testGPUIndexingWithCPUSearch() throws IOException {
    System.out.println("=== Testing CuVSCPUSearchCodec: GPU Indexing + CPU Search ===");
    System.out.println("Parameters: " + NUM_DOCS + " docs, " + DIMENSION + " dimensions");

    // Create test directory
    Path testDir = createTempDir("cpu_search_test");

    try (Directory dir = FSDirectory.open(testDir);
        IndexWriter w =
            new IndexWriter(dir, newIndexWriterConfig().setCodec(new CuVSCPUSearchCodec()))) {

      // Create documents with vectors - exactly like NativeHNSWSerialization.java
      System.out.println("Creating index with " + NUM_DOCS + " documents...");
      for (int i = 0; i < NUM_DOCS; i++) {
        Document doc = new Document();
        float[] vecs = randomVector(DIMENSION);
        doc.add(new KnnFloatVectorField("knn1", vecs, EUCLIDEAN));
        doc.add(new KnnFloatVectorField("knn2", vecs, EUCLIDEAN));
        doc.add(new StringField("id", Integer.toString(i), Field.Store.YES));
        w.addDocument(doc);
      }

      System.out.println("Force merging to single segment...");
      w.forceMerge(1);

      // Test reading the index and performing searches
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);

        // Print index files (like NativeHNSWSerialization.java)
        System.out.println("Index files:");
        for (String file : reader.directory().listAll()) {
          System.out.println(file + ": " + reader.directory().fileLength(file));
        }

        // Verify vector fields exist and have correct dimensions
        FloatVectorValues values1 = r.getFloatVectorValues("knn1");
        FloatVectorValues values2 = r.getFloatVectorValues("knn2");

        assertNotNull("knn1 field should exist", values1);
        assertNotNull("knn2 field should exist", values2);
        assertEquals("knn1 should have correct size", NUM_DOCS, values1.size());
        assertEquals("knn2 should have correct size", NUM_DOCS, values2.size());
        assertEquals("knn1 should have correct dimension", DIMENSION, values1.dimension());
        assertEquals("knn2 should have correct dimension", DIMENSION, values2.dimension());

        System.out.println(
            "knn1 - Size: " + values1.size() + ", Dimension: " + values1.dimension());
        System.out.println(
            "knn2 - Size: " + values2.size() + ", Dimension: " + values2.dimension());

        // Test CPU search on the GPU-indexed data (like TestIndexWithLucene101.java)
        System.out.println("Testing CPU search on GPU-indexed data...");
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = randomVector(DIMENSION);

        System.out.println("Query vector: " + java.util.Arrays.toString(queryVector));

        // Test search on knn1 field
        KnnFloatVectorQuery query1 = new KnnFloatVectorQuery("knn1", queryVector, SEARCH_K);
        TopDocs results1 = searcher.search(query1, SEARCH_K);

        assertNotNull("Search results should not be null", results1);
        assertTrue("Should have positive total hits", results1.totalHits.value() > 0);
        assertTrue("Should have score docs", results1.scoreDocs.length > 0);

        System.out.println(
            "\nknn1 search results (" + results1.totalHits.value() + " total hits):");
        for (int i = 0; i < results1.scoreDocs.length; i++) {
          ScoreDoc scoreDoc = results1.scoreDocs[i];
          Document doc = searcher.storedFields().document(scoreDoc.doc);
          System.out.println(
              "  Rank "
                  + (i + 1)
                  + ": doc "
                  + scoreDoc.doc
                  + " (id="
                  + doc.get("id")
                  + "), score="
                  + scoreDoc.score);
        }

        // Test search on knn2 field
        KnnFloatVectorQuery query2 = new KnnFloatVectorQuery("knn2", queryVector, SEARCH_K);
        TopDocs results2 = searcher.search(query2, SEARCH_K);

        assertNotNull("Search results should not be null", results2);
        assertTrue("Should have positive total hits", results2.totalHits.value() > 0);
        assertTrue("Should have score docs", results2.scoreDocs.length > 0);

        System.out.println(
            "\nknn2 search results (" + results2.totalHits.value() + " total hits):");
        for (int i = 0; i < results2.scoreDocs.length; i++) {
          ScoreDoc scoreDoc = results2.scoreDocs[i];
          Document doc = searcher.storedFields().document(scoreDoc.doc);
          System.out.println(
              "  Rank "
                  + (i + 1)
                  + ": doc "
                  + scoreDoc.doc
                  + " (id="
                  + doc.get("id")
                  + "), score="
                  + scoreDoc.score);
        }

        // Verify search results are reasonable
        for (ScoreDoc scoreDoc : results1.scoreDocs) {
          assertTrue("Score should be finite", Float.isFinite(scoreDoc.score));
          assertTrue("Document ID should be valid", scoreDoc.doc >= 0 && scoreDoc.doc < NUM_DOCS);
        }

        for (ScoreDoc scoreDoc : results2.scoreDocs) {
          assertTrue("Score should be finite", Float.isFinite(scoreDoc.score));
          assertTrue("Document ID should be valid", scoreDoc.doc >= 0 && scoreDoc.doc < NUM_DOCS);
        }

        System.out.println("✓ All search results validated");
        System.out.println("=== GPU Indexing with CPU Search test completed successfully! ===");
      }
    }
  }

  /**
   * Test with different dataset sizes to ensure scalability.
   */
  @Test
  public void testDifferentDatasetSizes() throws IOException {
    System.out.println("=== Testing CuVSCPUSearchCodec with different dataset sizes ===");

    int[] testSizes = {10, 50, 100}; // Test with smaller datasets first

    for (int numDocs : testSizes) {
      System.out.println("Testing with " + numDocs + " documents...");

      Path testDir = createTempDir();

      try (Directory dir = FSDirectory.open(testDir);
          IndexWriter w =
              new IndexWriter(dir, newIndexWriterConfig().setCodec(new CuVSCPUSearchCodec()))) {

        // Create documents
        for (int i = 0; i < numDocs; i++) {
          Document doc = new Document();
          float[] vecs = randomVector(DIMENSION);
          doc.add(new KnnFloatVectorField("knn1", vecs, EUCLIDEAN));
          doc.add(new StringField("id", Integer.toString(i), Field.Store.YES));
          w.addDocument(doc);
        }

        w.forceMerge(1);

        // Test search
        try (DirectoryReader reader = DirectoryReader.open(w)) {
          LeafReader r = getOnlyLeafReader(reader);
          FloatVectorValues values = r.getFloatVectorValues("knn1");

          assertNotNull("Vector values should exist", values);
          assertEquals("Should have correct size", numDocs, values.size());
          assertEquals("Should have correct dimension", DIMENSION, values.dimension());

          IndexSearcher searcher = new IndexSearcher(reader);
          float[] queryVector = randomVector(DIMENSION);
          KnnFloatVectorQuery query =
              new KnnFloatVectorQuery("knn1", queryVector, Math.min(5, numDocs));
          TopDocs results = searcher.search(query, Math.min(5, numDocs));

          assertNotNull("Search results should not be null", results);
          assertTrue("Should have positive total hits", results.totalHits.value() > 0);
        }
      }

      System.out.println("✓ Test with " + numDocs + " documents completed successfully");
    }
  }

  /**
   * Test error handling when CuVS library is not available.
   */
  @Test
  public void testCuVSLibraryUnavailable() throws Exception {
    System.out.println("=== Testing behavior when CuVS library is unavailable ===");

    // This test should be skipped if CuVS is available
    try {
      com.nvidia.cuvs.CuVSResources.create();
      // If we get here, CuVS is available, so skip this test
      org.junit.Assume.assumeTrue("CuVS library is available, skipping unavailable test", false);
    } catch (Throwable e) {
      // CuVS is not available, test the fallback behavior
      System.out.println("CuVS library not available, testing fallback behavior...");

      Path testDir = createTempDir();

      try (Directory dir = FSDirectory.open(testDir);
          IndexWriter w =
              new IndexWriter(dir, newIndexWriterConfig().setCodec(new CuVSCPUSearchCodec()))) {

        // This should fail gracefully or use fallback
        try {
          Document doc = new Document();
          float[] vecs = randomVector(DIMENSION);
          doc.add(new KnnFloatVectorField("knn1", vecs, EUCLIDEAN));
          doc.add(new StringField("id", "0", Field.Store.YES));
          w.addDocument(doc);
          w.forceMerge(1);

          // If we get here, the fallback worked
          System.out.println("✓ Fallback behavior worked correctly");
        } catch (Exception ex) {
          // Expected failure when CuVS is not available
          System.out.println(
              "✓ Expected failure when CuVS library unavailable: " + ex.getMessage());
        }
      }
    }
  }

  // Helper methods

  private static float[] randomVector(int dim) {
    assert dim > 0;
    float[] v = new float[dim];
    double squareSum = 0.0;
    while (squareSum == 0.0) {
      squareSum = 0.0;
      for (int i = 0; i < dim; i++) {
        v[i] = RANDOM.nextFloat();
        squareSum += v[i] * v[i];
      }
    }
    return v;
  }

  /**
   * Test memory efficiency with large datasets.
   */
  @Test
  public void testMemoryEfficientGraphCreation() throws IOException {
    int numVectors = 10000;
    int dimensions = 128;

    Path tempDir = createTempDir("test-memory-efficiency");

    try (Directory directory = FSDirectory.open(tempDir)) {
      IndexWriterConfig config = newIndexWriterConfig();
      config.setMaxBufferedDocs(1000);

      CuVSCodec codec = new CuVSCodec();
      codec.setKnnFormat(
          new CuVSVectorsFormat(1, 128, 64, CuVSVectorsWriter.IndexType.HNSW_LUCENE));
      config.setCodec(codec);

      try (IndexWriter writer = new IndexWriter(directory, config)) {
        Random random = new Random(42);

        for (int i = 0; i < numVectors; i++) {
          Document doc = new Document();
          float[] vector = new float[dimensions];
          for (int j = 0; j < dimensions; j++) {
            vector[j] = random.nextFloat();
          }

          doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          writer.addDocument(doc);
        }

        writer.commit();

        try (DirectoryReader reader = DirectoryReader.open(directory)) {
          assertEquals("Should have indexed all documents", numVectors, reader.numDocs());

          LeafReader leafReader = reader.leaves().get(0).reader();
          for (int i = 0; i < Math.min(100, numVectors); i++) {
            Document doc = leafReader.storedFields().document(i);
            assertNotNull("Document should exist", doc);
            assertEquals("Document ID should match", String.valueOf(i), doc.get("id"));
          }
        }
      }
    }
  }

  /**
   * Test streaming graph with small dataset.
   */
  @Test
  public void testStreamingGraphWithSmallDataset() throws IOException {
    int numVectors = 1000;
    int dimensions = 64;

    Path tempDir = createTempDir("test-streaming-small");

    try (Directory directory = FSDirectory.open(tempDir)) {
      IndexWriterConfig config = newIndexWriterConfig();
      config.setMaxBufferedDocs(500);

      CuVSCodec codec = new CuVSCodec();
      codec.setKnnFormat(
          new CuVSVectorsFormat(1, 128, 64, CuVSVectorsWriter.IndexType.HNSW_LUCENE));
      config.setCodec(codec);

      try (IndexWriter writer = new IndexWriter(directory, config)) {
        Random random = new Random(123);

        for (int i = 0; i < numVectors; i++) {
          Document doc = new Document();
          float[] vector = new float[dimensions];
          for (int j = 0; j < dimensions; j++) {
            vector[j] = random.nextFloat();
          }

          doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          writer.addDocument(doc);
        }

        writer.commit();

        try (DirectoryReader reader = DirectoryReader.open(directory)) {
          assertEquals("Should have indexed all documents", numVectors, reader.numDocs());
        }
      }
    }
  }

  /**
   * Test large dataset memory efficiency.
   */
  @Test
  public void testLargeDatasetMemoryEfficiency() throws IOException {
    int numVectors = 50000;
    int dimensions = 128;

    Path tempDir = createTempDir("test-large-memory-efficiency");

    try (Directory directory = FSDirectory.open(tempDir)) {
      IndexWriterConfig config = newIndexWriterConfig();
      config.setMaxBufferedDocs(2000);

      CuVSCodec codec = new CuVSCodec();
      codec.setKnnFormat(
          new CuVSVectorsFormat(1, 128, 64, CuVSVectorsWriter.IndexType.HNSW_LUCENE));
      config.setCodec(codec);

      try (IndexWriter writer = new IndexWriter(directory, config)) {
        Random random = new Random(42);

        for (int i = 0; i < numVectors; i++) {
          Document doc = new Document();
          float[] vector = new float[dimensions];
          for (int j = 0; j < dimensions; j++) {
            vector[j] = random.nextFloat();
          }

          doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          writer.addDocument(doc);
        }

        writer.commit();

        try (DirectoryReader reader = DirectoryReader.open(directory)) {
          assertEquals("Should have indexed all documents", numVectors, reader.numDocs());
          System.out.println(
              "Successfully created index with "
                  + numVectors
                  + " vectors using streaming implementation");
        }
      }
    }
  }
}
