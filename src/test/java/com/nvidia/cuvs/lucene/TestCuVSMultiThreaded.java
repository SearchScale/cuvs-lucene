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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.analysis.MockTokenizer;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.Ignore;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "prints info from within cuvs")
public class TestCuVSMultiThreaded extends LuceneTestCase {

  protected static Logger log = Logger.getLogger(TestCuVSMultiThreaded.class.getName());
  static final Codec codec = TestUtil.alwaysKnnVectorsFormat(new CuVSVectorsFormat());

  @Test
  public void testSingleIndexMultipleWriterThreads() throws Exception {
    assumeTrue("cuvs not supported", CuVSVectorsFormat.supported());

    try (Directory directory = newDirectory()) {
      Random random = random();
      int numThreads = random.nextInt(4) + 2; // 2-5 threads
      int docsPerThread = random.nextInt(50) + 20; // 20-70 docs per thread
      int dimensions = random.nextInt(128) + 64; // 64-192 dimensions

      log.info(
          "Testing with " + numThreads + " writer threads, " + docsPerThread + " docs per thread");

      ExecutorService executor = Executors.newFixedThreadPool(numThreads);
      CountDownLatch startLatch = new CountDownLatch(1);
      CountDownLatch completeLatch = new CountDownLatch(numThreads);
      AtomicInteger totalDocsWritten = new AtomicInteger(0);
      Set<String> writtenDocIds = ConcurrentHashMap.newKeySet();

      try (IndexWriter writer = createIndexWriter(directory)) {
        List<Future<?>> futures = new ArrayList<>();

        // Pre-generate seeds for each thread to avoid sharing Random instances
        long[] threadSeeds = new long[numThreads];
        for (int i = 0; i < numThreads; i++) {
          threadSeeds[i] = random.nextLong();
        }

        for (int threadId = 0; threadId < numThreads; threadId++) {
          final int tid = threadId;
          Future<?> future =
              executor.submit(
                  () -> {
                    try {
                      startLatch.await(); // All threads start together
                      Random threadRandom = new Random(threadSeeds[tid]);

                      for (int i = 0; i < docsPerThread; i++) {
                        String docId = "thread" + tid + "_doc" + i;
                        Document doc = new Document();
                        doc.add(new StringField("id", docId, Field.Store.YES));
                        doc.add(new StringField("threadId", String.valueOf(tid), Field.Store.YES));

                        float[] vector = generateRandomVector(dimensions, threadRandom);
                        doc.add(
                            new KnnFloatVectorField(
                                "vector", vector, VectorSimilarityFunction.EUCLIDEAN));

                        writer.addDocument(doc);
                        writtenDocIds.add(docId);
                        totalDocsWritten.incrementAndGet();

                        // Randomly yield to increase thread interleaving
                        if (threadRandom.nextInt(10) == 0) {
                          Thread.yield();
                        }
                      }
                    } catch (Exception e) {
                      throw new RuntimeException("Thread " + tid + " failed", e);
                    } finally {
                      completeLatch.countDown();
                    }
                  });
          futures.add(future);
        }

        startLatch.countDown(); // Start all threads

        // Wait for all threads to complete
        assertTrue(
            "All threads should complete within 30 seconds",
            completeLatch.await(30, TimeUnit.SECONDS));

        // Check for any exceptions
        for (Future<?> future : futures) {
          future.get(); // This will throw if the thread encountered an exception
        }

        writer.commit();
      }

      // Verify all documents were written correctly
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);

        assertEquals(
            "Total documents should match", numThreads * docsPerThread, totalDocsWritten.get());
        assertEquals("Reader should see all documents", totalDocsWritten.get(), reader.numDocs());

        // Verify vector search works
        float[] queryVector = generateRandomVector(dimensions, random);
        Query query = new KnnFloatVectorQuery("vector", queryVector, 10);
        TopDocs results = searcher.search(query, 10);

        assertTrue("Should find some results", results.totalHits.value() > 0);
        log.info("Vector search found " + results.totalHits.value() + " results");
      }

      executor.shutdown();
      assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
    }
  }

  @Test
  @Ignore("CUDA concurrent search errors cause cudaErrorInvalidValue and thread leaks")
  public void testSingleIndexMultipleQueryThreads() throws Exception {
    assumeTrue("cuvs not supported", CuVSVectorsFormat.supported());

    try (Directory directory = newDirectory()) {
      Random random = random();
      int numDocs = random.nextInt(200) + 100; // 100-300 documents
      int dimensions = random.nextInt(128) + 64; // 64-192 dimensions
      int numQueryThreads = random.nextInt(6) + 3; // 3-8 query threads
      int queriesPerThread = random.nextInt(20) + 10; // 10-30 queries per thread

      // First, create the index
      float[][] dataset = new float[numDocs][dimensions];
      try (IndexWriter writer = createIndexWriter(directory)) {
        for (int i = 0; i < numDocs; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));

          dataset[i] = generateRandomVector(dimensions, random);
          doc.add(
              new KnnFloatVectorField("vector", dataset[i], VectorSimilarityFunction.EUCLIDEAN));

          writer.addDocument(doc);
        }
        writer.commit();
      }

      // Now test concurrent queries
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);

        ExecutorService executor = Executors.newFixedThreadPool(numQueryThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch completeLatch = new CountDownLatch(numQueryThreads);
        AtomicInteger totalQueriesExecuted = new AtomicInteger(0);
        AtomicInteger totalResultsFound = new AtomicInteger(0);

        List<Future<?>> futures = new ArrayList<>();

        // Pre-generate seeds for each thread to avoid sharing Random instances
        long[] threadSeeds = new long[numQueryThreads];
        for (int i = 0; i < numQueryThreads; i++) {
          threadSeeds[i] = random.nextLong();
        }

        for (int threadId = 0; threadId < numQueryThreads; threadId++) {
          final int tid = threadId;
          Future<?> future =
              executor.submit(
                  () -> {
                    try {
                      startLatch.await(); // All threads start together
                      Random threadRandom = new Random(threadSeeds[tid]);

                      for (int i = 0; i < queriesPerThread; i++) {
                        float[] queryVector = generateRandomVector(dimensions, threadRandom);
                        int topK = threadRandom.nextInt(20) + 5; // 5-25 results

                        Query query = new KnnFloatVectorQuery("vector", queryVector, topK);
                        TopDocs results = searcher.search(query, topK);

                        totalQueriesExecuted.incrementAndGet();
                        totalResultsFound.addAndGet(results.scoreDocs.length);

                        // Verify results are valid
                        assertTrue("Should have results", results.scoreDocs.length > 0);
                        assertTrue(
                            "Results should not exceed topK", results.scoreDocs.length <= topK);

                        // Randomly yield to increase thread interleaving
                        if (threadRandom.nextInt(5) == 0) {
                          Thread.yield();
                        }
                      }
                    } catch (Exception e) {
                      throw new RuntimeException("Query thread " + tid + " failed", e);
                    } finally {
                      completeLatch.countDown();
                    }
                  });
          futures.add(future);
        }

        startLatch.countDown(); // Start all threads

        // Wait for all threads to complete
        assertTrue(
            "All query threads should complete within 30 seconds",
            completeLatch.await(30, TimeUnit.SECONDS));

        // Check for any exceptions
        for (Future<?> future : futures) {
          future.get();
        }

        log.info(
            "Executed "
                + totalQueriesExecuted.get()
                + " queries across "
                + numQueryThreads
                + " threads, found "
                + totalResultsFound.get()
                + " total results");

        assertEquals(
            "All queries should have been executed",
            numQueryThreads * queriesPerThread,
            totalQueriesExecuted.get());

        executor.shutdown();
        assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
      }
    }
  }

  @Test
  @Ignore("CUDA threading issues cause timeouts and instability")
  public void testMultipleIndicesFromDifferentThreads() throws Exception {
    assumeTrue("cuvs not supported", CuVSVectorsFormat.supported());

    Random random = random();
    int numIndices = random.nextInt(4) + 2; // 2-5 indices
    int docsPerIndex = random.nextInt(100) + 50; // 50-150 docs per index
    int dimensions = random.nextInt(128) + 64; // 64-192 dimensions

    ExecutorService executor = Executors.newFixedThreadPool(numIndices);
    CountDownLatch startLatch = new CountDownLatch(1);
    CountDownLatch completeLatch = new CountDownLatch(numIndices);
    List<Directory> directories = Collections.synchronizedList(new ArrayList<>());
    AtomicInteger successfulIndices = new AtomicInteger(0);

    List<Future<?>> futures = new ArrayList<>();

    // Pre-generate seeds for each thread to avoid sharing Random instances
    long[] threadSeeds = new long[numIndices];
    for (int i = 0; i < numIndices; i++) {
      threadSeeds[i] = random.nextLong();
    }

    for (int indexId = 0; indexId < numIndices; indexId++) {
      final int iid = indexId;
      Future<?> future =
          executor.submit(
              () -> {
                Directory directory = null;
                try {
                  startLatch.await(); // All threads start together
                  directory = newDirectory();
                  directories.add(directory);

                  Random threadRandom = new Random(threadSeeds[iid]);

                  try (IndexWriter writer = createIndexWriter(directory)) {
                    for (int i = 0; i < docsPerIndex; i++) {
                      Document doc = new Document();
                      doc.add(new StringField("id", "index" + iid + "_doc" + i, Field.Store.YES));
                      doc.add(new StringField("indexId", String.valueOf(iid), Field.Store.YES));

                      float[] vector = generateRandomVector(dimensions, threadRandom);
                      doc.add(
                          new KnnFloatVectorField(
                              "vector", vector, VectorSimilarityFunction.EUCLIDEAN));

                      writer.addDocument(doc);
                    }
                    writer.commit();
                  }

                  // Verify the index was created correctly
                  try (DirectoryReader reader = DirectoryReader.open(directory)) {
                    assertEquals(
                        "Index " + iid + " should have correct doc count",
                        docsPerIndex,
                        reader.numDocs());

                    IndexSearcher searcher = newSearcher(reader);
                    float[] queryVector = generateRandomVector(dimensions, threadRandom);
                    Query query = new KnnFloatVectorQuery("vector", queryVector, 10);
                    TopDocs results = searcher.search(query, 10);

                    assertTrue(
                        "Index " + iid + " should return results", results.totalHits.value() > 0);
                    successfulIndices.incrementAndGet();
                  }

                } catch (Exception e) {
                  throw new RuntimeException("Index creation thread " + iid + " failed", e);
                } finally {
                  completeLatch.countDown();
                }
              });
      futures.add(future);
    }

    startLatch.countDown(); // Start all threads

    // Wait for all threads to complete
    assertTrue(
        "All index creation threads should complete within 30 seconds",
        completeLatch.await(30, TimeUnit.SECONDS));

    // Check for any exceptions
    for (Future<?> future : futures) {
      future.get();
    }

    assertEquals("All indices should be created successfully", numIndices, successfulIndices.get());

    log.info("Successfully created " + numIndices + " indices concurrently");

    // Clean up directories
    for (Directory dir : directories) {
      dir.close();
    }

    executor.shutdown();
    assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
  }

  @Test
  @Ignore("CUDA concurrent access issues cause cudaErrorInvalidValue")
  public void testConcurrentIndexingAndQuerying() throws Exception {
    assumeTrue("cuvs not supported", CuVSVectorsFormat.supported());

    try (Directory directory = newDirectory()) {
      Random random = random();
      int dimensions = random.nextInt(128) + 64; // 64-192 dimensions
      int numWriterThreads = random.nextInt(3) + 1; // 1-3 writer threads
      int numQueryThreads = random.nextInt(4) + 2; // 2-5 query threads
      int docsPerWriter = random.nextInt(50) + 30; // 30-80 docs per writer
      int queriesPerThread = random.nextInt(30) + 20; // 20-50 queries per thread
      int initialDocs = random.nextInt(100) + 50; // 50-150 initial documents

      // First, create some initial documents
      try (IndexWriter writer = createIndexWriter(directory)) {
        for (int i = 0; i < initialDocs; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", "initial_" + i, Field.Store.YES));
          doc.add(new StringField("phase", "initial", Field.Store.YES));

          float[] vector = generateRandomVector(dimensions, random);
          doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.EUCLIDEAN));

          writer.addDocument(doc);
        }
        writer.commit();
      }

      // Now run concurrent indexing and querying
      ExecutorService executor = Executors.newFixedThreadPool(numWriterThreads + numQueryThreads);
      CountDownLatch startLatch = new CountDownLatch(1);
      CountDownLatch completeLatch = new CountDownLatch(numWriterThreads + numQueryThreads);

      AtomicInteger totalNewDocs = new AtomicInteger(0);
      AtomicInteger totalQueries = new AtomicInteger(0);
      AtomicInteger totalHits = new AtomicInteger(0);

      List<Future<?>> futures = new ArrayList<>();

      // Pre-generate seeds for all threads to avoid sharing Random instances
      long[] writerSeeds = new long[numWriterThreads];
      for (int i = 0; i < numWriterThreads; i++) {
        writerSeeds[i] = random.nextLong();
      }
      long[] querySeeds = new long[numQueryThreads];
      for (int i = 0; i < numQueryThreads; i++) {
        querySeeds[i] = random.nextLong();
      }

      // Create a single IndexWriter that will be shared among writer threads
      IndexWriter sharedWriter = createIndexWriter(directory);

      // Start writer threads
      for (int writerId = 0; writerId < numWriterThreads; writerId++) {
        final int wid = writerId;
        Future<?> future =
            executor.submit(
                () -> {
                  try {
                    startLatch.await();
                    Random threadRandom = new Random(writerSeeds[wid]);

                    for (int i = 0; i < docsPerWriter; i++) {
                      Document doc = new Document();
                      doc.add(new StringField("id", "writer" + wid + "_doc" + i, Field.Store.YES));
                      doc.add(new StringField("phase", "concurrent", Field.Store.YES));
                      doc.add(new StringField("writerId", String.valueOf(wid), Field.Store.YES));

                      float[] vector = generateRandomVector(dimensions, threadRandom);
                      doc.add(
                          new KnnFloatVectorField(
                              "vector", vector, VectorSimilarityFunction.EUCLIDEAN));

                      sharedWriter.addDocument(doc);
                      totalNewDocs.incrementAndGet();

                      // Periodically commit to make documents visible to readers
                      if (i % 10 == 0) {
                        sharedWriter.commit();
                      }

                      // Random delay
                      if (threadRandom.nextInt(20) == 0) {
                        Thread.sleep(threadRandom.nextInt(10));
                      }
                    }

                    sharedWriter.commit(); // Final commit
                  } catch (Exception e) {
                    throw new RuntimeException("Writer thread " + wid + " failed", e);
                  } finally {
                    completeLatch.countDown();
                  }
                });
        futures.add(future);
      }

      // Start query threads
      for (int queryId = 0; queryId < numQueryThreads; queryId++) {
        final int qid = queryId;
        Future<?> future =
            executor.submit(
                () -> {
                  try {
                    startLatch.await();
                    Random threadRandom = new Random(querySeeds[qid]);

                    // Small delay to let some documents get indexed
                    Thread.sleep(50);

                    for (int i = 0; i < queriesPerThread; i++) {
                      // Create a new reader for each query to see latest commits
                      try (DirectoryReader reader = DirectoryReader.open(directory)) {
                        IndexSearcher searcher = newSearcher(reader);

                        float[] queryVector = generateRandomVector(dimensions, threadRandom);
                        int topK = threadRandom.nextInt(15) + 5; // 5-20 results

                        Query query = new KnnFloatVectorQuery("vector", queryVector, topK);
                        TopDocs results = searcher.search(query, topK);

                        totalQueries.incrementAndGet();
                        totalHits.addAndGet(results.scoreDocs.length);

                        // Log periodically
                        if (i % 10 == 0) {
                          log.fine(
                              "Query thread "
                                  + qid
                                  + " executed "
                                  + i
                                  + " queries, current index size: "
                                  + reader.numDocs());
                        }

                        // Random delay between queries
                        if (threadRandom.nextInt(10) == 0) {
                          Thread.sleep(threadRandom.nextInt(5));
                        }
                      }
                    }
                  } catch (Exception e) {
                    throw new RuntimeException("Query thread " + qid + " failed", e);
                  } finally {
                    completeLatch.countDown();
                  }
                });
        futures.add(future);
      }

      startLatch.countDown(); // Start all threads

      // Wait for all threads to complete
      assertTrue(
          "All threads should complete within 60 seconds",
          completeLatch.await(60, TimeUnit.SECONDS));

      // Check for any exceptions
      for (Future<?> future : futures) {
        future.get();
      }

      sharedWriter.close();

      // Final verification
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        int expectedDocs = initialDocs + (numWriterThreads * docsPerWriter);
        assertEquals("Final document count should match", expectedDocs, reader.numDocs());

        log.info(
            "Concurrent test completed: "
                + totalNewDocs.get()
                + " docs indexed, "
                + totalQueries.get()
                + " queries executed, "
                + totalHits.get()
                + " total hits");
      }

      executor.shutdown();
      assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
    }
  }

  private IndexWriter createIndexWriter(Directory directory) throws IOException {
    IndexWriterConfig config =
        newIndexWriterConfig(new MockAnalyzer(random(), MockTokenizer.SIMPLE, true))
            .setCodec(codec)
            .setMergePolicy(newTieredMergePolicy());
    return new IndexWriter(directory, config);
  }

  private static float[] generateRandomVector(int dimensions, Random random) {
    float[] vector = new float[dimensions];
    for (int i = 0; i < dimensions; i++) {
      vector[i] = random.nextFloat() * 100;
    }
    return vector;
  }
}
