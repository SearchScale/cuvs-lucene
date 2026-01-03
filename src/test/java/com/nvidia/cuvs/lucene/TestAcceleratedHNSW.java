/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.createWriter;
import static com.nvidia.cuvs.lucene.TestUtils.generateExpectedResults;
import static com.nvidia.cuvs.lucene.TestUtils.generateRandomVector;
import static com.nvidia.cuvs.lucene.TestUtils.generateRandomVectors;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestAcceleratedHNSW extends LuceneTestCase {

  private static Logger log = Logger.getLogger(TestAcceleratedHNSW.class.getName());
  private static Random random;
  private static Path indexDirPath;
  private static String randomID;
  private static Codec codec;
  private static TestDataProvider testDataProvider;

  @Before
  public void beforeTest() throws Exception {
    assumeTrue(
        "cuVS not supported so skipping these tests",
        Lucene99AcceleratedHNSWVectorsFormat.supported());
    random = new Random();
    indexDirPath = Paths.get(UUID.randomUUID().toString());
    randomID = UUID.randomUUID().toString();
    testDataProvider = new TestDataProvider(random);
    codec =
        new Lucene101AcceleratedHNSWCodec(32, 128, 64, CagraGraphBuildAlgo.NN_DESCENT, 3, 16, 100);
  }

  @Test
  public void testAcceleratedHNSW() throws Exception {
    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        RandomIndexWriter indexWriter = createWriter(random, indexDirectory, codec)) {
      for (int i = 0; i < testDataProvider.getDatasetSize(); i++) {
        Document document = new Document();
        document.add(
            new StringField(TestDataProvider.ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(
            new KnnFloatVectorField(
                TestDataProvider.VECTOR_FIELD, testDataProvider.getDataset()[i], EUCLIDEAN));
        document.add(
            new KnnFloatVectorField(
                TestDataProvider.VECTOR_FIELD2, testDataProvider.getDataset2()[i], EUCLIDEAN));
        indexWriter.addDocument(document);
      }
      indexWriter.commit();
    }

    // Searching
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
      int vectorCount = 0;
      for (LeafReaderContext leafReaderContext : reader.leaves()) {
        LeafReader leafReader = leafReaderContext.reader();
        FloatVectorValues knnValues =
            leafReader.getFloatVectorValues(TestDataProvider.VECTOR_FIELD);
        assertNotNull(knnValues);
        log.log(
            Level.FINE,
            TestDataProvider.VECTOR_FIELD
                + " field: "
                + knnValues.size()
                + " vectors, "
                + knnValues.dimension()
                + " dimensions");
        vectorCount += knnValues.size();
        assertTrue(
            "Vector dimension mismatch", knnValues.dimension() == testDataProvider.getDimensions());
      }
      assertTrue("Dataset size mismatch", vectorCount == testDataProvider.getDatasetSize());

      log.log(Level.FINE, "Testing vector search queries...");
      IndexSearcher searcher = new IndexSearcher(reader);

      float[] queryVector = generateRandomVectors(random, 1, testDataProvider.getDimensions())[0];
      log.log(Level.FINER, "Query vector: " + Arrays.toString(queryVector));

      KnnFloatVectorQuery query =
          new KnnFloatVectorQuery(
              TestDataProvider.VECTOR_FIELD, queryVector, testDataProvider.getTopK());
      TopDocs results = searcher.search(query, testDataProvider.getTopK());

      log.log(Level.FINE, "Search results (" + results.totalHits + " total hits):");
      List<List<Integer>> expected =
          generateExpectedResults(
              testDataProvider.getTopK(),
              testDataProvider.getDataset(),
              new float[][] {queryVector});

      for (int i = 0; i < results.scoreDocs.length; i++) {
        ScoreDoc scoreDoc = results.scoreDocs[i];
        Document doc = searcher.storedFields().document(scoreDoc.doc);
        int id = Integer.valueOf(doc.get(TestDataProvider.ID_FIELD));
        log.log(
            Level.FINE,
            "  Rank "
                + (i + 1)
                + ": doc "
                + scoreDoc.doc
                + " (id="
                + id
                + "), score="
                + scoreDoc.score);
        assertTrue("Id: " + id + " expected but not found", expected.get(0).contains(id));
      }
      assertTrue(
          "TopK results not returned", results.scoreDocs.length == testDataProvider.getTopK());
    }
  }

  @Test
  public void testSingleVectorIndex() throws Exception {
    try (Directory indexDirectory = newDirectory()) {
      float[] vector = generateRandomVector(random, testDataProvider.getDimensions());
      IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);
      try (IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
        Document document = new Document();
        document.add(new StringField(TestDataProvider.ID_FIELD, randomID, Field.Store.YES));
        document.add(new KnnFloatVectorField(TestDataProvider.VECTOR_FIELD, vector, EUCLIDEAN));
        indexWriter.addDocument(document);
        indexWriter.commit();
      }

      // Verify the index can be opened and searched
      try (DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
        assertEquals(1, reader.numDocs());
        LeafReader leafReader = getOnlyLeafReader(reader);
        FloatVectorValues knnValues =
            leafReader.getFloatVectorValues(TestDataProvider.VECTOR_FIELD);
        assertNotNull(knnValues);
        assertEquals(1, knnValues.size());
        assertEquals(testDataProvider.getDimensions(), knnValues.dimension());

        // Test search functionality
        IndexSearcher searcher = new IndexSearcher(reader);
        KnnFloatVectorQuery query =
            new KnnFloatVectorQuery(TestDataProvider.VECTOR_FIELD, vector, 1);
        TopDocs results = searcher.search(query, 1);
        assertEquals(1, results.totalHits.value());
        assertEquals(1, results.scoreDocs.length);
        assertEquals(randomID, reader.storedFields().document(results.scoreDocs[0].doc).get("id"));
      }
    }
  }

  @After
  public void afterTest() throws Exception {
    File indexDirPathFile = indexDirPath.toFile();
    if (indexDirPathFile.exists() && indexDirPathFile.isDirectory()) {
      FileUtils.deleteDirectory(indexDirPathFile);
    }
  }
}
