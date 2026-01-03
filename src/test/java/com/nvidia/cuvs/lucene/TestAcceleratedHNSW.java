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
  private static int datasetSize;
  private static int dimensions;
  private static int topK;
  private static float[][] dataset;
  private static float[][] dataset2;
  private static String randomID;
  private static Codec codec;

  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector_field";
  private static final String VECTOR_FIELD2 = "vector_field2";
  private static final int DATASET_SIZE_LIMIT = 1000;
  private static final int DIMENSIONS_LIMIT = 256;
  private static final int TOP_K_LIMIT = 64;

  @Before
  public void beforeTest() throws Exception {
    assumeTrue(
        "cuVS not supported so skipping these tests",
        Lucene99AcceleratedHNSWVectorsFormat.supported());
    random = new Random();
    indexDirPath = Paths.get(UUID.randomUUID().toString());
    datasetSize = random.nextInt(200, DATASET_SIZE_LIMIT);
    dimensions = random.nextInt(8, DIMENSIONS_LIMIT);
    topK = Math.min(random.nextInt(2, TOP_K_LIMIT), datasetSize);
    dataset = generateRandomVectors(random, datasetSize, dimensions);
    dataset2 = generateRandomVectors(random, datasetSize, dimensions);
    randomID = UUID.randomUUID().toString();
    codec =
        new Lucene101AcceleratedHNSWCodec(32, 128, 64, CagraGraphBuildAlgo.NN_DESCENT, 3, 16, 100);
    log.log(Level.FINE, "Dataset size: " + datasetSize + "x" + dimensions + ", topK: " + topK);
  }

  @Test
  public void testAcceleratedHNSW() throws Exception {
    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        RandomIndexWriter indexWriter = createWriter(random, indexDirectory, codec)) {
      for (int i = 0; i < datasetSize; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        document.add(new KnnFloatVectorField(VECTOR_FIELD2, dataset2[i], EUCLIDEAN));
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
        FloatVectorValues knnValues = leafReader.getFloatVectorValues(VECTOR_FIELD);
        assertNotNull(knnValues);
        log.log(
            Level.FINE,
            VECTOR_FIELD
                + " field: "
                + knnValues.size()
                + " vectors, "
                + knnValues.dimension()
                + " dimensions");
        vectorCount += knnValues.size();
        assertTrue("Vector dimension mismatch", knnValues.dimension() == dimensions);
      }
      assertTrue("Dataset size mismatch", vectorCount == datasetSize);

      log.log(Level.FINE, "Testing vector search queries...");
      IndexSearcher searcher = new IndexSearcher(reader);

      float[] queryVector = generateRandomVectors(random, 1, dimensions)[0];
      log.log(Level.FINER, "Query vector: " + Arrays.toString(queryVector));

      KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK);
      TopDocs results = searcher.search(query, topK);

      log.log(Level.FINE, "Search results (" + results.totalHits + " total hits):");
      List<List<Integer>> expected =
          generateExpectedResults(topK, dataset, new float[][] {queryVector});

      for (int i = 0; i < results.scoreDocs.length; i++) {
        ScoreDoc scoreDoc = results.scoreDocs[i];
        Document doc = searcher.storedFields().document(scoreDoc.doc);
        int id = Integer.valueOf(doc.get(ID_FIELD));
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
      assertTrue("TopK results not returned", results.scoreDocs.length == topK);
    }
  }

  @Test
  public void testSingleVectorIndex() throws Exception {
    try (Directory indexDirectory = newDirectory()) {
      float[] vector = generateRandomVector(random, dimensions);
      IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);
      try (IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, randomID, Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
        indexWriter.addDocument(document);
        indexWriter.commit();
      }

      // Verify the index can be opened and searched
      try (DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
        assertEquals(1, reader.numDocs());
        LeafReader leafReader = getOnlyLeafReader(reader);
        FloatVectorValues knnValues = leafReader.getFloatVectorValues(VECTOR_FIELD);
        assertNotNull(knnValues);
        assertEquals(1, knnValues.size());
        assertEquals(dimensions, knnValues.dimension());

        // Test search functionality
        IndexSearcher searcher = new IndexSearcher(reader);
        KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, vector, 1);
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
