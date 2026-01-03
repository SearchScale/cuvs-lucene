/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.createWriter;
import static com.nvidia.cuvs.lucene.TestUtils.generateExpectedResults;
import static com.nvidia.cuvs.lucene.TestUtils.generateQueries;
import static com.nvidia.cuvs.lucene.TestUtils.generateRandomVector;
import static com.nvidia.cuvs.lucene.TestUtils.generateRandomVectors;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.English;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestAcceleratedHNSWRandomizedSearch extends LuceneTestCase {

  private static Logger log = Logger.getLogger(TestAcceleratedHNSWRandomizedSearch.class.getName());
  private static Codec codec;
  private static IndexSearcher searcher;
  private static IndexReader reader;
  private static Directory directory;
  private static Random random;
  private static int datasetSize;
  private static int dimensions;
  private static int topK;
  private static float[][] dataset;
  private static float[][] dataset2;
  private static float[] queryVector;
  private static int numQueries;

  private static final String ID_FIELD = "id";
  private static final String TEXT_FIELD = "some_text_field";
  private static final String VECTOR_FIELD = "vector_field";
  private static final String VECTOR_FIELD2 = "vector_field2";
  private static final int DATASET_SIZE_LIMIT = 1000;
  private static final int DIMENSIONS_LIMIT = 256;
  private static final int TOP_K_LIMIT = 64;
  private static final int NUM_QUERIES_LIMIT = 10;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue(
        "cuVS not supported so skipping these tests",
        Lucene99AcceleratedHNSWVectorsFormat.supported());
    directory = newDirectory();
    random = random();
    codec = TestUtil.alwaysKnnVectorsFormat(new Lucene99AcceleratedHNSWVectorsFormat());
    RandomIndexWriter writer = createWriter(random, directory, codec);

    datasetSize = random.nextInt(5, DATASET_SIZE_LIMIT);
    dimensions = random.nextInt(8, DIMENSIONS_LIMIT);
    topK = Math.min(random.nextInt(2, TOP_K_LIMIT), datasetSize);
    dataset = generateRandomVectors(random, datasetSize, dimensions);
    dataset2 = generateRandomVectors(random, datasetSize, dimensions);
    queryVector = generateRandomVector(random, dimensions);
    numQueries = Math.min(random.nextInt(1, NUM_QUERIES_LIMIT), datasetSize);

    log.log(
        Level.FINE,
        "Dataset size: "
            + datasetSize
            + "x"
            + dimensions
            + ", topK: "
            + topK
            + ", numQueries: "
            + numQueries);

    // Add documents
    for (int i = 0; i < datasetSize; i++) {
      Document doc = new Document();
      doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
      doc.add(newTextField(TEXT_FIELD, English.intToEnglish(i), Field.Store.YES));
      boolean skipVector = random.nextInt(10) < 4;
      if (!skipVector || datasetSize < 100) {
        doc.add(
            new KnnFloatVectorField(VECTOR_FIELD, dataset[i], VectorSimilarityFunction.EUCLIDEAN));
        doc.add(
            new KnnFloatVectorField(
                VECTOR_FIELD2, dataset2[i], VectorSimilarityFunction.EUCLIDEAN));
      }
      writer.addDocument(doc);
    }
    writer.commit();
    reader = writer.getReader();
    searcher = newSearcher(reader);
    writer.close();
  }

  @Test
  public void testVectorSearch() throws IOException {

    // Generate queries and expected results for each
    float[][] queries = generateQueries(random, dimensions, numQueries);
    List<List<Integer>> expected = generateExpectedResults(topK, dataset, queries);

    for (int i = 0; i < numQueries; i++) {
      log.log(Level.FINE, "Running query: " + (i + 1) + " of " + numQueries);
      Query query = new KnnFloatVectorQuery(VECTOR_FIELD, queries[i], topK);

      // Perform search
      ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;
      log.log(Level.FINE, "RESULTS: " + Arrays.toString(hits));
      log.log(Level.FINE, "EXPECTED: " + expected.get(i));

      // Iterate through the results and assert
      for (ScoreDoc hit : hits) {
        Document doc = reader.storedFields().document(hit.doc);
        int docId = Integer.parseInt(doc.get(ID_FIELD));
        log.log(Level.FINE, "\t" + doc.get(ID_FIELD) + ": " + hit.score);
        assertTrue("Result returned was not in topk*2: " + doc, expected.get(i).contains(docId));
      }
    }
  }

  @Test
  public void testVectorSearchWithFilter() throws IOException {
    // Find a document that has a vector by doing a search first
    Query unfiltered = new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, 1);
    ScoreDoc[] unfilteredHits = searcher.search(unfiltered, 1).scoreDocs;

    assertTrue(
        "Need at least one document with vector for filtering test", unfilteredHits.length > 0);

    Document doc = reader.storedFields().document(unfilteredHits[0].doc);
    String targetDocId = doc.get(ID_FIELD);

    // Create a filter that matches only the document we know has a vector
    Query filter = new TermQuery(new Term(ID_FIELD, targetDocId));

    // Test the new constructor with filter
    Query filteredQuery = new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK, filter);

    ScoreDoc[] filteredHits = searcher.search(filteredQuery, topK).scoreDocs;

    // Ensure we got some results
    assertTrue("Should have at least one result", filteredHits.length > 0);

    // Verify that all results match the filter
    for (ScoreDoc hit : filteredHits) {
      String docId = reader.storedFields().document(hit.doc).get("id");
      assertEquals("All results should match the filter", targetDocId, docId);
    }

    log.log(Level.FINE, "Prefiltering test passed with " + filteredHits.length + " results");
  }

  @AfterClass
  public static void afterClass() throws Exception {
    if (reader != null) reader.close();
    if (directory != null) directory.close();
    searcher = null;
    reader = null;
    directory = null;
  }
}
