/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.TestUtils.generateRandomVector;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestDeletes extends LuceneTestCase {

  @Test
  public void test() throws Exception {
    // assertTrue(false);
    Random random = new Random();
    Codec codec = new CuVS2510GPUSearchCodec();
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setMaxBufferedDocs(50)
            .setMergePolicy(new TieredMergePolicy())
            .setMergeScheduler(new SerialMergeScheduler())
            .setCodec(codec);

    try (Directory directory = new ByteBuffersDirectory()) {
      int datasetSize = random.nextInt(200, 1000); // 200-1200 documents
      int dimensions = random.nextInt(64, 256); // 64-320 dimensions
      float deletionProbability = random.nextFloat() * 0.4f + 0.1f; // 10-50% deletion rate

      float[][] dataset = generateDataset(random, datasetSize, dimensions);
      Set<Integer> deletedDocs = new HashSet<>();

      // Create index with all documents having vectors
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int i = 0; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          doc.add(
              new KnnFloatVectorField("vector", dataset[i], VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
        }

        // Delete
        for (int i = 0; i < datasetSize; i++) {
          if (random.nextFloat() < deletionProbability) {
            writer.deleteDocuments(new Term("id", String.valueOf(i)));
            deletedDocs.add(i);
          }
        }
        writer.commit();
      }

      int actD = datasetSize - deletedDocs.size();
      int topK = Math.min(1, actD);
      System.out.println("Total docs: " + datasetSize + " num delete docs:" + deletedDocs.size());

      // Search and verify deleted documents are not returned
      try (DirectoryReader reader = DirectoryReader.open(directory)) {

        IndexSearcher searcher = newSearcher(reader);
        // Use a random vector for query
        float[] queryVector = generateRandomVector(dimensions, random);

        GPUKnnFloatVectorQuery query =
            new GPUKnnFloatVectorQuery("vector", queryVector, topK, null, topK, 1);
        ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;
        // Verify we got results
        assertTrue("Should have search results", hits.length > 0);

        for (int i = 0; i < hits.length; i++) {
          ScoreDoc scoreDoc = hits[i];
          Document doc = searcher.storedFields().document(scoreDoc.doc);
          String id = doc.get("id");
          System.out.println(
              "Rank "
                  + (i + 1)
                  + ": doc "
                  + scoreDoc.doc
                  + " (id="
                  + id
                  + "), score="
                  + scoreDoc.score);
        }

        // Verify no deleted documents in results
        for (ScoreDoc hit : hits) {
          String docId = reader.storedFields().document(hit.doc).get("id");
          int id = Integer.parseInt(docId);
          assertFalse(
              "Deleted document " + id + " should not appear in results", deletedDocs.contains(id));
          System.out.println("Found non-deleted document: " + id + ", Score: " + hit.score);
        }

        // Verify deleted documents are truly deleted
        for (int deletedId : deletedDocs) {
          TopDocs result =
              searcher.search(new TermQuery(new Term("id", String.valueOf(deletedId))), 1);
          assertEquals(
              "Deleted document " + deletedId + " should not be found",
              0,
              result.totalHits.value());
        }
      }
    }
  }
}
