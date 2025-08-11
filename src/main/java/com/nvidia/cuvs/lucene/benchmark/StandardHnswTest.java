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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Test standard Lucene HNSW performance for comparison
 */
public class StandardHnswTest {

  public static void main(String[] args) throws IOException {
    System.out.println("🔍 Testing Standard Lucene HNSW for comparison");

    // Load SIFT data
    List<float[]> vectors =
        FvecsReader.readFvecs("/data/faiss/siftsmall/siftsmall_base.fvecs", 10000);
    List<float[]> queries =
        FvecsReader.readFvecs("/data/faiss/siftsmall/siftsmall_query.fvecs", 100);
    List<int[]> groundTruth =
        IvecsReader.readIvecs("/data/faiss/siftsmall/siftsmall_groundtruth.ivecs", 100);

    System.out.println(
        "✅ Loaded SIFT data: " + vectors.size() + " vectors, " + queries.size() + " queries");

    // Create standard Lucene HNSW index
    Path tempDir = Files.createTempDirectory("standard_hnsw_test");
    try (Directory directory = FSDirectory.open(tempDir)) {

      // Use default codec
      Lucene101Codec codec = new Lucene101Codec();

      // Index documents
      IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
      config.setCodec(codec);
      config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);

      long indexStart = System.currentTimeMillis();
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int i = 0; i < vectors.size(); i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("vector", vectors.get(i), EUCLIDEAN));
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
      }
      long indexTime = System.currentTimeMillis() - indexStart;

      // Search
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = new IndexSearcher(reader);

        long searchStart = System.currentTimeMillis();
        int totalHits = 0;
        int correctHits = 0;

        for (int i = 0; i < Math.min(queries.size(), 10); i++) { // Test first 10 queries
          KnnFloatVectorQuery query = new KnnFloatVectorQuery("vector", queries.get(i), 10);
          TopDocs topDocs = searcher.search(query, 10);
          totalHits += topDocs.scoreDocs.length;

          // Check against ground truth
          if (i < groundTruth.size()) {
            int[] truth = groundTruth.get(i);
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
              Document doc = searcher.storedFields().document(scoreDoc.doc);
              int docId = Integer.parseInt(doc.get("id"));
              for (int truthId : truth) {
                if (docId == truthId) {
                  correctHits++;
                  break;
                }
              }
            }
          }

          System.out.println("Query " + i + ": " + topDocs.scoreDocs.length + " hits");
        }
        long searchTime = System.currentTimeMillis() - searchStart;

        System.out.println("\n📊 Standard Lucene HNSW Results:");
        System.out.println("  - Indexing time: " + indexTime + " ms");
        System.out.println("  - Search time: " + searchTime + " ms");
        System.out.println("  - Total hits: " + totalHits);
        System.out.println("  - Average hits per query: " + (totalHits / 10.0));
        System.out.println("  - Correct hits: " + correctHits);
        System.out.println("  - Approximate recall: " + String.format("%.4f", correctHits / 100.0));
      }

    } finally {
      // Cleanup
      Files.walk(tempDir)
          .sorted((a, b) -> b.compareTo(a))
          .forEach(
              path -> {
                try {
                  Files.delete(path);
                } catch (IOException e) {
                  // ignore
                }
              });
    }
  }
}
