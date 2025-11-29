/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene.benchmarks;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
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
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.All)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1)
@Warmup(iterations = 0, time = 1)
@Measurement(iterations = 1, time = 1)
public class Benchmarks {

  private static Logger log = Logger.getLogger(Benchmarks.class.getName());
  private static Random random = new Random(222);
  private static Path indexDirPath = Paths.get(UUID.randomUUID().toString());

  @Benchmark
  public void benchmarkCagraToHnswIndexAndSearchFlow() throws Exception {

    Codec codec = new Lucene101AcceleratedHNSWCodec(32, 128, 64, 3, 16, 100);
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);

    final int COMMIT_FREQ = 100;
    final String ID_FIELD = "id";
    final String VECTOR_FIELD = "vector_field";

    int numDocs = 100;
    int dimension = 32;
    int topK = 5;
    int count = COMMIT_FREQ;
    float[][] dataset = generateDataset(random, numDocs, dimension);

    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
      for (int i = 0; i < numDocs; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        indexWriter.addDocument(document);
        count -= 1;
        if (count == 0) {
          indexWriter.commit();
          count = COMMIT_FREQ;
        }
      }
    }

    // Searching
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(indexDirectory)) {

      for (LeafReaderContext leafReaderContext : reader.leaves()) {
        LeafReader leafReader = leafReaderContext.reader();
        FloatVectorValues knnValues = leafReader.getFloatVectorValues(VECTOR_FIELD);
        log.info(
            VECTOR_FIELD
                + " field: "
                + knnValues.size()
                + " vectors, "
                + knnValues.dimension()
                + " dimensions");
      }

      IndexSearcher searcher = new IndexSearcher(reader);

      float[] queryVector = generateDataset(random, 1, dimension)[0];
      log.info("Query vector: " + Arrays.toString(queryVector));

      KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK);
      TopDocs results = searcher.search(query, topK);

      log.info("Search results (" + results.totalHits + " total hits):");
      Integer[] expected = new Integer[] {1869, 1803, 1302, 59, 1497, 108, 1411, 351, 1982};
      HashSet<Integer> expectedIds = new HashSet<Integer>(Arrays.asList(expected));

      for (int i = 0; i < results.scoreDocs.length; i++) {
        ScoreDoc scoreDoc = results.scoreDocs[i];
        Document doc = searcher.storedFields().document(scoreDoc.doc);
        String id = doc.get(ID_FIELD);
        log.info(
            "  Rank "
                + (i + 1)
                + ": doc "
                + scoreDoc.doc
                + " (id="
                + id
                + "), score="
                + scoreDoc.score);
      }
    }

    File indexDirPathFile = indexDirPath.toFile();
    if (indexDirPathFile.exists() && indexDirPathFile.isDirectory()) {
      FileUtils.deleteDirectory(indexDirPathFile);
    }
  }

  public static float[][] generateDataset(Random random, int size, int dimensions) {
    float[][] dataset = new float[size][dimensions];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < dimensions; j++) {
        dataset[i][j] = random.nextFloat() * 100;
      }
    }
    return dataset;
  }

  public static float[] generateRandomVector(int dimensions, Random random) {
    float[] vector = new float[dimensions];
    for (int i = 0; i < dimensions; i++) {
      vector[i] = random.nextFloat() * 100;
    }
    return vector;
  }

  public static float[][] generateQueries(Random random, int dimensions, int numQueries) {
    // Generate random query vectors
    float[][] queries = new float[numQueries][dimensions];
    for (int i = 0; i < numQueries; i++) {
      for (int j = 0; j < dimensions; j++) {
        queries[i][j] = random.nextFloat() * 100;
      }
    }
    return queries;
  }
}
