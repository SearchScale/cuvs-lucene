/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class PerfTests extends LuceneTestCase {

  private static final Logger log = Logger.getLogger(PerfTests.class.getName());
  private static final int NUM_VECTORS = 1_000_000;
  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector_field";
  private static IndexWriterConfig config;
  private static Codec codec;
  private static List<float[]> dataset;
  private static Path indexDirPath;
  private static int numThreads = 2;

  @BeforeClass
  public static void beforeClass() throws Exception {
    log.log(Level.INFO, "Starting perf tests ...");
    dataset = new ArrayList<float[]>();
    DatasetUtils.readDataFile("test-dataset/base.1M.fbin", NUM_VECTORS, null, dataset);
    codec = new Lucene101AcceleratedHNSWCodec();
    config = new IndexWriterConfig().setCodec(codec);
    config.setMaxBufferedDocs(NUM_VECTORS);
    config.setRAMBufferSizeMB(32767);
  }

  @Before
  public void beforeEach() {
    indexDirPath = Paths.get(UUID.randomUUID().toString());
  }

  @After
  public void afterEach() throws IOException {
    File indexDirPathFile = indexDirPath.toFile();
    if (indexDirPathFile.exists() && indexDirPathFile.isDirectory()) {
      FileUtils.deleteDirectory(indexDirPathFile);
    }
  }

  @Test
  public void gaugeIndexBuildTime() throws IOException, InterruptedException {
    StopWatch sw = StopWatch.createStarted();
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        IndexWriter writer = new IndexWriter(indexDirectory, config);
        ExecutorService pool = Executors.newFixedThreadPool(numThreads)) {
      AtomicInteger id = new AtomicInteger(1);
      for (int pi = 0; pi < numThreads; pi++) {
        pool.submit(
            () -> {
              while (id.getAndIncrement() <= NUM_VECTORS) {
                try {
                  Document document = new Document();
                  document.add(
                      new StringField(ID_FIELD, Integer.toString(id.get()), Field.Store.YES));
                  document.add(
                      new KnnFloatVectorField(VECTOR_FIELD, dataset.get(id.get() - 1), EUCLIDEAN));
                  writer.addDocument(document);
                } catch (IOException ex) {
                  throw new UncheckedIOException(ex);
                }
              }
            });
      }
      pool.shutdown();
      pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);

      log.log(Level.INFO, "Committing documents");
      writer.commit();
    }
    sw.stop();
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
      log.log(Level.INFO, "Number of segments: " + reader.leaves().size());
    }
    log.log(Level.INFO, "Index build time (ms) is: " + sw.getTime(TimeUnit.MILLISECONDS));
  }

  @AfterClass
  public static void afterClass() {
    // Cleanup
  }
}
