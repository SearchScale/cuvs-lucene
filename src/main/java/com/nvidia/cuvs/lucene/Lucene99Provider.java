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

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.List;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TaskExecutor;

public class Lucene99Provider {

  private static final String BASE = "org.apache.lucene.";
  private static final String CODECS = "codecs.lucene99.";
  private static final String BACKWARD_CODECS = "backward_codecs.lucene99.";

  private static final String LUCENE99_FLAT_VECTORS_FORMAT =
      BASE + CODECS + "Lucene99FlatVectorsFormat";
  private static final String LUCENE99_FLAT_VECTORS_FORMAT_FALLBACK =
      BASE + BACKWARD_CODECS + "Lucene99FlatVectorsFormat";

  private static final String LUCENE99_HNSW_VECTORS_FORMAT =
      BASE + CODECS + "Lucene99HnswVectorsFormat";
  private static final String LUCENE99_HNSW_VECTORS_FORMAT_FALLBACK =
      BASE + BACKWARD_CODECS + "Lucene99HnswVectorsFormat";

  private static final String LUCENE99_HNSW_VECTORS_READER =
      BASE + CODECS + "Lucene99HnswVectorsReader";
  private static final String LUCENE99_HNSW_VECTORS_READER_FALLBACK =
      BASE + BACKWARD_CODECS + "Lucene99HnswVectorsReader";

  private static final String LUCENE99_HNSW_VECTORS_WRITER =
      BASE + CODECS + "Lucene99HnswVectorsWriter";
  private static final String LUCENE99_HNSW_VECTORS_WRITER_FALLBACK =
      BASE + BACKWARD_CODECS + "Lucene99HnswVectorsWriter";

  private Class<?> lucene99FlatVectorsFormat;
  private Class<?> lucene99HnswVectorsFormat;
  private Class<?> lucene99HnswVectorsReader;
  private Class<?> lucene99HnswVectorsWriter;

  public Lucene99Provider() {
    try {
      lucene99FlatVectorsFormat = Class.forName(LUCENE99_FLAT_VECTORS_FORMAT);
      lucene99HnswVectorsFormat = Class.forName(LUCENE99_HNSW_VECTORS_FORMAT);
      lucene99HnswVectorsReader = Class.forName(LUCENE99_HNSW_VECTORS_READER);
      lucene99HnswVectorsWriter = Class.forName(LUCENE99_HNSW_VECTORS_WRITER);
    } catch (Exception e) {
      try {
        lucene99FlatVectorsFormat = Class.forName(LUCENE99_FLAT_VECTORS_FORMAT_FALLBACK);
        lucene99HnswVectorsFormat = Class.forName(LUCENE99_HNSW_VECTORS_FORMAT_FALLBACK);
        lucene99HnswVectorsReader = Class.forName(LUCENE99_HNSW_VECTORS_READER_FALLBACK);
        lucene99HnswVectorsWriter = Class.forName(LUCENE99_HNSW_VECTORS_WRITER_FALLBACK);
      } catch (ClassNotFoundException e1) {
        // Should not be reaching here in any situation
        e1.printStackTrace();
      }
    }
  }

  public FlatVectorsFormat getLucene99FlatVectorsFormatInstance() {
    Constructor<?> lucene99FlatVectorsFormatConstructor;
    try {
      lucene99FlatVectorsFormatConstructor =
          lucene99FlatVectorsFormat.getConstructor(FlatVectorsScorer.class);
      return (FlatVectorsFormat)
          lucene99FlatVectorsFormatConstructor.newInstance(DefaultFlatVectorScorer.INSTANCE);

    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  public KnnVectorsReader getLucene99HnswVectorsReaderInstance(
      SegmentReadState state, FlatVectorsReader reader) {
    try {
      Constructor<?> lucene99HnswVectorsReaderConstructor =
          lucene99HnswVectorsReader.getConstructor(SegmentReadState.class, FlatVectorsReader.class);
      return (KnnVectorsReader) lucene99HnswVectorsReaderConstructor.newInstance(state, reader);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  public KnnVectorsWriter getLucene99HnswVectorsWriterInstance(
      SegmentWriteState state,
      int maxConn,
      int beamWidth,
      FlatVectorsWriter writer,
      int numMergeWorkers,
      TaskExecutor executor) {
    try {
      Constructor<?> lucene99HnswVectorsWriterConstructor =
          lucene99HnswVectorsWriter.getConstructor(
              SegmentWriteState.class,
              Integer.TYPE,
              Integer.TYPE,
              FlatVectorsWriter.class,
              Integer.TYPE,
              TaskExecutor.class);
      return (KnnVectorsWriter)
          lucene99HnswVectorsWriterConstructor.newInstance(
              state, maxConn, beamWidth, writer, numMergeWorkers, executor);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  public int getDefaultBeamWidth() {
    Field f = null;
    try {
      f = lucene99HnswVectorsFormat.getField("DEFAULT_BEAM_WIDTH");
      return (int) f.getInt(null);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return -1;
  }

  public int getDefaultMaxConnection() {
    Field f = null;
    try {
      f = lucene99HnswVectorsFormat.getField("DEFAULT_MAX_CONN");
      return (int) f.getInt(null);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return -1;
  }

  public int getDefaultNumMergeWorker() {
    Field f = null;
    try {
      f = lucene99HnswVectorsFormat.getField("DEFAULT_NUM_MERGE_WORKER");
      return (int) f.getInt(null);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return -1;
  }

  public int getVersionCurrent() {
    Field f = null;
    try {
      f = lucene99HnswVectorsFormat.getField("VERSION_CURRENT");
      return (int) f.getInt(null);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return -1;
  }

  @SuppressWarnings("unchecked")
  public List<VectorSimilarityFunction> getSimilarityFunctions() {
    Field f = null;
    try {
      f = lucene99HnswVectorsReader.getField("SIMILARITY_FUNCTIONS");
      return (List<VectorSimilarityFunction>) f.get(null);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }
}
