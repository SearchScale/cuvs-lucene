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

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.lucene.CuVSVectorsWriter.IndexType;
import java.io.IOException;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * CuVS-based KnnVectorsFormat specifically designed for CPU search workflows.
 *
 * This format provides GPU-accelerated indexing (using CAGRA) with seamless CPU search
 * by serializing the GPU-built graph into Lucene's native HNSW format for reading.
 *
 * Key features:
 * - GPU indexing with CAGRA algorithm for high performance
 * - Automatic conversion to Lucene HNSW format for CPU search compatibility
 * - Robust error handling for CuVS library issues
 * - Fallback to standard Lucene HNSW when CuVS is unavailable
 */
public class CuVSCPUSearchVectorsFormat extends KnnVectorsFormat {

  private static final Logger LOG = Logger.getLogger(CuVSCPUSearchVectorsFormat.class.getName());

  // Format metadata
  static final String CUVS_CPU_SEARCH_META_CODEC_NAME = "Lucene102CuVSCPUSearchVectorsFormatMeta";
  static final String CUVS_CPU_SEARCH_META_CODEC_EXT = "vecm";
  static final String CUVS_CPU_SEARCH_INDEX_CODEC_NAME = "Lucene102CuVSCPUSearchVectorsFormatIndex";
  static final String CUVS_CPU_SEARCH_INDEX_EXT = "vcpu";

  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = VERSION_START;

  // Default parameters optimized for CPU search workflow
  public static final int DEFAULT_WRITER_THREADS = 32;
  public static final int DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 128;
  public static final int DEFAULT_GRAPH_DEGREE = 64; // Optimized for CPU search
  public static final IndexType DEFAULT_INDEX_TYPE = IndexType.HNSW_LUCENE;

  static CuVSResources resources = cuVSResourcesOrNull();

  /** The format for storing, reading, and merging raw vectors on disk. */
  private static final Lucene99FlatVectorsFormat flatVectorsFormat =
      new Lucene99FlatVectorsFormat(DefaultFlatVectorScorer.INSTANCE);

  final int maxDimensions = 4096;
  final int cuvsWriterThreads;
  final int intGraphDegree;
  final int graphDegree;
  final IndexType indexType;

  /**
   * Creates a CuVSCPUSearchVectorsFormat with default parameters optimized for CPU search.
   */
  public CuVSCPUSearchVectorsFormat() {
    this(
        DEFAULT_WRITER_THREADS,
        DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
        DEFAULT_GRAPH_DEGREE,
        DEFAULT_INDEX_TYPE);
  }

  /**
   * Creates a CuVSCPUSearchVectorsFormat with custom parameters.
   *
   * @param cuvsWriterThreads Number of threads for CuVS operations
   * @param intGraphDegree Intermediate graph degree for CAGRA construction
   * @param graphDegree Final graph degree for HNSW serialization
   * @param indexType Index type (should be HNSW_LUCENE for CPU search)
   */
  public CuVSCPUSearchVectorsFormat(
      int cuvsWriterThreads, int intGraphDegree, int graphDegree, IndexType indexType) {
    super("CuVSCPUSearchVectorsFormat");
    this.cuvsWriterThreads = cuvsWriterThreads;
    this.intGraphDegree = intGraphDegree;
    this.graphDegree = graphDegree;
    this.indexType = indexType;
  }

  private static CuVSResources cuVSResourcesOrNull() {
    try {
      resources = CuVSResources.create();
      return resources;
    } catch (UnsupportedOperationException uoe) {
      LOG.warning("CuVS is not supported on this platform or java version: " + uoe.getMessage());
    } catch (Throwable t) {
      if (t instanceof ExceptionInInitializerError ex) {
        t = ex.getCause();
      }
      LOG.warning("Exception occurred during creation of CuVS resources. " + t);
    }
    return null;
  }

  /** Tells whether the platform supports CuVS. */
  public static boolean supported() {
    if (resources == null) {
      try {
        resources = CuVSResources.create();
      } catch (Throwable t) {
        return false;
      }
    }
    return resources != null;
  }

  private static void checkSupported() {
    if (!supported()) {
      throw new UnsupportedOperationException("CuVS not supported on this platform");
    }
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    try {
      checkSupported();
      var flatWriter = flatVectorsFormat.fieldsWriter(state);
      LOG.info("Creating CuVS writer for GPU indexing with CPU search compatibility");
      return new CuVSVectorsWriter(
          state, cuvsWriterThreads, intGraphDegree, graphDegree, indexType, resources, flatWriter);
    } catch (Exception e) {
      LOG.warning(
          "Failed to create CuVS writer, falling back to standard Lucene HNSW: " + e.getMessage());
      // Fallback to standard Lucene HNSW format
      return new org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat(graphDegree, 200)
          .fieldsWriter(state);
    }
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    var flatReader = flatVectorsFormat.fieldsReader(state);

    // Always use Lucene HNSW reader for CPU search - this is the key difference
    // from the main CuVSVectorsFormat which tries to use CuVS for reading too
    LOG.info("Using Lucene99HnswVectorsReader for CPU search compatibility");
    return new Lucene99HnswVectorsReader(state, flatReader);
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return maxDimensions;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("CuVSCPUSearchVectorsFormat(");
    sb.append("cuvsWriterThreads=").append(cuvsWriterThreads);
    sb.append(", intGraphDegree=").append(intGraphDegree);
    sb.append(", graphDegree=").append(graphDegree);
    sb.append(", indexType=").append(indexType);
    sb.append(", resources=").append(resources != null ? "available" : "unavailable");
    sb.append(")");
    return sb.toString();
  }
}
