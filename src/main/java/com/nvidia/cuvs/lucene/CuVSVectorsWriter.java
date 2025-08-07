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

import static com.nvidia.cuvs.lucene.CuVSVectorsFormat.CUVS_INDEX_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVSVectorsFormat.CUVS_INDEX_EXT;
import static com.nvidia.cuvs.lucene.CuVSVectorsFormat.CUVS_META_CODEC_EXT;
import static com.nvidia.cuvs.lucene.CuVSVectorsFormat.CUVS_META_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVSVectorsFormat.VERSION_CURRENT;
import static com.nvidia.cuvs.lucene.CuVSVectorsReader.handleThrowable;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS;
import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceIndexParams;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.RowView;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.logging.Logger;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.packed.DirectMonotonicWriter;

/** KnnVectorsWriter for CuVS, responsible for merge and flush of vectors into GPU */
public class CuVSVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED = shallowSizeOfInstance(CuVSVectorsWriter.class);

  @SuppressWarnings("unused")
  private static final Logger log = Logger.getLogger(CuVSVectorsWriter.class.getName());

  /** The name of the CUVS component for the info-stream * */
  public static final String CUVS_COMPONENT = "CUVS";

  // The minimum number of vectors in the dataset required before
  // we attempt to build a Cagra index
  static final int MIN_CAGRA_INDEX_SIZE = 2;

  private final int cuvsWriterThreads;
  private final int intGraphDegree;
  private final int graphDegree;

  private final CuVSResources resources;
  private final IndexType indexType;

  private final FlatVectorsWriter flatVectorsWriter; // for writing the raw vectors
  private final List<CuVSFieldWriter> fields = new ArrayList<>();
  private final IndexOutput meta, cuvsIndex;
  private final IndexOutput hnswMeta, hnswVectorIndex;

  private final InfoStream infoStream;
  private boolean finished;

  String vemFileName;
  String vexFileName;

  /** The CuVS index Type. */
  public enum IndexType {
    /** Builds a Cagra index. */
    CAGRA(true, false, false, false),
    /** Builds a Brute Force index. */
    BRUTE_FORCE(false, true, false, false),
    /** Builds an HSNW index - suitable for searching on CPU. */
    HNSW(false, false, true, false),
    /** Builds a Cagra and a Brute Force index. */
    CAGRA_AND_BRUTE_FORCE(true, true, false, false),
    /** Builds a Lucene HNSW index via CAGRA. */
    HNSW_LUCENE(false, false, false, true);
    private final boolean cagra, bruteForce, hnsw, hnswLucene;

    IndexType(boolean cagra, boolean bruteForce, boolean hnsw, boolean hnswLucene) {
      this.cagra = cagra;
      this.bruteForce = bruteForce;
      this.hnsw = hnsw;
      this.hnswLucene = hnswLucene;
    }

    public boolean cagra() {
      return cagra;
    }

    public boolean bruteForce() {
      return bruteForce;
    }

    public boolean hnsw() {
      return hnsw;
    }

    public boolean hnswLucene() {
      return hnswLucene;
    }
  }

  public CuVSVectorsWriter(
      SegmentWriteState state,
      int cuvsWriterThreads,
      int intGraphDegree,
      int graphDegree,
      IndexType indexType,
      CuVSResources resources,
      FlatVectorsWriter flatVectorsWriter)
      throws IOException {
    super();
    this.indexType = indexType;
    this.cuvsWriterThreads = cuvsWriterThreads;
    this.intGraphDegree = intGraphDegree;
    this.graphDegree = graphDegree;
    this.resources = resources;
    this.flatVectorsWriter = flatVectorsWriter;
    this.infoStream = state.infoStream;

    vemFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, "vem");

    vexFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, "vex");

    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, CUVS_META_CODEC_EXT);
    String cagraFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, CUVS_INDEX_EXT);

    {
      hnswMeta = state.directory.createOutput(vemFileName, state.context);
      hnswVectorIndex = state.directory.createOutput(vexFileName, state.context);

      CodecUtil.writeIndexHeader(
          hnswMeta,
          "Lucene99HnswVectorsFormatMeta",
          Lucene99HnswVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          hnswVectorIndex,
          "Lucene99HnswVectorsFormatIndex",
          Lucene99HnswVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
    }

    boolean success = false;
    try {
      // Only create CAGRA files if not in HNSW_LUCENE mode
      if (indexType != IndexType.HNSW_LUCENE) {
        meta = state.directory.createOutput(metaFileName, state.context);
        cuvsIndex = state.directory.createOutput(cagraFileName, state.context);
        CodecUtil.writeIndexHeader(
            meta,
            CUVS_META_CODEC_NAME,
            VERSION_CURRENT,
            state.segmentInfo.getId(),
            state.segmentSuffix);
        CodecUtil.writeIndexHeader(
            cuvsIndex,
            CUVS_INDEX_CODEC_NAME,
            VERSION_CURRENT,
            state.segmentInfo.getId(),
            state.segmentSuffix);
      } else {
        // For HNSW_LUCENE, we don't need CAGRA files
        meta = null;
        cuvsIndex = null;
      }
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    var encoding = fieldInfo.getVectorEncoding();
    if (encoding != FLOAT32) {
      throw new IllegalArgumentException("expected float32, got:" + encoding);
    }
    var writer = Objects.requireNonNull(flatVectorsWriter.addField(fieldInfo));
    @SuppressWarnings("unchecked")
    var flatWriter = (FlatFieldVectorsWriter<float[]>) writer;
    var cuvsFieldWriter = new CuVSFieldWriter(fieldInfo, flatWriter);
    fields.add(cuvsFieldWriter);
    return cuvsFieldWriter;
  }

  static String indexMsg(int size, int... args) {
    StringBuilder sb = new StringBuilder("cagra index params");
    sb.append(": size=").append(size);
    sb.append(", intGraphDegree=").append(args[0]);
    sb.append(", actualIntGraphDegree=").append(args[1]);
    sb.append(", graphDegree=").append(args[2]);
    sb.append(", actualGraphDegree=").append(args[3]);
    return sb.toString();
  }

  private CagraIndexParams cagraIndexParams(int size) {
    if (size < 2) {
      // https://github.com/rapidsai/cuvs/issues/666
      throw new IllegalArgumentException("cagra index must be greater than 2");
    }
    // var minIntGraphDegree = Math.min(intGraphDegree, size - 1);
    // var minGraphDegree = Math.min(graphDegree, minIntGraphDegree);
    // log.info(indexMsg(size, intGraphDegree, minIntGraphDegree, graphDegree, minGraphDegree));

    return new CagraIndexParams.Builder()
        .withNumWriterThreads(cuvsWriterThreads)
        .withIntermediateGraphDegree(intGraphDegree)
        .withGraphDegree(graphDegree)
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .build();
  }

  static long nanosToMillis(long nanos) {
    return Duration.ofNanos(nanos).toMillis();
  }

  private void info(String msg) {
    if (infoStream.isEnabled(CUVS_COMPONENT)) {
      infoStream.message(CUVS_COMPONENT, msg);
    }
  }

  private void writeFieldInternal(FieldInfo fieldInfo, CuVSMatrix dataset) throws IOException {
    if (dataset.size() == 0) {
      writeEmpty(fieldInfo);
      return;
    }
    long cagraIndexOffset, cagraIndexLength = 0L;
    long bruteForceIndexOffset, bruteForceIndexLength = 0L;
    long hnswIndexOffset, hnswIndexLength = 0L;

    // workaround for the minimum number of vectors for Cagra
    IndexType indexType =
        this.indexType.cagra() && dataset.size() < MIN_CAGRA_INDEX_SIZE
            ? IndexType.BRUTE_FORCE
            : this.indexType;

    try {
      if (indexType.hnswLucene()) {
        log.info("Entered the writeFieldInternal's HNSW LUCENE block - writing only HNSW files");
        try {
          writeHnswOnlyIndex(dataset, fieldInfo);
        } catch (Throwable t) {
          handleThrowableWithIgnore(t, CANNOT_GENERATE_CAGRA);
          // workaround for cuVS issue
          indexType = IndexType.BRUTE_FORCE;
        }
        // For HNSW_LUCENE, we don't write any CAGRA data, so set lengths to 0
        cagraIndexLength = 0L;
        cagraIndexOffset = 0L;
        bruteForceIndexOffset = 0L;
        bruteForceIndexLength = 0L;
        hnswIndexOffset = 0L;
        hnswIndexLength = 0L;
      } else {
        cagraIndexOffset = cuvsIndex.getFilePointer();
        if (indexType.cagra()) {
          try {
            var cagraIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
            writeCagraIndex(cagraIndexOutputStream, dataset);
          } catch (Throwable t) {
            handleThrowableWithIgnore(t, CANNOT_GENERATE_CAGRA);
            // workaround for cuVS issue
            indexType = IndexType.BRUTE_FORCE;
          }
          cagraIndexLength = cuvsIndex.getFilePointer() - cagraIndexOffset;
        }

        bruteForceIndexOffset = cuvsIndex.getFilePointer();
        if (indexType.bruteForce()) {
          var bruteForceIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
          writeBruteForceIndex(bruteForceIndexOutputStream, dataset);
          bruteForceIndexLength = cuvsIndex.getFilePointer() - bruteForceIndexOffset;
        }

        hnswIndexOffset = cuvsIndex.getFilePointer();
        if (indexType.hnsw()) {
          var hnswIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
          if (dataset.size() > MIN_CAGRA_INDEX_SIZE) {
            try {
              writeHNSWIndex(hnswIndexOutputStream, dataset);
            } catch (Throwable t) {
              handleThrowableWithIgnore(t, CANNOT_GENERATE_CAGRA);
            }
          }
          hnswIndexLength = cuvsIndex.getFilePointer() - hnswIndexOffset;
        }
      }

      // StringBuilder sb = new StringBuilder("writeField ");
      // sb.append(": fieldInfo.name=").append(fieldInfo.name);
      // sb.append(", fieldInfo.number=").append(fieldInfo.number);
      // sb.append(", size=").append(vectors.length);
      // sb.append(", cagraIndexLength=").append(cagraIndexLength);
      // sb.append(", bruteForceIndexLength=").append(bruteForceIndexLength);
      // sb.append(", hnswIndexLength=").append(hnswIndexLength);
      // log.info(sb.toString());

      // Only write meta for non-HNSW_LUCENE modes
      if (indexType != IndexType.HNSW_LUCENE) {
        writeMeta(
            fieldInfo,
            (int) dataset.size(),
            cagraIndexOffset,
            cagraIndexLength,
            bruteForceIndexOffset,
            bruteForceIndexLength,
            hnswIndexOffset,
            hnswIndexLength);
      }
    } catch (Throwable t) {
      handleThrowable(t);
    }
  }

  private void writeHnswOnlyIndex(CuVSMatrix dataset, FieldInfo fieldInfo) throws Throwable {
    if (dataset.size() < 2) {
      throw new IllegalArgumentException(dataset.size() + " vectors, less than min [2] required");
    }
    CagraIndexParams params = cagraIndexParams((int) dataset.size());
    long startTime = System.nanoTime();
    CagraIndex index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(params).build();

    // Get the graph matrix from CAGRA index
    CuVSMatrix graphMatrix;
    try {
      graphMatrix = index.getGraph();
      info("Successfully got graph matrix from CAGRA index");
    } catch (Exception e) {
      info("Failed to get graph matrix: " + e.getMessage());
      graphMatrix = null;
    }

    int size = (int) dataset.size();
    int dimensions = fieldInfo.getVectorDimension();

    // Remember the vector index offset before writing
    long vectorIndexOffset = hnswVectorIndex.getFilePointer();

    // Write the graph to the vector index using streaming approach
    int[][] graphLevelNodeOffsets =
        writeGraphStreaming(size, dimensions, graphMatrix, hnswVectorIndex);

    // Calculate the length of written data
    long vectorIndexLength = hnswVectorIndex.getFilePointer() - vectorIndexOffset;

    // Write metadata
    writeMeta(
        hnswVectorIndex,
        hnswMeta,
        fieldInfo,
        vectorIndexOffset,
        vectorIndexLength,
        size,
        createMetadataGraph(size, graphLevelNodeOffsets), // Create a proper graph for metadata
        graphLevelNodeOffsets);

    long elapsedMillis = nanosToMillis(System.nanoTime() - startTime);
    info("HNSW-only graph created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");

    // Don't serialize CAGRA index - destroy it immediately
    index.destroyIndex();
  }

  /**
   * Creates a mock graph matrix for testing when the real graph is not available.
   * This creates a simple circular graph structure.
   */
  private CuVSMatrix createMockGraphMatrix(int size) {
    // REMOVED: This was a placeholder method that's no longer needed
    // The streaming approach doesn't need to create mock matrices
    throw new UnsupportedOperationException(
        "Mock graph matrix creation is not supported in streaming mode");
  }

  /**
   * Streaming version of writeGraph that processes one row at a time using RowView.
   * This avoids copying the entire graph matrix to the Java heap.
   */
  private int[][] writeGraphStreaming(
      int size, int dimensions, CuVSMatrix graphMatrix, IndexOutput vectorIndex)
      throws IOException {
    if (graphMatrix == null) {
      info("No graph matrix available, creating empty graph");
      return new int[0][0];
    }

    // For CAGRA graphs, we only have one level (level 0)
    int numLevels = 1;
    int maxConn = Math.max(10, (int) graphMatrix.columns()); // Use actual max connections

    int[][] offsets = new int[numLevels][];
    int[] scratch = new int[maxConn * 2];

    // Level 0 has all nodes
    int[] sortedNodes = new int[size];
    for (int i = 0; i < size; i++) {
      sortedNodes[i] = i;
    }
    offsets[0] = new int[size];
    int nodeOffsetId = 0;

    info("=== writeGraphStreaming: Level 0 has " + size + " nodes, processing with RowView ===");

    for (int node : sortedNodes) {
      // Get neighbors for this node using RowView - no copying to heap
      NeighborArray neighbors = getNeighborsFromRowView(graphMatrix, node, size);
      int neighborCount = neighbors.size();

      long offsetStart = vectorIndex.getFilePointer();
      int[] nnodes = neighbors.nodes();
      Arrays.sort(nnodes, 0, neighborCount);

      // Delta encoding to minimize storage
      int actualSize = 0;
      if (neighborCount > 0) {
        scratch[0] = nnodes[0];
        actualSize = 1;
      }
      for (int i = 1; i < neighborCount; i++) {
        assert nnodes[i] < size : "node too large: " + nnodes[i] + ">=" + size;
        if (nnodes[i - 1] == nnodes[i]) {
          continue;
        }
        scratch[actualSize++] = nnodes[i] - nnodes[i - 1];
      }

      // Write the size after duplicates are removed
      vectorIndex.writeVInt(actualSize);
      for (int i = 0; i < actualSize; i++) {
        vectorIndex.writeVInt(scratch[i]);
      }
      offsets[0][nodeOffsetId++] = Math.toIntExact(vectorIndex.getFilePointer() - offsetStart);
    }
    return offsets;
  }

  /**
   * Get neighbors for a node using RowView to avoid copying entire row to heap.
   * This method processes one row at a time using native memory access.
   */
  private NeighborArray getNeighborsFromRowView(CuVSMatrix graphMatrix, int node, int totalSize) {
    try {
      // Add debugging information about the graph matrix
      info(
          "Graph matrix info - size: "
              + graphMatrix.size()
              + ", columns: "
              + graphMatrix.columns());
      info("Requesting neighbors for node: " + node + " out of total size: " + totalSize);

      // Get the row view for the specific node using the correct CuVS API
      RowView row = graphMatrix.getRow(node);
      long rowSize = row.size();

      info("RowView obtained successfully, size: " + rowSize);

      // Create neighbor array with actual size
      NeighborArray neighbors = new NeighborArray((int) rowSize, true);

      // Process each neighbor in the row using robust data type handling
      for (int i = 0; i < rowSize; i++) {
        int neighbor = getNeighborFromRowViewRobust(row, i);
        if (neighbor >= 0 && neighbor < totalSize) {
          // Add neighbor with a score (1.0f - i * 0.001f for ordering)
          neighbors.addInOrder(neighbor, 1.0f - (i * 0.001f));
        }
      }

      info("Successfully processed " + neighbors.size() + " neighbors for node " + node);
      return neighbors;
    } catch (Exception e) {
      info("Error getting neighbors from RowView for node " + node + ": " + e.getMessage());
      e.printStackTrace();
      // Return empty neighbor array as fallback
      return new NeighborArray(0, true);
    }
  }

  /**
   * Extract a neighbor value from RowView at the specified index using robust data type handling.
   * This method tries FLOAT first (most common), then BYTE (for quantized indices).
   */
  private int getNeighborFromRowViewRobust(RowView row, int index) {
    // Try FLOAT first (most common case for graph matrices)
    try {
      return (int) row.getAsFloat(index);
    } catch (AssertionError e) {
      // Try BYTE next (for quantized indices)
      try {
        return (int) row.getAsByte(index);
      } catch (AssertionError e2) {
        // Both data types failed, return fallback value
        info("Both FLOAT and BYTE data types failed for index " + index + ", using fallback value");
        return -1;
      }
    } catch (Exception e) {
      info("Error accessing neighbor at index " + index + ": " + e.getMessage());
      return -1;
    }
  }

  /**
   * Create fallback neighbors when the graph matrix is not available or accessible.
   * This provides a simple circular graph structure as a fallback.
   */
  private NeighborArray createFallbackNeighbors(int node, int totalSize) {
    int degree = Math.min(10, totalSize - 1);
    NeighborArray neighbors = new NeighborArray(degree, true);
    for (int j = 0; j < degree; j++) {
      int neighbor = (node + j + 1) % totalSize;
      neighbors.addInOrder(neighbor, 1.0f - (j * 0.001f));
    }
    return neighbors;
  }

  /**
   * Create a graph object for metadata writing that matches the structure expected by writeMeta.
   * This creates a lightweight graph representation for metadata without copying the entire adjacency list.
   */
  private HnswGraph createMetadataGraph(int size, int[][] graphLevelNodeOffsets) {
    // Create a lightweight graph that doesn't store the full adjacency list
    // This is used only for metadata writing, not for actual graph traversal
    return new LightweightHnswGraph(size, graphLevelNodeOffsets);
  }

  private void writeNativeLuceneCagraIndex(OutputStream os, CuVSMatrix dataset, FieldInfo fieldInfo)
      throws Throwable {
    if (dataset.size() < 2) {
      throw new IllegalArgumentException(dataset.size() + " vectors, less than min [2] required");
    }
    CagraIndexParams params = cagraIndexParams((int) dataset.size());
    long startTime = System.nanoTime();
    CagraIndex index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(params).build();

    // Get the graph matrix from CAGRA index for streaming processing
    CuVSMatrix graphMatrix = null;
    try {
      graphMatrix = index.getGraph();
      info("Successfully got graph matrix from CAGRA index for streaming processing");
    } catch (Exception e) {
      info("getGraph() method failed: " + e.getMessage());
      // Continue with null graphMatrix - will use fallback neighbors
    }

    int size = (int) dataset.size();
    int dimensions = fieldInfo.getVectorDimension();

    info("Processing graph with " + size + " nodes and " + dimensions + " dimensions");

    // Remember the vector index offset before writing
    long vectorIndexOffset = hnswVectorIndex.getFilePointer();

    // Write the graph to the vector index using streaming approach
    int[][] graphLevelNodeOffsets =
        writeGraphStreaming(size, dimensions, graphMatrix, hnswVectorIndex);

    // Calculate the length of written data
    long vectorIndexLength = hnswVectorIndex.getFilePointer() - vectorIndexOffset;

    // Write metadata using lightweight graph
    writeMeta(
        hnswVectorIndex,
        hnswMeta,
        fieldInfo,
        vectorIndexOffset,
        vectorIndexLength,
        size,
        createMetadataGraph(size, graphLevelNodeOffsets),
        graphLevelNodeOffsets);

    long elapsedMillis = nanosToMillis(System.nanoTime() - startTime);
    info(
        "Native HNSW graph created in "
            + elapsedMillis
            + "ms, with "
            + dataset.size()
            + " vectors");

    // Serialize the CAGRA index to the cuvsIndex output stream
    Path tmpFile = Files.createTempFile(resources.tempDirectory(), "tmpindex", "cag");
    index.serialize(os, tmpFile);
    index.destroyIndex();
  }

  private void writeMeta(
      IndexOutput vectorIndex,
      IndexOutput meta,
      FieldInfo field,
      long vectorIndexOffset,
      long vectorIndexLength,
      int count,
      HnswGraph graph,
      int[][] graphLevelNodeOffsets)
      throws IOException {

    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(distFuncToOrd(field.getVectorSimilarityFunction()));
    meta.writeVLong(vectorIndexOffset);
    meta.writeVLong(vectorIndexLength);
    meta.writeVInt(field.getVectorDimension());
    meta.writeInt(count);
    int actualM = graph != null ? graph.maxConn() : 0;
    meta.writeVInt(actualM);
    // write graph nodes on each level
    if (graph == null) {
      meta.writeVInt(0);
    } else {
      meta.writeVInt(graph.numLevels());
      long valueCount = 0;
      for (int level = 0; level < graph.numLevels(); level++) {
        NodesIterator nodesOnLevel = graph.getNodesOnLevel(level);
        valueCount += nodesOnLevel.size();
        if (level > 0) {
          int[] nol = new int[nodesOnLevel.size()];
          int numberConsumed = nodesOnLevel.consume(nol);
          Arrays.sort(nol);
          assert numberConsumed == nodesOnLevel.size();
          meta.writeVInt(nol.length); // number of nodes on a level
          for (int i = nodesOnLevel.size() - 1; i > 0; --i) {
            nol[i] -= nol[i - 1];
          }
          for (int n : nol) {
            assert n >= 0 : "delta encoding for nodes failed; expected nodes to be sorted";
            meta.writeVInt(n);
          }
        } else {
          assert nodesOnLevel.size() == count : "Level 0 expects to have all nodes";
        }
      }
      long start = vectorIndex.getFilePointer();
      meta.writeLong(start);
      meta.writeVInt(16); // DIRECT_MONOTONIC_BLOCK_SHIFT);
      final DirectMonotonicWriter memoryOffsetsWriter =
          DirectMonotonicWriter.getInstance(
              meta, vectorIndex, valueCount, 16); // DIRECT_MONOTONIC_BLOCK_SHIFT);
      long cumulativeOffsetSum = 0;
      int totalOffsetsWritten = 0;
      for (int[] levelOffsets : graphLevelNodeOffsets) {
        System.out.println(
            "=== writeMeta: Writing offsets for level with "
                + levelOffsets.length
                + " entries ===");
        for (int v : levelOffsets) {
          memoryOffsetsWriter.add(cumulativeOffsetSum);
          cumulativeOffsetSum += v;
          totalOffsetsWritten++;
        }
      }
      System.out.println(
          "=== writeMeta: Total offsets written: "
              + totalOffsetsWritten
              + ", expected: "
              + valueCount
              + " ===");
      memoryOffsetsWriter.finish();
      meta.writeLong(vectorIndex.getFilePointer() - start);
    }
  }

  private int[][] writeGraph(MyOnHeapHnswGraph graph, IndexOutput vectorIndex) throws IOException {
    if (graph == null) return new int[0][0];
    // write vectors' neighbours on each level into the vectorIndex file
    int countOnLevel0 = graph.size();
    int[][] offsets = new int[graph.numLevels()][];
    int[] scratch = new int[graph.maxConn() * 2];
    for (int level = 0; level < graph.numLevels(); level++) {
      int[] sortedNodes = NodesIterator.getSortedNodes(graph.getNodesOnLevel(level));
      offsets[level] = new int[sortedNodes.length];
      int nodeOffsetId = 0;
      System.out.println(
          "=== writeGraph: Level "
              + level
              + " has "
              + sortedNodes.length
              + " nodes, expected "
              + (level == 0 ? countOnLevel0 : "unknown")
              + " ===");
      for (int node : sortedNodes) {
        NeighborArray neighbors = graph.getNeighbors(level, node);
        int size = neighbors.size();
        // Write size in VInt as the neighbors list is typically small
        long offsetStart = vectorIndex.getFilePointer();
        int[] nnodes = neighbors.nodes();
        Arrays.sort(nnodes, 0, size);
        // Now that we have sorted, do delta encoding to minimize the required bits to store the
        // information
        int actualSize = 0;
        if (size > 0) {
          scratch[0] = nnodes[0];
          actualSize = 1;
        }
        for (int i = 1; i < size; i++) {
          assert nnodes[i] < countOnLevel0 : "node too large: " + nnodes[i] + ">=" + countOnLevel0;
          if (nnodes[i - 1] == nnodes[i]) {
            continue;
          }
          scratch[actualSize++] = nnodes[i] - nnodes[i - 1];
        }
        // Write the size after duplicates are removed
        vectorIndex.writeVInt(actualSize);
        for (int i = 0; i < actualSize; i++) {
          vectorIndex.writeVInt(scratch[i]);
        }
        offsets[level][nodeOffsetId++] =
            Math.toIntExact(vectorIndex.getFilePointer() - offsetStart);
      }
    }
    return offsets;
  }

  private void writeCagraIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    if (dataset.size() < 2) {
      throw new IllegalArgumentException(dataset.size() + " vectors, less than min [2] required");
    }
    CagraIndexParams params = cagraIndexParams((int) dataset.size());
    long startTime = System.nanoTime();
    CagraIndex index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(params).build();
    long elapsedMillis = nanosToMillis(System.nanoTime() - startTime);
    info("Cagra index created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");
    Path tmpFile = Files.createTempFile(resources.tempDirectory(), "tmpindex", "cag");
    index.serialize(os, tmpFile);
    index.destroyIndex();
  }

  private void writeBruteForceIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    BruteForceIndexParams params =
        new BruteForceIndexParams.Builder()
            .withNumWriterThreads(32) // TODO: Make this configurable later.
            .build();
    long startTime = System.nanoTime();
    var index =
        BruteForceIndex.newBuilder(resources).withIndexParams(params).withDataset(dataset).build();
    long elapsedMillis = nanosToMillis(System.nanoTime() - startTime);
    info("bf index created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");
    index.serialize(os);
    index.destroyIndex();
  }

  private void writeHNSWIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    if (dataset.size() < 2) {
      throw new IllegalArgumentException(dataset.size() + " vectors, less than min [2] required");
    }
    CagraIndexParams indexParams = cagraIndexParams((int) dataset.size());
    long startTime = System.nanoTime();
    var index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(indexParams).build();
    long elapsedMillis = nanosToMillis(System.nanoTime() - startTime);
    info("HNSW index created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");
    Path tmpFile = Files.createTempFile("tmpindex", "hnsw");
    index.serializeToHNSW(os, tmpFile);
    index.destroyIndex();
  }

  @Override
  public void flush(int maxDoc, DocMap sortMap) throws IOException {
    flatVectorsWriter.flush(maxDoc, sortMap);
    for (var field : fields) {
      if (sortMap == null) {
        writeField(field);
      } else {
        writeSortingField(field, sortMap);
      }
    }
  }

  private void writeField(CuVSFieldWriter fieldData) throws IOException {
    // Use memory-efficient matrix creation from vector list
    List<float[]> vectors = fieldData.getVectors();
    CuVSMatrix dataset = createMatrixFromVectorList(vectors);
    if (dataset == null) {
      writeEmpty(fieldData.fieldInfo());
      return;
    }
    writeFieldInternal(fieldData.fieldInfo(), dataset);
  }

  private void writeSortingField(CuVSFieldWriter fieldData, Sorter.DocMap sortMap)
      throws IOException {
    DocsWithFieldSet oldDocsWithFieldSet = fieldData.getDocsWithFieldSet();
    final int[] new2OldOrd = new int[oldDocsWithFieldSet.cardinality()]; // new ord to old ord

    mapOldOrdToNewOrd(oldDocsWithFieldSet, sortMap, null, new2OldOrd, null);

    List<float[]> oldVectors = fieldData.getVectors();
    if (oldVectors.isEmpty()) {
      writeEmpty(fieldData.fieldInfo());
      return;
    }

    int vectorCount = oldVectors.size();

    // Create sorted array directly with pre-allocated size
    float[][] sortedVectors = new float[vectorCount][];
    for (int i = 0; i < vectorCount; i++) {
      sortedVectors[i] = oldVectors.get(new2OldOrd[i]);
    }

    CuVSMatrix dataset = CuVSMatrix.ofArray(sortedVectors);
    writeFieldInternal(fieldData.fieldInfo(), dataset);
  }

  private void writeEmpty(FieldInfo fieldInfo) throws IOException {
    // Only write meta for non-HNSW_LUCENE modes
    if (indexType != IndexType.HNSW_LUCENE) {
      writeMeta(fieldInfo, 0, 0L, 0L, 0L, 0L, 0L, 0L);
    }
  }

  private void writeMeta(
      FieldInfo field,
      int count,
      long cagraIndexOffset,
      long cagraIndexLength,
      long bruteForceIndexOffset,
      long bruteForceIndexLength,
      long hnswIndexOffset,
      long hnswIndexLength)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(distFuncToOrd(field.getVectorSimilarityFunction()));
    meta.writeInt(field.getVectorDimension());
    meta.writeInt(count);
    meta.writeVLong(cagraIndexOffset);
    meta.writeVLong(cagraIndexLength);
    meta.writeVLong(bruteForceIndexOffset);
    meta.writeVLong(bruteForceIndexLength);
    meta.writeVLong(hnswIndexOffset);
    meta.writeVLong(hnswIndexLength);
  }

  static int distFuncToOrd(VectorSimilarityFunction func) {
    for (int i = 0; i < SIMILARITY_FUNCTIONS.size(); i++) {
      if (SIMILARITY_FUNCTIONS.get(i).equals(func)) {
        return (byte) i;
      }
    }
    throw new IllegalArgumentException("invalid distance function: " + func);
  }

  // We currently ignore this, until cuVS supports tiered indices
  private static final String CANNOT_GENERATE_CAGRA =
      """
      Could not generate an intermediate CAGRA graph because the initial \
      kNN graph contains too many invalid or duplicated neighbor nodes. \
      This error can occur, for example, if too many overflows occur \
      during the norm computation between the dataset vectors\
      """;

  static void handleThrowableWithIgnore(Throwable t, String msg) throws IOException {
    String message = t.getMessage();
    if (message != null && message.contains(msg)) {
      return;
    }
    handleThrowable(t);
  }

  /** Creates CuVSMatrix directly from FloatVectorValues without intermediate List. */
  private CuVSMatrix getVectorDataMatrix(
      FloatVectorValues floatVectorValues, int vectorCount, int dimensions) throws IOException {
    if (vectorCount == 0) {
      return null;
    }

    // Pre-allocate exact size array instead of using ArrayList
    float[][] vectorArray = new float[vectorCount][];
    // Use ordinal-based mapping to ensure vectors are stored in the correct order
    // The ordinal from iter.index() should match the position that vectorValue(index) expects
    KnnVectorValues.DocIndexIterator iter = floatVectorValues.iterator();
    int rowIndex = 0;
    for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
      float[] vector = floatVectorValues.vectorValue(iter.index());
      if (vector != null) {
        int ordinal = iter.index();
        // Store vector at the ordinal position to match vectorValue(index) expectations
        vectorArray[ordinal] = vector;
        rowIndex++;
      }
    }

    // Resize if needed (though vectorCount should be accurate)
    if (rowIndex < vectorCount) {
      float[][] resized = new float[rowIndex][];
      System.arraycopy(vectorArray, 0, resized, 0, rowIndex);
      vectorArray = resized;
    }

    // Handle empty case that CuVSMatrix.ofArray doesn't support
    if (vectorArray.length == 0) {
      return null;
    }

    return CuVSMatrix.ofArray(vectorArray);
  }

  /** Creates CuVSMatrix from List<float[]> efficiently with pre-allocated array. */
  private CuVSMatrix createMatrixFromVectorList(List<float[]> vectors) {
    if (vectors.isEmpty()) {
      return null;
    }

    // Convert to array more efficiently
    return CuVSMatrix.ofArray(vectors.toArray(new float[vectors.size()][]));
  }

  /** Creates CuVSMatrix from merged vectors, using dense array without gaps. */
  private CuVSMatrix createMatrixFromMergedVectors(
      FloatVectorValues mergedVectorValues, int vectorCount, int dimensions) throws IOException {
    if (vectorCount == 0) {
      return null;
    }

    // Use a dense array approach to avoid null elements that CuVSMatrix.ofArray can't handle
    java.util.List<float[]> vectorList = new java.util.ArrayList<>();

    KnnVectorValues.DocIndexIterator iter = mergedVectorValues.iterator();
    for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
      int ordinal = iter.index();
      float[] vector = mergedVectorValues.vectorValue(ordinal);
      if (vector != null) {
        vectorList.add(vector.clone()); // Clone to ensure we have distinct arrays
      }
    }

    if (vectorList.isEmpty()) {
      return null;
    }

    return CuVSMatrix.ofArray(vectorList.toArray(new float[vectorList.size()][]));
  }

  /** Legacy method that copies vector values into a 2D array. */
  private static float[][] getVectorDataArray(FloatVectorValues floatVectorValues, int expectedSize)
      throws IOException {
    java.util.List<float[]> vectorList = new java.util.ArrayList<>();
    KnnVectorValues.DocIndexIterator iter = floatVectorValues.iterator();
    for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
      float[] vector = floatVectorValues.vectorValue(iter.index());
      if (vector != null) {
        vectorList.add(vector);
      }
    }
    return vectorList.toArray(new float[vectorList.size()][]);
  }

  /**
   * Merges CAGRA indexes using the native CuVS merge API instead of rebuilding from vectors.
   * Falls back to vector-based merge if native merge fails.
   */
  private void mergeCagraIndexes(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try {
      // Collect existing CAGRA indexes from merge segments
      List<CagraIndex> cagraIndexes = new ArrayList<>();

      // Get total vector count from the merged vector values to be accurate
      final FloatVectorValues mergedVectorValues =
          switch (fieldInfo.getVectorEncoding()) {
            case BYTE -> throw new AssertionError("bytes not supported");
            case FLOAT32 ->
                KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
          };
      int totalVectorCount = mergedVectorValues.size();

      for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
        var knnReader = mergeState.knnVectorsReaders[i];
        if (knnReader instanceof CuVSVectorsReader cuvsReader) {
          // Access the CAGRA index for this field from the reader
          CagraIndex cagraIndex = getCagraIndexFromReader(cuvsReader, fieldInfo.name);
          if (cagraIndex != null) {
            cagraIndexes.add(cagraIndex);
          }
        }
      }

      if (cagraIndexes.size() > 1) {
        // Use native CAGRA merge API when we have multiple valid indexes
        CagraIndex mergedIndex = CagraIndex.merge(cagraIndexes.toArray(new CagraIndex[0]));
        writeMergedCagraIndex(fieldInfo, mergedIndex, totalVectorCount);
        info(
            "Successfully merged " + cagraIndexes.size() + " CAGRA indexes using native merge API");
      } else {
        // Fall back to vector-based approach for single/no indexes
        fallbackToVectorMerge(fieldInfo, mergeState);
      }
    } catch (Throwable t) {
      info("Native CAGRA merge failed, falling back to vector-based merge: " + t.getMessage());
      fallbackToVectorMerge(fieldInfo, mergeState);
    }
  }

  /**
   * Fallback method that rebuilds indexes from merged vectors.
   * Used when native CAGRA merge is not possible or fails.
   */
  private void fallbackToVectorMerge(FieldInfo fieldInfo, MergeState mergeState)
      throws IOException {
    try {
      // Get the merged vectors from Lucene's merge process
      final FloatVectorValues mergedVectorValues =
          switch (fieldInfo.getVectorEncoding()) {
            case BYTE -> throw new AssertionError("bytes not supported");
            case FLOAT32 ->
                KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
          };

      int vectorCount = mergedVectorValues.size();
      int dimensions = fieldInfo.getVectorDimension();

      // Create CAGRA index from the properly merged vectors
      CuVSMatrix dataset =
          createMatrixFromMergedVectors(mergedVectorValues, vectorCount, dimensions);

      if (dataset == null) {
        writeEmpty(fieldInfo);
        return;
      }
      writeFieldInternal(fieldInfo, dataset);
      info("Completed fallback vector-based merge for field: " + fieldInfo.name);
    } catch (Throwable t) {
      handleThrowable(t);
    }
  }

  /**
   * Extracts the CAGRA index for a specific field from a CuVSVectorsReader.
   */
  private CagraIndex getCagraIndexFromReader(CuVSVectorsReader reader, String fieldName) {
    try {
      // Use reflection to access the private cuvsIndices field
      var cuvsIndicesField = reader.getClass().getDeclaredField("cuvsIndices");
      cuvsIndicesField.setAccessible(true);
      @SuppressWarnings("unchecked")
      var cuvsIndices = (IntObjectHashMap<CuVSIndex>) cuvsIndicesField.get(reader);

      // Find the field info for this field name
      var fieldInfosField = reader.getClass().getDeclaredField("fieldInfos");
      fieldInfosField.setAccessible(true);
      var fieldInfos = (FieldInfos) fieldInfosField.get(reader);

      FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldName);
      if (fieldInfo != null) {
        CuVSIndex cuvsIndex = cuvsIndices.get(fieldInfo.number);
        if (cuvsIndex != null) {
          return cuvsIndex.getCagraIndex();
        }
      }
    } catch (Exception e) {
      info("Failed to extract CAGRA index for field " + fieldName + ": " + e.getMessage());
    }
    return null;
  }

  /**
   * Writes a pre-built merged CAGRA index to the output.
   */
  private void writeMergedCagraIndex(FieldInfo fieldInfo, CagraIndex mergedIndex, int vectorCount)
      throws IOException {
    try {
      long cagraIndexOffset = cuvsIndex.getFilePointer();
      var cagraIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);

      // Serialize the merged index
      Path tmpFile = Files.createTempFile(resources.tempDirectory(), "mergedindex", "cag");
      mergedIndex.serialize(cagraIndexOutputStream, tmpFile);
      long cagraIndexLength = cuvsIndex.getFilePointer() - cagraIndexOffset;

      // Write metadata (assuming no brute force or HNSW indexes for merged result)
      // Only write meta for non-HNSW_LUCENE modes
      if (indexType != IndexType.HNSW_LUCENE) {
        writeMeta(fieldInfo, vectorCount, cagraIndexOffset, cagraIndexLength, 0L, 0L, 0L, 0L);
      }

      // Clean up the merged index
      mergedIndex.destroyIndex();
    } catch (Throwable t) {
      handleThrowable(t);
    }
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    flatVectorsWriter.mergeOneField(fieldInfo, mergeState);
    try {
      // Use CuVS CAGRA merge API
      if (indexType.cagra()) {
        mergeCagraIndexes(fieldInfo, mergeState);
      } else {
        // For non-CAGRA index types, use vector-based approach
        final FloatVectorValues mergedVectorValues =
            switch (fieldInfo.getVectorEncoding()) {
              case BYTE -> throw new AssertionError("bytes not supported");
              case FLOAT32 ->
                  KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
            };

        int vectorCount = mergedVectorValues.size();
        int dimensions = fieldInfo.getVectorDimension();

        CuVSMatrix dataset = getVectorDataMatrix(mergedVectorValues, vectorCount, dimensions);
        if (dataset == null) {
          writeEmpty(fieldInfo);
          return;
        }
        writeFieldInternal(fieldInfo, dataset);
      }
    } catch (Throwable t) {
      handleThrowable(t);
    }
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    flatVectorsWriter.finish();

    if (meta != null) {
      // write end of fields marker
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (cuvsIndex != null) {
      CodecUtil.writeFooter(cuvsIndex);
    }

    // HNSW
    {
      if (hnswMeta != null) {
        // write end of fields marker
        hnswMeta.writeInt(-1);
        CodecUtil.writeFooter(hnswMeta);
      }
      if (hnswVectorIndex != null) {
        CodecUtil.writeFooter(hnswVectorIndex);
      }
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, cuvsIndex, hnswMeta, hnswVectorIndex, flatVectorsWriter);
  }

  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    for (var field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }
}
