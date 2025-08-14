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

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.NeighborArray;

/**
 * Lightweight HNSW graph implementation that doesn't store the full adjacency list.
 * This is used only for metadata writing, not for actual graph traversal.
 */
public class LightweightHnswGraph extends HnswGraph {

  private final int size;
  private final int[][] graphLevelNodeOffsets;

  public LightweightHnswGraph(int size, int[][] graphLevelNodeOffsets) {
    this.size = size;
    this.graphLevelNodeOffsets = graphLevelNodeOffsets;
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public int numLevels() {
    // CAGRA graph has only one level
    return 1;
  }

  @Override
  public NodesIterator getNodesOnLevel(int level) {
    if (level == 0) {
      return new ArrayNodesIterator(size);
    } else {
      // No nodes on levels > 0 for CAGRA
      return new ArrayNodesIterator(0);
    }
  }

  public NeighborArray getNeighbors(int level, int node) {
    // This method is not used for metadata writing, so we can return empty neighbors
    // The actual neighbor data is written directly to the index file in writeGraphStreaming
    return new NeighborArray(0, true);
  }

  @Override
  public void seek(int level, int target) {
    // Not used for metadata writing
  }

  @Override
  public int nextNeighbor() {
    // Not used for metadata writing
    return NO_MORE_DOCS;
  }

  @Override
  public int entryNode() {
    // For a single-level graph, entry node is typically 0
    return 0;
  }

  @Override
  public int maxConn() {
    // Return a reasonable default for metadata
    return 128;
  }

  @Override
  public int neighborCount() {
    // Not used for metadata writing
    return 0;
  }

  // Simple implementation of NodesIterator for level 0
  private static class ArrayNodesIterator extends NodesIterator {
    private int current = -1;

    ArrayNodesIterator(int size) {
      super(size);
    }

    @Override
    public boolean hasNext() {
      return current + 1 < size;
    }

    @Override
    public int nextInt() {
      return ++current;
    }

    @Override
    public int consume(int[] dest) {
      int numToCopy = Math.min(dest.length, size - (current + 1));
      for (int i = 0; i < numToCopy; i++) {
        dest[i] = ++current;
      }
      return numToCopy;
    }
  }
}
