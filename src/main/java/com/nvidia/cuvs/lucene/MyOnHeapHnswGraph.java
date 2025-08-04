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

public class MyOnHeapHnswGraph extends HnswGraph {

  private final int size;
  private final int dimensions;
  private final int[][] adjacencyList;
  private final NeighborArray[] neighbors;

  public MyOnHeapHnswGraph(int size, int dimensions, int[][] adjacencyList) {
    this.size = size;
    this.dimensions = dimensions;
    this.adjacencyList = adjacencyList;
    this.neighbors = new NeighborArray[size];

    // Convert adjacency list to NeighborArray format
    for (int i = 0; i < size; i++) {
      if (adjacencyList[i] != null && adjacencyList[i].length > 0) {
        // Create NeighborArray with descending order (true)
        neighbors[i] = new NeighborArray(adjacencyList[i].length, true);
        // Add neighbors - assuming they are already sorted by distance
        for (int j = 0; j < adjacencyList[i].length; j++) {
          // Using placeholder scores for now - in real implementation these would be actual
          // distances
          neighbors[i].addInOrder(adjacencyList[i][j], 1.0f - (j * 0.001f));
        }
      } else {
        neighbors[i] = new NeighborArray(0, true);
      }
    }
  }

  public int size() {
    return size;
  }

  public int numLevels() {
    // CAGRA graph has only one level
    return 1;
  }

  public NodesIterator getNodesOnLevel(int level) {
    if (level == 0) {
      return new ArrayNodesIterator(size);
    } else {
      // No nodes on levels > 0 for CAGRA
      return new ArrayNodesIterator(0);
    }
  }

  public NeighborArray getNeighbors(int level, int node) {
    if (level == 0 && node < size) {
      return neighbors[node];
    }
    return null;
  }

  // Implementation of abstract methods from HnswGraph
  private int currentNode = -1;
  private int currentLevel = -1;
  private int neighborIndex = -1;

  @Override
  public void seek(int level, int target) {
    currentLevel = level;
    currentNode = target;
    neighborIndex = -1;
  }

  @Override
  public int nextNeighbor() {
    if (currentLevel == 0
        && currentNode >= 0
        && currentNode < size
        && neighbors[currentNode] != null) {
      neighborIndex++;
      if (neighborIndex < neighbors[currentNode].size()) {
        int neighborNode = neighbors[currentNode].nodes()[neighborIndex];
        // Add bounds check to prevent index out of bounds
        if (neighborNode >= 0 && neighborNode < size) {
          return neighborNode;
        } else {
          // Skip invalid neighbor and try next
          return nextNeighbor();
        }
      }
    }
    return NO_MORE_DOCS;
  }

  @Override
  public int entryNode() {
    // For a single-level graph, entry node is typically 0
    return 0;
  }

  @Override
  public int maxConn() {
    // Return the maximum degree across all nodes
    int max = 0;
    for (NeighborArray neighbor : neighbors) {
      if (neighbor != null) {
        max = Math.max(max, neighbor.size());
      }
    }
    return max;
  }

  @Override
  public int neighborCount() {
    if (currentNode >= 0 && currentNode < size && neighbors[currentNode] != null) {
      return neighbors[currentNode].size();
    }
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
