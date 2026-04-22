/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.lucene.search.TopKnnCollector;

/**
 * KnnCollector for cuVS used for search on the GPU.
 *
 * @since 25.10
 */
class GPUPerLeafBatchCuVSKnnCollector extends TopKnnCollector {

  private int iTopK;
  private int searchWidth;
  private int numQueries;
  private int queryVectorDimension;
  private Map<Integer, TopKnnCollector> collectors;

  /**
   * Initializes {@link GPUPerLeafBatchCuVSKnnCollector}
   *
   * @param topK the topK value
   * @param numQueries number of query vectors in the batch
   * @param iTopK the iTopK value
   * @param searchWidth the search width
   */
  public GPUPerLeafBatchCuVSKnnCollector(
      int topK,
      int numQueries,
      int queryVectorDimension,
      int visitLimit,
      int iTopK,
      int searchWidth) {
    super(topK, visitLimit);
    this.iTopK = iTopK > topK ? iTopK : topK;
    this.searchWidth = searchWidth;
    this.numQueries = numQueries;
    this.queryVectorDimension = queryVectorDimension;
    this.collectors = new ConcurrentHashMap<Integer, TopKnnCollector>();
    for (int i = 0; i < numQueries; i++) {
      collectors.put(i, new TopKnnCollector(topK, visitLimit));
    }
  }

  public int getiTopK() {
    return iTopK;
  }

  public int getSearchWidth() {
    return searchWidth;
  }

  public int getNumQueries() {
    return numQueries;
  }

  public int getQueryVectorDimension() {
    return queryVectorDimension;
  }

  public Map<Integer, TopKnnCollector> getCollectors() {
    return collectors;
  }
}
