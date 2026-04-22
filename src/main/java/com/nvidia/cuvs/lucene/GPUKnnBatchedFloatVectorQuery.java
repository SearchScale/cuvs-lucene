/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;

/**
 * Extends upon KnnFloatVectorQuery for GPU-only search.
 *
 * @since 25.10
 */
public class GPUKnnBatchedFloatVectorQuery extends KnnFloatVectorQuery {

  private final int iTopK;
  private final int searchWidth;
  private final int numQueries;
  private final int queryVectorDimension;

  /**
   * Initializes {@link GPUKnnBatchedFloatVectorQuery}
   *
   * @param field the vector field name
   * @param queries a batch of float query vectors
   * @param k the topK value
   * @param filter instance of the Query
   * @param iTopK the iTopK value
   * @param searchWidth the search width
   */
  public GPUKnnBatchedFloatVectorQuery(
      String field, float[][] queries, int k, Query filter, int iTopK, int searchWidth) {
    super(field, flatten(queries), k, filter);
    this.iTopK = iTopK;
    this.searchWidth = searchWidth;
    this.numQueries = queries.length;
    this.queryVectorDimension = queries.length != 0 ? queries[0].length : 0;
  }

  @Override
  protected TopDocs approximateSearch(
      LeafReaderContext context,
      Bits acceptDocs,
      int visitedLimit,
      KnnCollectorManager knnCollectorManager)
      throws IOException {

    GPUPerLeafBatchCuVSKnnCollector results =
        new GPUPerLeafBatchCuVSKnnCollector(
            k, numQueries, queryVectorDimension, visitedLimit, iTopK, searchWidth);

    LeafReader reader = context.reader();
    reader.searchNearestVectors(field, this.getTargetCopy(), results, acceptDocs);
    return results.getCollectors().get(0).topDocs();
  }

  private static float[] flatten(float[][] queries) {
    int n = queries.length;
    int d = queries[0].length;

    float[] packedQueries = new float[n * d];
    int offset = 0;
    for (float[] query : queries) {
      System.arraycopy(query, 0, packedQueries, offset, d);
      offset += d;
    }
    return packedQueries;
  }
}
