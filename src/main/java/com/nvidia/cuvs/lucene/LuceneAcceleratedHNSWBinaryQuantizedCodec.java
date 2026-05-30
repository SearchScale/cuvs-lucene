/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.LibraryException;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CuVS based codec for GPU based vector search
 *
 * @since 26.02
 */
public class LuceneAcceleratedHNSWBinaryQuantizedCodec extends FilterCodec {

  private static final Logger LOG =
      LoggerFactory.getLogger(LuceneAcceleratedHNSWBinaryQuantizedCodec.class);
  private static final String NAME = "Lucene101AcceleratedHNSWBinaryQuantizedCodec";

  private KnnVectorsFormat format;

  public LuceneAcceleratedHNSWBinaryQuantizedCodec() throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
  }

  public LuceneAcceleratedHNSWBinaryQuantizedCodec(String name, Codec delegate) {
    super(name, delegate);
    initializeFormat(new AcceleratedHNSWParams.Builder().build());
  }

  public LuceneAcceleratedHNSWBinaryQuantizedCodec(AcceleratedHNSWParams acceleratedHNSWParams)
      throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
    initializeFormat(acceleratedHNSWParams);
  }

  private void initializeFormat(AcceleratedHNSWParams acceleratedHNSWParams) {
    try {
      format = new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(acceleratedHNSWParams);
      setKnnFormat(format);
    } catch (LibraryException ex) {
      LOG.error("Couldn't load native library, possible classloader issue. {}", ex.getMessage());
    }
  }

  @Override
  public KnnVectorsFormat knnVectorsFormat() {
    return format;
  }

  public void setKnnFormat(KnnVectorsFormat format) {
    this.format = format;
  }
}
