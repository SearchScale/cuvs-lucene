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

import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;

/** CuVS based codec for GPU based vector search */
public class CuVSCPUSearchCodec extends FilterCodec {

  public CuVSCPUSearchCodec() {
    this("CuVSCPUSearchCodec", new Lucene101Codec());
  }

  public CuVSCPUSearchCodec(String name, Codec delegate) {
    super(name, delegate);
    initializeFormat();
  }

  private void initializeFormat() {
    KnnVectorsFormat format;
    try {
      // Use the dedicated CPU search format
      format = new CuVSCPUSearchVectorsFormat();
      setKnnFormat(format);
      Logger log = Logger.getLogger(CuVSCPUSearchCodec.class.getName());
      log.info("CuVSCPUSearchVectorsFormat initialized successfully");
    } catch (Exception ex) {
      Logger log = Logger.getLogger(CuVSCPUSearchCodec.class.getName());
      log.warning(
          "CuVS CPU search format initialization issue: "
              + ex.getMessage()
              + ". Falling back to standard Lucene HNSW...");

      try {
        // Fall back to standard Lucene HNSW format with optimized parameters for CPU search
        format = new Lucene99HnswVectorsFormat(64, 200);
        setKnnFormat(format);
        log.info("Fallback to Lucene99HnswVectorsFormat successful");
      } catch (Exception fallbackEx) {
        log.severe("Failed to create fallback HNSW format: " + fallbackEx.getMessage());
        setKnnFormat(null);
      }
    }
  }

  KnnVectorsFormat knnFormat = null;

  @Override
  public KnnVectorsFormat knnVectorsFormat() {
    return knnFormat;
  }

  public void setKnnFormat(KnnVectorsFormat format) {
    this.knnFormat = format;
  }
}
