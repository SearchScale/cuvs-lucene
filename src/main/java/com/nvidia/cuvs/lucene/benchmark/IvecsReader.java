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
package com.nvidia.cuvs.lucene.benchmark;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Reader for IVECS format files (commonly used for ground truth in SIFT dataset)
 * IVECS format: [dimension:int][vector:int[dimension]]...
 */
public class IvecsReader {

  public static List<int[]> readIvecs(String filename, int maxVectors) throws IOException {
    List<int[]> vectors = new ArrayList<>();

    try (RandomAccessFile file = new RandomAccessFile(filename, "r")) {
      long fileLength = file.length();
      long position = 0;
      int count = 0;

      while (position < fileLength && count < maxVectors) {
        // Read dimension (4 bytes, little-endian)
        file.seek(position);
        byte[] dimBytes = new byte[4];
        file.readFully(dimBytes);
        int dimension = ByteBuffer.wrap(dimBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();

        // Read vector data
        byte[] vectorBytes = new byte[dimension * 4]; // 4 bytes per int
        file.readFully(vectorBytes);

        // Convert bytes to ints
        ByteBuffer buffer = ByteBuffer.wrap(vectorBytes).order(ByteOrder.LITTLE_ENDIAN);
        int[] vector = new int[dimension];
        for (int i = 0; i < dimension; i++) {
          vector[i] = buffer.getInt();
        }

        vectors.add(vector);
        count++;
        position += 4 + (dimension * 4); // 4 bytes for dimension + vector data
      }
    }

    return vectors;
  }
}
