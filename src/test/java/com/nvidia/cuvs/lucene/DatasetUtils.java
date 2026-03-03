/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Reader utility for reading the fbin, ibin files.
 */
public class DatasetUtils {

  protected static Logger log = Logger.getLogger(DatasetUtils.class.getName());

  public static void readDataFile(String filePath, int numRows, List<int[]> its, List<float[]> fts)
      throws Exception {
    log.log(Level.INFO, "Reading " + numRows + " items from file: " + filePath);
    try (InputStream is = new FileInputStream(filePath)) {
      byte[] numVecBytes = is.readNBytes(4);
      ByteBuffer numVecBuffer = ByteBuffer.wrap(numVecBytes).order(ByteOrder.LITTLE_ENDIAN);
      int items = numVecBuffer.getInt();

      byte[] dimBytes = is.readNBytes(4);
      ByteBuffer dimBuffer = ByteBuffer.wrap(dimBytes).order(ByteOrder.LITTLE_ENDIAN);
      int dimension = dimBuffer.getInt();

      log.log(Level.INFO, "Available rows: " + items + ", dimension: " + dimension);
      int count = 0;
      float[] frow = null;
      int[] irow = null;
      if (filePath.endsWith("fbin")) {
        frow = new float[dimension];
      } else if (filePath.endsWith("ibin")) {
        irow = new int[dimension];
      }

      while (is.available() != 0 && (numRows == -1 || count < numRows)) {
        // Read dimension * 4 bytes (int values)
        byte[] vectorBytes = is.readNBytes(dimension * 4);
        if (vectorBytes.length != dimension * 4) {
          break; // End of file
        }

        ByteBuffer bb = ByteBuffer.wrap(vectorBytes).order(ByteOrder.LITTLE_ENDIAN);
        if (filePath.endsWith("fbin")) {
          for (int i = 0; i < dimension; i++) {
            frow[i] = bb.getFloat();
          }
          fts.add(frow.clone());
        } else if (filePath.endsWith("ibin")) {
          for (int i = 0; i < dimension; i++) {
            irow[i] = bb.getInt();
          }
          its.add(irow.clone());
        }
        count++;
        if (count % (int) (numRows / 4) == 0) {
          log.log(Level.INFO, "Read " + count + " items");
        }
      }
      log.log(Level.INFO, "Done! Read " + count + " out of a total of " + items + " items");
    } catch (Exception e) {
      throw e;
    }
  }
}
