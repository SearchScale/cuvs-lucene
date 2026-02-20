/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class PerfTests extends LuceneTestCase {

  private static final Logger log = Logger.getLogger(PerfTests.class.getName());

  @BeforeClass
  public static void beforeClass() throws Exception {
    // Setup
    List<float[]> c = new ArrayList<float[]>();
    DatasetUtils.readDataFile("test-dataset/base.1M.fbin", 100, null, c);
  }

  @Test
  public void myTest() {
    log.log(Level.INFO, "my Test!");
  }

  @AfterClass
  public static void afterClass() {
    // Cleanup
  }
}
