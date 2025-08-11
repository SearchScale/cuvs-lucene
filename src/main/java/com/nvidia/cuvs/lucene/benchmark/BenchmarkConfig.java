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

/**
 * Configuration class for CuVS-Lucene benchmarks
 * Compatible with vectorsearch-benchmarks JSON format
 */
public class BenchmarkConfig {

  public String benchmarkID;
  public String datasetFile;
  public int indexOfVector;
  public String vectorColName;
  public int numDocs;
  public int vectorDimension;
  public String queryFile;
  public Integer queryIndexOfVector; // Column index for query vectors (null = use indexOfVector)
  public int numQueriesToRun;
  public int flushFreq;
  public int topK;
  public int numIndexThreads;
  public int cuvsWriterThreads;
  public int queryThreads;
  public boolean createIndexInMemory;
  public boolean cleanIndexDirectory;
  public boolean saveResultsOnDisk;
  public boolean hasColNames;
  public String algoToRun;
  public String groundTruthFile;
  public String cuvsIndexDirPath;
  public String hnswIndexDirPath;
  public boolean loadVectorsInMemory;
  public String description;

  // HNSW parameters
  public int hnswMaxConn = 32; // default
  public int hnswBeamWidth = 256; // default

  // CAGRA parameters
  public int cagraIntermediateGraphDegree = 128; // default
  public int cagraGraphDegree = 64; // default
  public int cagraITopK = 64;
  public int cagraSearchWidth = 64;

  public void debugPrintArguments() {
    System.out.println("📋 Benchmark Configuration:");
    System.out.println("  - Benchmark ID: " + benchmarkID);
    System.out.println("  - Dataset file: " + datasetFile);
    System.out.println("  - Index of vector field: " + indexOfVector);
    System.out.println("  - Vector field name: " + vectorColName);
    System.out.println("  - Number of documents: " + numDocs);
    System.out.println("  - Vector dimensions: " + vectorDimension);
    System.out.println("  - Query file: " + queryFile);
    System.out.println("  - Query index of vector field: " + queryIndexOfVector);
    System.out.println("  - Number of queries: " + numQueriesToRun);
    System.out.println("  - Flush frequency: " + flushFreq);
    System.out.println("  - Top-K: " + topK);
    System.out.println("  - Index threads: " + numIndexThreads);
    System.out.println("  - Query threads: " + queryThreads);
    System.out.println("  - Create in memory: " + createIndexInMemory);
    System.out.println("  - Clean index directory: " + cleanIndexDirectory);
    System.out.println("  - Save results: " + saveResultsOnDisk);
    System.out.println("  - Has column names: " + hasColNames);
    System.out.println("  - Algorithm: " + algoToRun);
    System.out.println("  - Ground truth file: " + groundTruthFile);
    System.out.println("  - Load vectors in memory: " + loadVectorsInMemory);
    if (description != null) {
      System.out.println("  - Description: " + description);
    }

    System.out.println("🔧 Algorithm Parameters:");
    System.out.println("  - HNSW Max Connections: " + hnswMaxConn);
    System.out.println("  - HNSW Beam Width: " + hnswBeamWidth);
    System.out.println("  - CAGRA Intermediate Graph Degree: " + cagraIntermediateGraphDegree);
    System.out.println("  - CAGRA Graph Degree: " + cagraGraphDegree);
    System.out.println("  - CAGRA I-Top-K: " + cagraITopK);
    System.out.println("  - CAGRA Search Width: " + cagraSearchWidth);
  }
}
