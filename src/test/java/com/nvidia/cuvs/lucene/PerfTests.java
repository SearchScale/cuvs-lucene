/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.DatasetUtils.readDataFile;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import com.sun.management.OperatingSystemMXBean;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import tools.jackson.databind.ObjectMapper;

@SuppressSysoutChecks(bugUrl = "")
public class PerfTests extends LuceneTestCase {

  private static final Logger log = Logger.getLogger(PerfTests.class.getName());
  private static final int NUM_VECTORS = 1_000_000;
  private static final int NUM_QUERIES = 10_000;
  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector_field";
  private static IndexWriterConfig config;
  private static Codec codec;
  private static List<float[]> dataset;
  private static List<float[]> queries;
  private static List<int[]> neighbors;
  private static Path indexDirPath;
  private static int numThreads = 1;
  private static Gauges gauges;
  private static Map<String, Object> reports;

  static {
    reports = new LinkedHashMap<String, Object>();
  }

  public PerfTests(@Name("codec") Codec codec) {
    PerfTests.codec = codec;
  }

  @ParametersFactory
  public static List<Object[]> parameters() throws Exception {
    return Arrays.asList(
        new Object[][] {{new Lucene101AcceleratedHNSWCodec()}
          //          {new LuceneAcceleratedHNSWBinaryQuantizedCodec()},
          //          {new LuceneAcceleratedHNSWScalarQuantizedCodec()},
          //          {new CuVS2510GPUSearchCodec()}
        });
  }

  @BeforeClass
  public static void beforeClass() throws Exception {
    log.log(Level.INFO, "Starting perf tests ...");
    reports.put("Hardware", Gauges.getHardwareInformation());
    dataset = new ArrayList<float[]>();

    queries = new ArrayList<float[]>();
    neighbors = new ArrayList<int[]>();
    readDataFile("test-dataset/base.1M.fbin", NUM_VECTORS, null, dataset);
    readDataFile("test-dataset/queries.fbin", NUM_QUERIES, null, queries);
    readDataFile("test-dataset/groundtruth.1M.neighbors.ibin", NUM_QUERIES, neighbors, null);
    Map<String, Object> datasetMap = new LinkedHashMap<String, Object>();
    datasetMap.put("Dataset", "Wikipedia 10Mx768");
    datasetMap.put("vectors", dataset.size());
    datasetMap.put("dimensions", dataset.isEmpty() ? 0 : dataset.get(0).length);
    datasetMap.put("queries", queries.size());
    reports.put("Dataset", datasetMap);
  }

  @SuppressWarnings("unchecked")
  @Before
  public void beforeEach() throws IOException {
    String testName = getTestName().split("\\{")[0].trim();
    log.log(Level.INFO, "Running test: " + testName);
    if (!reports.containsKey(testName)) {
      Map<String, Object> testMap = new LinkedHashMap<String, Object>();
      reports.put(testName, testMap);
    }
    Map<String, Object> testMap = (Map<String, Object>) reports.get(testName);
    Map<String, Object> testCodecMap = new LinkedHashMap<String, Object>();
    testMap.put(codec.getName(), testCodecMap);

    indexDirPath = Paths.get(UUID.randomUUID().toString());
    config = new IndexWriterConfig().setCodec(codec);
    config.setMaxBufferedDocs(NUM_VECTORS);
    config.setRAMBufferSizeMB(32767);
    gauges = new Gauges();
    gauges.start();
  }

  @SuppressWarnings("unchecked")
  @After
  public void afterEach() throws IOException {
    gauges.stop();
    String testName = getTestName().split("\\{")[0].trim();
    Map<String, Object> testMap = (Map<String, Object>) reports.get(testName);
    Map<String, Object> testCodecMap = (Map<String, Object>) testMap.get(codec.getName());
    testCodecMap.put("Metrics", gauges.getMetrics());
    File indexDirPathFile = indexDirPath.toFile();
    if (indexDirPathFile.exists() && indexDirPathFile.isDirectory()) {
      FileUtils.deleteDirectory(indexDirPathFile);
    }
  }

  @SuppressWarnings("unchecked")
  @Test
  public void gaugeIndexBuildTime() throws IOException, InterruptedException {
    StopWatch sw = StopWatch.createStarted();
    AtomicInteger id = new AtomicInteger(1);
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        IndexWriter writer = new IndexWriter(indexDirectory, config);
        ExecutorService pool = Executors.newFixedThreadPool(numThreads)) {
      for (int pi = 0; pi < numThreads; pi++) {
        pool.submit(
            () -> {
              while (id.getAndIncrement() < 10000) {
                try {
                  Document document = new Document();
                  document.add(
                      new StringField(ID_FIELD, Integer.toString(id.get()), Field.Store.YES));
                  document.add(
                      new KnnFloatVectorField(VECTOR_FIELD, dataset.get(id.get() - 1), EUCLIDEAN));
                  writer.addDocument(document);
                } catch (IOException ex) {
                  throw new UncheckedIOException(ex);
                }
              }
            });
      }
      pool.shutdown();
      pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);

      log.log(Level.INFO, "Committing documents");
      writer.commit();
    }
    sw.stop();

    String testName = getTestName().split("\\{")[0].trim();
    Map<String, Object> testMap = (Map<String, Object>) reports.get(testName);
    Map<String, Object> testCodecMap = (Map<String, Object>) testMap.get(codec.getName());
    Map<String, Object> testInfoMap = new LinkedHashMap<String, Object>();
    testInfoMap.put("Num Documents", id.get() - 1);
    testInfoMap.put("Index Build Time [ms]", sw.getTime());
    testCodecMap.put("Details", testInfoMap);

    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
      log.log(Level.INFO, "Number of segments: " + reader.leaves().size());
      testInfoMap.put("Num Segments", reader.leaves().size());
    }
    log.log(Level.INFO, "Index build time is: " + sw.getTime(TimeUnit.MILLISECONDS) + " ms");
  }

  @AfterClass
  public static void afterClass() {
    // Pretty-print JSON report
    ObjectMapper mapper = new ObjectMapper();
    String ppJ = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(reports);
    log.log(Level.INFO, ppJ);
  }

  private class Gauges {

    private static final int TIME_INTERVAL_MS = 10;
    private static final long BYTES_IN_MEGABYTE = 1024L * 1024L;
    private ExecutorService executor;
    private boolean running;
    private Map<String, Object> metrics;
    private Callable<Map<String, Object>> task =
        () -> {
          MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
          OperatingSystemMXBean osBean =
              ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

          while (running) {
            String[] gpum =
                runProc(
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.total,memory.used",
                        "--format=csv,noheader,nounits")
                    .split(",");
            MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
            MemoryUsage nonHeapUsage = memoryBean.getNonHeapMemoryUsage();
            Double cpuSystemLoad = osBean.getCpuLoad() * 100;
            Double cpuProcessLoad = osBean.getProcessCpuLoad() * 100;
            Double gpuLoad = Double.valueOf(gpum[0].trim());
            Double gpuAvailableMemory = Double.valueOf(gpum[1].trim());
            Double gpuMemory = Double.valueOf(gpum[2].trim());
            Double gpuMemUtilization = (gpuMemory / gpuAvailableMemory) * 100;
            putMetric(metrics, "CPU_SYSTEM_LOAD", cpuSystemLoad);
            putMetric(metrics, "CPU_PROCESS_LOAD", cpuProcessLoad);
            putMetric(metrics, "GPU_LOAD", gpuLoad);
            putMetric(metrics, "GPU_MEMORY", gpuMemory);
            putMetric(metrics, "GPU_MEMORY_UTILIZATION", gpuMemUtilization);
            putMetric(metrics, "HEAP_MEMORY", heapUsage.getUsed() / BYTES_IN_MEGABYTE);
            putMetric(metrics, "NON_HEAP_MEMORY", nonHeapUsage.getUsed() / BYTES_IN_MEGABYTE);
            Thread.sleep(TIME_INTERVAL_MS);
          }
          return null;
        };

    private void putMetric(Map<String, Object> m, String key, double value) {
      if (!m.containsKey("INIT_" + key)) {
        m.put("INIT_" + key, value);
        m.put("MAX_" + key, value);
      } else {
        double cscl = (double) m.get("MAX_" + key);
        m.put("MAX_" + key, Math.max(cscl, value));
      }
    }

    public static Map<String, String> getHardwareInformation() throws IOException {
      List<String> cpuInfo = Files.readAllLines(Paths.get("/proc/cpuinfo"));
      Map<String, String> hwm = new LinkedHashMap<String, String>();
      if (!cpuInfo.isEmpty()) {
        for (String info : cpuInfo) {
          if (info.contains("model name") || info.contains("siblings")) {
            String[] inf = info.split(":");
            hwm.put("CPU " + inf[0].trim(), inf[1].trim());
          }
          if (hwm.size() == 2) { // Just get the CPU model and thread count
            break;
          }
        }
      }
      String op =
          runProc(
              "nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits");
      String[] sp = op.split(",");
      hwm.put("GPU model name", sp[0].trim());
      hwm.put("GPU memory [MB]", sp[1].trim());
      return hwm;
    }

    private static String runProc(String... command) throws IOException {
      ProcessBuilder processBuilder = new ProcessBuilder(command);
      Process process = processBuilder.start();
      BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
      String line;
      StringBuilder output = new StringBuilder();
      while ((line = reader.readLine()) != null) {
        output.append(line).append("\n");
      }
      reader.close();
      return output.toString();
    }

    public void start() {
      executor = Executors.newSingleThreadExecutor();
      metrics = new LinkedHashMap<String, Object>();
      running = true;
      executor.submit(task);
    }

    public void stop() {
      running = false;
      executor.shutdown();
    }

    public Map<String, Object> getMetrics() {
      return metrics;
    }
  }
}
