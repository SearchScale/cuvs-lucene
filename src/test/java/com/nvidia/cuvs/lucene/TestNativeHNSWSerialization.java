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

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.util.Arrays;

import com.nvidia.cuvs.lucene.CuVSVectorsWriter.IndexType;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.BeforeClass;

@SuppressSysoutChecks(bugUrl = "Ignore the sysout")
public class TestNativeHNSWSerialization extends BaseKnnVectorsFormatTestCase {

  @BeforeClass
  public static void beforeClass() {
    assumeTrue("cuvs is not supported", CuVSVectorsFormat.supported());
  }

  @Override
  protected Codec getCodec() {
    return TestUtil.alwaysKnnVectorsFormat(
        new CuVSVectorsFormat(
            CuVSVectorsFormat.DEFAULT_WRITER_THREADS,
                CuVSVectorsFormat.DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
            CuVSVectorsFormat.DEFAULT_GRAPH_DEGREE, IndexType.HNSW_LUCENE));
	  //return TestUtil.getDefaultCodec();
  }

  @Override
  public void testSparseVectors() throws Exception {
    // Override to only test float vectors
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      int numDocs = atLeast(100);
      int dimension = atLeast(10);
      int numIndexed = 0;
      for (int i = 0; i < numDocs; i++) {
        Document doc = new Document();
        if (random().nextInt(4) == 3) {
          doc.add(new KnnFloatVectorField("knn1", randomVector(dimension), EUCLIDEAN));
          doc.add(new KnnFloatVectorField("knn2", randomVector(dimension), EUCLIDEAN));
          numIndexed++;
        }
        doc.add(new StringField("id", Integer.toString(i), Field.Store.YES));
        w.addDocument(doc);
      }
      w.forceMerge(1);

      // TODO Write an assert for a KNN query
      try (DirectoryReader reader = DirectoryReader.open(w)) {
    	  
    	//System.out.println("Files in this directory: " + Arrays.toString(reader.directory().listAll()).replaceAll(" ", "\n"));
    	for (String file: reader.directory().listAll()) {
    		System.out.println(file + ": " + reader.directory().fileLength(file));
    	}

    	/*IndexInput ip = reader.directory().openInput("_0_CuVSVectorsFormat_0.vemf", IOContext.READONCE);
    	byte[] b = new byte[(int)ip.length()];
    	ip.readBytes(b, 0, (int)ip.length());
    	System.out.println("Contents: " + new String(b));*/
    	
        LeafReader r = getOnlyLeafReader(reader);
        FloatVectorValues values = r.getFloatVectorValues("knn");

        if (numIndexed == 0) {
          assertNull(values);
        } else {
          assertNotNull(values);
          assertEquals(numIndexed, values.size());
          assertEquals(dimension, values.dimension());
        }
      }
    }
  }

  // Override byte vector tests to skip them since we only support float32
  @Override
  public void testRandomBytes() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testSortedIndexBytes() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testMergingWithDifferentByteKnnFields() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testVectorValuesReportCorrectDocs() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testMismatchedFields() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testRandomExceptions() throws Exception {
    // Skip - this test uses byte vectors which are not supported
  }

  @Override
  public void testByteVectorScorerIteration() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testEmptyByteVectorData() throws Exception {
    // Skip - CuVS only supports float32 vectors
  }

  @Override
  public void testCheckIntegrityReadsAllBytes() throws Exception {
    // Skip - may use byte vectors
  }
}
