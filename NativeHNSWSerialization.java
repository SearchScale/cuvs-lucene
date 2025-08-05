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
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import org.apache.lucene.codecs.Codec;
import com.nvidia.cuvs.lucene.CuVSCodec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class NativeHNSWSerialization {

  public static Random random = new Random(42);

  public static void main(String[] args) throws IOException {

    Codec codec = new CuVSCodec();
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false); // Disable CFS to see individual index files

    // Override to only test float vectors
    try (Directory dir = FSDirectory.open(Paths.get("helloindex"));
        IndexWriter w = new IndexWriter(dir, config)) {
      int numDocs = 100;
      int dimension = 10;
      int numIndexed = 0;
      for (int i = 0; i < numDocs; i++) {
        Document doc = new Document();
        var vecs = randomVector(dimension);
        doc.add(new KnnFloatVectorField("knn1", vecs, EUCLIDEAN));
        doc.add(new KnnFloatVectorField("knn2", vecs, EUCLIDEAN));
        numIndexed++;

        doc.add(new StringField("id", Integer.toString(i), Field.Store.YES));
        w.addDocument(doc);
      }
      w.forceMerge(1);

      // TODO Write an assert for a KNN query
      try (DirectoryReader reader = DirectoryReader.open(w)) {

        // System.out.println("Files in this directory: " +
        // Arrays.toString(reader.directory().listAll()).replaceAll(" ", "\n"));
        for (String file : reader.directory().listAll()) {
          System.out.println(file + ": " + reader.directory().fileLength(file));
        }

        /*IndexInput ip = reader.directory().openInput("_0_CuVSVectorsFormat_0.vemf", IOContext.READONCE);
        byte[] b = new byte[(int)ip.length()];
        ip.readBytes(b, 0, (int)ip.length());
        System.out.println("Contents: " + new String(b));*/

        LeafReader r = getOnlyLeafReader(reader);
        FloatVectorValues values1 = r.getFloatVectorValues("knn1");
        FloatVectorValues values2 = r.getFloatVectorValues("knn2");

        if (values1 != null) {
          System.out.println(
              "knn1 - Size: " + values1.size() + ", Dimension: " + values1.dimension());
        }
        if (values2 != null) {
          System.out.println(
              "knn2 - Size: " + values2.size() + ", Dimension: " + values2.dimension());
        }
      }
    }
  }

  public static LeafReader getOnlyLeafReader(IndexReader reader) {
    List<LeafReaderContext> subReaders = reader.leaves();
    if (subReaders.size() != 1) {
      throw new IllegalArgumentException(
          reader + " has " + subReaders.size() + " segments instead of exactly one");
    }
    return subReaders.get(0).reader();
  }

  public static float[] randomVector(int dim) {
    assert dim > 0;
    float[] v = new float[dim];
    double squareSum = 0.0;
    // keep generating until we don't get a zero-length vector
    while (squareSum == 0.0) {
      squareSum = 0.0;
      for (int i = 0; i < dim; i++) {
        v[i] = random.nextFloat();
        squareSum += v[i] * v[i];
      }
    }
    return v;
  }
}
