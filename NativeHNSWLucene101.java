import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
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

public class NativeHNSWLucene101 {

  public static Random random = new Random(42);

  public static void main(String[] args) throws IOException {

    // Use standard Lucene101Codec instead of CuVSCodec
    Codec codec = new Lucene101Codec();
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false); // Disable CFS to see individual index files

    // Create index with pure Lucene101 codec
    try (Directory dir = FSDirectory.open(Paths.get("myindex_lucene101"));
        IndexWriter w = new IndexWriter(dir, config)) {
      int numDocs = 100;
      int dimension = 10;
      int numIndexed = 0;
      for (int i = 0; i < numDocs; i++) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("knn1", randomVector(dimension), EUCLIDEAN));
        doc.add(new KnnFloatVectorField("knn2", randomVector(dimension), EUCLIDEAN));
        numIndexed++;

        doc.add(new StringField("id", Integer.toString(i), Field.Store.YES));
        w.addDocument(doc);
      }
      w.forceMerge(1);

      // Verify the index can be read
      try (DirectoryReader reader = DirectoryReader.open(w)) {

        System.out.println("=== Pure Lucene101 Index Created Successfully ===");
        for (String file : reader.directory().listAll()) {
          System.out.println(file + ": " + reader.directory().fileLength(file));
        }

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
        
        System.out.println("✓ Index created with Lucene101Codec and is readable!");
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