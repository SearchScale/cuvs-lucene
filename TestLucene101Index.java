import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;

public class TestLucene101Index {
    private static final Random random = new Random(42);
    
    public static void main(String[] args) throws IOException {
        System.out.println("=== Testing Index with Lucene101 Codec ===");
        
        // Explicitly set the codec to Lucene101Codec
        Codec codec = new Lucene101Codec();
        System.out.println("Using codec: " + codec.getName());
        
        try (Directory dir = FSDirectory.open(Paths.get("myindex_lucene101"))) {
            System.out.println("\n1. Opening index with Lucene101Codec...");
            
            // Set the codec in IndexWriterConfig (even though we're only reading)
            //IndexWriterConfig config = new IndexWriterConfig().setCodec(codec);
            
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                System.out.println("✓ Successfully opened index");
                
                // List all files in the index
                System.out.println("\nIndex files:");
                for (String file : dir.listAll()) {
                    System.out.println("  " + file + " (" + dir.fileLength(file) + " bytes)");
                }
                
                // Get codec information
                LeafReader leafReader = getOnlyLeafReader(reader);
                System.out.println("\nSegment info: " + leafReader.getMetaData().toString());
                
                // Check vector fields
                FloatVectorValues knn1Values = leafReader.getFloatVectorValues("knn1");
                FloatVectorValues knn2Values = leafReader.getFloatVectorValues("knn2");
                
                if (knn1Values != null) {
                    System.out.println("✓ knn1 field: " + knn1Values.size() + " vectors, " + knn1Values.dimension() + " dimensions");
                } else {
                    System.out.println("✗ knn1 field: not found");
                }
                
                if (knn2Values != null) {
                    System.out.println("✓ knn2 field: " + knn2Values.size() + " vectors, " + knn2Values.dimension() + " dimensions");
                } else {
                    System.out.println("✗ knn2 field: not found");
                }
                
                // Test 2: Perform vector search queries
                System.out.println("\n2. Testing vector search queries...");
                testVectorSearch(reader);
                
            } catch (Exception e) {
                System.out.println("✗ Failed to open index: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
    
    private static void testVectorSearch(IndexReader reader) throws IOException {
        IndexSearcher searcher = new IndexSearcher(reader);
        
        // Create a random query vector (same dimension as indexed vectors)
        float[] queryVector = randomVector(10);
        System.out.println("Query vector: " + java.util.Arrays.toString(queryVector));
        
        // Test search on knn1 field
        try {
            KnnFloatVectorQuery query1 = new KnnFloatVectorQuery("knn1", queryVector, 5);
            TopDocs results1 = searcher.search(query1, 5);
            
            System.out.println("\n✓ knn1 search results (" + results1.totalHits + " total hits):");
            for (int i = 0; i < results1.scoreDocs.length; i++) {
                ScoreDoc scoreDoc = results1.scoreDocs[i];
                Document doc = searcher.storedFields().document(scoreDoc.doc);
                System.out.println("  Rank " + (i+1) + ": doc " + scoreDoc.doc + 
                                 " (id=" + doc.get("id") + "), score=" + scoreDoc.score);
            }
            
            if (results1.scoreDocs.length > 0) {
                System.out.println("✓ Vector search on field 'knn1' successful!");
            }
        } catch (Exception e) {
            System.out.println("✗ knn1 search failed: " + e.getMessage());
            e.printStackTrace();
        }
        
        // Test search on knn2 field
        try {
            KnnFloatVectorQuery query2 = new KnnFloatVectorQuery("knn2", queryVector, 5);
            TopDocs results2 = searcher.search(query2, 5);
            
            System.out.println("\n✓ knn2 search results (" + results2.totalHits + " total hits):");
            for (int i = 0; i < results2.scoreDocs.length; i++) {
                ScoreDoc scoreDoc = results2.scoreDocs[i];
                Document doc = searcher.storedFields().document(scoreDoc.doc);
                System.out.println("  Rank " + (i+1) + ": doc " + scoreDoc.doc + 
                                 " (id=" + doc.get("id") + "), score=" + scoreDoc.score);
            }
            
            if (results2.scoreDocs.length > 0) {
                System.out.println("✓ Vector search on field 'knn2' successful!");
            }
        } catch (Exception e) {
            System.out.println("✗ knn2 search failed: " + e.getMessage());
            e.printStackTrace();
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