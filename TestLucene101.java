import org.apache.lucene.codecs.lucene101.*;

public class TestLucene101 {
    public static void main(String[] args) {
        System.out.println("Lucene101 classes available");
        try {
            Class.forName("org.apache.lucene.codecs.lucene101.Lucene101FlatVectorsFormat");
            System.out.println("Lucene101FlatVectorsFormat exists");
        } catch (ClassNotFoundException e) {
            System.out.println("Lucene101FlatVectorsFormat does not exist");
        }
        
        try {
            Class.forName("org.apache.lucene.codecs.lucene101.Lucene101HnswVectorsFormat");
            System.out.println("Lucene101HnswVectorsFormat exists");
        } catch (ClassNotFoundException e) {
            System.out.println("Lucene101HnswVectorsFormat does not exist");
        }
    }
}