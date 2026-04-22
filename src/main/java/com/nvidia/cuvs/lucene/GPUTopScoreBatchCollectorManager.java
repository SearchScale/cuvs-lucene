package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.Collection;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.search.TopScoreDocCollectorManager;

public class GPUTopScoreBatchCollectorManager extends TopScoreDocCollectorManager {

  public GPUTopScoreBatchCollectorManager(int numHits, ScoreDoc after, int totalHitsThreshold) {
    super(numHits, after, totalHitsThreshold);
  }

  public GPUTopScoreBatchCollectorManager(int numHits, int totalHitsThreshold) {
    super(numHits, totalHitsThreshold);
  }

  @Override
  public TopDocs reduce(Collection<TopScoreDocCollector> collectors) throws IOException {
    return super.reduce(collectors);
  }
}
