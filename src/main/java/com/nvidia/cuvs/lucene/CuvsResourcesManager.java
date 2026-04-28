package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo.IVF_PQ;
import static com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CuVSIvfPqIndexParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.GPUInfoProvider;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;

public class CuvsResourcesManager {

  private static final Logger log = Logger.getLogger(Utils.class.getName());
  private static final int RESOURCES_POOL_SIZE = 5;
  private static final CuVSProvider PROVIDER = CuVSProvider.provider();
  private static final GPUInfoProvider GPU_INFO_PROVIDER = PROVIDER.gpuInfoProvider();

  private ManagedCuVSResources[] pool;
  private ReentrantLock lock;
  private Condition resourcesAvailable;
  private AtomicLong reserveMemory;
  private long totalDeviceMemory;

  public CuvsResourcesManager() {
    pool = new ManagedCuVSResources[RESOURCES_POOL_SIZE];
    lock = new ReentrantLock();
    resourcesAvailable = lock.newCondition();
    for (int i = 0; i < RESOURCES_POOL_SIZE; i++) {
      pool[i] = new ManagedCuVSResources(getCuVSResourceInstance());
    }
    CuVSResources rx = getCuVSResourceInstance();
    reserveMemory = new AtomicLong();
    totalDeviceMemory = GPU_INFO_PROVIDER.getCurrentInfo(rx).totalDeviceMemoryInBytes();
    rx.close();
  }

  private ManagedCuVSResources getAvailableResourcesFromPool() {
    try {
      lock.lock();
      for (int i = 0; i < RESOURCES_POOL_SIZE; i++) {
        if (pool[i] != null && !pool[i].isLocked()) {
          log.log(Level.INFO, "returning resource id: " + i);
          return pool[i];
        }
      }
    } finally {
      lock.unlock();
    }
    return null;
  }

  private int getNumLockedResources() {
    try {
      lock.lock();
      int res = 0;
      for (int i = 0; i < RESOURCES_POOL_SIZE; i++) {
        if (pool[i].isLocked()) {
          res += 1;
        }
      }
      return res;
    } finally {
      lock.unlock();
    }
  }

  public ManagedCuVSResources acquireResource(long rows, long dimension, CagraIndexParams params)
      throws InterruptedException {
    try {
      lock.lock();
      long neededMem = getEstimatedMemoryRequirement(rows, dimension, params);

      if (neededMem > totalDeviceMemory) {
        throw new RuntimeException("Not enough GPU device memory available");
      }

      while (getNumLockedResources() == RESOURCES_POOL_SIZE
          || (totalDeviceMemory - reserveMemory.get()) < neededMem) {
        resourcesAvailable.await();
      }
      reserveMemory.addAndGet(neededMem);

      ManagedCuVSResources res = getAvailableResourcesFromPool();
      assert res != null;

      res.setNeededMemory(neededMem);
      res.lock();
      return res;
    } finally {
      lock.unlock();
    }
  }

  public void releaseResource(ManagedCuVSResources resource) {
    try {
      lock.lock();
      reserveMemory.addAndGet(-resource.getNeededMemory());
      resource.unlock();
      resourcesAvailable.signalAll();
    } finally {
      lock.unlock();
    }
  }

  public void shutdown() {
    try {
      lock.lock();
      for (int i = 0; i < RESOURCES_POOL_SIZE; i++) {
        if (pool[i] != null && !pool[i].isLocked() && pool[i].getResource() != null) {
          pool[i].getResource().close();
        }
      }
    } finally {
      lock.unlock();
    }
  }

  private static CuVSResources getCuVSResourceInstance() {
    try {
      return CuVSResources.create();
    } catch (UnsupportedOperationException uoe) {
      log.log(
          Level.WARNING,
          "cuVS is not supported on this platform or java version: " + uoe.getMessage());
    } catch (Throwable t) {
      if (t instanceof ExceptionInInitializerError ex) {
        t = ex.getCause();
      }
      log.log(Level.WARNING, "Exception occurred during creation of cuVS resources. " + t);
    }
    return null;
  }

  private long getEstimatedMemoryRequirement(long rows, long dimension, CagraIndexParams params) {
    CagraGraphBuildAlgo buildAlgo = params.getCagraGraphBuildAlgo();
    if (buildAlgo.equals(NN_DESCENT)) {
      return 2 * rows * dimension * Float.BYTES;
    } else if (buildAlgo.equals(IVF_PQ)) {
      CuVSIvfPqIndexParams ip = params.getCuVSIvfPqParams().getIndexParams();
      long approximatedIvfBytes =
          (long)
              (rows * (ip.getPqDim() * (ip.getPqBits() / 8.0) + Float.BYTES)
                  + (long) ip.getnLists() * Integer.BYTES);
      return 2 * approximatedIvfBytes;
    } else {
      throw new IllegalArgumentException("Unsupported CAGRA build algo");
    }
  }

  class ManagedCuVSResources {

    private final CuVSResources resource;
    private final ReentrantLock lock;
    private long neededMemory;

    public ManagedCuVSResources(CuVSResources resource) {
      this.resource = resource;
      lock = new ReentrantLock();
    }

    public CuVSResources getResource() {
      return resource;
    }

    public long getNeededMemory() {
      return neededMemory;
    }

    public void setNeededMemory(long neededMemory) {
      this.neededMemory = neededMemory;
    }

    public void lock() {
      lock.lock();
    }

    public void unlock() {
      lock.unlock();
    }

    public boolean isLocked() {
      return lock.isLocked();
    }
  }
}
