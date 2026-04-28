package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.CuVSResourcesInfo;
import com.nvidia.cuvs.GPUInfoProvider;
import com.nvidia.cuvs.spi.CuVSProvider;
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

  public CuvsResourcesManager() {
    log.log(Level.INFO, "Initializing CuvsResourcesManager");
    pool = new ManagedCuVSResources[RESOURCES_POOL_SIZE];
    lock = new ReentrantLock();
    resourcesAvailable = lock.newCondition();
    for (int i = 0; i < RESOURCES_POOL_SIZE; i++) {
      pool[i] = new ManagedCuVSResources(getCuVSResourceInstance());
    }
  }

  private ManagedCuVSResources getAvailableResourcesFromPool() {
    log.log(Level.INFO, "getAvailableResourcesFromPool");
    for (int i = 0; i < RESOURCES_POOL_SIZE; i++) {
      if (pool[i] != null && !pool[i].isLocked()) {
        return pool[i];
      }
    }
    return null;
  }

  private int getNumLockedResources() {
    log.log(Level.INFO, "getNumLockedResources");
    int res = 0;
    for (int i = 0; i < RESOURCES_POOL_SIZE; i++) {
      if (pool[i].isLocked()) {
        res += 1;
      }
    }
    return res;
  }

  public ManagedCuVSResources acquireResource(long rows, long dimension)
      throws InterruptedException {
    log.log(Level.INFO, "acquireResource");
    try {
      lock.lock();

      if (getNumLockedResources() == RESOURCES_POOL_SIZE) {
        resourcesAvailable.await();
      }

      ManagedCuVSResources res = getAvailableResourcesFromPool();
      assert res != null;

      CuVSResourcesInfo info = GPU_INFO_PROVIDER.getCurrentInfo(res.getResource());
      long totalMem = info.totalDeviceMemoryInBytes();
      long freeMem = info.freeDeviceMemoryInBytes();
      long neededMem = getEstimatedMemoryRequirement(rows, dimension);

      if (neededMem > totalMem) {
        throw new RuntimeException("Not enough GPU device memory available");
      }

      // Wait for enough device memory to become available
      while (neededMem > freeMem) {
        freeMem = info.freeDeviceMemoryInBytes();
      }
      res.lock();
      return res;

    } finally {
      lock.unlock();
    }
  }

  public void releaseResource(ManagedCuVSResources resource) {
    log.log(Level.INFO, "releaseResource");
    try {
      lock.lock();
      resource.unlock();
      resourcesAvailable.signalAll();
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

  private long getEstimatedMemoryRequirement(long rows, long dimension) {
    return 2 * rows * dimension * Float.BYTES;
  }

  class ManagedCuVSResources {

    private final CuVSResources resource;
    private final ReentrantLock lock;

    public ManagedCuVSResources(CuVSResources resource) {
      this.resource = resource;
      lock = new ReentrantLock();
    }

    public CuVSResources getResource() {
      return resource;
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
