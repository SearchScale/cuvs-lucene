package com.nvidia.cuvs.lucene;

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
    log.log(Level.INFO, "Initializing CuvsResourcesManager " + Thread.currentThread().getName());
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
      log.log(Level.INFO, "getAvailableResourcesFromPool ");
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
      log.log(Level.INFO, "getNumLockedResources");
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

  public ManagedCuVSResources acquireResource(long rows, long dimension)
      throws InterruptedException {
    log.log(Level.INFO, "acquireResource");
    try {
      lock.lock();
      long neededMem = getEstimatedMemoryRequirement(rows, dimension);

      if (neededMem > totalDeviceMemory) {
        throw new RuntimeException("Not enough GPU device memory available");
      }

      long avm = totalDeviceMemory - reserveMemory.get();
      while (getNumLockedResources() == RESOURCES_POOL_SIZE || avm < neededMem) {
        avm = totalDeviceMemory - reserveMemory.get();
        resourcesAvailable.await();
      }
      reserveMemory.addAndGet(neededMem);

      ManagedCuVSResources res = getAvailableResourcesFromPool();
      assert res != null : "Should not be reaching here.";

      res.setNeededMemory(neededMem);
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
      reserveMemory.addAndGet(-resource.getNeededMemory());
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
