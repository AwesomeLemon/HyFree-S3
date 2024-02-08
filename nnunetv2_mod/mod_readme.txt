Cloned nnunet, commit hash: f19c5e4

Modified for my purposes:
- eval initialization
- linear warmup
- preprocessor that stores stats
- HD95
- do not run actual validation

Diff from my attempt to fix multiprocessing (I did not copy these changes):
+++ /export/scratch1/home/aleksand/miniconda3/envs/autoshare/lib/python3.10/site-packages/nnunetv2/inference/data_iterators.py	2024-01-12 21:44:19.431139466 +0100
@@ -1,4 +1,4 @@
-import multiprocessing
+import torch.multiprocessing as multiprocessing
 import queue
 from torch.multiprocessing import Event, Process, Queue, Manager

@@ -114,7 +114,16 @@
         if pin_memory:
             [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
         yield item
-    [p.join() for p in processes]
+    # [p.join() for p in processes]
+    # abort_event.set()  # Signal all subprocesses to terminate
+    # sleep(3)
+    for p in processes:
+        p.join(timeout=5)  # Wait for 5 seconds for natural termination
+        # if p.is_alive():
+        #     p.terminate()  # Forcefully terminate if still alive
+    # print('Joined processes! A')
+    # sleep(3)
+