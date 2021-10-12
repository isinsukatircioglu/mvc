# Perspective Crop Layer

Two versions are available:

 * pcl.py implements the main functionality, hardcoded for taking the target position and intrinsics in pytorch format (position in range -1..1 and scale 0..1)
 * pcl_complex.py offers functionality to use pixel coordinates or normalized image coordinates (position from 0..1) as input. It it was used for debugging.
 
These two versions can be tested with the respective demo scripts:

 * pcl_demo.py
 * pcl_complex_demo.py
 
 Note, only pcl_demo.py has code for displaying the 3D pose (but should be equivalent for both)