results of the indexes prediction using the best unet model for all the possible analyzed methods.

---------the "average on distance" threshold method was used here (see my thesis, methods chapter).-------------------------------------

each folder is named either:

filter: use of a kalman filter
nofilter: no kalman
distance: tapse indexes are calculated as the maximum distance that a point travels throughout the cardiac cicle
projection: tapses are calculated by fitting a line and the distance is then calculated using the point+s projections on those lines.
min, max, mean: the minimum, maximum or mean value of the index across the heartbeat is taken.
spline: a spline is fitted on the three points to calculate the areas (DRVA, SRVA, RVFAC). 

see methods of my thesis for other info.

best combination should take the best value out of every other method.
best_rvfac overestimates RVDA so that the bias is near 0 (because RVSA is also always a little overestimated)
best_rvlsffw does basically the same for rvlsffw.

Each folder contains:

best_unet.xlsx: the values predicted by the unet
bland_altman_stats.xlsx: values of bias, loa but also correlation indexes for each index predicted
per_patient_errors: well, autoexplicative I would say.


Bland_altman_summary.xlsx contains the stats divided by index and method. Useful for comparison.