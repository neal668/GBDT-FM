# GBDT-FM
An implementation of GBDT+FM 

The implementation of FM (factorization Machines) comes from Zhengruifeng (https://github.com/zhengruifeng/spark-libFM)

In this implementation, GBDT is used for feature transformation and FM optimize those 0-1 features for CTR prediction with SGD.

# args for input
args（0）： HDFS file
args（1）： repeat time of algorithm e.g. 5
args（2）： number of trees of GBDT e.g. 30
args（3）： number of iteration of FM e.g. 20
args（4）： step size of FM e.g. 0.1
other parameters in GBDT e.g. depth of tree and that of in FM e.g. bias, regularization can be set mannually.
