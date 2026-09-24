{ pkgs, accelstorePackages, redis-cvm-store }:

accelstorePackages ++ [ redis-cvm-store ]
