{ lib
, stdenv
, accelstore
, pkg-config
, which
, python3
, redis-src
, redisVersion ? "8.9.241"
}:

stdenv.mkDerivation {
  pname = "redis-cvm-store";
  version = redisVersion;

  src = redis-src;

  nativeBuildInputs = [ pkg-config which python3 ];

  buildInputs = [ ];

  makeFlags = [
    "PREFIX=${placeholder "out"}"
    "ACCELSTORE=yes"
    "ACCELSTORE_LIBDIR=${accelstore}/lib"
    "ACCELSTORE_INCDIR=${accelstore}/include"
  ];

  enableParallelBuilding = true;

  preBuild = ''
    if [ ! -e ${accelstore}/lib/libpyaccelstore.so ]; then
      echo "redis-cvm-store: ${accelstore} carries no lib/libpyaccelstore.so;" >&2
      echo "  the AccelStore build must install the bridge (CMakeLists.txt)." >&2
      exit 1
    fi
    if [ ! -e ${accelstore}/include/pyas_shim.h ]; then
      echo "redis-cvm-store: ${accelstore} carries no include/pyas_shim.h." >&2
      exit 1
    fi
  '';

  doCheck = false;

  postInstall = ''
    if ! $out/bin/redis-server --version | grep -q .; then
      echo "redis-cvm-store: installed redis-server does not run" >&2
      exit 1
    fi
  '';

  meta = with lib; {
    description = "Redis with an AccelStore (SPDK) persistence backend";
    homepage = "https://github.com/TUM-DSE/redis-CVM-Store";
    license = licenses.agpl3Only;
    platforms = [ "x86_64-linux" ];
    mainProgram = "redis-server";
  };
}
