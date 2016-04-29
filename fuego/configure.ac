dnl Process this file with autoconf to produce a configure script.

AC_INIT([fuego], [1.1.SVN])
AC_PREREQ([2.59])
AC_CONFIG_SRCDIR([go/GoBoard.cpp])
AC_CONFIG_HEADERS(config.h)
AC_OPENMP
AC_PROG_CC
AC_PROG_LIBTOOL
AM_PROG_CC_C_O
AM_INIT_AUTOMAKE
AM_MAINTAINER_MODE
AC_PROG_CXX
AC_PROG_RANLIB
AX_BOOST_BASE([1.50.0])
AX_BOOST_THREAD
AX_BOOST_SYSTEM
AX_BOOST_DATE_TIME
AX_BOOST_FILESYSTEM
AX_BOOST_PROGRAM_OPTIONS
AX_BOOST_UNIT_TEST_FRAMEWORK

####################
# Setup default CUDA paths
CUDA_INSTALL_PATH="/usr/local/depot/cuda-6.5"
CUDA_SDK_INSTALL_PATH="/usr/local/depot/cuda-6.5"

# Check: enable CUDA?
AC_MSG_CHECKING([whether to use CUDA library])
AC_ARG_ENABLE(cuda,
    [AS_HELP_STRING(--enable-cuda,enable CUDA support)],
    cudaval="yes",
    cudaval="no")
AC_MSG_RESULT([$cudaval])

# check to see if CUDA been selected
if test "$cudaval" == "no"; then
    AC_MSG_WARN([Not using CUDA!])
fi

# Setup custom CUDA paths
AC_ARG_WITH([cuda],
   [AS_HELP_STRING(
        [--with-cuda=PATH],
        [prefix where CUDA is installed @<:@default=auto@:>@])],
   [CUDA_INSTALL_PATH=$withval],
   [with_cuda=auto])
AC_ARG_WITH([cuda-sdk],
   [AS_HELP_STRING(
        [--with-cuda-sdk=PATH],
        [prefix where CUDA SDK is installed @<:@default=auto@:>@])],
   [CUDA_SDK_INSTALL_PATH=$withval],
   [with_cuda_sdk=auto])

# Setup nvcc flags
if test  "$cudaval" = "yes"; then
    NVCCFLAGS="$NVCCFLAGS"
    CUDA_CFLAGS="$CUDA_CFLAGS"
    CUDA_CFLAGS="$CUDA_CFLAGS -I$CUDA_SDK_INSTALL_PATH/C/common/inc/"
    CUDA_CFLAGS="$CUDA_CFLAGS -I$CUDA_INSTALL_PATH/include"
    CUDA_LDFLAGS="-L$CUDA_INSTALL_PATH/lib64"
    CUDA_LIBS="-lcuda -lcudart -lcufft"
    NVCCFLAGS="$NVCCFLAGS $CUDA_CFLAGS $CUDA_LDFLAGS $CUDA_LIBS"
    NVCC="nvcc"
    AC_SUBST(NVCC)
    AC_SUBST(NVCCFLAGS)
    AC_DEFINE([_CUDA],[1],[Defined if CUDA should be used])
    eval "ARRAY${N}='-D_CUDA'"
    N=`expr $N + 1`
fi

##################

AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUDA_LDFLAGS)
AC_SUBST(NVCC)
AC_OUTPUT


AC_ARG_ENABLE(max-size,
  [  --enable-max-size=n     define maximum board size (2..25, default 19)])

AH_TEMPLATE([SG_DEFINE_MAX_SIZE],
[Maximum board size for class SgPoint and related classes (2..25, default 19)])
if test "$enable_max_size" ; then
   AC_DEFINE_UNQUOTED(SG_DEFINE_MAX_SIZE, $enable_max_size)
fi


dnl Check for __sync_synchronize
dnl
AC_MSG_CHECKING([for __sync_synchronize])
AC_LINK_IFELSE([AC_LANG_SOURCE([int main (int argc, char** argv) {__sync_synchronize();}])],
	       [has_synchronize=yes],
	       [has_synchronize=no])
AC_MSG_RESULT([$has_synchronize])
		
if test "x$has_synchronize" = "xyes"
then
	AC_DEFINE(HAVE_SYNC_SYNCHRONIZE,1,"define if __sync_synchronize() is available")
fi


AC_ARG_ENABLE([assert],
	      AS_HELP_STRING([--enable-assert], [enable assertions (default is no)]),
	      [assert=$enableval],
	      [assert=no])
if test "x$assert" = "xno"
then
	CXXFLAGS="$CXXFLAGS -DNDEBUG"
else
	CXXFLAGS="$CXXFLAGS"
fi


dnl ./configure switch to enable explicit synchronization of CPU caches.
dnl
AC_ARG_ENABLE([cache-sync],
	      AS_HELP_STRING([--enable-cache-sync],
	      [Enable explicit synchronization of the CPU cache.
	      Required for most non-Intel architectures to support
	      lock-free multi-threading (default is no)]),
	      [cachesync=$enableval],
	      [cachesync=no])

if test "x$cachesync" = "xyes"
then
	if test "x$has_synchronize" = "xyes"
	then 
		AC_DEFINE(ENABLE_CACHE_SYNC, 1, [define to enable explicit synchronization of the CPU cache])
	else
		AC_MSG_ERROR(Explicit cache synchronization requires __sync_synchronize() builtin);	   fi

fi

AC_ARG_ENABLE(uct-value-type,
  [  --enable-uct-value-type=t  floating point type used in SgUctSearch (float|double)])
AH_TEMPLATE([SG_UCT_VALUE_TYPE],
[Floating point type used for counts and values in SgUctSearch (float|double)])
if test "$enable_uct_value_type" ; then
   AC_DEFINE_UNQUOTED(SG_UCT_VALUE_TYPE, $enable_uct_value_type)
fi

AC_CANONICAL_HOST
AC_SUBST(host_cpu)
AC_DEFINE_UNQUOTED(HOST_CPU, "$host_cpu",
  [Define the canonical host CPU type.]
)

AC_CHECK_HEADERS([sys/sysctl.h])
AX_CXXFLAGS_WARN_ALL
AX_CXXFLAGS_GCC_OPTION(-Wextra)

AC_OUTPUT([Makefile book/Makefile regression/Makefile misctests/Makefile fuegomain/Makefile fuegotest/Makefile go/Makefile gouct/Makefile gtpengine/Makefile features/Makefile simpleplayers/Makefile smartgame/Makefile unittestmain/Makefile])
