.cu.o:
 $(NVCC) -gencode=arch=compute_13,code=sm_13 -o $@ -c $<
.cu.lo:
 $(top_srcdir)/cudalt.py $@ $(NVCC) \
 -gencode=arch=compute_13,code=sm_13 \
 --compiler-options=\"$(CFLAGS) \
 $(DEFAULT_INCLUDES) $(INCLUDES) \
 $(AM_CPPFLAGS) $(CPPFLAGS) \" -c $<